#!/usr/bin/env python3
"""PureLogic — a terminal agent TUI on a local LLM.

A terminal agent interface pre-wired to a local LLM (llama.cpp, OpenAI-compatible),
driving the GVS5H manager–worker scaffold (bundled in codebase/v2-current/escalation;
paper & repo: https://github.com/slee-persis/GVS5H).

Modes
  harness (default)  your prompt runs the full agent loop: the manager plans, an
                     ideation pass proposes approaches, then manager→worker stages
                     work over a shared workspace until the problem is solved (or
                     the stage budget runs out), then finalize. Workers have a
                     write_file tool, so files are written directly into the
                     workspace — large artifacts never get truncated in a reply.
                     A live activity panel shows the current phase, stage budget,
                     task checklist, thinking/generating token rates (tok/s) and a
                     tail of what the model is writing right now.
  chat               streaming chat REPL against the model (thinking shown live in
                     dim italics while it streams).

Working directory
  Artifacts are written to a `Workspace` subfolder of the directory you launch
  from (override with --workspace-dir or GVS5H_WS_DIR). Each run gets its own
  subfolder with task.md, plan.md, tasks.json, notes.md, answer.md / solution.py,
  any files the agent wrote, and transcript.jsonl (every model call).

Screenshots
  Set PURELOGIC_SHOT_DIR=<dir> and the TUI renders the live activity panel to
  <dir>/activity.png (while the model is working, preferably while it is writing
  a file) and the final result panel to <dir>/result.png (110x48 truecolor).

File-writing calls
  Worker/finalize calls that get the write_file tool use a higher output cap
  (--file-cap / GVS5H_FILE_CAP, default 20480 — the endpoint accepts up to
  20480). Thinking is OFF for these calls by default (--file-think off): this
  endpoint does not reliably honor chat_template_kwargs.thinking_budget, and an
  unbounded deep-thinking pass can consume the entire output budget and leave
  no room for the file (observed: ~17k tokens of reasoning, empty answer).
  With thinking off the whole cap goes to file content; --file-think budget
  restores a capped think (GVS5H_THINK_BUDGET, default 3000 tokens).
  If a write_file call is still cut off mid-file (finish_reason=length with
  truncated tool JSON), the harness salvages the partial content, writes it,
  and tells the model to continue with mode='append'.

Everything is pre-wired via defaults below and overridable with CLI flags or env
vars (GVS5H_* are kept for scaffold compatibility).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ESCALATION_DIR = REPO_ROOT / "codebase" / "v2-current" / "escalation"

# --- prewire defaults (CLI > env > these) -----------------------------------
DEFAULT_BASE = "http://192.168.1.69:8080/v1"
DEFAULT_MODEL = ("/home/user/.local/share/localllm-qwen38/models/hauhau/"
                 "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf")
DEFAULT_STAGES = 6
DEFAULT_CAP = 8192
DEFAULT_FILE_CAP = 20480   # worker/finalize calls: the whole cap goes to file content
DEFAULT_TEMP = 0.3

# when True, do_harness() also prints scaffold progress lines to stderr
# (set by --run one-shot mode; interactive mode keeps stderr quiet)
_LOG_STDERR = False

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
ANSWER_CHARS = 6000          # cap for rendered answer panels
MAX_TOOL_ROUNDS = 32         # tool-call rounds per worker call (file can span many writes)
MAX_RETRIES = 40             # infra retries per model call
CHARS_PER_TOKEN = 3.6        # rough live estimate while streaming (exact usage arrives at call end)
TAIL_CHARS = 104             # live "what is it writing" tail shown in the panel (one line)


def spin(t=None):
    return SPINNER_FRAMES[int((t if t is not None else time.time()) * 10) % len(SPINNER_FRAMES)]


def fmt_elapsed(s):
    s = max(0, int(s))
    m, sec = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m:02d}:{sec:02d}"


def fmt_k(n):
    n = n or 0
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:,}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PureLogic — terminal agent TUI")
    p.add_argument("--base", default=os.environ.get("GVS5H_BASE", DEFAULT_BASE),
                   help="OpenAI-compatible base URL (default: prewired local llama.cpp)")
    p.add_argument("--model", default=os.environ.get("GVS5H_MODEL", DEFAULT_MODEL),
                   help="model id as served by the endpoint")
    p.add_argument("--mode", choices=["harness", "chat"], default="harness",
                   help="mode (default: harness = full agent loop)")
    p.add_argument("--run", metavar="PROMPT", default=None,
                   help="run one prompt non-interactively and exit")
    p.add_argument("--spec", choices=["general", "code", "math"], default="general",
                   help="harness task spec (default: general)")
    p.add_argument("--stages", "--iters", dest="stages", type=int, default=None,
                   help="manager->worker stage budget (default 6)")
    p.add_argument("--cap", type=int, default=None,
                   help="max output tokens per model call (default 8192)")
    p.add_argument("--file-cap", type=int, default=None,
                   help="max output tokens for worker/finalize calls that write files "
                        "(default 20480, so the whole budget goes to file content)")
    p.add_argument("--file-think", choices=["off", "budget"], default="off",
                   help="thinking on file-writing calls: off (default, full cap for "
                        "content) or budget (cap it with GVS5H_THINK_BUDGET)")
    p.add_argument("--temp", type=float, default=None, help="temperature")
    p.add_argument("--workspace-dir", default=os.environ.get("GVS5H_WS_DIR", None),
                   help="root for per-run artifact folders (default: ./Workspace)")
    p.add_argument("--no-think", action="store_true",
                   help="disable the model's thinking (chat_template_kwargs.enable_thinking=false)")
    p.add_argument("--no-stream", action="store_true", help="--run chat: print only the final answer")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="--run: print a JSON result object")
    return p.parse_args(argv)


class Cfg:
    def __init__(self, args):
        self.base = args.base.rstrip("/")
        self.model = args.model
        self.stages = args.stages or int(os.environ.get("GVS5H_ITERS", DEFAULT_STAGES))
        self.cap = args.cap or int(os.environ.get("GVS5H_CAP", DEFAULT_CAP))
        self.file_cap = args.file_cap or int(os.environ.get("GVS5H_FILE_CAP", DEFAULT_FILE_CAP))
        self.think_budget = int(os.environ.get("GVS5H_THINK_BUDGET", 3000))
        self.file_think = args.file_think
        self.temp = float(os.environ["GVS5H_TEMP"]) if os.environ.get("GVS5H_TEMP") else (
            args.temp if args.temp is not None else DEFAULT_TEMP)
        self.no_think = args.no_think
        self.mode = args.mode
        self.spec = args.spec
        ws = Path(args.workspace_dir) if args.workspace_dir else Path.cwd() / "Workspace"
        self.ws_dir = ws.resolve()
        self.ws_dir.mkdir(parents=True, exist_ok=True)


def apply_env(cfg):
    """Set the scaffold's env vars BEFORE it is imported (they are read at import time)."""
    os.environ.setdefault("ESCALATION_LOCAL_BASE", cfg.base + "/chat/completions")
    os.environ.setdefault("ESCALATION_LOCAL_KEY", "local")
    os.environ.setdefault("MULTIAGENT_MODEL", "local:" + cfg.model)
    os.environ.setdefault("MULTIAGENT_MAX_ITERS", str(cfg.stages))
    os.environ.setdefault("MULTIAGENT_STRICT_FORMAT", "1")
    os.environ.setdefault("MULTIAGENT_WS", str(cfg.ws_dir))
    os.environ.setdefault("ESCALATION_CLOUD_TIMEOUT", "3600")  # local generations are slow
    os.environ.setdefault("ESCALATION_TIMEOUT", "3600")
    os.environ.setdefault("ESCALATION_CLOUD_MAX_TOKENS", str(cfg.cap))
    if cfg.no_think:
        os.environ.setdefault("ESCALATION_LOCAL_EXTRA",
                              json.dumps({"chat_template_kwargs": {"enable_thinking": False}}))


def import_scaffold(cfg):
    apply_env(cfg)
    sys.path.insert(0, str(ESCALATION_DIR))
    import orchestrator  # noqa: E402
    import multiagent  # noqa: E402
    return orchestrator, multiagent


GENERAL_SPEC = {
    "kind": "general",
    "solver_system": (
        "You are an expert problem-solver. Think carefully and reason step by step, then "
        "give the complete answer. If the task asks you to PRODUCE an artifact (a program, "
        "an HTML page, a document, data, ...), put the COMPLETE artifact in your answer "
        "(full file contents, nothing elided), in a single fenced code block when it is "
        "code. On the FINAL line of your answer, output exactly "
        "'ANSWER: <a concise description of the result>'."
    ),
    "critic_system": (
        "You are a meticulous reviewer. Given a problem and a candidate answer, check the "
        "reasoning and the final answer for errors (including elided or broken artifacts). "
        "If fully correct, reply with exactly 'APPROVED' on the first line; otherwise reply "
        "'REJECTED' on the first line followed by the specific errors."
    ),
}
SPECS = {"general": GENERAL_SPEC}  # code/math come from the scaffold once imported


def local_extra(cfg):
    extra = {}
    if cfg.no_think:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    return extra or None


# =============================================================================
#  The model layer: streaming calls + write_file tool loop (replaces the
#  scaffold's chat so the TUI sees every call live and workers can write files)
# =============================================================================

TOOL_DEFS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": ("Write text content to a file in the shared workspace. Use "
                        "mode='write' to create/replace a file, mode='append' to add to "
                        "the end. Call it repeatedly (write, then append) to build a file "
                        "larger than one reply."),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Relative file path inside the workspace, e.g. 'page.html' or 'src/main.py'."},
                "content": {"type": "string", "description": "The exact text to write."},
                "mode": {"type": "string", "enum": ["write", "append"],
                         "description": "'write' creates/replaces, 'append' adds to the end (default 'write')."},
            },
            "required": ["path", "content"],
        },
    },
}]

TOOL_NOTE_GENERAL = """

FILE TOOL — you have a tool: write_file(path, content, mode="write"|"append").
When the task is to PRODUCE A FILE (an HTML page, a program, a document, a dataset, ...),
write that file with the tool INSTEAD of pasting it in your reply:
- Use mode="write" for the first chunk and mode="append" for each following chunk,
  continuing until the file is complete.
- Each call can carry at most ~15,000 characters of content: size your chunks so
  one call never runs long, and keep going until the file is complete.
- The file must be COMPLETE and SELF-CONTAINED: never elide, abbreviate, or write
  "(rest of file omitted)".
- Keep your reply sections concise afterwards: in ANSWER state the file path and a
  one-line result, still ending with the final 'ANSWER: ...' line.
"""

TOOL_NOTE_CODE = """

FILE TOOL — you have a tool: write_file(path, content, mode="write"|"append").
Write the complete program to solution.py using the tool (mode="write" first, then
mode="append" for the rest) INSTEAD of pasting it in your reply. Keep each call to
at most ~15,000 characters of content. The program in solution.py must be COMPLETE
and SELF-CONTAINED: never elide or abbreviate it.
In the CODE section put only a one-line pointer, e.g. '(complete program written to
solution.py)', and keep NOTES concise.
"""


class LiveState:
    """Shared live state: written by the model-call threads, read by the panel.

    The panel renders ~8x/s; dict/attr reads are cheap and the GIL keeps them
    consistent enough for a status display."""

    def __init__(self):
        self.kind = "general"
        self.lock = threading.Lock()
        self.role = None          # role of the in-flight call (plan/ideation/worker:1/...)
        self.call_t0 = None       # when the in-flight call started
        self.gen_chars = 0        # content chars generated in the in-flight call
        self.think_chars = 0      # reasoning chars in the in-flight call
        self.tool_chars = 0       # tool-call argument chars in the in-flight call
        self.tool_path = None
        self.tool_mode = "write"
        self.tail = ""            # what the model is writing right now (dim tail)
        self.last_log = ""        # latest scaffold log line
        self.stage = 0            # current manager→worker stage (worker id)
        self.files = []           # [(relpath, bytes)] files written by the agent this run
        self.retries = 0          # infra retries across the run
        self.inflight = 0         # scaffold calls currently in flight

    def begin_call(self, role):
        with self.lock:
            self.role = role
            self.call_t0 = time.time()
            self.gen_chars = 0
            self.think_chars = 0
            self.tool_chars = 0
            self.tool_path = None
            self.tool_mode = "write"
            self.tail = ""
            m = re.match(r"worker:(\d+)", role)
            if m:
                self.stage = max(self.stage, int(m.group(1)))

    def note_delta(self, kind, chunk):
        if not chunk:
            return
        with self.lock:
            if kind == "reasoning":
                self.think_chars += len(chunk)
                self.tail = _one_line(self.tail + chunk)
            elif kind == "content":
                self.gen_chars += len(chunk)
                self.tail = _one_line(self.tail + chunk)
            elif kind == "tool":
                self.tool_chars += len(chunk)
                self.tail = _one_line(self.tail + chunk)

    def note_tool_path(self, path, mode):
        with self.lock:
            self.tool_path = path
            if mode:
                self.tool_mode = mode

    def note_file(self, relpath, size):
        with self.lock:
            self.files.append((relpath, size))


TAIL_LIMIT = TAIL_CHARS * 2


def _one_line(s, limit=TAIL_LIMIT):
    return re.sub(r"\s+", " ", s)[-limit:]


# --- one streaming call (with retries) ---------------------------------------

class _InfraError(RuntimeError):
    pass


def _stream_one(cfg, body, live):
    """One streaming request; returns (content, tool_calls, usage, finish, reasoning)."""
    req = urllib.request.Request(
        cfg.base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "purelogic-tui/2.0"})
    content_parts, reasoning_parts = [], []
    tool_calls = {}
    args_buf = {}
    finish = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices") or []:
                d = c.get("delta") or {}
                rd = d.get("reasoning_content")
                if rd:
                    reasoning_parts.append(rd)
                    live.note_delta("reasoning", rd)
                cd = d.get("content")
                if cd:
                    content_parts.append(cd)
                    live.note_delta("content", cd)
                for tc in d.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    t = tool_calls.setdefault(i, {"id": None, "name": None})
                    if tc.get("id"):
                        t["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        t["name"] = fn["name"]
                    a = fn.get("arguments")
                    if a:
                        args_buf[i] = args_buf.get(i, "") + a
                        live.note_delta("tool", a)
                        m = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', args_buf[i])
                        if m and (not live.tool_path
                                  or live.tool_path != _unesc(m.group(1))):
                            live.note_tool_path(_unesc(m.group(1)), None)
                        mm = re.search(r'"mode"\s*:\s*"(append|write)"', args_buf[i])
                        if mm:
                            live.note_tool_path(live.tool_path, mm.group(1))
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    tcs = []
    for i in sorted(tool_calls):
        t = dict(tool_calls[i])
        t["args"] = args_buf.get(i, "")
        tcs.append(t)
    return "".join(content_parts), tcs, (usage or {}), finish, "".join(reasoning_parts)


def _unesc(s):
    try:
        return json.loads(f'"{s}"')
    except ValueError:
        return s


def _stream_with_retry(cfg, messages, tools, temperature, live, cap=None, extra=None):
    body = {
        "model": cfg.model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": cap or cfg.cap,
        "stream_options": {"include_usage": True},
    }
    merged = local_extra(cfg) or {}
    if extra:
        if "chat_template_kwargs" in extra and "chat_template_kwargs" in merged:
            merged["chat_template_kwargs"] = {**merged["chat_template_kwargs"], **extra["chat_template_kwargs"]}
        merged.update(extra)
    if merged:
        body.update(merged)
    if tools:
        body["tools"] = tools
    attempt = 0
    while True:
        attempt += 1
        try:
            return _stream_one(cfg, body, live)
        except urllib.error.HTTPError as e:
            snippet = ""
            try:
                snippet = e.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                live.retries += 1
                live.last_log = f"endpoint error {e.code}; retry {attempt}/{MAX_RETRIES - 1}"
                time.sleep(min(5.0 * attempt, 30.0))
                continue
            raise _InfraError(f"HTTP {e.code} from {cfg.base}: {snippet}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < MAX_RETRIES:
                live.retries += 1
                live.last_log = f"endpoint unreachable ({type(e).__name__}); retry {attempt}/{MAX_RETRIES - 1}"
                time.sleep(min(5.0 * attempt, 30.0))
                continue
            raise _InfraError(f"endpoint unreachable: {e}") from e


# --- the write_file tool -------------------------------------------------------

BOOKKEEPING = {"task.md", "plan.md", "tasks.json", "transcript.jsonl", "notes.md",
               "answer.md", "solution.py"}


def _safe_ws_path(ws, p):
    p = (p or "").strip().strip('"').lstrip("./")
    root = Path(ws).resolve()
    pp = (root / p).resolve()
    if pp != root and root not in pp.parents:
        raise ValueError(f"path escapes the workspace: {p!r}")
    return pp


def _salvage_tool_args(raw):
    """Recover (path, content, closed) from a write_file args JSON truncated by
    the token cap. `closed` is True when the content string was fully generated
    (only the JSON tail was cut). Returns None when nothing usable was written.
    """
    m_path = re.search(r'"path"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if not m_path:
        return None
    rest = raw[m_path.end():]
    m_content = re.search(r'"content"\s*:\s*"', rest)
    if not m_content:
        return None
    body = rest[m_content.end():]
    # walk to the first UNESCAPED closing quote (the value may contain \")
    out, i, closed = [], 0, False
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            if i + 1 < len(body):
                out.append(body[i:i + 2])
                i += 2
                continue
            break  # dangling escape exactly at the cut point
        if ch == '"':
            closed = True
            break
        out.append(ch)
        i += 1
    if closed and "}" in body[i + 1:]:  # object closed: JSON was whole
        return None
    return _unesc(m_path.group(1)), _unesc("".join(out)), closed


def _exec_tool(ws, tc, live):
    name = tc.get("name")
    raw = tc.get("args") or "{}"
    try:
        args = json.loads(raw)
    except ValueError:
        salv = _salvage_tool_args(raw)
        if salv is None:
            return ("ERROR: could not parse tool arguments as JSON (the call was "
                    "cut off before a usable file chunk). Reply with a smaller "
                    "write_file chunk (under ~15,000 characters) and try again.")
        path, content, closed = salv
        if len(content) < 200:
            return ("ERROR: the write_file call was cut off after only "
                    f"{len(content)} characters of content. Start over with a "
                    "smaller first chunk (under ~15,000 characters).")
        m_mode = re.search(r'"mode"\s*:\s*"(append|write)"', raw)
        mode = m_mode.group(1) if m_mode else "write"
        try:
            pp = _safe_ws_path(ws, path)
            pp.parent.mkdir(parents=True, exist_ok=True)
            with open(pp, "w" if mode == "write" else "a", encoding="utf-8") as f:
                f.write(content)
            rel = pp.relative_to(Path(ws).resolve()).as_posix()
            size = pp.stat().st_size
            live.note_file(rel, size)
            live.last_log = (f"[write_file salvaged] {mode} {rel} "
                             f"(+{len(content):,} chars, {size:,} bytes)")
            if closed:
                return (f"WROTE {size:,} chars to {rel} (mode={mode}); the call was "
                        f"truncated after the file content was complete, so the "
                        f"file should be whole — verify it and finish with your "
                        f"ANSWER line.")
            return (f"PARTIAL: wrote {size:,} chars to {rel} (mode={mode}); the "
                    f"call was cut off at the token limit mid-file. Continue with "
                    f"write_file mode='append', starting exactly where the text "
                    f"stopped — do not repeat anything already in the file.")
        except Exception as e:  # noqa: BLE001
            return f"ERROR: could not write salvaged partial file: {type(e).__name__}: {e}"
    if name != "write_file":
        return f"ERROR: unknown tool {name!r}. Only write_file is available."
    path = args.get("path")
    content = args.get("content")
    mode = (args.get("mode") or "write")
    if not isinstance(path, str) or not isinstance(content, str):
        return "ERROR: write_file needs string 'path' and 'content'."
    if mode not in ("write", "append"):
        return "ERROR: mode must be 'write' or 'append'."
    try:
        pp = _safe_ws_path(ws, path)
        pp.parent.mkdir(parents=True, exist_ok=True)
        with open(pp, "w" if mode == "write" else "a", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: could not write file: {type(e).__name__}: {e}"
    rel = pp.relative_to(Path(ws).resolve()).as_posix()
    size = pp.stat().st_size
    live.note_file(rel, size)
    live.last_log = f"[write_file] {mode} {rel} (+{len(content):,} chars, {size:,} bytes)"
    verb = "wrote" if mode == "write" else "appended"
    return f"OK: {verb} {len(content):,} chars to {rel} (file is now {size:,} bytes)."


def _ws_mtimes(ws):
    out = {}
    for f in Path(ws).iterdir():
        if f.is_file():
            out[f.name] = f.stat().st_mtime_ns
    return out


def _ws_artifact_advanced(ws, before, kind):
    """Did the tool layer produce/advance the real artifact file during this worker call?"""
    after = _ws_mtimes(ws)
    if kind == "code":
        return "solution.py" in after and after["solution.py"] > before.get("solution.py", 0)
    return any(n not in before or after[n] > before[n]
               for n in after if n not in BOOKKEEPING)


# --- replacement for the scaffold's _chat (records the transcript exactly as
#     the original did, but every call streams and workers get the tool loop) ---

def install_model_layer(cfg, live, multiagent):
    def _chat2(ws, role, messages, temperature, meta=None):
        m = meta if meta is not None else {}
        resp = tui_model_call(cfg, live, ws, role, messages, temperature, m)
        multiagent._record(ws, {
            "t": time.time(), "role": role, "request": messages, "response": resp,
            "reasoning": m.get("reasoning"), "thinking_blocks": m.get("thinking_blocks"),
            "reasoning_is_summary": m.get("reasoning_is_summary"),
            "finish_reason": m.get("finish_reason"),
            "completion_tokens": m.get("completion_tokens"),
            "prompt_tokens": m.get("prompt_tokens"), "provider": m.get("provider"),
            "attempts": m.get("attempts"), "discarded": m.get("discarded"),
            "infra_exhausted": m.get("infra_exhausted")})
        return resp

    def _worker2(problem, spec, ws, task, log, finalize=False):
        before = _ws_mtimes(ws)
        status, nexts, summary, wrote = _orig_worker(problem, spec, ws, task, log, finalize)
        if not wrote:
            wrote = _ws_artifact_advanced(ws, before, spec["kind"])
        return status, nexts, summary, wrote

    _orig_worker = multiagent._worker
    multiagent._chat = _chat2
    multiagent._worker = _worker2


def tui_model_call(cfg, live, ws, role, messages, temperature, meta):
    """A model call for the scaffold. Worker/finalize calls get the write_file tool."""
    is_file_role = role == "finalize" or role.startswith("worker:")
    tools = None
    msgs = list(messages)
    if is_file_role:
        tools = TOOL_DEFS
        note = {"code": TOOL_NOTE_CODE}.get(live.kind, TOOL_NOTE_GENERAL)
        if live.kind == "math":
            note = ""
        if note:
            first = dict(messages[0])
            first["content"] = str(first.get("content", "")) + note
            msgs = [first] + list(messages[1:])
    live.begin_call(role)
    live.inflight += 1
    try:
        return _tui_model_call_inner(cfg, live, ws, role, msgs, temperature, tools,
                                     is_file_role, meta)
    finally:
        live.inflight = max(0, live.inflight - 1)


def _tui_model_call_inner(cfg, live, ws, role, msgs, temperature, tools, is_file_role,
                          meta):
    cap = max(cfg.cap, cfg.file_cap) if (is_file_role and tools) else cfg.cap
    extra = None
    if is_file_role and tools and not cfg.no_think:
        if cfg.file_think == "budget" and cfg.think_budget > 0:
            extra = {"chat_template_kwargs": {"thinking_budget": cfg.think_budget}}
        else:
            # This endpoint does not reliably honor thinking_budget: on the
            # uncensored Qwen3.8 build the model was observed thinking ~17k
            # tokens and truncating with an EMPTY answer (no file at all).
            # Executor roles write directly; planners keep their thinking.
            extra = {"chat_template_kwargs": {"enable_thinking": False}}
    content = ""
    total_comp = 0
    last_prompt = 0
    reasoning_all = []
    rounds = 0
    while True:
        content, tcs, usage, finish, reasoning = _stream_with_retry(
            cfg, msgs, tools, temperature, live, cap=cap, extra=extra)
        reasoning_all.append(reasoning)
        total_comp += usage.get("completion_tokens") or 0
        last_prompt = usage.get("prompt_tokens") or last_prompt
        if tcs and tools and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            asst = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [{"id": t.get("id") or f"call_{rounds}_{i}", "type": "function",
                                "function": {"name": t.get("name", "write_file"),
                                             "arguments": t.get("args") or "{}"}}
                               for i, t in enumerate(tcs)],
            }
            msgs.append(asst)
            for i, t in enumerate(tcs):
                res = _exec_tool(ws, t, live)
                msgs.append({"role": "tool",
                             "tool_call_id": asst["tool_calls"][i]["id"],
                             "content": res})
            continue
        break
    meta.update({
        "finish_reason": finish,
        "completion_tokens": total_comp,
        "prompt_tokens": last_prompt,
        "reasoning": "".join(reasoning_all),
        "provider": "local",
        "attempts": rounds + 1,
        "discarded": None,
        "infra_exhausted": False,
    })
    return content


# --- transcript / workspace introspection ------------------------------------

def read_transcript(ws):
    path = Path(ws) / "transcript.jsonl"
    recs = []
    if path and Path(path).exists():
        for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                recs.append(json.loads(ln))
            except ValueError:
                pass
    calls = [r for r in recs if not r.get("_meta")]
    return {
        "n_calls": len(calls),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in calls),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in calls),
        "truncated_calls": sum(1 for r in calls if r.get("finish_reason") == "length"),
        "last_role": (calls[-1].get("role") if calls else None),
        "calls": calls,
    }


def read_tasks(ws):
    p = Path(ws) / "tasks.json" if ws else None
    if p and p.exists():
        try:
            return json.loads(p.read_text(errors="replace"))
        except ValueError:
            pass
    return []


ROLE_NAME = {
    "primary_plan": "plan",
    "ideation": "ideation",
    "primary_manage": "manage",
    "cutoff_summary": "compact",
    "finalize": "finalize",
}


def role_label(role):
    if not role:
        return "starting"
    if role.startswith("worker:"):
        return f"worker {role.split(':', 1)[1]}"
    return ROLE_NAME.get(role, role)


def extract_artifact(answer):
    """Fallback: pull a self-contained artifact out of a general-spec answer when the
    agent did not use the write_file tool."""
    if not answer:
        return None
    fences = re.findall(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", answer, re.S)
    exts = {"python": "py", "html": "html", "htm": "html", "js": "js", "javascript": "js",
            "json": "json", "css": "css", "sql": "sql", "bash": "sh", "sh": "sh", "": "txt"}
    best = None
    for lang, body in fences:
        body = body.strip("\n")
        if len(body) >= 200 and (best is None or len(body) > len(best[1])):
            best = ("artifact." + exts.get(lang.lower(), "txt"), body)
    if best:
        return best
    stripped = answer.lstrip()
    if stripped[:3].upper() in ("<!D", "<HT") or stripped.lower().startswith("<html"):
        return ("artifact.html", answer.strip())
    return None


# =============================================================================
#  --run (non-interactive)
# =============================================================================

def run_once(cfg, args, orchestrator, multiagent, live):
    prompt = args.run
    if cfg.mode == "chat":
        messages = [{"role": "system",
                     "content": "You are PureLogic, a capable assistant running on a "
                                "local LLM. Be direct and concrete."},
                    {"role": "user", "content": prompt}]
        live.begin_call("chat")
        t0 = time.time()
        content, _tcs, usage, finish, reasoning = _stream_with_retry(
            cfg, messages, None, cfg.temp, live)
        ok = bool(content.strip())
        if args.as_json:
            print(json.dumps({"mode": "chat", "ok": ok, "answer": content,
                              "reasoning": reasoning, "usage": usage,
                              "finish_reason": finish,
                              "elapsed_s": round(time.time() - t0, 1),
                              "tok_per_s": round((usage.get("completion_tokens") or 0)
                                                 / max(time.time() - t0, 0.001), 1)},
                             indent=2))
            return 0 if ok else 1
        if args.no_stream:
            if reasoning:
                print(f"[thinking] {reasoning.strip()[:400]}", file=sys.stderr)
            print(content)
            return 0 if ok else 1
        print()
        return 0 if ok else 1

    # harness mode — routed through the TUI's live loop, so one-shot runs show
    # the activity panel and capture PURELOGIC_SHOT_DIR screenshots like the
    # interactive mode does
    global _LOG_STDERR
    _LOG_STDERR = True
    live.kind = cfg.spec
    t0 = time.time()
    tui = build_tui(cfg, orchestrator, multiagent, live)
    tui.render_initial()
    tui.do_harness(prompt)
    answer = tui.last_answer
    if tui.last_error:
        print(f"harness failed: {tui.last_error}", file=sys.stderr)
        if args.as_json:
            print(json.dumps({"mode": "harness", "ok": False, "error": tui.last_error}))
        return 1
    ws = tui.last_ws
    stats = read_transcript(ws) if ws else {}
    elapsed = time.time() - t0
    files_written = list(live.files)
    artifact = None
    if cfg.spec == "general" and not files_written and answer and ws:
        artifact = extract_artifact(answer)
        if artifact:
            (Path(ws) / artifact[0]).write_text(artifact[1], encoding="utf-8")
            files_written.append((artifact[0], os.path.getsize(Path(ws) / artifact[0])))
    if args.as_json:
        files = sorted(f.name for f in Path(ws).iterdir() if f.is_file()) if ws else []
        result = {
            "mode": "harness", "spec": cfg.spec,
            "ok": bool(answer and answer.strip()) or bool(files_written),
            "answer": answer, "workspace": ws, "files": files,
            "files_written": [{"path": p, "bytes": b} for p, b in files_written],
            "elapsed_s": round(elapsed, 1),
            "tok_per_s": round((stats.get("completion_tokens") or 0) / max(elapsed, 0.001), 1),
            "n_calls": stats.get("n_calls"),
            "completion_tokens": stats.get("completion_tokens"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "truncated_calls": stats.get("truncated_calls"),
            "finish_reason": tui.last_status.get("finish_reason"),
            "infra_retries": live.retries,
        }
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    print()
    print("=" * 62)
    print(f"RESULT ({cfg.spec}) — {stats.get('n_calls', 0)} calls, "
          f"{stats.get('completion_tokens', 0)} output tokens, {elapsed:.0f}s "
          f"({(stats.get('completion_tokens') or 0) / max(elapsed, 0.001):.1f} tok/s)")
    print("=" * 62)
    print(answer or "(no answer produced)")
    if ws:
        print(f"\nartifacts: {ws}")
        for f in sorted(Path(ws).iterdir()):
            if f.is_file():
                tag = "  <- written by agent" if f.name in [p for p, _ in files_written] else ""
                print(f"  {f.name:<16} {f.stat().st_size:>10,} bytes{tag}")
    return 0 if (answer and answer.strip()) or files_written else 1


# =============================================================================
#  interactive TUI
# =============================================================================

def build_tui(cfg, orchestrator, multiagent, live):
    from rich import box
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.text import Text

    console = Console(highlight=False)
    model_short = cfg.model.rstrip("/").split("/")[-1] or cfg.model
    if len(model_short) > 46:
        model_short = model_short[:22] + " … " + model_short[-22:]

    # --- markup discipline (rich 15: Text() does NOT parse markup) -------------
    def M(s):
        return Text.from_markup(s)

    def P(s, style=""):
        return Text(str(s), style=style)

    def T(*parts):
        out = Text()
        for part in parts:
            if isinstance(part, Text):
                out.append_text(part)
            else:
                out.append(str(part))
        return out

    def bar(frac, width=14):
        frac = max(0.0, min(1.0, frac))
        filled = int(round(width * frac))
        return T(Text("▓" * filled, style="magenta"),
                 Text("░" * (width - filled), style="dim"))

    def task_marks(tasks):
        if not tasks:
            return P("—")
        marks = []
        for t in tasks[:14]:
            st = t.get("status")
            if st == "done":
                marks.append(Text("✓ ", style="green"))
            elif st == "in_progress":
                marks.append(Text("→ ", style="yellow"))
            else:
                marks.append(Text("· ", style="dim"))
        done_n = sum(1 for t in tasks if t.get("status") == "done")
        out = Text()
        for mk in marks:
            out.append_text(mk)
        out.append_text(Text(f"({done_n}/{len(tasks)})", style="dim"))
        return out

    def phase_text():
        role = live.role
        if not role:
            return "starting"
        if role.startswith("worker:") or role == "finalize":
            if live.tool_path:
                suffix = " (append)" if live.tool_mode == "append" else ""
                return f"{role_label(role)} · writing {live.tool_path}{suffix}"
            if live.think_chars > live.gen_chars and live.think_chars > 200:
                return f"{role_label(role)} · thinking"
            return f"{role_label(role)} · answering"
        return {"primary_plan": "manager · planning",
                "ideation": "ideation pass",
                "primary_manage": "manager · reviewing",
                "cutoff_summary": "context compaction",
                "finalize": "finalizing answer"}.get(role, "working")

    def live_stats():
        """Stats for the in-flight call (live estimates until usage arrives)."""
        if not live.call_t0:
            return None
        el = time.time() - live.call_t0
        gen_tok = (live.gen_chars + live.tool_chars) / CHARS_PER_TOKEN
        think_tok = live.think_chars / CHARS_PER_TOKEN
        parts = [P(fmt_elapsed(el), "dim")]
        if gen_tok >= 1:
            parts.append(P(f"  gen {gen_tok:,.0f} tok @ {gen_tok / max(el, 0.1):.1f} t/s"))
        if think_tok >= 1:
            parts.append(P(f"  think {think_tok:,.0f} tok", "dim"))
        return T(*parts)

    class Activity:
        """Renderable for the live panel; reads shared state each refresh."""

        def __init__(self, kind):
            self.kind = kind          # "harness" | "chat"
            self.state = None         # set by the owner

        def _harness(self):
            s = self.state
            el_run = time.time() - s.get("t0", time.time())
            budget = max(1, s.get("stages", 1))
            stage = live.stage
            frac = (stage - 0.5) / budget if 0 < stage <= budget else (0.0 if stage == 0
                                                                       else 1.0)
            ls = live_stats()
            if ls is not None:
                lead = T(M(f"{spin()} "), M(f"[bold]{phase_text()}[/]"), M("  "), ls)
            else:
                lead = T(M(f"{spin()} "), M(f"[bold]{phase_text()}[/]"), M("  "),
                         P(fmt_elapsed(el_run), "dim"),
                         P(f"  {s.get('calls', 0)} calls · {fmt_k(s.get('tokens', 0))} tok out",
                           "dim"))
            row2 = T(M("stage  "), bar(frac),
                     M(f"  {stage}/{budget}   "), M("tasks "), task_marks(s.get("tasks") or []))
            calls_n = s.get("calls", 0) + live.inflight
            row3 = [P(f"run {fmt_elapsed(el_run)}", "dim"),
                    P(f" · {calls_n} calls", "dim")]
            if live.inflight:
                est = (live.gen_chars + live.tool_chars) / CHARS_PER_TOKEN
                row3.append(P(f" ({live.inflight} running · +{fmt_k(int(est))} live)", "cyan"))
            row3.append(P(f" · in {fmt_k(s.get('ptokens', 0))} · out {fmt_k(s.get('tokens', 0))}",
                          "dim"))
            rows = [lead, row2, T(*row3)]
            tail = live.tail if (live.role and (live.gen_chars or live.think_chars
                                                or live.tool_chars)) else ""
            if tail:
                rows.append(P("  " + tail, "dim"))
            if live.last_log:
                rows.append(P("  " + live.last_log[:100], "dim"))
            title = M(f"[bold magenta]AGENT RUN[/] [dim]· {s.get('spec', '')} · "
                      f"{datetime.now():%H:%M:%S}[/]")
            return Panel(Group(*rows), box=box.ROUNDED, border_style="magenta",
                         title=title, padding=(0, 1))

        def _chat(self):
            s = self.state
            el = time.time() - s.get("t0", time.time())
            ls = live_stats()
            if live.think_chars > live.gen_chars and live.think_chars > 0:
                phase = "model thinking…"
            elif live.gen_chars > 0:
                phase = "generating answer…"
            else:
                phase = "waiting for model…"
            lead = T(M(f"{spin()} "), M(f"[bold]{phase}[/]"), M("  "),
                     P(fmt_elapsed(el), "dim"))
            if ls is not None:
                lead.append_text(M("  "))
                lead.append_text(ls)
            rows = [lead]
            if live.tail:
                rows.append(P(live.tail, "dim italic" if live.think_chars > live.gen_chars
                              else ""))
            title = M(f"[bold cyan]CHAT[/] [dim]· {model_short} · {datetime.now():%H:%M:%S}[/]")
            return Panel(Group(*rows), box=box.ROUNDED, border_style="cyan",
                         title=title, padding=(0, 1))

        def __rich_console__(self, c, options):
            if self.kind == "harness":
                yield self._harness()
            else:
                yield self._chat()

    class TUI:
        def __init__(self):
            self.mode = cfg.mode
            self.spec = cfg.spec
            self.history = []
            self.blocks = []
            self.lock = threading.Lock()
            self.quit = False
            self.last_ws = None
            self.last_answer = None
            self.last_error = None
            self.last_status = {}
            self.header_block = None
            self.hint_block = None
            self.last_prompt = ""
            self.shot_dir = os.environ.get("PURELOGIC_SHOT_DIR")
            if self.shot_dir:
                Path(self.shot_dir).mkdir(parents=True, exist_ok=True)

        # ---- screenshots (PURELOGIC_SHOT_DIR set) ----------------------------
        def _shot_lines(self, renderables, width):
            from rich.console import Console as RConsole
            c = RConsole(width=width, force_terminal=True, color_system="truecolor",
                         legacy_windows=False)
            lines, cur = [], []
            for b in renderables:
                if b is None:  # e.g. header/hint blocks before render_initial()
                    continue
                for seg in c.render(b):
                    if seg.is_control:
                        for ch in seg.text:
                            if ch == "\n":
                                lines.append(cur)
                                cur = []
                        continue
                    # rich 15: segments already carry a resolved Style (or None)
                    style = seg.style
                    for ch in seg.text:
                        if ch == "\n":
                            lines.append(cur)
                            cur = []
                        else:
                            cur.append((ch, style))
            lines.append(cur)
            while lines and not any(ch != " " for ch, _ in lines[-1]):
                lines.pop()
            return lines

        def _shot_save(self, name, renderables, width=110, height=48):
            try:
                from PIL import Image, ImageDraw, ImageFont
                lines = self._shot_lines(renderables, width)
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
                except Exception:  # noqa: BLE001
                    font = ImageFont.load_default()
                cw = font.getlength("M")
                lh = int(font.size * 1.38)
                default_fg = (218, 218, 220)
                default_bg = (13, 13, 20)

                def fg_of(style):
                    rgb = None
                    if style is not None and style.color is not None \
                            and not style.color.is_default:
                        try:
                            rgb = style.color.get_truecolor()
                        except Exception:  # noqa: BLE001
                            rgb = None
                    if rgb is None:
                        rgb = default_fg
                    if style is not None:
                        if getattr(style, "bold", False):
                            rgb = tuple(min(255, int(v * 1.35 + 30)) for v in rgb)
                        if getattr(style, "dim", False):
                            rgb = tuple(int(v * 0.6) for v in rgb)
                    return rgb

                img = Image.new("RGB", (int(cw * width), int(lh * height)), default_bg)
                dr = ImageDraw.Draw(img)
                for y, line in enumerate(lines[:height]):
                    for x, (ch, style) in enumerate(line[:width]):
                        if ch == " ":
                            continue
                        dr.text((int(cw * x), int(lh * y)), ch, font=font,
                                fill=fg_of(style))
                img.save(str(Path(self.shot_dir) / name))
                return True
            except Exception as e:  # noqa: BLE001
                print(f"screenshot failed: {e}", file=sys.stderr)
                return False

        def add(self, *renderables):
            with self.lock:
                self.blocks.extend(renderables)
                if len(self.blocks) > 400:
                    self.blocks = self.blocks[-300:]
            for r in renderables:
                console.print(r)   # stream the transcript to the terminal

        # ---- header ----
        def render_header(self):
            mode_style = "bold magenta" if self.mode == "harness" else "bold cyan"
            mode_word = "AGENT · HARNESS" if self.mode == "harness" else "CHAT"
            row1 = T(M("[bold white]◆ PureLogic[/]"), M(" [dim]· terminal agent ·[/] "),
                     M(f"[{mode_style}]{mode_word}[/]"))
            if self.mode == "harness":
                row1.append_text(M(f" [dim]· spec {self.spec}[/]"))
            row1.append(" " * 2)
            row2 = T(M("[dim]model[/] "), P(model_short, "cyan"),
                     M(" [dim]@[/] "), P(cfg.base, "dim"))
            row3 = T(M("[dim]workspace[/] "), P(str(cfg.ws_dir), "bold"))
            return Panel(Group(row1, row2, row3), box=box.DOUBLE, border_style="blue",
                         padding=(0, 1),
                         title=M(f"[dim]{datetime.now():%H:%M:%S}[/]"))

        # ---- chat ----
        def do_chat(self, prompt):
            self.history.append({"role": "user", "content": prompt})
            messages = ([{"role": "system",
                          "content": "You are PureLogic, a capable assistant running on a "
                                     "local LLM. Be direct and concrete."}]
                         + self.history[-40:])
            act = Activity("chat")
            act.state = {"t0": time.time()}
            result = {}
            live.begin_call("chat")

            def worker():
                try:
                    content, _tcs, usage, finish, reasoning = _stream_with_retry(
                        cfg, messages, None, cfg.temp, live)
                    result.update(content=content, reasoning=reasoning, usage=usage,
                                  finish=finish)
                except Exception as e:  # noqa: BLE001
                    result["error"] = f"{type(e).__name__}: {e}"

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            with Live(act, console=console, refresh_per_second=8, transient=True):
                while t.is_alive():
                    time.sleep(0.1)
            if result.get("error"):
                self.add(T(M("[bold red]✗ error:[/] "), P(result["error"])))
                return
            self.history.append({"role": "assistant", "content": result.get("content", "")})
            blocks = []
            reasoning = (result.get("reasoning") or "").strip()
            if reasoning:
                if len(reasoning) > 1200:
                    reasoning = reasoning[:1200] + " …(truncated)"
                blocks.append(T(M("[bold magenta]◆ thinking[/]")))
                blocks.append(P(reasoning, "dim italic"))
            content = result.get("content") or ""
            if content.strip():
                blocks.append(Markdown(content))
            usage = result.get("usage") or {}
            el = time.time() - act.state["t0"]
            if usage.get("completion_tokens"):
                blocks.append(P(f"{usage.get('completion_tokens')} completion tokens · "
                                f"{usage.get('completion_tokens') / max(el, 0.001):.1f} tok/s · "
                                f"{fmt_elapsed(el)}", "dim"))
            self.add(Panel(Group(*blocks) if blocks else P("(empty response)"),
                           box=box.SQUARE, border_style="cyan",
                           title=M(f"[cyan]{model_short}[/]")))

        # ---- harness ----
        def do_harness(self, problem):
            self.add(Rule(style="magenta"))
            live.kind = self.spec
            live.stage = 0
            live.files = []
            live.last_log = ""
            ws = {"v": None}
            status_out = {}
            act = Activity("harness")
            act.state = {"t0": time.time(), "stages": cfg.stages, "calls": 0, "tokens": 0,
                         "ptokens": 0, "tasks": [], "spec": self.spec}
            error = {"v": None}

            def log(msg):
                m = msg.strip()
                live.last_log = m[:110]
                if _LOG_STDERR:
                    print(f"{datetime.now():%H:%M:%S} {m}", file=sys.stderr, flush=True)

            def solve():
                try:
                    answer = multiagent.multiagent_solve(
                        problem, SPECS[self.spec], log=log, status_out=status_out)
                    self.last_answer = answer
                    result = {"v": answer}
                except Exception as e:  # noqa: BLE001
                    result = {"v": None}
                    error["v"] = f"{type(e).__name__}: {e}"
                    self.last_error = error["v"]
                    log(f"ERROR: {error['v']}")
                ws["v"] = status_out.get("ws")

            def poll():
                st = act.state
                w = ws["v"] or status_out.get("ws")
                if w:
                    tr = read_transcript(w)
                    st["calls"] = tr["n_calls"]
                    st["tokens"] = tr["completion_tokens"]
                    st["ptokens"] = tr["prompt_tokens"]
                    st["tasks"] = read_tasks(w)

            def prompt_echo():
                return T(M("[bold cyan]❯[/] "), P(problem))

            shot_tool = False
            last_shot_t = 0.0
            t = threading.Thread(target=solve, daemon=True)
            t.start()
            with Live(act, console=console, refresh_per_second=8, transient=True):
                last_poll = 0.0
                while t.is_alive():
                    now = time.time()
                    if now - last_poll >= 0.8:
                        try:
                            poll()
                        except Exception:  # noqa: BLE001
                            pass
                        last_poll = now
                    # activity screenshot: prefer the model mid-file-write, else the
                    # first call that has produced a visible stream (throttled to 5s)
                    if self.shot_dir and live.role and now - last_shot_t > 5.0:
                        live_chars = live.tool_chars + live.gen_chars + live.think_chars
                        if (live.tool_path and live.tool_chars > 800) or \
                                (not shot_tool and live_chars > 3000):
                            if self._shot_save(
                                    "activity.png",
                                    [self.header_block, self.hint_block,
                                     prompt_echo(), act]):
                                last_shot_t = now
                                if live.tool_path:
                                    shot_tool = True
                    time.sleep(0.15)
                try:
                    poll()
                except Exception:  # noqa: BLE001
                    pass
            self.last_status = status_out
            self.add_harness_result(ws["v"], status_out, time.time() - act.state["t0"],
                                    error["v"], problem=problem)

        def add_harness_result(self, ws, status_out, elapsed, error, problem=None):
            st = read_transcript(ws) if ws else {}
            answer = None
            if ws:
                wsd = Path(ws)
                for name in ("answer.md", "solution.py"):
                    p = wsd / name
                    if p.exists() and p.stat().st_size:
                        answer = p.read_text(encoding="utf-8", errors="replace")
                        break
            if not answer:
                answer = ""
            files_written = list(live.files)
            # fallback artifact extraction (only when the agent used no file writes)
            if self.spec != "code" and not files_written:
                art = extract_artifact(answer)
                if art and ws and not (Path(ws) / art[0]).exists():
                    (Path(ws) / art[0]).write_text(art[1], encoding="utf-8")
                    files_written.append((art[0], os.path.getsize(Path(ws) / art[0])))
            has_answer = bool(answer.strip())
            ok = (error is None) and (has_answer or bool(files_written))
            blocks = []
            meta = T(M(f"[bold]{'✓' if ok else '✗'}[/] "),
                     P(f"{fmt_elapsed(elapsed)} · {st.get('n_calls', 0)} calls · "
                       f"{st.get('completion_tokens', 0):,} tok out @ "
                       f"{(st.get('completion_tokens') or 0) / max(elapsed, 0.001):.1f} t/s · "
                       f"in {fmt_k(st.get('prompt_tokens', 0))} · "
                       f"{st.get('truncated_calls', 0)} truncated · "
                       f"finish={status_out.get('finish_reason', '-')}", "dim"))
            if error:
                blocks.append(T(M("[bold red]✗ failed:[/] "), P(error)))
            blocks.append(meta)
            if answer:
                if self.spec == "code" and ws and (Path(ws) / "solution.py").exists():
                    blocks.append(Syntax(answer, "python", word_wrap=True))
                elif files_written and self.spec != "code" and \
                        any(p.lower().endswith(".html") for p, _ in files_written):
                    html_path, _ = next((p, b) for p, b in files_written
                                        if p.lower().endswith(".html"))
                    html = (Path(ws) / html_path).read_text(encoding="utf-8", errors="replace")
                    blocks.append(Syntax(html[:ANSWER_CHARS * 2], "html", word_wrap=True))
                else:
                    shown = answer
                    if len(shown) > ANSWER_CHARS:
                        shown = shown[:ANSWER_CHARS] + "\n… (truncated)"
                    blocks.append(Markdown(shown))
            elif not error:
                blocks.append(P("no answer produced", "red"))
            if ws:
                if files_written:
                    fw = T(M("[bold]files written by the agent:[/]"))
                    blocks.append(fw)
                    for rel, size in files_written:
                        blocks.append(T(P("  "), M(f"[green]{rel}[/]"),
                                        P(f"  {size:,} bytes", "dim")))
                calls = st.get("calls") or []
                if calls:
                    blocks.append(M("[bold]timeline:[/]"))
                    for c in calls[-12:]:
                        mark = M("[yellow]⚠[/]") if c.get("finish_reason") == "length" \
                            else M("[green]✓[/]")
                        when = time.strftime("%H:%M:%S", time.localtime(c.get("t", 0)))
                        blocks.append(T(P("  "), P(when, "dim"),
                                        M(f"  [bold]{role_label(c.get('role')):<13}[/]"),
                                        P(f" {c.get('completion_tokens', 0):>6,} tok  ", "dim"),
                                        mark))
                blocks.append(Rule(style="dim"))
                blocks.append(T(M("[bold]artifacts →[/] "), P(ws, "bold green")))
                files = T(M("[dim]files:[/] "))
                for f in sorted(Path(ws).iterdir()):
                    if f.is_file():
                        files.append(f" {f.name}")
                        files.append_text(Text(f" {f.stat().st_size:,}B", style="dim"))
                blocks.append(files)
            self.last_ws = ws
            panel = Panel(Group(*blocks), box=box.DOUBLE,
                          border_style="green" if ok else "red",
                          title=M(f"[bold]RESULT[/] · {self.spec} · "
                                  f"{datetime.now():%H:%M:%S}"))
            self.add(panel)
            if self.shot_dir:
                self._shot_save("result.png",
                                [T(M("[bold cyan]❯[/] "), P(problem or self.last_prompt)),
                                 Rule(style="magenta"), panel,
                                 T(M("[bold cyan]❯[/] "))])

        # ---- commands ----
        def handle(self, text):
            t = text.strip()
            if not t:
                return
            if not t.startswith("/"):
                if self.mode == "chat":
                    self.do_chat(t)
                else:
                    self.last_prompt = t
                    self.do_harness(t)
                return
            parts = t[1:].split(maxsplit=1)
            cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
            if cmd in ("quit", "exit", "q"):
                self.quit = True
            elif cmd == "help":
                self.add(Panel(Group(
                    M("[bold]/mode[/] [dim]harness|chat   switch mode (default harness)[/]"),
                    M("[bold]/spec[/] [dim]general|code|math   harness task spec[/]"),
                    M("[bold]/stages[/] [dim]N    manager→worker stage budget (default 6)[/]"),
                    M("[bold]/cap[/] [dim]N     max output tokens per model call (file calls use[/] "
                      "[dim]--file-cap[/] [dim], default 20480)[/]"),
                    M("[bold]/temp[/] [dim]F    temperature[/]"),
                    M("[bold]/think[/] [dim]on|off   model thinking[/]"),
                    M("[bold]/new[/] [dim]clear chat history + redraw header[/]"),
                    M("[bold]/ws[/] [dim]files in the last harness workspace[/]"),
                    M("[bold]/tasks[/] [dim]last harness task list[/]"),
                    M("[bold]/transcript[/] [dim][N]  last harness model calls[/]"),
                    M("[bold]/open[/] [dim]open the last workspace in Explorer[/]"),
                    M("[bold]/save[/] [dim][PATH]  save chat as markdown[/]"),
                    M("[bold]/quit[/]"),
                    M(""),
                    M("[dim]a [bold]stage[/] is one manager→worker round: the manager "
                      "reviews progress, picks the next task, and a worker does it — "
                      "workers can write files with their write_file tool. Repeat until "
                      "the manager says done or the stage budget runs out. "
                      "[bold]tasks[/] is the manager's checklist; a stage usually "
                      "completes one of them.[/]"),
                ), title="commands", border_style="blue"))
            elif cmd == "mode":
                if arg in ("chat", "harness"):
                    self.mode = arg
                    self.add(T(M("mode → "), M(f"[bold]{arg}[/]")))
                else:
                    self.add(T(M("mode: "), M(f"[bold]{self.mode}[/]")))
            elif cmd == "spec":
                if arg in ("general", "code", "math"):
                    self.spec = arg
                    self.add(T(M("harness spec → "), M(f"[bold]{self.spec}[/]")))
            elif cmd in ("stages", "iters"):
                try:
                    cfg.stages = max(1, int(arg or cfg.stages))
                    multiagent.MAX_ITERS = cfg.stages
                    self.add(T(M("stages → "), M(f"[bold]{cfg.stages}[/]")))
                except ValueError:
                    self.add(T(M("[yellow]stages expects an integer[/]")))
            elif cmd == "cap":
                try:
                    cfg.cap = max(16, int(arg or cfg.cap))
                    self.add(T(M("cap → "), M(f"[bold]{cfg.cap}[/]")))
                except ValueError:
                    self.add(T(M("[yellow]cap expects an integer[/]")))
            elif cmd == "temp":
                try:
                    cfg.temp = float(arg or cfg.temp)
                    self.add(T(M("temp → "), M(f"[bold]{cfg.temp}[/]")))
                except ValueError:
                    self.add(T(M("[yellow]temp expects a number[/]")))
            elif cmd == "think":
                if arg in ("on", "off"):
                    cfg.no_think = (arg == "off")
                    self.add(T(M("thinking → "),
                               M(f"[bold]{'off' if cfg.no_think else 'on'}[/]")))
            elif cmd == "new":
                self.history = []
                self.blocks.clear()
                self.add(self.render_header())
                self.add(P("chat history cleared", "dim"))
            elif cmd == "ws":
                if not self.last_ws:
                    self.add(P("no harness workspace yet", "dim"))
                    return
                for f in sorted(Path(self.last_ws).iterdir()):
                    if f.is_file():
                        self.add(T(P("  "), M(f"[bold]{f.name}[/]"),
                                   P(f"  {f.stat().st_size:,} bytes", "dim")))
                self.add(P(f"  {self.last_ws}", "dim"))
            elif cmd == "tasks":
                tasks = read_tasks(self.last_ws) if self.last_ws else []
                if not tasks:
                    self.add(P("no tasks yet", "dim"))
                    return
                for tsk in tasks:
                    mark = {"done": M("[green]✓[/]"), "in_progress": M("[yellow]→[/]")}.get(
                        tsk.get("status"), M("[dim]·[/]"))
                    self.add(T(P("  "), mark, P(" " + str(tsk.get("desc", ""))[:100])))
            elif cmd == "transcript":
                if not self.last_ws:
                    self.add(P("no transcript yet", "dim"))
                    return
                n = int(arg) if arg.isdigit() else 8
                tr = read_transcript(self.last_ws)
                for r in tr["calls"][-n:]:
                    self.add(T(P("  "), M(f"[bold]{role_label(r.get('role'))}[/]"),
                               P(f"  {r.get('completion_tokens', '?')} tok · "
                                 f"finish={r.get('finish_reason') or '-'}", "dim")))
                    resp = (r.get("response") or "")[:150].replace("\n", " ")
                    self.add(P("      " + resp, "dim"))
            elif cmd == "open":
                if not self.last_ws:
                    self.add(P("no harness workspace yet", "dim"))
                    return
                import subprocess
                if os.name == "nt":
                    os.startfile(self.last_ws)  # noqa: S606
                else:
                    subprocess.Popen(["xdg-open", self.last_ws])
                self.add(P(f"opened {self.last_ws}", "dim"))
            elif cmd == "save":
                path = Path(arg) if arg else Path.cwd() / f"purelogic-chat-{datetime.now():%Y%m%d-%H%M%S}.md"
                path.write_text("\n\n".join(
                    ("> " if h["role"] == "user" else "") + h["content"]
                    for h in self.history), encoding="utf-8")
                self.add(T(P("saved → "), P(str(path), "dim")))
            else:
                self.add(T(M(f"[yellow]unknown command: /{cmd}[/]"),
                           P("  (try /help)", "dim")))

        # ---- main loop ----
        def prompt_input(self):
            try:
                console.print(M("[bold cyan]❯[/] "), end="")
                return input()
            except EOFError:
                return "/quit"

        def render_initial(self):
            self.header_block = self.render_header()
            self.hint_block = T(M("[dim]type a task and the agent team solves it — manager "
                                   "plans, workers work (and write files with a "
                                   "[bold]write_file[/] tool), the manager signs off. "
                                   "[bold]/help[/] for commands, [bold]/quit[/] to exit[/]"))
            self.add(self.header_block)
            self.add(self.hint_block)

        def run(self):
            self.render_initial()
            while not self.quit:
                try:
                    text = self.prompt_input()
                except KeyboardInterrupt:
                    console.print(M("[yellow]Ctrl+C to interrupt — /quit to exit.[/]"), end="")
                    continue
                if text is None:
                    break
                try:
                    self.handle(text)
                except Exception as e:  # noqa: BLE001
                    self.add(T(M(f"[bold red]{type(e).__name__}:[/] "), P(str(e))))

    return TUI()


def main(argv=None):
    args = parse_args(argv)
    cfg = Cfg(args)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    orchestrator, multiagent = import_scaffold(cfg)
    SPECS.update({"code": orchestrator.CODE_SPEC, "math": orchestrator.MATH_SPEC})
    live = LiveState()
    live.kind = cfg.spec
    install_model_layer(cfg, live, multiagent)
    if args.run:
        return run_once(cfg, args, orchestrator, multiagent, live)
    tui = build_tui(cfg, orchestrator, multiagent, live)
    tui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
