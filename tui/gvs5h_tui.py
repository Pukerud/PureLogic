#!/usr/bin/env python3
"""GVS5H agent TUI — PureLogic.

A terminal agent interface pre-wired to a local LLM (llama.cpp, OpenAI-compatible),
driving the GVS5H manager-worker scaffold in `codebase/v2-current/escalation`.

Modes
  harness (default)  your prompt runs the full GVS5H loop: manager plans, an
                     ideation pass, then manager/worker iterations over a shared
                     workspace, then finalize. A live activity panel shows the
                     current phase, iteration budget, call/token counters and the
                     task checklist while it works.
  chat               streaming chat REPL against the local model (reasoning shown
                     live in dim italics while it streams).

Working directory
  Artifacts are written to a `Workspace` subfolder of the directory you launch
  from (override with --workspace-dir or GVS5H_WS_DIR). Each run gets its own
  subfolder with task.md, plan.md, tasks.json, notes.md, answer.md /
  solution.py and transcript.jsonl.

Quick start (repo root):
  python tui/gvs5h_tui.py                          # interactive agent (harness)
  python tui/gvs5h_tui.py --run "problem"          # one harness run, non-interactive
  python tui/gvs5h_tui.py --run "hi" --mode chat   # quick chat
  python tui/gvs5h_tui.py --run "problem" --mode harness --spec code --iters 4
  python tui/gvs5h_tui.py --run "problem" --json   # machine-readable result

Everything is pre-wired via defaults below and overridable with CLI flags or the
same env vars the scaffold uses (ESCALATION_*, MULTIAGENT_*).
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
DEFAULT_ITERS = 6
DEFAULT_CAP = 8192
DEFAULT_TEMP = 0.3

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
ANSWER_CHARS = 6000          # cap for rendered answer panels
LOG_KEEP = 3                 # last log lines kept for the activity panel


def spin(t=None):
    return SPINNER_FRAMES[int((t if t is not None else time.time()) * 10) % len(SPINNER_FRAMES)]


def fmt_elapsed(s):
    s = max(0, int(s))
    m, sec = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m:02d}:{sec:02d}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GVS5H agent TUI (PureLogic)")
    p.add_argument("--base", default=os.environ.get("GVS5H_BASE", DEFAULT_BASE),
                   help="OpenAI-compatible base URL (default: prewired local llama.cpp)")
    p.add_argument("--model", default=os.environ.get("GVS5H_MODEL", DEFAULT_MODEL),
                   help="model id as served by the endpoint")
    p.add_argument("--mode", choices=["harness", "chat"], default="harness",
                   help="mode (default: harness = full GVS5H agent loop)")
    p.add_argument("--run", metavar="PROMPT", default=None,
                   help="run one prompt non-interactively and exit")
    p.add_argument("--spec", choices=["general", "code", "math"], default="general",
                   help="harness task spec (default: general)")
    p.add_argument("--iters", type=int, default=None, help="manager iteration budget")
    p.add_argument("--cap", type=int, default=None,
                   help="max output tokens per model call (default 8192)")
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
        self.iters = args.iters or int(os.environ.get("GVS5H_ITERS", DEFAULT_ITERS))
        self.cap = args.cap or int(os.environ.get("GVS5H_CAP", DEFAULT_CAP))
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
    os.environ.setdefault("MULTIAGENT_MAX_ITERS", str(cfg.iters))
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


# --- OpenAI-compatible client (streaming for chat, scaffold client for harness) ---

def stream_chat(cfg, messages, temperature=None, max_tokens=None, on_delta=None):
    """Stream an OpenAI-compatible chat completion.

    on_delta(reasoning: str, content: str) is called per SSE delta.
    Returns (content, reasoning, usage_dict)."""
    body = {
        "model": cfg.model,
        "messages": messages,
        "stream": True,
        "temperature": cfg.temp if temperature is None else temperature,
        "max_tokens": max_tokens or cfg.cap,
    }
    extra = local_extra(cfg)
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        cfg.base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "gvs5h-tui/1.0"})
    content, reasoning, usage = [], [], None
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
                rd, cd = d.get("reasoning_content") or "", d.get("content") or ""
                if rd:
                    reasoning.append(rd)
                if cd:
                    content.append(cd)
                if on_delta and (rd or cd):
                    on_delta(rd, cd)
    return "".join(content), "".join(reasoning), usage


def chat_completion(cfg, messages, temperature=None, max_tokens=None, meta=None):
    """Non-streaming chat via the scaffold's own OpenAI-compatible client (with its
    retry/watchdog/reasoning capture)."""
    import orchestrator  # local import: already on sys.path after import_scaffold
    orchestrator.CLOUD_MAX_TOKENS = max_tokens or cfg.cap
    return orchestrator.openai_chat(
        cfg.base + "/chat/completions", os.environ.get("ESCALATION_LOCAL_KEY", "local"),
        cfg.model, messages, cfg.temp if temperature is None else temperature,
        local_extra(cfg), meta)


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


ROLE_PHASE = {
    "primary_plan": "manager · planning",
    "ideation": "ideation pass",
    "primary_manage": "manager · reviewing",
    "cutoff_summary": "context compaction",
    "finalize": "finalizing answer",
}


def phase_for(role):
    if not role:
        return "starting"
    if role.startswith("worker:"):
        return f"worker {role.split(':', 1)[1]} working"
    return ROLE_PHASE.get(role, "working")


def extract_artifact(answer):
    """Pull a self-contained artifact out of a general-spec answer.

    Returns (filename, content) or None. Heuristic: the largest fenced block in the
    answer, or the whole answer when it looks like a complete HTML document."""
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

def run_once(cfg, args, orchestrator, multiagent):
    prompt = args.run
    if cfg.mode == "chat":
        messages = [{"role": "system",
                     "content": "You are PureLogic, a capable assistant running on a "
                                "local LLM. Be direct and concrete."},
                    {"role": "user", "content": prompt}]
        if args.as_json:
            t0 = time.time()
            content, reasoning, usage = chat_completion(cfg, messages)
            result = {"mode": "chat", "ok": bool(content.strip()), "answer": content,
                      "reasoning": reasoning, "usage": usage,
                      "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.no_stream:
            content, reasoning, usage = chat_completion(cfg, messages)
            if reasoning:
                print(f"[thinking] {reasoning.strip()[:400]}", file=sys.stderr)
            print(content)
            return 0 if content.strip() else 1
        content, reasoning, usage = stream_chat(
            cfg, messages, on_delta=lambda rd, cd: (cd is not None and print(cd, end="", flush=True)))
        print()
        return 0 if content.strip() else 1

    # harness mode
    spec = SPECS[cfg.spec] if cfg.spec == "general" else getattr(
        orchestrator, "CODE_SPEC" if cfg.spec == "code" else "MATH_SPEC")
    status_out = {}

    def log(msg):
        print(f"{datetime.now():%H:%M:%S} {msg.strip()}", file=sys.stderr, flush=True)

    t0 = time.time()
    try:
        answer = multiagent.multiagent_solve(prompt, spec, log=log, status_out=status_out)
    except Exception as e:  # noqa: BLE001
        print(f"harness failed: {type(e).__name__}: {e}", file=sys.stderr)
        if args.as_json:
            print(json.dumps({"mode": "harness", "ok": False, "error": str(e)}))
        return 1
    ws = status_out.get("ws")
    stats = read_transcript(ws) if ws else {}
    artifact = None
    if cfg.spec == "general" and answer and ws:
        artifact = extract_artifact(answer)
        if artifact:
            (Path(ws) / artifact[0]).write_text(artifact[1], encoding="utf-8")
    if args.as_json:
        files = sorted(str(f.name) for f in Path(ws).iterdir() if f.is_file()) if ws else []
        result = {"mode": "harness", "spec": cfg.spec, "ok": bool(answer and answer.strip()),
                  "answer": answer, "workspace": ws, "files": files,
                  "artifact": f"{ws}/{artifact[0]}" if artifact else None,
                  "elapsed_s": round(time.time() - t0, 1),
                  "n_calls": stats.get("n_calls"),
                  "completion_tokens": stats.get("completion_tokens"),
                  "truncated_calls": stats.get("truncated_calls"),
                  "finish_reason": status_out.get("finish_reason")}
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    print()
    print("=" * 62)
    print(f"HARNESS RESULT ({cfg.spec}) — {stats.get('n_calls', 0)} calls, "
          f"{stats.get('completion_tokens', 0)} output tokens, {time.time() - t0:.0f}s")
    print("=" * 62)
    print(answer or "(no answer produced)")
    if ws:
        print(f"\nartifacts: {ws}")
        for f in sorted(Path(ws).iterdir()):
            if f.is_file():
                print(f"  {f.name:<16} {f.stat().st_size:>9,} bytes")
    return 0 if answer and answer.strip() else 1


# =============================================================================
#  interactive TUI
# =============================================================================

def build_tui(cfg, orchestrator, multiagent):
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

    # --- markup discipline ----------------------------------------------------
    # Static UI strings that intentionally contain [markup] go through M();
    # dynamic content (answers, log lines, paths, tasks) is always plain Text.
    def M(s):
        return Text.from_markup(s)

    def P(s, style=""):
        return Text(s, style=style)

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
        for m in marks:
            out.append_text(m)
        out.append_text(Text(f"({done_n}/{len(tasks)})", style="dim"))
        return out

    class Activity:
        """Renderable for the live panel; reads shared state each refresh."""

        def __init__(self, kind):
            self.kind = kind          # "harness" | "chat"
            self.state = None         # set by the owner

        def _harness(self):
            s = self.state
            t0 = s.get("t0") or time.time()
            el = time.time() - t0
            budget = max(1, s.get("iters", 1))
            done = s.get("iter", 0)
            phase = s.get("phase", "starting")
            tasks = s.get("tasks") or []
            last = s.get("last_log")
            rows = [
                T(M(f"{spin()} "), M(f"[bold]{phase}[/]"),
                  M("   "), P(f"{fmt_elapsed(el)}", "dim"),
                  M("   "), P(f"{s.get('calls', 0)} calls · {s.get('tokens', 0):,} tok", "dim")),
                T(M("iter  "), bar(done / budget),
                  M(f"  {done}/{budget}   "), M("tasks "), task_marks(tasks)),
            ]
            if last:
                rows.append(P(last[:110], "dim"))
            title = M(f"[bold magenta]AGENT RUN[/] [dim]· {s.get('spec', '')} · "
                      f"{datetime.now():%H:%M:%S}[/]")
            return Panel(Group(*rows), box=box.ROUNDED, border_style="magenta",
                         title=title, padding=(0, 1))

        def _chat(self):
            s = self.state
            t0 = s.get("t0") or time.time()
            el = time.time() - t0
            phase = s.get("phase", "waiting")
            if phase == "waiting":
                lead = T(M(f"{spin()} "), M("[bold]waiting for model…[/]"))
                tail = None
            elif phase == "thinking":
                lead = T(M(f"{spin()} "), M("[bold]model thinking…[/]"),
                         M("   "), P(fmt_elapsed(el) + f" · {s.get('rtokens', 0)} tok", "dim"))
                tail = P((s.get("rb") or "")[-700:], "dim italic")
            else:
                lead = T(M(f"{spin()} "), M("[bold]generating answer…[/]"),
                         M("   "), P(fmt_elapsed(el) + f" · {s.get('ctokens', 0)} tok", "dim"))
                tail = P((s.get("cb") or "")[-900:])
            rows = [lead]
            if tail:
                rows.append(tail)
            title = M(f"[bold cyan]CHAT[/] [dim]· {model_short} · {datetime.now():%H:%M:%S}[/]")
            return Panel(Group(*rows), box=box.ROUNDED, border_style="cyan",
                         title=title, padding=(0, 1))

        def __rich_console__(self, c, options):
            if self.kind == "harness":
                yield self._harness()
            else:
                yield self._chat()

    # --- the TUI ---------------------------------------------------------------
    class TUI:
        def __init__(self):
            self.mode = cfg.mode
            self.spec = cfg.spec
            self.history = []
            self.blocks = []          # scrollback renderables
            self.lock = threading.Lock()
            self.quit = False
            self.last_ws = None
            self.last_answer = None
            self.cancel_hinted = False

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
            row1 = T(M("[bold white]◆ GVS5H[/]"), M(" [dim]· PureLogic agent ·[/] "),
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
            # the typed prompt is already on the terminal (input echo)
            self.history.append({"role": "user", "content": prompt})
            messages = ([{"role": "system",
                          "content": "You are PureLogic, a capable assistant running on a "
                                     "local LLM. Be direct and concrete."}]
                         + self.history[-40:])
            act = Activity("chat")
            act.state = {"t0": time.time(), "phase": "waiting", "rb": "", "cb": "",
                         "rtokens": 0, "ctokens": 0}
            result = {}

            def on_delta(rd, cd):
                st = act.state
                if rd:
                    st["rb"] += rd
                    st["rtokens"] += len(rd)
                    st["phase"] = "thinking"
                if cd:
                    st["cb"] += cd
                    st["ctokens"] += len(cd)
                    st["phase"] = "answer"

            def worker():
                try:
                    content, reasoning, usage = stream_chat(cfg, messages, on_delta=on_delta)
                    result.update(content=content, reasoning=reasoning, usage=usage)
                except Exception as e:  # noqa: BLE001
                    result["error"] = f"{type(e).__name__}: {e}"

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            with Live(act, console=console, refresh_per_second=8, transient=True):
                while t.is_alive():
                    time.sleep(0.1)
            if result.get("error"):
                self.add(T(M("[bold red]✗ error: ", "bold red"), P(result["error"])))
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
            if usage.get("completion_tokens"):
                blocks.append(P(f"{usage.get('completion_tokens')} completion tokens · "
                                f"{fmt_elapsed(time.time() - act.state['t0'])}", "dim"))
            self.add(Panel(Group(*blocks) if blocks else P("(empty response)"),
                           box=box.SQUARE, border_style="cyan",
                           title=M(f"[cyan]{model_short}[/]")))

        # ---- harness ----
        def do_harness(self, problem):
            self.add(Rule(style="magenta"))
            ws = {"v": None}
            status_out = {}
            act = Activity("harness")
            act.state = {"t0": time.time(), "phase": "starting", "iter": 0, "calls": 0,
                         "tokens": 0, "tasks": [], "last_log": "", "iters": cfg.iters,
                         "spec": self.spec}
            error = {"v": None}

            def log(msg):
                m = msg.strip()
                with self.lock:
                    act.state["last_log"] = m[:110]

            def solve():
                spec = SPECS[self.spec] if self.spec == "general" else getattr(
                    orchestrator, "CODE_SPEC" if self.spec == "code" else "MATH_SPEC")
                try:
                    answer = multiagent.multiagent_solve(problem, spec, log=log,
                                                         status_out=status_out)
                    result = {"v": answer}
                except Exception as e:  # noqa: BLE001
                    result = {"v": None}
                    error["v"] = f"{type(e).__name__}: {e}"
                    log(f"ERROR: {error['v']}")
                ws["v"] = status_out.get("ws")

            def poll():
                st = act.state
                w = ws["v"] or status_out.get("ws")
                if w:
                    tr = read_transcript(w)
                    st["calls"] = tr["n_calls"]
                    st["tokens"] = tr["completion_tokens"]
                    st["iter"] = min(cfg.iters, sum(
                        1 for c in tr["calls"] if str(c.get("role", "")).startswith("worker:")))
                    st["tasks"] = read_tasks(w)
                    st["phase"] = phase_for(tr["last_role"])

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
                    time.sleep(0.15)
                try:
                    poll()
                except Exception:  # noqa: BLE001
                    pass
            self.add_harness_result(ws["v"], status_out, time.time() - act.state["t0"],
                                    error["v"])

        def add_harness_result(self, ws, status_out, elapsed, error):
            st = read_transcript(ws) if ws else {}
            # answer text: prefer answer.md, then solution.py, then returned answer
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
            ok = (error is None) and bool(answer.strip())
            blocks = []
            if error:
                blocks.append(T(M("[bold red]✗ failed: ", "bold red"), P(error)))
            if answer:
                if ws and (Path(ws) / "solution.py").exists() and self.spec == "code":
                    blocks.append(Syntax(answer, "python", word_wrap=True))
                else:
                    art = extract_artifact(answer) if self.spec != "code" else None
                    if art and art[0].endswith(".html"):
                        blocks.append(Syntax(art[1][:ANSWER_CHARS * 2], "html", word_wrap=True))
                    else:
                        shown = answer
                        if len(shown) > ANSWER_CHARS:
                            shown = shown[:ANSWER_CHARS] + "\n… (truncated)"
                        blocks.append(Markdown(shown))
            else:
                blocks.append(P("no answer produced", "red"))
            if ws:
                # convenience artifact for general runs
                if self.spec != "code" and not (Path(ws) / "artifact.html").exists() \
                        and not (Path(ws) / "artifact.txt").exists():
                    art = extract_artifact(answer)
                    if art:
                        (Path(ws) / art[0]).write_text(art[1], encoding="utf-8")
                meta = T(M(f"[bold]{'✓' if ok else '✗'}[/] "),
                         P(f"{fmt_elapsed(elapsed)} · {st.get('n_calls', 0)} calls · "
                           f"{st.get('completion_tokens', 0):,} tok out · "
                           f"{st.get('truncated_calls', 0)} truncated · "
                           f"finish={status_out.get('finish_reason', '-')}", "dim"))
                files = T(M("[dim]files:[/] "))
                for f in sorted(Path(ws).iterdir()):
                    if f.is_file():
                        files.append(f" {f.name}")
                        files.append_text(Text(f" {f.stat().st_size:,}B", style="dim"))
                wspath = T(M("[bold]artifacts →[/] "), P(ws, "bold green"))
                blocks.insert(0, meta)
                blocks.append(Rule(style="dim"))
                blocks.append(wspath)
                blocks.append(files)
            self.last_ws = ws
            self.add(Panel(Group(*blocks), box=box.DOUBLE,
                           border_style="green" if ok and not error else "red",
                           title=M(f"[bold]RESULT[/] · {self.spec} · "
                                   f"{datetime.now():%H:%M:%S}")))

        # ---- commands ----
        def handle(self, text):
            t = text.strip()
            if not t:
                return
            if not t.startswith("/"):
                if self.mode == "chat":
                    self.do_chat(t)
                else:
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
                    M("[bold]/iters[/] [dim]N    manager iteration budget[/]"),
                    M("[bold]/cap[/] [dim]N     max output tokens per model call[/]"),
                    M("[bold]/temp[/] [dim]F    temperature[/]"),
                    M("[bold]/think[/] [dim]on|off   model thinking[/]"),
                    M("[bold]/new[/] [dim]clear chat history + redraw header[/]"),
                    M("[bold]/ws[/] [dim]files in the last harness workspace[/]"),
                    M("[bold]/tasks[/] [dim]last harness task list[/]"),
                    M("[bold]/transcript[/] [dim][N]  last harness model calls[/]"),
                    M("[bold]/open[/] [dim]open the last workspace in Explorer[/]"),
                    M("[bold]/save[/] [dim][PATH]  save chat as markdown[/]"),
                    M("[bold]/quit[/]"),
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
            elif cmd == "iters":
                try:
                    cfg.iters = max(1, int(arg or cfg.iters))
                    multiagent.MAX_ITERS = cfg.iters
                    self.add(T(M("iters → "), M(f"[bold]{cfg.iters}[/]")))
                except ValueError:
                    self.add(T(M("[yellow]iters expects an integer[/]")))
            elif cmd == "cap":
                try:
                    cfg.cap = max(16, int(arg or cfg.cap))
                    orchestrator.CLOUD_MAX_TOKENS = cfg.cap
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
                    os.environ["ESCALATION_LOCAL_EXTRA"] = (
                        json.dumps({"chat_template_kwargs": {"enable_thinking": False}})
                        if cfg.no_think else "")
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
                    self.add(T(P("  "), M(f"[bold]{r.get('role', '?')}[/]"),
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

        def run(self):
            self.add(self.render_header())
            self.add(T(M("[dim]type a task and the agent will run the full GVS5H loop — "
                         "[bold]/help[/] for commands, [bold]/quit[/] to exit[/]")))
            while not self.quit:
                try:
                    text = self.prompt_input()
                except KeyboardInterrupt:
                    self.cancel_hinted = not self.cancel_hinted
                    if self.cancel_hinted:
                        console.print(M("[yellow]Ctrl+C again to quit.[/]"), end="")
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
    if args.run:
        return run_once(cfg, args, orchestrator, multiagent)
    tui = build_tui(cfg, orchestrator, multiagent)
    tui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
