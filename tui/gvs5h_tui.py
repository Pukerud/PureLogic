#!/usr/bin/env python3
"""GVS5H harness TUI — PureLogic.

A terminal chat + agent harness pre-wired to a local LLM (llama.cpp, OpenAI-compatible),
driving the GVS5H manager-worker scaffold in `codebase/v2-current/escalation`.

Modes
  chat    streaming chat REPL against the local model (reasoning shown dim)
  harness full GVS5H manager-worker loop (plan -> tasks -> workers -> finalize) with a
          live panel: iteration budget, call/token counts, task list, then the artifact

Quick start (repo root):
  python tui/gvs5h_tui.py                          # interactive, chat mode
  python tui/gvs5h_tui.py --run "Say hi"           # one prompt, scriptable
  python tui/gvs5h_tui.py --run "problem" --mode harness --spec code --iters 4 --json

Everything is pre-wired via defaults below and overridable with CLI flags or the same
env vars the scaffold uses (ESCALATION_*, MULTIAGENT_*).
"""

from __future__ import annotations

import argparse
import json
import os
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GVS5H harness TUI (PureLogic)")
    p.add_argument("--base", default=os.environ.get("GVS5H_BASE", DEFAULT_BASE),
                   help="OpenAI-compatible base URL (default: prewired local llama.cpp)")
    p.add_argument("--model", default=os.environ.get("GVS5H_MODEL", DEFAULT_MODEL),
                   help="model id as served by the endpoint")
    p.add_argument("--mode", choices=["chat", "harness"], default=None,
                   help="mode for --run (default: chat)")
    p.add_argument("--run", metavar="PROMPT", default=None,
                   help="run one prompt non-interactively and exit")
    p.add_argument("--spec", choices=["general", "code", "math"], default="general",
                   help="harness task spec (default: general)")
    p.add_argument("--iters", type=int, default=None, help="manager iteration budget")
    p.add_argument("--cap", type=int, default=None,
                   help="max output tokens per model call (default 8192)")
    p.add_argument("--temp", type=float, default=None, help="temperature")
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
        self.mode = args.mode or "chat"
        self.spec = args.spec
        self.ws_root = REPO_ROOT / "tui_workspaces"
        self.ws_root.mkdir(exist_ok=True)


def apply_env(cfg):
    """Set the scaffold's env vars BEFORE it is imported (they are read at import time).
    Real env vars always win over CLI-derived values (setdefault semantics, reversed:
    here we only fill in what the user did not already set)."""
    os.environ.setdefault("ESCALATION_LOCAL_BASE", cfg.base + "/chat/completions")
    os.environ.setdefault("ESCALATION_LOCAL_KEY", "local")
    os.environ.setdefault("MULTIAGENT_MODEL", "local:" + cfg.model)
    os.environ.setdefault("MULTIAGENT_MAX_ITERS", str(cfg.iters))
    os.environ.setdefault("MULTIAGENT_STRICT_FORMAT", "1")
    os.environ.setdefault("MULTIAGENT_WS", str(cfg.ws_root))
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
        "give the complete answer. On the FINAL line of your answer, output exactly "
        "'ANSWER: <your concise final answer>'."
    ),
    "critic_system": (
        "You are a meticulous reviewer. Given a problem and a candidate answer, check the "
        "reasoning and the final answer for errors. If fully correct, reply with exactly "
        "'APPROVED' on the first line; otherwise reply 'REJECTED' on the first line "
        "followed by the specific errors."
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
    extra = local_extra(cfg)
    if extra:
        import copy
        extra = copy.deepcopy(extra)
    from orchestrator import openai_chat
    return openai_chat(cfg.base + "/chat/completions", os.environ.get("ESCALATION_LOCAL_KEY", "local"),
                       cfg.model, messages, cfg.temp if temperature is None else temperature,
                       extra, meta)


# --- transcript / workspace introspection ------------------------------------

def read_transcript(ws):
    path = Path(ws) / "transcript.jsonl"
    recs = []
    if path.exists():
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
        "calls": calls,
    }


def read_tasks(ws):
    p = Path(ws) / "tasks.json"
    if p.exists():
        try:
            return json.loads(p.read_text(errors="replace"))
        except ValueError:
            pass
    return []


# =============================================================================
#  --run (non-interactive)
# =============================================================================

def run_once(cfg, args, orchestrator, multiagent):
    prompt = args.run
    if cfg.mode == "chat":
        messages = [{"role": "system", "content": "You are PureLogic, a capable assistant "
                                                  "running on a local LLM. Be direct and concrete."},
                    {"role": "user", "content": prompt}]
        if args.as_json:
            t0 = time.time()
            content, reasoning, usage = chat_completion(cfg, messages)
            result = {"mode": "chat", "ok": bool(content.strip()), "answer": content,
                      "reasoning": reasoning, "usage": usage, "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.no_stream:
            content, reasoning, usage = chat_completion(cfg, messages)
            if reasoning:
                print(f"[thinking] {reasoning.strip()[:400]}", file=sys.stderr)
            print(content)
            return 0 if content.strip() else 1
        # stream to stdout
        def show(rd, cd):
            if cd:
                print(cd, end="", flush=True)

        content, reasoning, usage = stream_chat(cfg, messages, on_delta=show)
        print()
        return 0 if content.strip() else 1

    # harness mode
    spec = SPECS[cfg.spec]
    if cfg.spec != "general":
        spec = getattr(orchestrator, "CODE_SPEC" if cfg.spec == "code" else "MATH_SPEC")
    status_out = {}

    def log(msg):
        print(f"  {datetime.now().strftime('%H:%M:%S')} {msg.strip()}", file=sys.stderr, flush=True)

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
    if args.as_json:
        result = {"mode": "harness", "spec": cfg.spec, "ok": bool(answer.strip()),
                  "answer": answer, "workspace": ws, "elapsed_s": round(time.time() - t0, 1),
                  "n_calls": stats.get("n_calls"), "completion_tokens": stats.get("completion_tokens"),
                  "truncated_calls": stats.get("truncated_calls"),
                  "finish_reason": status_out.get("finish_reason")}
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    print("\n" + "=" * 62)
    print(f"HARNESS RESULT ({cfg.spec}) — {stats.get('n_calls')} calls, "
          f"{stats.get('completion_tokens')} output tokens, {time.time() - t0:.0f}s")
    print("=" * 62)
    print(answer or "(no answer produced)")
    if ws:
        print(f"\nworkspace: {ws}")
    return 0 if answer.strip() else 1


# =============================================================================
#  interactive TUI
# =============================================================================

def build_tui(cfg, orchestrator, multiagent):
    from rich import box
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.text import Text

    console = Console(highlight=False)
    model_short = cfg.model.rstrip("/").split("/")[-1] or cfg.model

    class TUI:
        def __init__(self):
            self.mode = cfg.mode if cfg.mode == "harness" else "chat"
            self.spec = cfg.spec
            self.history = []
            self.lines = []
            self.lock = threading.Lock()
            self.quit = False
            self.last_ws = None
            self.last_answer = None
            self.harness = {"running": False, "iter": 0, "t0": None, "phase": "",
                            "calls": 0, "tokens": 0, "touched": 0}
            self.streaming = None  # (reasoning_buf, content_buf)
            self.height = max(10, console.height or 30)

        # -- renderables ------------------------------------------------
        def add(self, *renderables):
            with self.lock:
                self.lines.extend(renderables)
                if len(self.lines) > 1500:
                    self.lines = self.lines[-1200:]

        def header(self):
            mode_style = "bold cyan" if self.mode == "chat" else "bold magenta"
            badge = f"[{mode_style}{'CHAT' if self.mode == 'chat' else 'HARNESS'}[/]"
            if self.mode == "harness":
                badge += f"[bold magenta]·{self.spec}[/]"
            left = f"[bold white]GVS5H[/] [dim]PureLogic harness[/]  {badge}"
            right = f"[dim]{datetime.now().strftime('%H:%M:%S')}[/]"
            line1 = Text()
            line1.append(left)
            line1.append(" " * max(1, (console.width or 100) - len(left) - len(right) + 2))
            line1.append(right)
            line2 = Text.assemble(
                f"[dim]model:[/] [cyan]{model_short}[/]  [dim]@ {cfg.base}[/]   ",
                f"[dim]iters:[/] {cfg.iters}   [dim]cap:[/] {cfg.cap}   [dim]temp:[/] {cfg.temp}",
                f"[dim]  think:[/] {'off' if cfg.no_think else 'on'}[/]")
            return Panel(Group(line1, line2), box=box.ROUNDED, border_style="blue",
                         padding=(0, 1))

        def status_line(self):
            h = self.harness
            if h["running"] and h["t0"]:
                el = time.time() - h["t0"]
                m, s = divmod(int(el), 60)
                budget = max(cfg.iters, 1)
                done = min(h["iter"], budget)
                bar_w = 24
                filled = int(bar_w * done / budget)
                bar = Text()
                bar.append("▓" * filled, style="magenta")
                bar.append("░" * (bar_w - filled), style="dim")
                tasks = read_tasks(self.last_ws) if self.last_ws else []
                done_n = sum(1 for t in tasks if t.get("status") == "done")
                phase = h["phase"][:60]
                return Group(
                    Text.assemble(f"[bold magenta]HARNESS[/] [bold]iter {done}/{budget}[/]  {bar}  "
                                  f"[dim]{m:02d}:{s:02d}[/]  {h['calls']} calls · "
                                  f"{h['tokens']:,} tok  [dim]{phase}[/]"),
                    Text.assemble(f"[bold]tasks:[/] "
                                  f"{' '.join('✓' if t.get('status') == 'done' else '•' for t in tasks[:16]) or '—'}"
                                  f" [dim]({done_n}/{len(tasks)})[/]  [dim]{self.last_ws or ''}[/]"))
            return Text(f"[dim]mode:[/] {self.mode}   [dim]spec:[/] {self.spec}   "
                        f"[dim]iters:[/] {cfg.iters}   [dim]cap:[/] {cfg.cap}   "
                        f"[/][dim]type a prompt, or /help for commands[/]")

        def __rich_console__(self, c, options):
            body = self.lines[-(self.height - 8):]
            streaming = None
            if self.streaming:
                rb, cb = self.streaming
                parts = []
                if rb:
                    parts.append(Text(rb[-3000:], style="dim italic"))
                parts.append(Text(cb))
                streaming = Group(*parts) if parts else Text("…", style="dim")
            parts = list(body)
            if streaming is not None:
                parts.append(streaming)
            body_r = Group(*parts) if parts else Text("…", style="dim")
            layout = Layout()
            layout.split_column(
                Layout(self.header(), name="header", size=4),
                Layout(body_r, name="body"),
                Layout(self.status_line(), name="status", size=3),
            )
            yield layout

        def _refresh_live(self):
            live = getattr(self, "_live", None)
            if live is not None:
                try:
                    live.refresh()
                except Exception:  # noqa: BLE001
                    pass

        # -- chat --------------------------------------------------------
        def do_chat(self, prompt):
            self.add(Text.assemble("[bold cyan]❯[/] [cyan]", prompt))
            self.history.append({"role": "user", "content": prompt})
            messages = ([{"role": "system",
                          "content": "You are PureLogic, a capable assistant running on a "
                                     "local LLM. Be direct and concrete."}]
                         + self.history[-40:])
            rb, cb = [], []
            self.streaming = (rb, cb)
            error = None
            last_flush = [0.0]

            def on_delta(r, c):
                if r:
                    rb.append(r)
                if c:
                    cb.append(c)
                now = time.time()
                if now - last_flush[0] > 0.1:
                    last_flush[0] = now
                    self._refresh_live()

            try:
                content, reasoning, usage = stream_chat(cfg, messages, on_delta=on_delta)
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                content = reasoning = ""
            self.streaming = None
            if error:
                self.add(Text(f"[bold red]error:[/] {error}"))
                return
            self.history.append({"role": "assistant", "content": content})
            blocks = []
            if reasoning.strip():
                blocks.append(Text("◆ thinking", style="dim magenta"))
                blocks.append(Text(reasoning.strip()[:6000], style="dim italic"))
            if content.strip():
                blocks.append(Markdown(content))
            if usage:
                ut = usage.get("completion_tokens")
                blocks.append(Text(f"[dim]({ut} tokens)[/]"))
            self.add(Panel(Group(*blocks), box=box.SQUARE, border_style="cyan",
                           padding=(0, 1), title=f"[cyan]{model_short}[/]"))

        # -- harness ------------------------------------------------------
        def harness_log(self, msg):
            m = msg.strip()
            style, icon = "dim", "·"
            if m.startswith("[primary"):
                style, icon = "bold magenta", "◆"
            elif m.startswith("[worker") or m.startswith("[finalize"):
                style, icon = "green", "▸"
            elif m.startswith("[samples"):
                style, icon = "yellow", "✔"
            elif "CUT OFF" in m or "gave up" in m:
                style, icon = "red", "✖"
            self.add(Text.assemble(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/] ",
                                   f"[{style}]{icon}[/]", f" [{style}]{m}[/]"))
            if "plan +" in m:
                self.harness["phase"] = "planning"
            elif m.startswith("[worker") or m.startswith("[finalize"):
                self.harness["phase"] = "worker"
            elif m.startswith("[samples"):
                self.harness["phase"] = "sample tests"
            elif m.startswith("[primary-manage"):
                self.harness["phase"] = "managing"

        def do_harness(self, problem):
            self.add(Text.assemble("[bold cyan]❯[/] [cyan]", problem),
                     Rule("harness", style="magenta"))
            ws = None
            status_out = {}

            def solve():
                nonlocal ws
                spec = SPECS[self.spec] if self.spec == "general" else getattr(
                    orchestrator, "CODE_SPEC" if self.spec == "code" else "MATH_SPEC")
                try:
                    answer = multiagent.multiagent_solve(problem, spec, log=self.harness_log,
                                                         status_out=status_out)
                except Exception as e:  # noqa: BLE001
                    answer = None
                    self.harness_log(f"ERROR: {type(e).__name__}: {e}")
                ws = status_out.get("ws")
                self.last_ws = ws
                self.last_answer = answer
                self.harness["running"] = False

            self.harness.update(running=True, iter=0, t0=time.time(), phase="starting",
                                calls=0, tokens=0)
            self.last_ws = None
            t = threading.Thread(target=solve, daemon=True)
            t.start()
            while self.harness["running"]:
                time.sleep(0.5)
                self._harness_stats(status_out, t)
            self.add_harness_result(problem)

        def _harness_stats(self, status_out, thread):
            # refresh call/token counts from the transcript (throttled to ~1/s)
            now = time.time()
            if now - self.harness["touched"] < 1.0:
                return
            self.harness["touched"] = now
            ws = status_out.get("ws") or self.last_ws
            if ws:
                self.last_ws = ws
                st = read_transcript(ws)
                self.harness["calls"] = st["n_calls"]
                self.harness["tokens"] = st["completion_tokens"]
                # The scaffold's budget counts worker iterations (plan + ideation run
                # before the loop and are free), so mirror that: one bar unit per worker call.
                self.harness["iter"] = min(
                    cfg.iters, sum(1 for c in st["calls"]
                                   if c.get("role", "").startswith("worker:")))

        def add_harness_result(self, problem):
            ws, answer = self.last_ws, self.last_answer
            st = read_transcript(ws) if ws else {}
            ok = bool(answer and answer.strip())
            blocks = []
            if answer and self.spec == "code" and "```" in answer:
                code = answer.strip().strip("`").removeprefix("python").strip()
                code = answer.strip()
                if code.startswith("```python"):
                    code = code.removeprefix("```python").removesuffix("```").strip()
                blocks.append(Syntax(code, "python", word_wrap=True))
            elif answer:
                blocks.append(Markdown(answer))
            else:
                blocks.append(Text("no answer produced", style="red"))
            meta = (f"[dim]{st.get('n_calls', 0)} calls · {st.get('completion_tokens', 0):,} "
                    f"output tokens · {st.get('truncated_calls', 0)} truncated"
                    + (f" · ws: {ws}" if ws else "") + "[/]")
            self.add(Panel(Group(meta, Rule(), *blocks),
                           box=box.DOUBLE, border_style="green" if ok else "red",
                           title=f"HARNESS RESULT · {self.spec}"))
            if ws:
                self.add(Text(f"[dim]workspace:[/] {ws}   [dim](/ws, /tasks, /transcript to inspect)[/]"))

        # -- commands ------------------------------------------------------
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
                self.add(Panel(
                    "\n".join([
                        "[bold]/mode[/] chat|harness   switch mode",
                        "[bold]/spec[/] general|code|math   harness task spec",
                        "[bold]/iters[/] N       manager iteration budget",
                        "[bold]/cap[/] N         max output tokens per call",
                        "[bold]/temp[/] F        temperature",
                        "[bold]/model[/] ID      switch model (chat + harness)",
                        "[bold]/think[/] on|off  toggle model thinking",
                        "[bold]/new[/]           clear chat history",
                        "[bold]/ws[/]            last harness workspace + files",
                        "[bold]/tasks[/]         last harness task list",
                        "[bold]/transcript[/] [N]  last harness model calls",
                        "[bold]/save[/] [PATH]   save chat as markdown",
                        "[bold]/quit[/]",
                    ]), title="commands", border_style="blue"))
            elif cmd == "mode":
                if arg in ("chat", "harness"):
                    self.mode = arg
                    self.add(Text(f"mode -> [bold]{arg}[/]"))
                else:
                    self.add(Text(f"current mode: [bold]{self.mode}[/]"))
            elif cmd == "spec":
                if arg in SPECS or arg in ("code", "math") or (
                        self.spec == arg and arg in ("general",)):
                    if arg in ("general", "code", "math"):
                        self.spec = arg
                    self.add(Text(f"harness spec -> [bold]{self.spec}[/]"))
            elif cmd == "iters":
                cfg.iters = max(1, int(arg or cfg.iters))
                multiagent.MAX_ITERS = cfg.iters
                self.add(Text(f"iters -> [bold]{cfg.iters}[/]"))
            elif cmd == "cap":
                cfg.cap = max(16, int(arg or cfg.cap))
                orchestrator.CLOUD_MAX_TOKENS = cfg.cap
                self.add(Text(f"cap -> [bold]{cfg.cap}[/]"))
            elif cmd == "temp":
                cfg.temp = float(arg or cfg.temp)
                self.add(Text(f"temp -> [bold]{cfg.temp}[/]"))
            elif cmd == "model":
                if arg:
                    cfg.model = arg
                    model_short  # no-op keep
                    self.add(Text(f"model -> [bold]{cfg.model.split('/')[-1]}[/]"))
            elif cmd == "think":
                if arg in ("on", "off"):
                    cfg.no_think = (arg == "off")
                    os.environ["ESCALATION_LOCAL_EXTRA"] = (
                        json.dumps({"chat_template_kwargs": {"enable_thinking": False}})
                        if cfg.no_think else "")
                    self.add(Text(f"thinking -> [bold]{'off' if cfg.no_think else 'on'}[/]"))
            elif cmd == "new":
                self.history = []
                self.add(Text("chat history cleared", style="dim"))
            elif cmd == "ws":
                if not self.last_ws:
                    self.add(Text("no harness workspace yet", style="dim"))
                    return
                ws = Path(self.last_ws)
                for f in sorted(ws.iterdir()):
                    self.add(Text(f"  [bold]{f.name}[/] [dim]{f.stat().st_size:,} bytes[/]"))
                self.add(Text(f"[dim]{self.last_ws}[/]"))
            elif cmd == "tasks":
                tasks = read_tasks(self.last_ws) if self.last_ws else []
                if not tasks:
                    self.add(Text("no tasks yet", style="dim"))
                    return
                for t in tasks:
                    mark = {"done": "[green]✓[/]", "in_progress": "[yellow]→[/]"}.get(
                        t.get("status"), "[dim]•[/]")
                    self.add(Text.assemble(f"  {mark} ", t.get("desc", "")))
            elif cmd == "transcript":
                if not self.last_ws:
                    self.add(Text("no transcript yet", style="dim"))
                    return
                n = int(arg) if arg.isdigit() else 8
                st = read_transcript(self.last_ws)
                for r in st["calls"][-n:]:
                    self.add(Text.assemble(
                        "  [bold]", str(r.get("role", "?")), "[/] ",
                        f"[dim]{r.get('completion_tokens', '?')} tok · "
                        f"finish={r.get('finish_reason') or '-'}[/]"))
                    resp = (r.get("response") or "")
                    resp = resp[:160].replace("\n", " ")
                    self.add(Text("      " + resp, style="dim"))
            elif cmd == "save":
                path = Path(arg) if arg else Path.cwd() / f"purelogic-chat-{datetime.now():%Y%m%d-%H%M%S}.md"
                path.write_text("\n\n".join(
                    ("> " if h["role"] == "user" else "") + h["content"] for h in self.history),
                    encoding="utf-8")
                self.add(Text(f"saved -> {path}", style="dim"))
            else:
                self.add(Text(f"unknown command: /{cmd}  (try /help)", style="yellow"))

        # -- main loop ------------------------------------------------------
        def run(self):
            self.add(Text.assemble("[dim]GVS5H harness — prewired to ",
                                   f"[cyan]{model_short}[/] [dim]@ {cfg.base}[/] · "
                                   "type a prompt or [bold]/help[/] [dim]· /quit to exit[/]"))
            from rich.live import Live as _Live
            with _Live(self, console=console, refresh_per_second=4) as live:
                self._live = live
                while not self.quit:
                    live.stop()
                    try:
                        text = input("❯ ")
                    except EOFError:
                        break
                    except KeyboardInterrupt:
                        if self.harness["running"]:
                            text = ""
                        else:
                            text = "/quit"
                    live.start()
                    if text:
                        try:
                            self.handle(text)
                        except Exception as e:  # noqa: BLE001
                            self.add(Text(f"[bold red]{type(e).__name__}: {e}[/]"))
                    live.refresh()

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
    if cfg.mode == "harness":
        multiagent.MAX_ITERS = cfg.iters
    if args.run:
        return run_once(cfg, args, orchestrator, multiagent)
    tui = build_tui(cfg, orchestrator, multiagent)
    tui.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
