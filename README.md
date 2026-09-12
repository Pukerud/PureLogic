# PureLogic — a terminal agent on your local LLM

**Type a task. Watch an agent team solve it. Get the files it wrote.**

PureLogic is a terminal **agent** TUI pre-wired to a local LLM (llama.cpp,
OpenAI-compatible). You type a task — *"build me a single HTML file that does X"* —
and a manager–worker team runs the job: the manager plans, a first worker ideates
approaches, then the manager and workers loop over a shared workspace until the
manager signs off. Workers have a real **`write_file` tool**, so the artifact you
asked for is *written as a file in your workspace* — not pasted (and possibly
truncated) into a chat reply.

While it works, a live panel shows exactly what the agent is doing:

![PureLogic agent TUI — live activity panel](tui/screenshot-activity.png)

- **phase** — `manager · planning` → `ideation pass` → `worker 2 · writing page.html (append)` …, with the current call's elapsed time and live thinking-token count
- **stage** — the current manager→worker round out of the stage budget, with the task checklist progress (`tasks ✓ · · · (2/4)`) as `tasks.json` fills
- **run line** — run-level stats: total run elapsed · calls so far (incl. in-flight) · `N running` · live `+N` tokens this call · cumulative `in`/`out` tokens
- **thinking / generating throughput** — live token estimates and `tok/s` while the model streams
- **tail** — the last line the model is writing *right now* (its reply, its thinking, or file content)

When the run finishes you get a result panel with the final answer, the files the
agent wrote, a per-call timeline, and the artifact folder:

![PureLogic agent TUI — result panel](tui/screenshot-result.png)

## Quick start

Windows (recommended — handles Python, `rich`, UTF-8 console and the workspace path):

```
powershell -File tui\start_tui.ps1            # interactive agent
powershell -File tui\start_tui.ps1 --mode chat --run "hi"
powershell -File tui\start_tui.ps1 --spec code --run "write a snake game in one file"
```

Anywhere else (Python 3.10+, `rich`):

```
python tui/purelogic_tui.py                   # interactive agent (harness mode)
python tui/purelogic_tui.py --run "problem"   # one agent run, non-interactive
python tui/purelogic_tui.py --run "hi" --mode chat
python tui/purelogic_tui.py --run "problem" --spec code --stages 4
python tui/purelogic_tui.py --run "problem" --json   # machine-readable result
```

**Where things land:** every run writes to a `Workspace` subfolder of the directory
you launch from — one subfolder per prompt (`<md5>`), holding `task.md`, `plan.md`,
`tasks.json`, `notes.md`, `answer.md` / `solution.py`, any files the agent wrote
(e.g. `page.html`), and `transcript.jsonl` (every model call, with reasoning and
token counts). Override the root with `--workspace-dir` or `GVS5H_WS_DIR`.

## How the agent works

```
your prompt
    │
    ├─ 1. manager plans            → plan.md + task list (tasks.json)
    ├─ 2. ideation pass            → 3-6 distinct approaches in notes.md
    └─ 3. stages (manager→worker)  → repeat until "done" or the stage budget runs out:
           manager reviews progress, curates the task list, picks ONE next task
           worker does it — thinking + reasoning, then answers; if the task
           produces a file, the worker calls write_file (write, then append,
           until the file is complete) and keeps the reply section short
    └─ 4. result                   → final answer + artifact files + stats
```

A **stage** is one manager→worker round. `tasks` is the manager's checklist — a
stage usually completes one task, and the checklist can span several stages.
The budget is set with `/stages N` (default 6).

### The `write_file` tool

Workers (and the finalize pass) are given an OpenAI-style function tool:

```
write_file(path, content, mode="write"|"append")
```

- Files are sandboxed to the run's workspace folder.
- A file can span many calls (`write` then `append` ×N, up to 32 tool rounds
  per call chain), so large artifacts (a 400-line HTML page, a full program)
  are never truncated by the per-call token cap — the model simply keeps writing.
- File-writing calls get a higher output cap (`--file-cap` /
  `GVS5H_FILE_CAP`, default 20480) **and thinking is off by default**
  (`--file-think off`). This endpoint does not reliably honor
  `chat_template_kwargs.thinking_budget`: the prewired 27B will otherwise
  spend its entire per-call budget on reasoning (~17k tokens of thinking,
  empty answer, no file). With thinking off the whole cap goes to file
  content. `--file-think budget` restores a capped think
  (`GVS5H_THINK_BUDGET`, default 3000 tokens) if you want it.
- If a write is still cut off mid-file (`finish_reason=length` with a
  truncated tool JSON), the harness **salvages the partial content**, writes
  it, and tells the model to continue with `mode='append'` — nothing the
  model wrote is lost.
- Every write shows up live in the activity panel
  (`worker 2 · writing page.html (append)`) and is listed in the result panel.

## Interactive commands

| command | what it does |
|---|---|
| `/mode harness\|chat` | switch mode (harness = the agent loop, default) |
| `/spec general\|code\|math` | harness task spec |
| `/stages N` | manager→worker stage budget (default 6; `/iters` still works) |
| `/cap N` | max output tokens per model call (default 8192) |
| `/temp F` | temperature |
| `/think on\|off` | model thinking (reasoning) |
| `/new` | clear chat history, redraw header |
| `/ws` | files in the last run's workspace |
| `/tasks` | last run's task list |
| `/transcript [N]` | last N model calls with token counts |
| `/open` | open the last workspace in Explorer |
| `/save [PATH]` | save the chat as markdown |
| `/help`, `/quit` | |

## Configuration

Prewired to a local llama.cpp server (Qwen3.8-27B-Uncensored-HauhauCS Q8_K_P,
256k context) — base `http://192.168.1.69:8080/v1`, model id exactly as served.
Everything is overridable: CLI flag > env > built-in default.

| flag | env | default | meaning |
|------|-----|---------|---------|
| `--base` | `GVS5H_BASE` | local llama.cpp URL | any OpenAI-compatible base URL |
| `--model` | `GVS5H_MODEL` | local Qwen3.8-27B gguf path | model id sent to the endpoint |
| `--stages` / `--iters` | `GVS5H_ITERS` | 6 | manager→worker stage budget |
| `--cap` | `GVS5H_CAP` | 8192 | max output tokens per model call |
| `--file-cap` | `GVS5H_FILE_CAP` | 20480 | output cap for worker/finalize calls that write files |
| `--file-think` | — | `off` | thinking on file-writing calls: `off` (full cap for content) or `budget` |
| `—` | `GVS5H_THINK_BUDGET` | 3000 | thinking-token cap when `--file-think budget` |
| `--temp` | `GVS5H_TEMP` | 0.3 | temperature |
| `--workspace-dir` | `GVS5H_WS_DIR` | `./Workspace` | artifact root |
| `--no-think` | — | off | disable thinking (`chat_template_kwargs.enable_thinking=false`) |

`GVS5H_*` env names are kept for compatibility with the scaffold. The TUI talks
to the endpoint with its own streaming client (SSE), so the live panel updates
*while* a call is generating — thinking vs. answer tokens, `tok/s`, tool calls,
file writes — instead of jumping from 0 to 1 call when a request finishes.
Infrastructure errors (server briefly down, timeouts, HTTP 429/5xx) are retried
in place and shown in the panel.

Wiring: the TUI sets the scaffold's env vars and routes the model with the
`local:` prefix, then replaces the scaffold's model-call function with the
streaming + tool layer above. `transcript.jsonl` keeps the exact same format the
scaffold wrote before (request, response, reasoning, token counts, finish
reason, attempts).

## The GVS5H scaffold (reference)

PureLogic drives the **GVS5H** manager–worker scaffold — *"Five Qwen3.8-27B
Models Match Claude Fable 5 on LiveCodeBench Hard"* — whose manager plans,
curates a task list, and dispatches workers over a shared file workspace, with
a sample-test verifier, cut-off summarizer and context bounds. The scaffold
code ships under `codebase/v2-current/escalation/`; the paper, the full
benchmark runs and the reproduction notes live in the GVS5H repository:

**https://github.com/slee-persis/GVS5H**

This bundle also carries the paper source (`paper/`), both scaffold versions
(`codebase/`) and the archived run data (`runs/`) so every figure is
reproducible from the copies here.

## Verified end-to-end

All runs below were live against the llama.cpp server
(Qwen3.8-27B-Uncensored-HauhauCS-Aggressive Q8_K_P @
`http://192.168.1.69:8080/v1`) with the prewired defaults. Evidence artifacts
in `tui/evidence/` (machine-readable results, run logs, full transcripts):

- **Tool loop, non-interactive.** `--run "Create a file named hello.txt …"` —
  the worker called `write_file`, the file landed in the run workspace, exit 0.
- **Large artifact, file written.** A full-page canvas animation prompt that the
  pre-tool build truncated at the 8192-token cap on every worker call (6×
  `finish=length`, no file) now completes: the worker writes the HTML across
  several `write_file` calls and the result panel lists the file.
  (`tui/evidence/tui_evidence_file/`)
- **Live agent run, interactive.** Through `start_tui.ps1` in a PTY:
  `What is 12 * 13? Answer with just the final integer.` — live panel ticked
  through every phase with tok/s and the task checklist; result
  `✓ … 5 calls · … tok · finish=stop`, `ANSWER: 156`, tasks 4/4 done, `/quit`
  exit 0. (`tui/evidence/tui_evidence_live156/`)
- **Windows launcher.** Python discovery, `rich` install check, UTF-8 console
  (`chcp 65001` + restore), `Workspace` path from the launch directory, arg
  passthrough — all exercised; script stays pure ASCII for PowerShell 5.1.

Rendering note: the TUI targets rich ≥ 15, where `Text`/`Text.assemble` no
longer parse `[markup]` — static UI strings go through `Text.from_markup` and
dynamic content (answers, logs, paths, file tails) is always plain `Text`.

## Repository layout

```
tui/            the PureLogic agent TUI (this README's subject) + launcher
codebase/       GVS5H scaffold, v1 and v2-current (see link above)
paper/          the GVS5H paper (ICLR 2027 format), source + plots
runs/           archived benchmark workspaces & graded results
```
