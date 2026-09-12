# Live in-TUI harness run — "What is 12 * 13?" (final build, verified 2026-09-12)

Final agent TUI build (`tui/gvs5h_tui.py`, agent-first rewrite) launched interactively
through `powershell -File tui\start_tui.ps1` in a real PTY, from `C:\Temp\PureLogic`.
Commands: `/iters 2`, then the prompt
`What is 12 * 13? Answer with just the final integer.` (spec `general`, default).

## Observed live activity panel (from the running TUI)

The panel rendered in place (rich `Live`) and ticked while the model was working:

- `starting` → `manager · planning` (after the plan landed: `1 calls · 285 tok`)
- `ideation pass` (log line `[worker:ideate] proposed 5 approaches`, `2 calls · 911 tok`)
- `worker 1 working` (log line `[worker] task 1 -> solved, +0 next steps`,
  `4 calls · 1,633 tok`)
- iteration bar `iter ▓▓▓▓▓▓▓░░░░░░░ 1/2` with task checklist `tasks ✓ · · · (1/4)`
  filling in as `tasks.json` was written
- elapsed timer and call/token counters updated every refresh; spinner animated
  (`⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏` braille frames)

## Result panel

```
RESULT · general · 08:44:44
✓ 02:02 · 5 calls · 1,903 tok out · 0 truncated · finish=stop
156
ANSWER: 156
artifacts → C:\temp\PureLogic\Workspace\bd6360dab0a0
files: answer.md 18B notes.md 212B plan.md 294B solution.py 0B task.md 52B
       tasks.json 493B transcript.jsonl 21,337B
```

The run wrote to `C:\Temp\PureLogic\Workspace\bd6360dab0a0` — the `Workspace`
subfolder of the directory the launcher was run from (per-prompt md5 subfolder).

## Transcript timeline (transcript.jsonl, all finish=stop, no infra failures)

1. `primary_plan` — PLAN + 4 tasks (285 completion tokens)
2. `ideation` — 5 distinct approaches with pitfalls (626)
3. `primary_manage` — STATUS continue, task 1 done, next = compute product (355)
4. `worker:1` — computes 12×13 = 156, verifies (120+36), STATUS solved (367)
5. `primary_manage` — STATUS done, all 4 tasks marked done (270)

`tasks.json` (final): 4/4 `done`. `answer.md`: `156` / `ANSWER: 156`.
`scaffold_notes.md` is the worker's notes as written to the workspace.

`/quit` then returned the PTY to the shell with exit code 0.
