# Live in-TUI harness run — "What is 12 * 13?" (verified 2026-09-11)

Interactive TUI (`tui/gvs5h_tui.py`), `/spec general`, `/mode harness`, `/iters 2`,
`/cap 4096`, then the prompt `What is 12 * 13? Answer with just the final integer.`
Workspace: `tui_workspaces/bd6360dab0a0/` (gitignored; artifacts copied here).

## Observed live status bar (from the running TUI)

- `HARNESS iter 2/2  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  03:53  7 calls · 8,284 tok  [managing]`
  with `tasks: ✓ ✓ • (2/3)` and the workspace path — captured while the finalize call
  was still in flight (elapsed timer ticking 03:53 → 03:56 between refreshes).
- Result panel after completion: `HARNESS RESULT · general` —
  `8 calls · 8,693 output tokens · 0 truncated · ws: C:\temp\PureLogic\tui_workspaces\bd6360dab0a0`
  (the +1 call / +409 tokens vs. the live snapshot is the finalize call, confirming
  the counters track the transcript as it fills).
- TUI returned to the prompt afterwards.

## Outcome

- Timeline: plan + 4 initial tasks → ideation (6 approaches) → worker 1 solved →
  manage → worker 2 solved → manage → worker 0 (finalize) solved.
- `answer.md`: computes 12 × 13 two independent ways (direct decomposition and
  (10+2)(10+3)), both agree, `ANSWER: 156`.
