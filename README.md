# GVS5H: Five Qwen3.8-27B Models Match Claude Fable 5 on LiveCodeBench Hard

**Fable 5 Level Coding for a Fifth the Price - or on a Single GPU**

Everything behind the paper's numbers: the paper itself, both versions of the
manager–worker scaffold, the benchmark harness they call, and the complete transcripts and
workspaces of every run the paper reports.

Every figure in the paper was recomputed from the copies in `runs/` before this bundle was
written, and reproduces exactly — see §4.

**No credentials are included.** `escalation/.env`, `.env.groq` and `.env.openrouter` are
excluded, as are any `*.key` / `.credentials*` files, and the tree was scanned for
key-shaped strings before packing. Model routing, caps and reasoning mode are all
environment-driven; supply your own keys.

---

## 0. `tui/` — GVS5H harness TUI (PureLogic)

A terminal **agent** interface pre-wired to a local LLM (llama.cpp, OpenAI-compatible),
driving the v2 manager–worker scaffold from §2. You type tasks in a live terminal and the
GVS5H team does the work. Two modes:

- **harness (default)** — your prompt runs the full GVS5H loop (`multiagent_solve`):
  manager plan → ideation worker → manager/worker iterations over a shared workspace
  (`plan.md`, `notes.md`, `solution.py`/`answer.md`, `tasks.json`) → finalize. While it
  works, a live **activity panel** shows an animated spinner with the current phase
  (manager planning / ideation / worker N / finalizing), elapsed time, call and output-
  token counters, an iteration progress bar and the task checklist (`✓ ✓ → ·`); the run
  then ends in a **result panel** with the final artifact (code/HTML syntax-highlighted,
  or the answer as markdown), stats and the artifact folder path. General-spec answers
  that contain a complete file are additionally saved as `artifact.<ext>` in the
  workspace so the result is a real file, not just text.
- **chat** — streaming chat REPL against the local model; a live panel shows
  `waiting → thinking → generating` with the reasoning streaming in dim italics, then the
  answer is rendered as markdown.

Working directory: artifacts go to a **`Workspace`** subfolder of the directory you run
the launcher from (each run gets its own subfolder). Override with `--workspace-dir` or
`GVS5H_WS_DIR`.

Quick start (anywhere; Python 3.10+, `rich` for the interactive UI):

```
python tui/gvs5h_tui.py                        # interactive agent (harness mode)
python tui/gvs5h_tui.py --run "<problem>"      # one harness run, non-interactive
python tui/gvs5h_tui.py --run "<problem>" --spec code --iters 4
python tui/gvs5h_tui.py --run "Say hi" --mode chat
python tui/gvs5h_tui.py --run "<problem>" --json   # machine-readable result
```

Windows: `powershell -File tui\start_tui.ps1` — finds a Python 3 (`python` → `py -3` →
`python3`), installs `rich` once if it is missing, switches the console to UTF-8 so the
glyphs render, prints the `Workspace` path it will use, and launches the TUI from your
current directory (extra args pass straight through, e.g.
`powershell -File tui\start_tui.ps1 --mode chat --run "hi"`). The original console code
page is restored on exit. Note for maintainers: the script is kept pure ASCII and avoids
`[void](nativeCmd 2>$null)` (a PowerShell 5.1 parse/execution crash) and quote-free
`-c` probes (PS 5.1 strips embedded double quotes from native arguments).

It is prewired to the local llama.cpp server (Qwen3.8-27B-Uncensored-HauhauCS Q8_K_P,
256k context): base `http://192.168.1.69:8080/v1`, model id exactly as served by the
endpoint. Everything is overridable — CLI flag > env > built-in default:

| flag | env | default | meaning |
|------|-----|---------|---------|
| `--base` | `GVS5H_BASE` | local llama.cpp URL | any OpenAI-compatible base URL |
| `--model` | `GVS5H_MODEL` | local Qwen3.8-27B gguf path | model id sent to the endpoint |
| `--iters` | `GVS5H_ITERS` | 6 | manager iteration budget (`MULTIAGENT_MAX_ITERS`) |
| `--cap` | `GVS5H_CAP` | 8192 | max output tokens per model call |
| `--temp` | `GVS5H_TEMP` | 0.3 | temperature |
| `--no-think` | — | off | disable thinking (`chat_template_kwargs.enable_thinking=false`) |

Harness specs: `--spec general|code|math` (default `general`; `code`/`math` use the
scaffold's `CODE_SPEC`/`MATH_SPEC` verbatim). Interactive commands: `/help`, `/mode`,
`/spec`, `/iters`, `/cap`, `/temp`, `/think on|off`, `/new`, `/ws`, `/tasks`,
`/transcript`, `/open` (open the last workspace in Explorer), `/save`, `/quit`.

Wiring: the TUI sets the scaffold's env vars and routes model names with the `local:`
prefix, dispatched in `escalation/orchestrator.py` (`chat()`) to any OpenAI-compatible
`/v1/chat/completions` endpoint — llama.cpp included (its `reasoning_content` field is
captured into the transcript like the other providers). Per-prompt workspaces land in
`<cwd>/Workspace/<md5(prompt)>/` (gitignored), each with the full `transcript.jsonl` of
every model call. Rendering note: the TUI targets rich ≥ 15, where `Text`/`Text.assemble`
no longer parse `[markup]` — static UI strings go through `Text.from_markup` and dynamic
content (answers, logs, paths) is always plain `Text`.

### Verified end-to-end (2026-09-11)

All of the below was run live against the llama.cpp server (Qwen3.8-27B-Uncensored-HauhauCS-Aggressive
Q8_K_P @ `http://192.168.1.69:8080/v1`, 256k context) with the prewired defaults:

- **Endpoint probe.** Non-streaming 183-token completion in 6.7 s (~27 tok/s); SSE streaming emits
  `content` + `reasoning_content` deltas and terminates on `[DONE]` (66 chunks).
- **Non-interactive chat.** `python tui/gvs5h_tui.py --run "Say hi"` streams and exits 0.
- **Harness, general spec** (`--run "What is 17 * 243? Give the decimal result and the same value in
  hexadecimal (0x...)." --spec general --json`). Full loop — manager plan → ideation → worker → finalize —
  in 955 s, 5 calls / 5 407 output tokens; final answer `4131 (decimal), 0x1023 (hex)`, verified by
  two independent methods in the transcript. The log also caught the scaffold's retry loop working:
  13+ consecutive `URLError` retries while the local server was briefly unreachable, then clean recovery.
- **Harness, code spec** (`--run "Write a complete, self-contained Python program that reads one
  integer n from standard input and prints the sum of the first n positive integers on a single line.
  If n is 0 or negative, print 0." --spec code --json`). 294 s, 7 calls / 1 693 output tokens; produced
  a working `sys.stdin`-reading program (`n*(n+1)//2`); both worker tasks marked solved.
- **Interactive TUI.** Chat mode streams the model's thinking (dim) and renders the answer as markdown
  (5-line haiku request; saved transcript below). Harness mode runs the same loop in a background
  thread with a live status bar — iteration budget, elapsed time, call/token counts, phase and task
  list — and a result panel with the final artifact and workspace paths.
- **Live status bar, interactive.** In the TUI: `/spec general`, `/mode harness`, `/iters 2`, `/cap 4096`,
  then `What is 12 * 13? Answer with just the final integer.` While the run was in flight the status
  bar ticked (elapsed `03:53` → `03:56`, `7 calls · 8,284 tok`, tasks `✓ ✓ • (2/3)`); the result panel
  then showed `8 calls · 8,693 output tokens · 0 truncated` — the +1 call / +409 tokens being the
  finalize call that landed after the live snapshot, confirming the counters track the workspace
  transcript as it fills. `answer.md`: `156`, cross-checked two independent ways.
- **Windows launcher.** `powershell -File tui\start_tui.ps1 --run "Say hi in exactly three words"`
  streamed the reply and exited 0 (Python discovery, rich check, UTF-8 console and arg
  passthrough all exercised); an interactive launch in a console rendered the full TUI (banner,
  model/endpoint, status bar) and `/quit` exited cleanly with 0.

### Verified end-to-end (2026-09-12, agent TUI v2)

Rebuilt the TUI as an agent-first interface (harness default, live activity panel,
`Workspace` working dir) and re-verified against the same llama.cpp server:

- **Markup fix (rich 15).** The raw `[bold …]` tags in the v1 screenshot were a rich 15.0.0
  behaviour change: `Text()`/`Text.append()`/`Text.assemble()` no longer parse `[markup]`
  (only `Console.print(str)` and `Panel(str)` do). All static UI strings now go through
  `Text.from_markup`, dynamic content stays plain `Text`; rendered output shows no stray tags.
- **Launcher + workspace path.** `powershell -File tui\start_tui.ps1 --run "Say hi in exactly
  three words" --mode chat` run from `C:\Temp` printed the banner `GVS5H TUI - agent artifacts
  will be written to: C:\Temp\Workspace`, streamed the reply and exited 0 (the script no longer
  changes directory; the TUI uses the caller's CWD).
- **Full agent loop, non-interactive.** `--run "Write a minimal valid HTML5 page titled Ping …"` 
  `--iters 2` ran plan → ideation → manage → worker → manage → finalize, wrote the run folder
  under `C:\Temp\Workspace\<md5>\` and saved the extracted page as `artifact.html` (see evidence
  below).
- **Live agent run, final build.** Through `start_tui.ps1` in a PTY: `/iters 2`, then
  `What is 12 * 13? Answer with just the final integer.` The activity panel ticked live through
  `starting` → `manager · planning` → `ideation pass` → `worker 1 working` (counters
  `1 calls · 285 tok` → `4 calls · 1,633 tok`, iteration bar `1/2`, task checklist
  `✓ · · · (1/4)` as `tasks.json` filled), then the result panel:
  `✓ 02:02 · 5 calls · 1,903 tok out · 0 truncated · finish=stop`, `ANSWER: 156`, artifacts →
  `C:\Temp\PureLogic\Workspace\bd6360dab0a0` (the `Workspace` subfolder of the directory the
  launcher ran from). Transcript: `primary_plan` → `ideation` (5 approaches) →
  `primary_manage` → `worker:1` (solved) → `primary_manage` (done) — 5/5 calls `finish=stop`,
  tasks 4/4 done (`tui/evidence/tui_evidence_live156/`); the rebuilt header (model + endpoint +
  workspace path) rendered once, and `/quit` exited 0.

Evidence artifacts in `tui/evidence/`: `out_e2e_general.json` / `out_e2e_code.json` (the `--json`
results above, incl. workspace path, call and token counts), `out_e2e_general.log` / `out_e2e_code.log`
(the live run logs), `tui_evidence_chat.md` (saved interactive chat transcript),
`tui_evidence_harness/` (a full in-TUI harness run: `transcript.jsonl`, `solution.py`, `plan.md`,
`tasks.json`), and `tui_evidence_live156/` (the live in-TUI 12×13 run on the final build:
`notes.md` with the observed live-panel and result-panel values, `answer.md`, `tasks.json`,
`plan.md`, `scaffold_notes.md`, and the full 5-call `transcript.jsonl`).

---

## 1. `paper/`

`zero_shot_self_orchestration_with_ledger_based_control_for_improved_llm_coding_performance_2026-08-25.tex` and the PDF built from
it, plus the generated `fig-*.tex` figure files, `references.bib`, and, in `plots/`, every
PNG the document references.

The manuscript is formatted for **ICLR 2027**. `iclr2027_conference.sty`, `.bst`,
`fancyhdr.sty` and `natbib.sty` are checked in beside it so it builds from a clean clone;
the upstream `iclr-2027-style-files.zip` and the `iclr2027/` directory it unpacks to are
gitignored.

It builds in place:

```bash
latexmk -pdf zero_shot_self_orchestration_with_ledger_based_control_for_improved_llm_coding_performance_2026-08-25.tex
```

**One line differs from the repository copy**: line 24, `\newcommand{\plotdir}{plots}`,
points at `./plots` here instead of the repo's `../escalation/plots`.

**Anonymity is a toggle**: `\finalversionfalse` on line 21 builds the double-blind
submission — anonymous author block, review line numbers, no repository URLs.
`\finalversiontrue` restores the author block, the acknowledgements, the
author-contribution statement and the links. Submit with it false.

Main text (§1–§7) is 9 pages, ICLR's strict limit. The AI use, ethics and reproducibility
statements, the references and the appendices do not count against it.

The `fig-*.tex` files are generated, not hand-written — captions and footnotes live in the
plot script that draws each chart, and `run_bench_script/make_figures_tex.py` turns them
into LaTeX. Edit a caption in the plot script and re-run that, never in `fig-*.tex`.
`plot_bars.py` draws the three compact panels of Figure 5; the other scripts draw one
figure each — Figures 1–4 and 6–8.

Cross-references are `\label`/`\ref`, including from the generated captions, so the float
and section numbering survives reordering. The `label=` keys in `make_figures_tex.py` and
the `\cite` keys in `references.bib` are load-bearing: renaming one silently breaks a
generated caption.

---

## 2. `codebase/`

### the original scaffold
Produced the **OpenRouter-served model set** (§5.4).

### the updated scaffold
Produced the **seven first-party arms** (§5.1–§5.3).

The paper (Appendix A) names four differences, all acting on the manager arm alone. They are
verifiable directly in `escalation/multiagent.py`:

| | v1 | v2 |
|---|---|---|
| round budget (`MULTIAGENT_MAX_ITERS`) | 4 | 10 |
| sample-test verifier (§3 step 5) | absent | present |
| cut-off summarizer | absent | present |
| size bounds on workspace files | absent | present |

Because the single-call baseline is one call under either version, manager-minus-single
deltas are **not strictly comparable** between the two sets. That is why the paper reports
the OpenRouter models as their own condition (§5.4) rather than pooling them with §5.1.

Two further differences worth knowing, which the paper does not enumerate:

- **Problem selection.** v1 selects the latest 100 hard problems at run time
  (`--lcb 100`). Id pinning arrived later, so `escalation/lcb100_hardest_v6.json` — the
  frozen list of 100 `question_id`s — exists only under `v2-current/`. Both resolve to the
  same 100 problems; the pinned file makes it reproducible rather than date-dependent.
- **One post-paper fix is present in v2.** `orchestrator.py` now compares a suspected
  provider clamp against the cap actually sent rather than the configured cap. This was
  written *after* the runs and did not affect them (it only bites when a 400/context error
  shrinks the cap mid-run, which happened to a model not in the paper).

Two analysis tools under `v2-current/escalation/` were written for this revision of the
paper and are worth calling out:

- **`regrade.py`** re-scores stored generations against the *fixed* evaluator without
  re-running any model. Appendix C explains why this was necessary; §4 below shows how to repeat
  it. It writes `<name>.regraded.json` beside each input and never modifies the input.
- **`capmatch_q38.py`** replays Qwen3.8-27B's single-call generations against a 128k output
  cap, token-exactly, using the serving stack's own tokenizer. That arm was generated at
  250k while its manager twin ran at 128k; the paper reports it cap-matched (Appendix B) so the
  pair is like-for-like. It writes `*.cap128k.json`.

### `livecodebench/`
The benchmark harness both versions import for dataset loading, code extraction and
hidden-test grading (`escalation/run_bench.py` puts it on `sys.path`). Shared by v1 and
v2 — neither scaffold vendors its own copy.

**This copy carries the evaluator fix of Appendix C.** Upstream's stdin mock implemented
`MockBuffer.readline()` as a stateless expression that returned line 1 on every call, so
any solution reading multi-line input through `sys.stdin.buffer.readline()` was scored
wrong however correct it was. See `lcb_runner/evaluation/testing_util.py`. Every number in
the paper is reported after fixing it.

---

## 3. `runs/` — every run the paper reports

| directory | condition | paper location | workspaces |
|---|---|---|---|
| `firstparty-128k-reasoning-on-5pass/` | 128k cap, reasoning on, ×5, first-party APIs — Qwen3.8-27B, GPT-5.6-Luna, GPT-5.6-Terra | §5.1–§5.3, Figures 2–4 | 3,200 |
| `fable5-128k-reasoning-on-5pass/` | 128k cap, reasoning on, ×5, Anthropic API — Claude Fable 5, single-call only | §5.1–§5.3, Figures 2–4 | 500 |
| `16k-reasoning-off-5pass/` | 16k cap, reasoning off, ×5 | §5.4, Figures 5b and 7 | 4,500 |
| `128k-reasoning-off-1pass/` | 128k cap, reasoning off, ×1 | §5.4, Figures 5c and 8 | 800 |
| `128k-reasoning-on-1pass/` | 128k cap, reasoning on, ×1 | §5.4, Figures 5a and 6 | 901 |
| `q9-reasoning-on-archived/` | Qwen3.5-9B with thinking on | §6.1 (limitation) | — |

`per_problem_tokens.json` at the top of `runs/` holds the per-problem, per-pass token
counts every dollar figure in §5.2 is computed from, including calls that were retried and
discarded — those were generated and would be billed. Regenerate it with
`run_bench_script/extract_tokens.py`.

The six symlinks in `runs/` (`4models-1pass-reason-on`, `fable5-5pass-single`,
`models-lcb-5pass`, `128k-clean`, `results_think_high`, `ws_think_high`) are the names the
plot scripts address these directories by. They exist so every script runs unmodified;
extract with a tool that preserves symlinks, or re-point the paths at the top of each
script.

### What a run directory holds

`results/` (graded JSON, one file per config/pass) and `ws/` (per-problem workspaces). A
workspace holds what the roles actually read and wrote:

```
<hash>/task.md          the problem statement
<hash>/plan.md          the manager's overarching plan
<hash>/tasks.json       the task list [{id, desc, status, result}]
<hash>/notes.md         accumulated ideas / findings / partial proofs
<hash>/solution.py      the code that was graded
<hash>/transcript.jsonl every model call, in order, with its role
```

Workspace directory names are content hashes, not question ids. To map one to a problem,
match `solution.py` against the `code` field of the corresponding record in `results/`.

Result records carry `question_id`, `code`, `passed`, and — in the 128k conditions —
`status` ∈ {ok, truncated, empty_stop, empty, error}, `finish_reason` and
`completion_tokens`. The 16k ×5 runs predate that instrumentation and carry only the
extracted code and its pass/fail, which is why §5.4's 16k column is a broader *no-code*
count rather than a strict truncation count.

### Which file is the graded one

Three suffixes appear side by side, and the paper reports the last of them:

- `<name>.json` — as generated, scored by the evaluator as it stood at run time.
- `<name>.cap128k.json` — Qwen3.8-27B's single arm only: the same generations truncated to
  128k output tokens and the solution re-extracted from that prefix (Appendix B).
- `<name>.regraded.json` — re-scored on the fixed evaluator (Appendix C). Each record keeps
  `passed_before_regrade` beside the corrected `passed`, and the file keeps
  `pass@1_before_regrade`, so the correction is auditable rather than silent.

**Every figure and table in the paper reads the `.regraded.json` files**, and for
Qwen3.8-27B's single arm the `.cap128k.regraded.json` twin.

`q9-reasoning-on-archived/` is the evidence for §6.1's claim that Qwen3.5-9B with reasoning
enabled was untestable: these attempts were archived rather than graded, and the paper
reports the model as a limitation rather than a data point.

---

## 4. Reproducing

**The figures.** With `uv` available, from
`codebase/v2-current/escalation/run_bench_script/`:

```bash
uv run --with matplotlib --with numpy --with scipy python plot_agent_loop_flowchart.py   # Figure 1
uv run --with matplotlib --with numpy --with scipy python plot_4new_5pass_reason_on.py   # Figure 2
uv run --with matplotlib --with numpy --with scipy python plot_cost_5_pass.py            # Figure 3 + Table 1
uv run --with matplotlib --with numpy --with scipy python plot_cost_vs_score.py          # Figure 4
uv run --with matplotlib --with numpy --with scipy python plot_bars.py                   # Figures 5a-5c
uv run --with matplotlib --with numpy --with scipy python plot_128k_reason_on_1_pass.py  # Figure 6
uv run --with matplotlib --with numpy --with scipy python plot_16k_reason_off_5_pass.py  # Figure 7
uv run --with matplotlib --with numpy --with scipy python plot_128k_reason_off_1_pass.py # Figure 8
uv run --with matplotlib --with numpy --with scipy python make_figures_tex.py            # -> paper/fig-*.tex
```

Those nine commands write exactly the ten files `paper/plots/` holds and nothing else —
one vector PDF per figure, light theme only. Each prints its table to stdout first; those
printed numbers are the ones in the paper.

`plot_q38_vs_fable5_5_pass.py` still runs and is still where the manager-vs-Fable-5
numbers are computed, but the paper does not include its chart — Figure 2 carries the same
comparison — so it is not in the list above and a default run writes nothing for it.
`plot_agent_loop_flowchart.py` likewise keeps `FIG_46710A5`, the original scaffold's
diagram, defined but not emitted: it is the only drawing of the version behind §5.4.

**The grading.** To repeat the Appendix C correction from the raw generations:

```bash
python escalation/regrade.py runs/<condition>/results/*.json
```

**The runs themselves.** Both scaffolds are driven through `escalation/run_bench.py`,
which expects the benchmark harness importable from the repository root:

```bash
python escalation/run_bench.py --engine {single,multiagent} --only lcb --lcb 100 \
       --ids-file escalation/lcb100_hardest_v6.json --parallel N --out results.json
```

`run_bench_script/run_4models_1pass_reason_on.sh` is the driver for the three first-party
API arms and `run_fable5_5pass_single.sh` for Fable 5, both including the per-provider
caveats. Model routing, caps and reasoning mode are environment-driven; see the header
comments in `orchestrator.py`, and Appendix B of the paper for what each provider's thinking
control was set to.

---

## 5. What was deliberately left out

- **Credentials**: `.env*`, `*.key`, `.credentials*` — see the note at the top.
- **`.gitignore`**, per request.
- **One-off scripts not part of the benchmark**: `aggregate_big.py` and `run_big.sh`, which
  serve the exploratory LCB+AIME sweep that §6.1 lists among the benchmarks run but not
  reported.
- **Superseded reruns** inside the first-party sweep — the `.clampbug`, `.pre-notesfix` and
  `.pre-planfix` workspaces. The five graded passes per arm are what ship.
- **`LiveCodeBench/output/` and `LiveCodeBench/claude_transcripts/`** (~177 MB). These
  belong to a *separate* experiment in the same repository — a wrapper that benchmarks the
  `claude -p` CLI as an agent — which shares no data with this paper.
- **Dark-theme chart variants**, reproducible from the plot scripts. The light variants the
  paper uses ship in `paper/plots/`.
