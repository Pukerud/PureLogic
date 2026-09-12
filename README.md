# PureLogic

PureLogic is a terminal agent TUI for a local OpenAI-compatible language model. It
keeps the existing chat, general, code, and math workflows, and adds a first-class
visual/web workflow for producing and checking frontend artifacts.

## Quick start

Windows:

```powershell
powershell -File tui\start_tui.ps1
powershell -File tui\start_tui.ps1 --spec visual --run "Build an accessible weather dashboard"
powershell -File tui\start_tui.ps1 --spec code --run "Write a program that reads stdin"
powershell -File tui\start_tui.ps1 --mode chat --run "Explain recursion"
```

Other platforms:

```bash
python tui/purelogic_tui.py
python tui/purelogic_tui.py --spec visual --run "Create a responsive landing page"
python tui/purelogic_tui.py --run "Create a small SVG animation" --json
```

The default `--spec general` smart-routes prompts containing clear web/UI terms to
the visual workflow. Use `--no-auto-visual` when a general prompt must remain text
only. Explicit `--spec code`, `--spec math`, `--spec visual`, or the equivalent
`/spec` command always wins. `--task-mode` and `--visual` are convenience aliases.

## Modes

- `--mode harness` (default): manager planning, ideation, worker stages, and final
  sign-off over a shared run workspace.
- `--mode chat`: streaming text chat; no browser or artifact workflow is involved.
- `--spec code`: preserves the existing `solution.py` workflow and sample-test
  behavior.
- `--spec math` and `--spec general`: preserve text/math behavior.
- `--spec visual`: workers can write complete HTML, CSS, JavaScript, SVG, JSON, or
  other requested assets with `write_file`. The manager receives browser findings
  after each implementation stage and can assign fixes for another check.

Interactive commands include `/mode`, `/spec`, `/stages`, `/cap`, `/temp`,
`/inspect auto|required|off`, `/vision auto|required|off`, `/tasks`, `/ws`,
`/transcript`, `/open`, `/save`, and `/quit`.

## Visual inspection

Visual inspection is optional and does not affect text-only runs.

1. The worker writes the requested files, preferably with an `index.html` entrypoint.
2. PureLogic launches a headless Chromium context with Playwright against a local
   `file://` URL.
3. It collects title, visible text, landmarks, interactive controls and labels,
   missing image alt text, console messages, page errors, failed requests, and a
   screenshot.
4. The screenshot is sent to a configured vision endpoint through the same streaming
   model abstraction. The browser evidence and optional vision findings are returned
   to the manager as bounded worker feedback.
5. A browser finding keeps the loop in `continue` state and supplies a concrete fix
   task, so the next worker can repair and re-check the page.

Install the browser dependency when needed:

```bash
python -m pip install playwright
python -m playwright install chromium
```

If Playwright is absent, `--inspect auto` reports a clear unavailable warning and
the visual run remains usable; `--inspect required` fails the run clearly. Use
`--inspect off` to disable browser inspection explicitly. The browser context,
profile, screenshot temporary file, and temporary directory are closed and removed
after every check.

Vision is independent of browser inspection. Configure an OpenAI-compatible
multimodal endpoint without putting credentials in code or documentation:

```powershell
$env:PURELOGIC_VISION_BASE = "http://localhost:9000/v1"
$env:PURELOGIC_VISION_MODEL = "a-vision-model-id"
powershell -File tui\start_tui.ps1 --spec visual --vision required --run "Build a card grid"
```

The same settings can be passed as `--vision-base` and `--vision-model`. Use
`--vision auto` (default) to report an unavailable endpoint without breaking the
visual run, `--vision required` to fail clearly when it is missing or unusable, or
`--vision off` to skip screenshot review. Endpoint access and model compatibility
are deployment-specific; a live vision call is not required for DOM-only operation.

## Workspaces and cleanup

Each prompt gets a deterministic folder below `Workspace/` or `GVS5H_WS_DIR`:

```text
<workspace-root>/<prompt-hash>/
  requested files                 final artifacts written by workers
  answer.md                       concise final agent response
  inspection.json                 latest bounded visual inspection result
  result.json                     concise run summary
  transcript.jsonl                only with --keep-transcript/--verbose
```

Manager plans, task lists, notes, and transcripts are used while the run is active.
By default, task/plan/note files and the transcript are removed at completion, so
the workspace retains requested artifacts plus concise results. `--keep-transcript`
or `--verbose` opts into the full model transcript. `--keep-evidence` or `--verbose`
opts into `visual-evidence/latest.png`; otherwise browser screenshots are deleted.
The TUI live panel is in-memory and remains functional without persistent logs.

Rerunning the same prompt clears its run-owned folder first, preventing stale
artifacts from being inspected or returned. Paths supplied to `write_file` are
sandboxed to that run folder.

## Configuration

| option | environment | purpose |
|---|---|---|
| `--base` | `GVS5H_BASE` | text model base URL |
| `--model` | `GVS5H_MODEL` | text model id |
| `--workspace-dir` | `GVS5H_WS_DIR` | artifact root |
| `--stages` | `GVS5H_ITERS` | manager-to-worker budget |
| `--cap` | `GVS5H_CAP` | normal output cap |
| `--file-cap` | `GVS5H_FILE_CAP` | file-worker output cap |
| `--inspect` | `PURELOGIC_INSPECT` | `auto`, `required`, or `off` |
| `--vision` | `PURELOGIC_VISION` | screenshot review policy |
| `--vision-base` | `PURELOGIC_VISION_BASE` | multimodal endpoint base |
| `--vision-model` | `PURELOGIC_VISION_MODEL` | multimodal model id |
| `--keep-transcript` | `PURELOGIC_KEEP_TRANSCRIPT` | retain full transcript |
| `--keep-evidence` | `PURELOGIC_KEEP_EVIDENCE` | retain latest screenshot |

Text model defaults remain compatible with the original local llama.cpp setup.
Every option has a CLI override. Credentials must be provided by the deployment
environment or host-owned secret mechanism, never in source or checked-in docs.

## Repository layout

```text
tui/                              TUI, launcher, visual inspector
tests/                            dependency-light visual inspector tests
codebase/v2-current/escalation/   retained manager/worker and model abstraction
```

The former paper, benchmark, archived-run, vendored LiveCodeBench, old-scaffold,
and checked-in TUI evidence trees are outside the runtime product and are no longer
tracked.

## Verification

From the repository root:

```bash
python -m compileall -q tui tests codebase/v2-current/escalation
python tui/purelogic_tui.py --help
python -m unittest discover -s tests -v
python tui/purelogic_tui.py --spec visual --inspect off --run "Create a page" --json
git diff --check
```

The smoke commands that contact the configured text model require that endpoint to
be running. A live browser test additionally requires the Playwright package and
Chromium installation; the deterministic inspector test does not require either.
