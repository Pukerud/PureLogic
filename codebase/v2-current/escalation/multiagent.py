"""Multi-agent collaborative solver: a primary "manager" plus worker subagents,
coordinating over a SHARED FILESYSTEM WORKSPACE. Emulates the Claude-Code
orchestrator/subagent pattern, but every role is the SAME model
(default groq:qwen/qwen3.6-27b) invoked in a fresh context.

Control flow (as requested):
  1. PRIMARY writes an overarching plan and seeds the task list.
  2. The FIRST WORKER just thinks about the problem and proposes approaches.
  3. Loop: PRIMARY decides which task to prioritize and spawns a worker; the
     worker completes it, updates the shared code/proof, and proposes next steps.
     Primary stops when it judges the problem solved (or the iteration budget
     runs out).
  4. A FINALIZE worker emits the definitive artifact, which is then graded.

Shared workspace (real files, so every role sees the same evolving state):
    <ws>/task.md      the problem statement
    <ws>/plan.md      the primary's overarching plan
    <ws>/notes.md     curated ideas / proofs / findings       (read by every role; each
                      worker REWRITES it rather than appending, so it stays organised and
                      bounded -- append-only growth used to blow past the context window)
    <ws>/solution.py  current best code   (code problems)     (read by every role)
    <ws>/answer.md    current best answer  (math problems)     (read by every role)
    <ws>/transcript.jsonl  every call for this problem (see _record); not read by any role
    <ws>/tasks.json   debug dump of the task list [{id, desc, status, result}] -- written
                      after each manager round but NOT read back by any role. The live task
                      list is the in-memory `tasks` value threaded through the loop and
                      re-rendered into the manager's prompt; `id` is renumbered from 1 on
                      each curation and `result` is always "" (neither is used downstream).
"""

import os
import re
import sys
import time
import json
import hashlib
import subprocess

from orchestrator import chat  # groq:/ollama dispatch

MODEL = os.environ.get("MULTIAGENT_MODEL", "groq:qwen/qwen3.6-27b")
MAX_ITERS = int(os.environ.get("MULTIAGENT_MAX_ITERS", "10"))  # primary->worker cycles

# Some models answer the PROBLEM instead of filling in the manager's response format, and
# emit no "### " headers at all -- so _sections() finds nothing and plan/ideation both parse
# to zero items, silently collapsing the manager into a single worker round. Verified on
# Muse-Glimmer-30B: 87k characters of thinking on the plan call that never once mention the
# orchestrator role, opening with "### Question" and closing with "produce final code".
# The same model obeys an explicit prohibition perfectly when one is stated, so the fix is
# to demand the literal headers rather than to describe the sections.
# Gated per-arm so every other model keeps the byte-identical prompt the plotted runs used:
# "1" = on, "0" = off, "auto" (default) = sniff the model name.
# SET IT EXPLICITLY for a local arm: a litellm route name ("small-model") says nothing about
# the weights behind it, so auto-detection silently fails there and the manager collapses
# again with no error.
STRICT_FORMAT = os.environ.get("MULTIAGENT_STRICT_FORMAT", "auto")


def _strict():
    return STRICT_FORMAT == "1" or (
        STRICT_FORMAT == "auto" and any(s in MODEL.lower() for s in ("muse", "glimmer")))


def _format_rule(first, *rest):
    """Mandatory-format addendum: name the headers as literal lines and forbid solving."""
    others = "".join(f", then the literal line '### {r}'" for r in rest)
    return ("\n\nTHE FORMAT IS MANDATORY AND YOUR REPLY IS PARSED BY A PROGRAM. "
            f"Begin your reply with the literal line '### {first}'{others}. "
            "Write nothing before the first header and nothing outside these sections. "
            "Do NOT solve the problem and do NOT write code -- a later worker does that. "
            "A reply without these exact header lines is discarded.")
MAX_TASKS = int(os.environ.get("MULTIAGENT_MAX_TASKS", "12"))  # cap on live task list
WS_ROOT = os.environ.get("MULTIAGENT_WS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ws"))


# --- workspace helpers -----------------------------------------------------

def _slug(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _read(ws, name):
    p = os.path.join(ws, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def _write(ws, name, content):
    with open(os.path.join(ws, name), "w", encoding="utf-8") as f:
        f.write(content)


def _append(ws, name, content):
    with open(os.path.join(ws, name), "a", encoding="utf-8") as f:
        f.write(content)


# --- transcript logging ----------------------------------------------------
# Every agent call for a problem is appended to <ws>/transcript.jsonl: the first
# line is a _meta record, then one record per call
# {role, request, response, reasoning, discarded, token counts}.

def _record(ws, rec):
    # utf-8: the thinking text contains characters (e.g. superscript digits) that the
    # Windows cp1252 default codec cannot encode, which would crash mid-run.
    with open(os.path.join(ws, "transcript.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _chat(ws, role, messages, temperature, meta=None):
    """Model call that also transcripts the full request/response for this problem.
    If `meta` is passed it is populated with {finish_reason, completion_tokens}.

    The record holds the WHOLE generation, not just what gets graded: `reasoning` is the
    model's thinking (empty when reasoning is off), and `discarded` holds any attempt that
    was rerouted/retried away -- otherwise those tokens leave no trace anywhere.
    """
    m = meta if meta is not None else {}
    resp = chat(MODEL, messages, temperature=temperature, meta=m)
    _record(ws, {"t": time.time(), "role": role, "request": messages, "response": resp,
                 "reasoning": m.get("reasoning"), "thinking_blocks": m.get("thinking_blocks"),
                 # Anthropic returns a SUMMARY of the thinking; vLLM/DashScope return the real
                 # chain of thought in the same `reasoning` field. Without this flag the two are
                 # indistinguishable on disk, and any later analysis of thinking length or
                 # content would silently compare a summary against a transcript.
                 "reasoning_is_summary": m.get("reasoning_is_summary"),
                 "finish_reason": m.get("finish_reason"), "completion_tokens": m.get("completion_tokens"),
                 "prompt_tokens": m.get("prompt_tokens"), "provider": m.get("provider"),
                 "attempts": m.get("attempts"), "discarded": m.get("discarded"),
                 "infra_exhausted": m.get("infra_exhausted")})
    return resp


# --- parsing helpers -------------------------------------------------------

def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code(text):
    """Remove fenced code blocks. Ideation must contribute approaches in prose, never a
    finished program -- otherwise the manager reads it as solved and skips the worker loop."""
    return re.sub(r"```.*?```", "[code omitted -- approach only]", text, flags=re.DOTALL).strip()


def _sections(text):
    """Split a model reply into {HEADER: body}. Accepts '### H', '**H**', 'H:'."""
    out, cur, buf = {}, None, []
    for line in _strip_think(text).splitlines():
        s = line.strip()
        m = (re.match(r"^#{1,3}\s*([A-Za-z_]+)\s*$", s)
             or re.match(r"^\*\*([A-Za-z_]+)\*\*\s*:?\s*$", s)
             or re.match(r"^([A-Z_]{3,})\s*:\s*$", s))
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).upper(), []
        else:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf).strip()
    return out


def _bullets(text):
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*\S)", line)
        if m:
            items.append(m.group(1).strip())
    return items


def _extract_py(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", _strip_think(text), re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_tasks(text):
    """Parse the primary's curated list: lines like '- [done] ...' / '- [todo] ...'."""
    out = []
    for b in _bullets(text):
        m = re.match(r"\[\s*([a-z_ ]+?)\s*\]\s*(.*)", b, re.I)
        if m:
            raw = m.group(1).lower().replace(" ", "").replace("_", "")
            desc = m.group(2).strip()
            status = "done" if raw == "done" else ("in_progress" if raw in ("wip", "inprogress") else "pending")
        else:
            desc, status = b, "pending"
        if desc:
            out.append({"id": len(out) + 1, "desc": desc, "status": status, "result": ""})
    return out[:MAX_TASKS]


ANS_RE = re.compile(r"ANSWER:\s*\S", re.I)  # any non-empty final answer (int or expression)
# A genuine final answer is short. Anything longer is a reasoning dump, not an answer.
MAX_ANSWER_CHARS = 20000
# plan.md is injected into EVERY later prompt, so it must stay small. ~4k chars is about
# 1k tokens -- ample for a 3-6 sentence strategy, and cheap to carry through 20+ calls.
MAX_PLAN_CHARS = 4000


def _has_answer(ws, spec):
    """True if the workspace already holds a usable final artifact."""
    if spec["kind"] == "code":
        return bool(_read(ws, "solution.py").strip())
    return bool(ANS_RE.search(_read(ws, "answer.md")))


# --- roles -----------------------------------------------------------------

def _save_tasks(ws, tasks):
    _write(ws, "tasks.json", json.dumps(tasks, indent=2))


def _add_tasks(tasks, descs):
    have = {t["desc"].lower() for t in tasks}
    nid = max([t["id"] for t in tasks], default=0)
    for d in descs:
        d = d.strip()
        if d and d.lower() not in have and len(tasks) < MAX_TASKS:
            nid += 1
            tasks.append({"id": nid, "desc": d, "status": "pending", "result": ""})
            have.add(d.lower())


def _primary_plan(problem, spec, ws, log):
    kind = spec["kind"]
    sys = (
        "You are the PRIMARY orchestrator (manager) of a small team of workers, all "
        f"expert at {'competitive programming' if kind == 'code' else 'olympiad mathematics'}. "
        "Given a problem, produce a short overarching plan to solve it, then a task list "
        "the workers can pick up. Respond with EXACTLY these sections:\n"
        "### PLAN\n<3-6 sentence strategy>\n"
        "### TASKS\n<3-6 bullet tasks, each a concrete unit of work>"
    )
    if _strict():
        sys += _format_rule("PLAN", "TASKS")
    reply = _chat(ws, "primary_plan", [
        {"role": "system", "content": sys},
        {"role": "user", "content": problem},
    ], temperature=0.3)
    sec = _sections(reply)
    # Bounded on write. plan.md goes into every subsequent prompt, so one oversized value
    # poisons the whole problem. The old fallback wrote the ENTIRE reply whenever no PLAN
    # section parsed -- and a plan call that truncates at the token cap parses no sections at
    # all, so it wrote a ~128k-token dump. Measured 2026-08-13 on muse: two problems carried
    # a 205KB and a 483KB plan.md, overflowed the window on every later call, and produced no
    # code at all. Same failure the answer.md guard below already prevents.
    _write(ws, "plan.md", (sec.get("PLAN") or reply.strip())[:MAX_PLAN_CHARS])
    tasks = []
    _add_tasks(tasks, _bullets(sec.get("TASKS", "")))
    log(f"    [primary] plan + {len(tasks)} initial tasks")
    return tasks


def _ideation_worker(problem, spec, ws, log):
    """First worker: think about the problem and PROPOSE approaches (returns a list).
    Its proposals are folded into the task list by the primary, not appended blindly."""
    sys = (
        "You are the FIRST WORKER. Do NOT solve the problem and do NOT write any code. "
        "Just think about it: identify the core difficulty, then list SEVERAL DISTINCT "
        "candidate approaches (genuinely different algorithms / data structures / problem "
        "reductions, not variations of one idea), and note pitfalls for each. Describe each "
        "approach in prose only -- absolutely no code blocks; a later worker will implement. "
        "Respond with EXACTLY:\n### NOTES\n<your analysis>\n"
        "### NEXT\n<bullet list of distinct approaches to try next>"
    )
    if _strict():
        sys += _format_rule("NOTES", "NEXT")
    reply = _chat(ws, "ideation", [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"PROBLEM:\n{problem}\n\nPLAN:\n{_read(ws, 'plan.md')}"},
    ], temperature=0.4)
    sec = _sections(reply)
    # Strip any code the ideation wrote anyway: if a full solution reaches the notes, the
    # manager reads it as "already solved", marks every task done, and the worker loop never
    # runs. Ideation must contribute approaches, not a finished program.
    # Bounded, for the same reason as plan.md: this is the first thing written to notes.md and
    # it is injected into every later prompt. `reply.strip()` is the fallback when ideation
    # truncates and no section parses -- uncapped, that is a ~128k-token dump seeding the file
    # every worker then reads. (Written, not appended, only because notes.md is empty here.)
    _append(ws, "notes.md",
            f"\n## ideation\n{_strip_code(sec.get('NOTES') or reply.strip())[:MAX_PLAN_CHARS * 2]}\n")
    proposals = [_strip_code(p) for p in _bullets(sec.get("NEXT", ""))]
    log(f"    [worker:ideate] proposed {len(proposals)} approaches")
    return proposals


def _primary_manage(problem, spec, ws, tasks, proposals, last_summary, log):
    """Primary monitors progress, curates the task list, and decides done/next.

    Reviews the current solution + the latest worker's result, then rewrites the task
    list (merge duplicates, mark done, fold in only genuinely new proposals) and either
    declares the problem solved or names the single next task.
    Returns (status, next_desc, curated_tasks).
    """
    kind = spec["kind"]
    cur = _read(ws, "solution.py" if kind == "code" else "answer.md").strip()
    task_lines = "\n".join(
        f"- [{'done' if t['status'] == 'done' else 'todo'}] {t['desc']}" for t in tasks
    ) or "(none)"
    prop_lines = "\n".join(f"- {p}" for p in proposals) or "(none)"
    sys = (
        "You are the PRIMARY orchestrator and manager. You OWN the task list and decide "
        "when the problem is solved. Review the current progress and the latest worker's "
        "result, then:\n"
        "- The LATEST WORKER RESULT may include a SAMPLE TESTS verdict from actually running "
        "the code. Treat it as ground truth: only set STATUS 'done' if the solution PASSED the "
        "sample tests; if it FAILED, you MUST set STATUS 'continue' and choose a task that "
        "fixes the failing case or switches to a different approach.\n"
        "- If the current solution/answer is complete and correct, set STATUS to 'done'.\n"
        "- Otherwise CURATE the task list: merge duplicates, drop finished or irrelevant "
        "items, mark completed ones [done], and fold in ONLY genuinely new sub-tasks from "
        "the proposals. Then choose the single most valuable next task.\n"
        "- IMPORTANT: if the current solution keeps failing, or the last worker made no real "
        "progress, do NOT keep refining the same idea. Switch to a DIFFERENT approach "
        "(a different algorithm / data structure / reduction) from the notes, or ask for a "
        "new one. You have many rounds -- use them to try distinct approaches, not to polish "
        "a stuck one.\n"
        "Respond with EXACTLY these sections:\n"
        "### STATUS\n<done|continue>\n"
        "### NEXT\n<exact text of the ONE task to do next; omit if done>\n"
        "### TASKS\n<curated list, one per line, each '- [done] ...' or '- [todo] ...'>"
    )
    if _strict():
        sys += _format_rule("STATUS", "NEXT", "TASKS")
    reply = _chat(ws, "primary_manage", [
        {"role": "system", "content": sys},
        {"role": "user", "content": (
            f"PROBLEM:\n{problem}\n\nCURRENT {'SOLUTION' if kind == 'code' else 'ANSWER'}:\n"
            f"{cur or '(none yet)'}\n\nNOTES:\n{_read(ws, 'notes.md')}\n\n"
            f"CURRENT TASK LIST:\n{task_lines}\n\nLATEST WORKER RESULT: {last_summary}\n\n"
            f"PROPOSED NEW STEPS:\n{prop_lines}"
        )},
    ], temperature=0.2)
    sec = _sections(reply)
    curated = _parse_tasks(sec.get("TASKS", "")) or tasks
    status = "done" if sec.get("STATUS", "").strip().lower().startswith("done") else "continue"
    if status == "done" and not cur:  # invariant: can't be done with no answer produced yet
        status = "continue"
    nxt = (_bullets(sec.get("NEXT", ""))[:1] or [sec.get("NEXT", "").strip()])[0]
    nxt = re.sub(r"^[-*]\s*", "", nxt).strip()
    if status == "continue" and not nxt:  # manager wants to continue but named no task
        todo = [t for t in curated if t["status"] != "done"]
        nxt = todo[0]["desc"] if todo else ""
    if not cur and not nxt:
        # No solution produced yet and no task chosen -- typically the manager marked every
        # task done off the ideation. Require a real implementation round: a WORKER must
        # actually write the solution before the problem can be considered solved.
        status, nxt = "continue", "Implement the full working solution for the most promising approach in the notes."
    log(f"    [primary-manage] {status}" + (f", next: {nxt[:60]}" if status == "continue" else ""))
    return status, nxt, curated


def _summarize_cutoff(ws, reply, task_desc):
    """A worker ran out of tokens mid-thought. Summarize its partial attempt (in a fresh,
    cheap call) so the ideas are not lost and the manager can act on them."""
    txt = _strip_think(reply) or reply
    snippet = txt if len(txt) <= 9000 else txt[:3500] + "\n...[middle omitted]...\n" + txt[-5500:]
    sysmsg = (
        "A worker's solution attempt was CUT OFF when it hit the token limit. Summarize its "
        "partial attempt in 3-5 sentences: which approach it was pursuing, what it established "
        "or ruled out, how far it got, and what remained unfinished. Be concrete so another "
        "worker can resume or judge it. Do NOT try to finish the solution yourself."
    )
    s = _chat(ws, "cutoff_summary", [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": f"TASK: {task_desc}\n\nCUT-OFF ATTEMPT:\n{snippet}"},
    ], temperature=0.2)
    return _strip_think(s).strip() or "(the cut-off attempt could not be summarized)"


def _worker(problem, spec, ws, task, log, finalize=False):
    kind = spec["kind"]
    cur = _read(ws, "solution.py" if kind == "code" else "answer.md")
    # NOTES is a REWRITE of the whole file, not an addition to it. See the _write below.
    notes_fmt = (
        "### NOTES\n<the COMPLETE notes file, rewritten. You are shown the current NOTES "
        "above: fold your findings into them, keep what still matters, and DELETE anything "
        "superseded, disproven, or now obvious. This REPLACES the file, so whatever you omit "
        "is gone. Organise it as '- **Topic:** ...' bullets, under ~800 words. Do NOT use "
        "markdown headings (#, ##), bold-only lines, or ALLCAPS: lines anywhere inside this "
        "section -- the reply is split on those, so they would truncate your notes.>\n"
    )
    if kind == "code":
        out_fmt = (
            "### CODE\n```python\n<the FULL updated self-contained program>\n```\n"
            + notes_fmt +
            "### NEXT\n<bullet list of remaining steps, or 'none'>\n### STATUS\n<solved|continue>"
        )
    else:
        out_fmt = (
            "### ANSWER\n<full worked solution, ending with the exact final "
            "ANSWER: line your instructions require>\n"
            + notes_fmt +
            "### NEXT\n<bullet list of remaining checks, or 'none'>\n### STATUS\n<solved|continue>"
        )
    goal = (
        "Produce the DEFINITIVE final solution now, using all notes and current work."
        if finalize else f"Complete this task: {task['desc']}"
    )
    sys = (
        f"You are a WORKER subagent, {spec['solver_system']} "
        "You share a workspace with the team. Build on the current work and notes where "
        "useful -- but if your task is to try a different approach, write a FRESH solution "
        "for that approach instead of patching the stuck one. "
        f"Respond with EXACTLY these sections:\n{out_fmt}"
    )
    wmeta = {}
    reply = _chat(ws, "finalize" if finalize else f"worker:{task['id']}", [
        {"role": "system", "content": sys},
        {"role": "user", "content": (
            f"PROBLEM:\n{problem}\n\nPLAN:\n{_read(ws, 'plan.md')}\n\n"
            f"NOTES:\n{_read(ws, 'notes.md')}\n\nCURRENT WORK:\n{cur or '(none yet)'}\n\n"
            f"YOUR TASK: {goal}"
        )},
    ], temperature=0.2, meta=wmeta)
    sec = _sections(reply)
    wrote = False  # did THIS call produce a fresh artifact? (else solution.py/answer.md is stale)
    if kind == "code":
        code = _extract_py(sec.get("CODE", "")) or _extract_py(reply)
        if code:
            _write(ws, "solution.py", code)
            wrote = True
    else:
        ans = sec.get("ANSWER", "").strip()
        if not ans and ANS_RE.search(reply) and len(reply) < MAX_ANSWER_CHARS:
            # No '### ANSWER' header, but a short reply that does state a final answer.
            ans = _strip_think(reply).strip()
        # A reply with neither the section nor a short final answer is a failed or truncated
        # call. Writing its raw text here is what let a 100k-char reasoning dump become the
        # workspace's "current answer", which the manager could never make progress against.
        # Only overwrite if this response has a parseable final answer (or there is
        # nothing yet): a rambling/truncated call must not destroy a good prior answer.
        if ans and (ANS_RE.search(ans) or not _read(ws, "answer.md").strip()):
            _write(ws, "answer.md", ans)
            wrote = True
    if sec.get("NOTES"):
        # REWRITE, not append. The worker is handed the current notes and returns a curated
        # replacement, so superseded material is deleted instead of accumulating. Append-only
        # growth is what pushed manager prompts past ~91k tokens against a 131,072 window --
        # the point where no max_tokens shrink can make the request fit and the call fails
        # outright (8 problems, 26 exhausted calls, in the 2026-08-12 muse manager run).
        # A cut-off reply is DISCARDED, not appended and not written: its NOTES section is
        # half-written, so it can neither be trusted to replace the file nor appended without
        # reintroducing unbounded growth (appending cut-off notes is what still drove three
        # problems to a ~400KB notes.md on 2026-08-13, even after the rewrite landed). The
        # partial thinking is not lost -- it reaches the manager via the digest in `summary`.
        if wmeta.get("finish_reason") != "length":
            # Hard-capped as well as instructed (~800 words): the prompt asks for brevity, but
            # nothing enforces it, and every file read into a prompt needs a bound in code.
            _write(ws, "notes.md", sec["NOTES"].strip()[:MAX_PLAN_CHARS * 2] + "\n")
    nexts = [b for b in _bullets(sec.get("NEXT", "")) if b.lower() != "none"]
    status = sec.get("STATUS", "continue").strip().lower()
    if wmeta.get("finish_reason") == "length" and not finalize:
        # Worker ran out of tokens mid-thought: it did NOT finish. Summarize the cut-off
        # attempt and tell the manager EXPLICITLY (instead of handing back silence) so it can
        # pivot -- e.g. try a simpler/different approach. The digest goes ONLY into the
        # manager-facing summary below, never into notes.md: appending a digest per cut-off is
        # unbounded, and muse truncates often enough to reach ~400KB of notes that way.
        digest = _summarize_cutoff(ws, reply, task["desc"])
        summary = (f"worker EXCEEDED THE TOKEN LIMIT and was cut off before finishing task: "
                   f"{task['desc'][:80]}. No complete solution was produced -- the approach was "
                   f"too long to finish in one call, so prefer a simpler or different approach "
                   f"next. Summary of its partial thinking: {digest[:500]}")
        log(f"    [worker] task {task['id']} -> CUT OFF at token limit; summarized thinking, told manager")
        return "continue", nexts, summary, wrote
    summary = f"worker reported '{status}' on: {task['desc'][:80]}. " + (sec.get("NOTES", "")[:200])
    log(f"    [worker] task {task['id']} -> {status}, +{len(nexts)} next steps")
    return status, nexts, summary, wrote


# --- sample-test execution (feedback so the manager knows if the code actually works) -----

def _run_samples(ws, tests):
    """Run the current solution.py against the problem's PUBLIC stdin sample tests and report
    the result, so the manager has a real pass/fail signal instead of guessing by reading.
    Functional/call-based tests are left to the final grader. Returns a dict with keys
    {ran, passed, total, fail:{input,expected,got}}."""
    code = _read(ws, "solution.py")
    stdin_tests = [t for t in (tests or [])
                   if "stdin" in str(getattr(t, "testtype", "")).lower()]
    if not code.strip() or not stdin_tests:
        return {"ran": False}
    sol = os.path.join(ws, "solution.py")
    passed, fail = 0, None
    for t in stdin_tests:
        inp = getattr(t, "input", "") or ""
        exp = (getattr(t, "output", "") or "").strip()
        try:
            # sys.executable, NOT bare "python3": the gate's verdict is only useful if it is
            # computed in the SAME interpreter that grades. /usr/bin/python3 here has neither
            # numpy nor numba, while the project venv has numpy -- so a bare "python3" made
            # every numpy solution "fail" its samples, which flipped the manager's hard gate
            # and made it burn all 10 rounds fixing code that was already correct.
            r = subprocess.run([sys.executable, sol], input=inp, capture_output=True,
                               text=True, timeout=10)
            got = (r.stdout or "").strip()
            if r.returncode != 0 and not got:
                got = f"<runtime error: {(r.stderr or '')[:200]}>"
        except subprocess.TimeoutExpired:
            got = "<timed out (>10s)>"
        except Exception as e:  # noqa: BLE001
            got = f"<error: {e}>"
        if got == exp:
            passed += 1
        elif fail is None:
            fail = {"input": inp[:600], "expected": exp[:400], "got": got[:400]}
    return {"ran": True, "passed": passed, "total": len(stdin_tests), "fail": fail}


def _sample_feedback(res):
    """Turn a _run_samples result into a sentence the manager can act on."""
    if not res or not res.get("ran"):
        return ""
    if res["passed"] == res["total"]:
        return f"[SAMPLE TESTS: PASSED all {res['total']} public samples -- the solution looks correct.] "
    f = res.get("fail") or {}
    return ("[SAMPLE TESTS: FAILED -- passed {p}/{t}. The current solution is WRONG. First failing "
            "case: input={i!r} expected={e!r} got={g!r}. Fix the bug or, if this approach keeps "
            "failing, switch to a DIFFERENT approach.] ").format(
                p=res["passed"], t=res["total"], i=f.get("input", ""),
                e=f.get("expected", ""), g=f.get("got", ""))


# --- public entry point ----------------------------------------------------

def multiagent_solve(problem_text, spec, log=None, status_out=None, tests=None):
    """Same signature as escalate(): returns final answer text for grading.
    `tests` (optional) are the problem's public test cases; when given, each worker's
    solution is run against them and the pass/fail is fed to the manager.
    If `status_out` is passed, it gets the last call's finish_reason plus a count of
    truncated (finish_reason=length) calls over the whole problem."""
    log = log or (lambda *a, **k: None)
    ws = os.path.join(WS_ROOT, _slug(problem_text))
    os.makedirs(ws, exist_ok=True)
    # Clean-slate reset: the ws dir is keyed by md5(problem_text), so a re-run of the same
    # problem reuses it. Clear ALL artifacts, not just task/notes/transcript -- otherwise the
    # prior run's plan.md/solution.py/answer.md/tasks.json survive, the manager reads a stale
    # solution.py as "CURRENT SOLUTION" (defeating the not-cur invariant below), and it can be
    # returned verbatim as this run's answer.
    for f in ("task.md", "notes.md", "transcript.jsonl", "plan.md",
              "solution.py", "answer.md", "tasks.json"):
        _write(ws, f, "")
    _write(ws, "task.md", problem_text)
    _record(ws, {"_meta": True, "t": time.time(), "model": MODEL,
                 "kind": spec["kind"], "max_iters": MAX_ITERS, "problem": problem_text})
    if status_out is not None:
        status_out["ws"] = ws  # expose early so live UIs can track the workspace as it fills

    tasks = _primary_plan(problem_text, spec, ws, log)
    proposals = _ideation_worker(problem_text, spec, ws, log)
    # Primary folds the plan + ideation into one curated list and picks the first task.
    status, next_desc, tasks = _primary_manage(
        problem_text, spec, ws, tasks, proposals, "ideation complete", log)
    _save_tasks(ws, tasks)

    # Primary manages the loop: worker does the chosen task, then the primary reviews
    # progress, re-curates the list, and decides whether the problem is done.
    iters, prev_desc = 0, None
    while status == "continue" and next_desc and iters < MAX_ITERS:
        # No-progress guard: if the manager hands back the very same task it just assigned,
        # the worker achieved nothing and another identical cycle will too. Each cycle can
        # cost a full CLOUD_MAX_TOKENS generation, so stop rather than spend the budget.
        if prev_desc is not None and next_desc.strip().lower() == prev_desc.strip().lower():
            log(f"    [primary] reissued the same task; no progress, stopping after {iters} iters")
            break
        prev_desc = next_desc
        iters += 1
        task = {"id": iters, "desc": next_desc, "status": "in_progress", "result": ""}
        _, nexts, summary, wrote = _worker(problem_text, spec, ws, task, log)
        # Run the public sample tests so the manager gets a real pass/fail signal (and won't
        # declare a wrong solution "done"). This is what makes the extra rounds + "try a
        # different approach" rule actually engage on failure. Only grade if THIS worker
        # actually wrote code -- otherwise we'd re-grade the previous round's solution.py and
        # hand the manager a verdict about work this round didn't do.
        res = None
        if spec["kind"] == "code" and tests and wrote:
            res = _run_samples(ws, tests)
            if res.get("ran"):
                summary = _sample_feedback(res) + summary
                log(f"    [samples] {res['passed']}/{res['total']} public tests passed")
        status, next_desc, tasks = _primary_manage(
            problem_text, spec, ws, tasks, nexts, summary, log)
        # Hard guard: never accept a solution that fails the public samples, no matter what
        # the manager said -- keep iterating (fix / different approach) until they pass.
        if res and res.get("ran") and res["passed"] < res["total"] and status == "done":
            status = "continue"
            if not next_desc:
                next_desc = "The solution fails the public sample tests; fix it or try a different approach."
            log("    [primary-manage] overriding 'done' -- sample tests still failing")
        _save_tasks(ws, tasks)

    # Finalize only if the primary didn't already sign off on a usable answer (a redundant
    # finalize can ramble past the token cap and destroy a correct intermediate answer).
    if status == "done" and _has_answer(ws, spec):
        log("    [finalize] skipped (primary marked done)")
    else:
        _worker(problem_text, spec, ws, {"id": 0, "desc": "finalize"}, log, finalize=True)

    if status_out is not None:
        status_out["ws"] = ws  # so a result row can be traced back to its transcript
        # summarize call outcomes across the whole problem from the transcript
        recs = []
        for ln in open(os.path.join(ws, "transcript.jsonl")):
            r = json.loads(ln)
            if not r.get("_meta"):
                recs.append(r)
        status_out["finish_reason"] = recs[-1].get("finish_reason") if recs else None
        status_out["truncated_calls"] = sum(1 for r in recs if r.get("finish_reason") == "length")
        status_out["n_calls"] = len(recs)
        # If the problem ended with NO artifact and the gateway had given up on providers,
        # the empty result is an infra failure, not the model answering wrong -- flag it so
        # the grader excludes it from pass@1 instead of counting it as a fail.
        final_empty = (not _read(ws, "solution.py").strip() if spec["kind"] == "code"
                       else not ANS_RE.search(_read(ws, "answer.md")))
        if final_empty and any(r.get("infra_exhausted") for r in recs):
            status_out["infra_fail"] = True

    if spec["kind"] == "code":
        code = _read(ws, "solution.py")
        return f"```python\n{code}\n```" if code else ""
    return _read(ws, "answer.md")


def single_solve(problem_text, spec, log=None, status_out=None, tests=None):
    """Single-shot baseline: one model call, no orchestration. Same signature as
    escalate()/multiagent_solve() (`tests` accepted for interface parity, unused here);
    transcripts each problem like the multi-agent path.
    If `status_out` (a dict) is passed, it gets {finish_reason, completion_tokens} of the call."""
    log = log or (lambda *a, **k: None)
    ws = os.path.join(WS_ROOT, _slug(problem_text))
    os.makedirs(ws, exist_ok=True)
    _write(ws, "task.md", problem_text)
    _write(ws, "transcript.jsonl", "")
    _record(ws, {"_meta": True, "t": time.time(), "model": MODEL,
                 "kind": spec["kind"], "engine": "single", "problem": problem_text})
    meta = {}
    answer = _chat(ws, "single", [
        {"role": "system", "content": spec["solver_system"]},
        {"role": "user", "content": problem_text},
    ], temperature=0.2, meta=meta)
    if status_out is not None:
        status_out.update(meta)
        status_out["ws"] = ws  # so a result row can be traced back to its transcript
    log(f"    [single] {len(answer)} chars  finish={meta.get('finish_reason')}")
    return answer
