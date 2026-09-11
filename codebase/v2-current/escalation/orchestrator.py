"""Hierarchical multi-model escalation over local ollama.

Configuration (as requested):
  - A ladder of qwen3.5 models ordered by size/capability.
  - For each problem: the SMALLEST model solves, then a second instance of the
    same model critiques. Solver and critic pass the answer back and forth until
    the critic is satisfied (APPROVED) or a per-layer round cap is hit.
  - The satisfied answer is then handed UP to the next (larger) layer as a
    reference candidate, which runs the same solve<->critique loop to improve it.
  - The top layer's approved answer is the final output.

Talks to ollama's /api/chat directly (no extra deps; stdlib urllib only).
"""

import os
import sys
import json
import time
import subprocess
import threading
import urllib.request
import urllib.error

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Cloud passthrough: a ladder entry like "groq:openai/gpt-oss-120b" routes to
# Groq's OpenAI-compatible endpoint instead of local ollama. Any provider with
# an OpenAI-compatible /v1/chat/completions works via ESCALATION_OPENAI_BASE.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = os.environ.get(
    "ESCALATION_OPENAI_BASE", "https://api.groq.com/openai/v1/chat/completions")
# Disable hidden reasoning by default: the free-tier token/min limit is tiny and
# qwen3.6's <think> blocks burn ~10x the tokens. "none" or "default".
GROQ_REASONING = os.environ.get("ESCALATION_GROQ_REASONING", "none")

# OpenRouter: a ladder/model entry like "openrouter:qwen/qwen3-30b-a3b" routes here.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# "none" -> ask OpenRouter to disable reasoning; "" -> leave to model default.
OPENROUTER_REASONING = os.environ.get("ESCALATION_OR_REASONING", "none")
# Optional provider allowlist (comma-separated OpenRouter provider names). When set,
# routing is restricted to these (which must support the full output cap, so no low
# provider limit can clamp a long generation). openai_chat additionally reroutes AMONG
# the allowed providers on any failure/empty reply. Empty -> all providers eligible.
OPENROUTER_PROVIDERS = [p.strip() for p in os.environ.get("ESCALATION_OR_PROVIDERS", "").split(",") if p.strip()]

# Size-ordered ladder (small -> large). Names resolved against `ollama list`.
LADDER = os.environ.get(
    "ESCALATION_LADDER",
    "qwen3.5:2b,qwen3.5:9b,qwen3.5:35b,qwen3.5:122b",
).split(",")

MAX_ROUNDS = int(os.environ.get("ESCALATION_MAX_ROUNDS", "2"))  # solver<->critic rounds per layer
THINK = os.environ.get("ESCALATION_THINK", "0") == "1"          # enable model "thinking" (slower)
REQUEST_TIMEOUT = int(os.environ.get("ESCALATION_TIMEOUT", "1200"))
# Cloud calls should be fast; a hung provider must abort quickly and let retry
# re-route to a different provider rather than blocking on the 20-min local cap.
# This is a HARD wall-clock cap (a socket timeout alone doesn't bound total time:
# some providers trickle bytes slowly, keeping the read alive for minutes).
CLOUD_TIMEOUT = int(os.environ.get("ESCALATION_CLOUD_TIMEOUT", "120"))
# Cap generation length so no single call can run away (0 = unset).
CLOUD_MAX_TOKENS = int(os.environ.get("ESCALATION_CLOUD_MAX_TOKENS", "8000"))
# `claude:` models shell out to the Claude Code CLI on subscription auth (no API key).
CLAUDE_TIMEOUT = int(os.environ.get("ESCALATION_CLAUDE_TIMEOUT", "1800"))
CLAUDE_ATTEMPTS = int(os.environ.get("ESCALATION_CLAUDE_ATTEMPTS", "3"))

# --- first-party providers -------------------------------------------------
# OpenRouter aggregates, and its routing turned out to be the main source of run-to-run
# noise (silent provider swaps, clamped output caps, stalls). These prefixes go straight
# to each vendor's own endpoint instead, so a run is reproducible against one backend.
#
#   openai:gpt-5.6-luna      -> OpenAI            (GPT-5 series: max_completion_tokens)
#   dashscope:qwen3.8-max    -> Alibaba Model Studio, OpenAI-compatible
#   anthropic:claude-opus-5  -> Anthropic Messages API (native, via the anthropic SDK)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# GPT-5 rejects `max_tokens` (must be `max_completion_tokens`) and non-default `temperature`.
# Sending either is a 400, which openai_chat would otherwise misread as a context overflow.
OPENAI_REASONING = os.environ.get("ESCALATION_OPENAI_REASONING", "")  # "" | low|medium|high

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
# Legacy international host, which is workspace-free. Most regions are NOT: Singapore,
# Tokyo, Frankfurt, Hong Kong and Beijing all require the account's workspace id inside the
# hostname (https://<WorkspaceId>.<region>.maas.aliyuncs.com/compatible-mode/v1/...), so this
# default only works for international / US (dashscope-us.aliyuncs.com) accounts.
DASHSCOPE_URL = os.environ.get(
    "ESCALATION_DASHSCOPE_BASE",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")

# max_tokens does NOT bound thinking here. Measured on qwen3.8-max: a request with
# max_tokens=100 generated reasoning for a full 5 minutes and never returned, and a real
# benchmark prompt ran past 72 minutes without finishing -- max_tokens appears to govern only
# the visible answer. `thinking_budget` is the control that works: budget 256 came back in
# 39s with reasoning_tokens exactly 256. Leave this at 0 and a dashscope arm has NO output
# limit at all, which silently invalidates a capped experiment and bills for the privilege.
DASHSCOPE_THINKING_BUDGET = int(os.environ.get("ESCALATION_DASHSCOPE_THINKING_BUDGET", "0"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Local OpenAI-compatible server (llama.cpp / vLLM / TabbyAPI / ...): a model entry like
# "local:my-model.gguf" routes here. No key required by default. ESCALATION_LOCAL_EXTRA may
# hold extra JSON body fields, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}'.
LOCAL_API_KEY = os.environ.get("ESCALATION_LOCAL_KEY", "local")
LOCAL_URL = os.environ.get(
    "ESCALATION_LOCAL_BASE", "http://127.0.0.1:8080/v1/chat/completions")
try:
    LOCAL_EXTRA = json.loads(os.environ.get("ESCALATION_LOCAL_EXTRA", "") or "{}")
except ValueError:
    LOCAL_EXTRA = {}

# Opus 5 caps output at 128K regardless of what CLOUD_MAX_TOKENS asks for, so this arm
# cannot match a 200k cap; the call clamps rather than 400s, and records that it did.
ANTHROPIC_MAX_OUTPUT = 128000
# Thinking depth. "high" IS the API default, and it is pinned here rather than left implicit
# precisely because it is a default: an unsent parameter is whatever the API decides it means
# on the day the run happens, which is not a property a benchmark cell can carry. Sent
# explicitly, the run_config records a value the next pass can be held to.
ANTHROPIC_EFFORT = "high"


def ollama_chat(model, messages, temperature=0.2, num_ctx=16384, meta=None):
    num_ctx = int(os.environ.get("ESCALATION_NUM_CTX", num_ctx))
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": THINK,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=data, headers={"Content-Type": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                out = json.loads(r.read())
            if meta is not None:
                meta["finish_reason"] = out.get("done_reason")
                meta["completion_tokens"] = out.get("eval_count")
                meta["prompt_tokens"] = out.get("prompt_eval_count")
                # think=True makes ollama return the chain of thought in its own field,
                # separate from content. Capture it so the transcript holds the whole
                # generation, not just the visible answer.
                meta["reasoning"] = (out.get("message") or {}).get("thinking") or ""
            return out["message"]["content"]
        except Exception as e:  # noqa: BLE001 - ollama can drop/reload; retry
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    return ""


def openai_chat(url, api_key, model, messages, temperature=0.2, extra=None, meta=None,
                token_param="max_tokens", send_temperature=True):
    """Call an OpenAI-compatible /v1/chat/completions endpoint (e.g. Groq, OpenRouter).

    Each attempt has a HARD wall-clock cap: a watchdog abandons a provider that
    stalls (some trickle bytes slowly and evade the socket timeout) so the retry
    re-routes to a different, faster provider.

    `token_param` / `send_temperature` exist because OpenAI's GPT-5 series renamed
    max_tokens -> max_completion_tokens and rejects a non-default temperature; both
    would otherwise come back as a 400 and be mistaken for a context overflow.
    """
    base = {"model": model, "messages": messages, "stream": False}
    if send_temperature:
        base["temperature"] = temperature
    if CLOUD_MAX_TOKENS > 0:
        base[token_param] = CLOUD_MAX_TOKENS
    if extra:
        base.update(extra)
    prov0 = dict(base.get("provider") or {})  # {only?, sort?} the caller requested
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # Groq/Cloudflare returns 403 for urllib's default UA; set an explicit one.
        "User-Agent": "model-test/1.0",
    }

    def _do(box, data):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=CLOUD_TIMEOUT) as r:
                box["out"] = json.loads(r.read())
        except urllib.error.HTTPError as e:
            box["err"] = e
            box["code"] = e.code
            try:
                box["body"] = e.read().decode()[:300]
            except Exception:  # noqa: BLE001
                box["body"] = ""
        except Exception as e:  # noqa: BLE001 - captured for the caller to classify
            box["err"] = e

    # A "fail" must mean the model answered wrong, never that a provider stalled, dropped
    # the stream, returned an error object, clamped output below the cap, or handed back a
    # reasoning-only reply. So every such outcome REROUTES to a different provider (via
    # OpenRouter's `ignore`) and retries; only a genuine completion (non-empty content, not
    # clamped short) is returned. Give up only after exhausting the attempt budget.
    ATTEMPTS = 16
    only = list(prov0.get("only") or [])
    cur_max = base.get(token_param)  # auto-reduced on a 400 (prompt + max_tokens > context)
    ignore, last, got_response = [], "", False
    for attempt in range(ATTEMPTS):
        body = dict(base)
        if cur_max:
            body[token_param] = cur_max
        prov = dict(prov0)
        # Rotate the preferred provider every attempt. On a timeout we get no response, so we
        # cannot `ignore` the provider that stalled -- rotating the order guarantees the next
        # attempt starts with a DIFFERENT provider instead of re-hitting the stalled one.
        if only:
            k = attempt % len(only)
            prov["order"] = only[k:] + only[:k]
        if ignore:
            prov["ignore"] = list(ignore)
        if prov:
            body["provider"] = prov
        box = {}
        th = threading.Thread(target=_do, args=(box, json.dumps(body).encode()), daemon=True)
        th.start()
        th.join(CLOUD_TIMEOUT + 5)
        if th.is_alive():  # hard cap: abandon the stalled provider (daemon thread leaks, dies later)
            print(f"    [cloud] hard timeout >{CLOUD_TIMEOUT}s, abandoning provider (attempt {attempt + 1})", file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"hard timeout >{CLOUD_TIMEOUT}s (no response)"})
            continue
        out = box.get("out")
        if out is None:
            e = box.get("err")
            if meta is not None:  # infra failure: no model output, but record that it happened
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"{type(e).__name__}: {str(e)[:200]}",
                     "http_code": box.get("code"), "body": box.get("body")})
            # A 400 means "context overflow" ONLY if the body says so. Every other 400 --
            # bad key, exhausted credit, unsupported parameter, unknown model -- is fatal and
            # will not fix itself; shrinking max_tokens 16 times just buries the real message
            # and, if a later attempt happens to succeed, silently runs at a smaller cap than
            # the experiment specifies. Seen for real: litellm returns 400 "No connected db."
            # on an auth failure, and OpenAI 400s on `max_tokens` for the GPT-5 series.
            body_txt = (box.get("body") or "").lower()
            ctx_err = any(s in body_txt for s in (
                "context", "maximum context length", "too long", "exceeds", "max_model_len"))
            if box.get("code") == 400 and ctx_err and cur_max and cur_max > 40000:
                # Shrink the output budget (the prompt is fixed) and retry. Reduce in a SMALL
                # step and stop the moment a request succeeds, so we land just under the
                # ceiling and never over-shrink the output into truncation. Floor at 40k --
                # ample for any solution + reasoning; a prompt too big to leave even that is
                # pathological.
                cur_max = max(40000, cur_max - 24000)
                print(f"    [cloud] HTTP 400 (prompt+max_tokens > context); reducing {token_param} to {cur_max}", file=sys.stderr, flush=True)
                continue
            if box.get("code") == 400 and ctx_err:
                # Already at the 40k floor and STILL over the window: the prompt alone does not
                # fit, so no output budget makes this request legal and retrying is pointless.
                # Without this the call matched neither the shrink branch (floor reached) nor
                # the abort branch below (which requires `not ctx_err`), so it fell through to
                # the generic retry and burned all 16 attempts -- ~8 minutes of backoff per
                # call. Cost 26 exhausted calls across 8 problems in the 2026-08-12 muse
                # manager run, once append-only notes pushed prompts past ~91k tokens.
                print(f"    [cloud] prompt alone exceeds the context window at {token_param}"
                      f"={cur_max}; not retryable, aborting", file=sys.stderr, flush=True)
                break
            if box.get("code") in (400, 401, 402, 403, 404) and not ctx_err:
                print(f"    [cloud] HTTP {box.get('code')} is not retryable, aborting: "
                      f"{(box.get('body') or '')[:200]}", file=sys.stderr, flush=True)
                break
            if isinstance(e, urllib.error.HTTPError) and e.code == 429:
                ra = e.headers.get("retry-after")
                try:
                    wait = float(ra) if ra else min(30.0, 5.0 * (attempt + 1))
                except ValueError:
                    wait = min(30.0, 5.0 * (attempt + 1))
                print(f"    [cloud] 429 rate-limited, sleep {wait:.0f}s (attempt {attempt + 1})", file=sys.stderr, flush=True)
                time.sleep(wait + 1)
                continue
            print(f"    [cloud] {type(e).__name__} on attempt {attempt + 1}, retrying", file=sys.stderr, flush=True)
            time.sleep(3 * (attempt + 1))
            continue
        served = out.get("provider")
        choices = out.get("choices")
        if not choices:  # provider returned an error object / malformed response -> reroute
            print(f"    [cloud] no choices from {served} ({str(out.get('error'))[:80]}); rerouting", file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "provider": served, "why": "no choices in response",
                     "body": str(out.get("error"))[:500]})
            if served:
                ignore.append(served)
            time.sleep(2)
            continue
        got_response = True
        choice = choices[0]
        msg = choice.get("message") or {}
        finish = choice.get("finish_reason")
        content = msg.get("content") or ""
        # OpenRouter calls the thinking "reasoning"; vLLM's --reasoning-parser splits it into
        # "reasoning_content". Read it ALWAYS, not just when content is empty: with reasoning
        # on it is most of the generation, and dropping it once content is non-empty means the
        # transcript keeps only the visible answer and the thinking is gone for good.
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if not content:  # some providers put the whole answer under reasoning; use it
            content = reasoning
        usage = out.get("usage") or {}
        ntok = usage.get("completion_tokens")
        if meta is not None:
            meta["finish_reason"] = finish
            meta["completion_tokens"] = ntok
            meta["prompt_tokens"] = usage.get("prompt_tokens")
            meta["reasoning"] = reasoning
            meta["provider"] = served
            meta["attempts"] = attempt + 1
        # A provider clamping output below our cap is an infra cutoff, not the model's doing.
        # Compare against cur_max -- the cap ACTUALLY SENT -- not CLOUD_MAX_TOKENS. After the
        # 400/context shrink above lowers cur_max (e.g. 128000 -> 104000), a reply that fills
        # the reduced budget is a legitimate truncation, but against the original cap it looks
        # clamped (104000 < 0.9*128000) and gets discarded and retried. That made every call
        # after a shrink unsatisfiable: 16 attempts x a full 104k-token generation each, ~7h
        # per call. Cost muse's manager p3 sixteen hours on a single problem (arc191_c, 15 of
        # 16 attempts discarded as "clamped"), while luna and terra -- which never shrink,
        # having context to spare -- were untouched.
        cap_used = cur_max or CLOUD_MAX_TOKENS
        clamped = finish == "length" and cap_used and (ntok or 0) < 0.9 * cap_used
        if content.strip() and not clamped:
            return content  # genuine completion (correct or not) -- this is what gets graded
        why = f"clamped at {ntok} tok (< cap)" if clamped else f"empty reply (finish={finish})"
        print(f"    [cloud] {why} from {served}; rerouting to another provider (attempt {attempt + 1})", file=sys.stderr, flush=True)
        if meta is not None:
            # A rerouted attempt is still real model output on a graded problem; keep it (with
            # its thinking) rather than discarding it to stderr, so the transcript accounts for
            # every token generated, not only the attempt that happened to come back clean.
            meta.setdefault("discarded", []).append(
                {"attempt": attempt + 1, "provider": served, "why": why, "finish_reason": finish,
                 "completion_tokens": ntok, "content": content, "reasoning": reasoning})
        last = content or last
        if served:
            ignore.append(served)
        time.sleep(1)
    # Exhausted the attempt budget without a usable completion. If no attempt produced any
    # text, this is an INFRA failure, not the model answering wrong -- flag it so the grader
    # can exclude it from pass@1 rather than scoring it as a fail.
    n_tried = len((meta or {}).get("discarded") or []) or ATTEMPTS
    print(f"    [cloud] gave up after {n_tried} attempt(s) without a clean completion "
          f"(got_response={got_response})", file=sys.stderr, flush=True)
    if meta is not None and not last.strip():
        meta["infra_exhausted"] = True
    return last


def claude_cli_chat(model, messages, temperature=0.2, meta=None):
    """One-shot completion through the `claude -p` CLI, on the machine's subscription auth.

    NOT equivalent to the HTTP paths above, and deliberately so -- the CLI exposes no
    max_tokens, no temperature and no reasoning switch, so ESCALATION_CLOUD_MAX_TOKENS and
    the reasoning knobs do NOT apply to this arm. `temperature` is accepted for signature
    parity only. --tools "" keeps it a single model turn rather than an agent loop.
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    body = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose",
           "--no-session-persistence", "--tools", "", "--model", model]
    if system:
        cmd += ["--append-system-prompt", system]
    for attempt in range(CLAUDE_ATTEMPTS):
        try:
            # Prompt goes on stdin, never as a trailing positional: --tools is variadic and
            # would swallow it as a tool name.
            proc = subprocess.run(cmd, input=body, capture_output=True, text=True,
                                  timeout=CLAUDE_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"    [claude] timeout >{CLAUDE_TIMEOUT}s (attempt {attempt + 1})",
                  file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"timeout >{CLAUDE_TIMEOUT}s (no response)"})
            continue
        text, usage, stop, err, turns = "", {}, None, False, None
        thinking, blocks = [], 0
        for line in proc.stdout.splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "assistant":
                # Extended thinking arrives as `thinking` content blocks on the assistant
                # events; the terminal result event carries only the visible answer, so
                # without this the thinking is never seen by the caller. NOTE: `claude -p`
                # returns these blocks with an empty `thinking` string and only a signature
                # -- the CLI does not expose the thinking TEXT. So this arm can record THAT
                # the model thought (block count) but not WHAT it thought; that text is not
                # retrievable over this transport at all.
                for b in (ev.get("message") or {}).get("content") or []:
                    if b.get("type") == "thinking":
                        blocks += 1
                        if b.get("thinking"):
                            thinking.append(b["thinking"])
            if ev.get("type") == "result":
                text = ev.get("result") or ""
                usage = ev.get("usage") or {}
                stop = ev.get("stop_reason")
                err = bool(ev.get("is_error"))
                turns = ev.get("num_turns")
        if meta is not None:
            # Map the CLI's vocabulary onto the harness's: _classify_status keys off "length".
            meta["finish_reason"] = ("length" if stop == "max_tokens"
                                     else "error" if err else stop or "stop")
            meta["completion_tokens"] = usage.get("output_tokens")
            meta["prompt_tokens"] = usage.get("input_tokens")
            meta["reasoning"] = "\n\n".join(thinking)
            meta["thinking_blocks"] = blocks  # >0 with empty reasoning = thought, text withheld
            meta["num_turns"] = turns
            meta["attempts"] = attempt + 1
        if text.strip() and not err:
            return text
        print(f"    [claude] empty/error reply (stop={stop} rc={proc.returncode}); "
              f"retrying (attempt {attempt + 1})", file=sys.stderr, flush=True)
        if meta is not None:  # keep the failed attempt (and why it failed) out of /dev/null
            meta.setdefault("discarded", []).append(
                {"attempt": attempt + 1, "why": f"empty/error (stop={stop} rc={proc.returncode})",
                 "finish_reason": stop, "completion_tokens": usage.get("output_tokens"),
                 "content": text, "reasoning": "\n\n".join(thinking), "stderr": proc.stderr})
        time.sleep(3 * (attempt + 1))
    if meta is not None:
        meta["infra_exhausted"] = True
    return ""


def anthropic_chat(model, messages, temperature=0.2, meta=None):
    """One-shot completion through Anthropic's first-party Messages API.

    Not interchangeable with the OpenAI-compatible path, and the differences are forced
    by the API rather than chosen here:
      - Opus 5 caps output at 128K. A larger ESCALATION_CLOUD_MAX_TOKENS is CLAMPED (not
        rejected), and the clamp is recorded, so a run cannot silently claim a cap it never
        actually had. max_tokens bounds THINKING PLUS ANSWER TOGETHER, so a problem can
        spend the whole cap reasoning and return no text at all; that lands as `truncated`
        and is a real outcome, not a bug to be capped around -- 128K is the ceiling.
      - Thinking is adaptive and on by default, and RAW thinking tokens are never returned.
        `meta["reasoning"]` therefore holds Anthropic's SUMMARY of the thinking, not the
        chain of thought -- unlike the vLLM arms, where it is the real thing.
        `meta["reasoning_is_summary"]` marks that in the transcript so the two cannot be
        read as the same kind of evidence later.
      - `temperature` is not sent: non-default sampling is rejected alongside thinking.
      - Streaming is required at these output sizes; the final message is reassembled here.

    Thinking DEPTH is the one thing here that is chosen rather than forced: ANTHROPIC_EFFORT
    is sent explicitly (see its definition).
    """
    import anthropic  # imported lazily so the other arms need no anthropic dependency

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=CLOUD_TIMEOUT)
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [{"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"]
    want = CLOUD_MAX_TOKENS or ANTHROPIC_MAX_OUTPUT
    max_out = min(want, ANTHROPIC_MAX_OUTPUT)
    for attempt in range(CLAUDE_ATTEMPTS):
        try:
            kw = {"model": model, "max_tokens": max_out, "messages": msgs,
                  # display:"summarized" is the most thinking this API will hand back.
                  "thinking": {"type": "adaptive", "display": "summarized"},
                  "output_config": {"effort": ANTHROPIC_EFFORT}}
            if system:
                kw["system"] = system
            with client.messages.stream(**kw) as stream:
                msg = stream.get_final_message()
        except Exception as e:  # noqa: BLE001 - recorded, then retried
            print(f"    [anthropic] {type(e).__name__} on attempt {attempt + 1}: {str(e)[:200]}",
                  file=sys.stderr, flush=True)
            if meta is not None:
                meta.setdefault("discarded", []).append(
                    {"attempt": attempt + 1, "why": f"{type(e).__name__}: {str(e)[:200]}"})
            time.sleep(3 * (attempt + 1))
            continue
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        thinking = "\n\n".join(getattr(b, "thinking", "") or ""
                               for b in msg.content if getattr(b, "type", "") == "thinking")
        if meta is not None:
            meta["finish_reason"] = "length" if msg.stop_reason == "max_tokens" else msg.stop_reason
            meta["completion_tokens"] = msg.usage.output_tokens
            meta["prompt_tokens"] = msg.usage.input_tokens
            meta["reasoning"] = thinking
            meta["reasoning_is_summary"] = True  # NOT the raw chain of thought
            meta["attempts"] = attempt + 1
            if want > ANTHROPIC_MAX_OUTPUT:
                meta["cap_clamped_to"] = ANTHROPIC_MAX_OUTPUT
        # A safety refusal is a real, final answer-shaped outcome, not an infra failure:
        # return it so the grader scores it rather than retrying into the same wall.
        if msg.stop_reason == "refusal":
            print("    [anthropic] stop_reason=refusal", file=sys.stderr, flush=True)
        return text
    if meta is not None:
        meta["infra_exhausted"] = True
    return ""


def chat(model, messages, temperature=0.2, num_ctx=16384, meta=None):
    """Dispatch to a cloud provider for prefixed models, else local ollama.

    If `meta` (a dict) is passed, it is populated with {finish_reason, completion_tokens}
    from the underlying call, so callers can classify truncated/empty_stop outcomes.
    """
    if model.startswith("groq:"):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set but ladder uses a groq: model")
        extra = {"reasoning_effort": GROQ_REASONING} if GROQ_REASONING else None
        return openai_chat(GROQ_URL, GROQ_API_KEY, model[len("groq:"):], messages, temperature, extra, meta)
    if model.startswith("claude:"):  # subscription auth via the Claude Code CLI
        return claude_cli_chat(model[len("claude:"):], messages, temperature, meta)
    if model.startswith("openai:"):  # OpenAI first-party
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set but model uses an openai: prefix")
        name = model[len("openai:"):]
        gpt5 = name.startswith("gpt-5")  # renamed token param, fixed temperature
        extra = {"reasoning_effort": OPENAI_REASONING} if OPENAI_REASONING else None
        return openai_chat(OPENAI_URL, OPENAI_API_KEY, name, messages, temperature, extra, meta,
                           token_param="max_completion_tokens" if gpt5 else "max_tokens",
                           send_temperature=not gpt5)
    if model.startswith("dashscope:"):  # Alibaba Model Studio, OpenAI-compatible
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY not set but model uses a dashscope: prefix")
        extra = {"thinking_budget": DASHSCOPE_THINKING_BUDGET} if DASHSCOPE_THINKING_BUDGET else None
        if not extra:
            print("    [dashscope] WARNING: ESCALATION_DASHSCOPE_THINKING_BUDGET is unset -- "
                  "max_tokens does not bound thinking on this provider, so this call is "
                  "effectively uncapped", file=sys.stderr, flush=True)
        return openai_chat(DASHSCOPE_URL, DASHSCOPE_API_KEY, model[len("dashscope:"):],
                           messages, temperature, extra, meta)
    if model.startswith("anthropic:"):  # Anthropic first-party Messages API
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set but model uses an anthropic: prefix")
        return anthropic_chat(model[len("anthropic:"):], messages, temperature, meta)
    if model.startswith("openrouter:"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set but model uses an openrouter: prefix")
        prov = {"sort": "throughput"}  # prefer fast providers; openai_chat reroutes on failure
        if OPENROUTER_PROVIDERS:  # restrict to high-output-cap providers so nothing clamps
            prov["only"] = OPENROUTER_PROVIDERS
        extra = {"provider": prov}
        if OPENROUTER_REASONING == "none":
            extra["reasoning"] = {"enabled": False}
        elif OPENROUTER_REASONING in ("low", "medium", "high"):
            extra["reasoning"] = {"effort": OPENROUTER_REASONING}
        elif OPENROUTER_REASONING.startswith("budget:"):
            # bounded reasoning token budget. Models without native effort levels
            # emulate `effort:high` as ~80% of max_tokens (~100k here) -> runaway
            # thinking that stalls small models / truncates the answer. A budget
            # caps thinking and forces a clean thinking->answer transition instead.
            extra["reasoning"] = {"max_tokens": int(OPENROUTER_REASONING.split(":", 1)[1])}
        # any other value (e.g. "default") -> send nothing; model reasons at its default
        return openai_chat(OPENROUTER_URL, OPENROUTER_API_KEY, model[len("openrouter:"):], messages, temperature, extra, meta)
    if model.startswith("local:"):  # any OpenAI-compatible local server
        return openai_chat(LOCAL_URL, LOCAL_API_KEY, model[len("local"):], messages,
                           temperature, LOCAL_EXTRA or None, meta)
    return ollama_chat(model, messages, temperature=temperature, num_ctx=num_ctx, meta=meta)


# --- Task specifications ---------------------------------------------------
# A task spec adapts the loop to a domain (code vs math) via prompt fragments.

CODE_SPEC = {
    "kind": "code",
    "solver_system": (
        "You are an elite competitive programmer. Solve the given problem in Python. "
        "Think carefully about algorithmic complexity and edge cases. "
        "Output EXACTLY ONE complete, self-contained Python program inside a single "
        "```python ...``` fenced block, and nothing else after it."
    ),
    "critic_system": (
        "You are a ruthless code reviewer for competitive programming. You are given a "
        "problem statement and a candidate Python solution. Check for: wrong algorithm, "
        "incorrect edge cases, off-by-one errors, wrong time/space complexity for the "
        "constraints, and INPUT/OUTPUT FORMAT mismatches (stdin/stdout vs function). "
        "If and only if the solution is fully correct and complete, reply with exactly "
        "'APPROVED' on the first line. Otherwise reply 'REJECTED' on the first line "
        "followed by a numbered list of concrete, fixable problems."
    ),
}

MATH_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are an elite mathematician solving an AIME problem. The answer is an integer "
        "from 0 to 999. Reason step by step, then on the FINAL line output exactly "
        "'ANSWER: <integer>'."
    ),
    "critic_system": (
        "You are a meticulous math grader. You are given an AIME problem and a candidate "
        "solution. Check the reasoning and the final integer answer for errors. If the "
        "solution is fully correct, reply with exactly 'APPROVED' on the first line. "
        "Otherwise reply 'REJECTED' on the first line followed by the specific errors."
    ),
}


MATH500_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are an elite mathematician. Solve the problem. Reason step by step, then on the FINAL "
        "line output exactly 'ANSWER: <final answer>' with the answer in simplest form (a number or "
        "simplified expression, written as it would appear in a textbook)."
    ),
    "critic_system": (
        "You are a meticulous math grader. Given a problem and a candidate solution, check the "
        "reasoning and final answer. If fully correct, reply exactly 'APPROVED' on the first line; "
        "otherwise reply 'REJECTED' then the specific errors."
    ),
}

GPQA_SPEC = {
    "kind": "math",
    "solver_system": (
        "You are a PhD-level expert in physics, chemistry, and biology answering a multiple-choice "
        "question with options A, B, C, D. Reason carefully, then on the FINAL line output exactly "
        "'ANSWER: <letter>' where <letter> is the single correct option (A, B, C, or D)."
    ),
    "critic_system": (
        "You are a rigorous science reviewer. Given a multiple-choice question and a candidate "
        "answer, verify the reasoning and the chosen letter. If fully correct reply 'APPROVED' on "
        "the first line; otherwise reply 'REJECTED' then the errors."
    ),
}

HLE_SPEC = {
    "kind": "math",
    # 'ANSWER:' rather than upstream HLE's 'Exact Answer:' -- multiagent.py's ANS_RE gates
    # whether a worker may overwrite answer.md, so keep this repo's existing convention.
    "solver_system": (
        "You are answering a question from Humanity's Last Exam: an extremely difficult, "
        "expert-level question that may come from any academic discipline. It may be "
        "multiple-choice or require an exact free-form answer (a number, expression, name, or "
        "short phrase). If it is multiple-choice, answer with the option LETTER only -- note "
        "there may be more than four options. Reason carefully, then finish with exactly "
        "these two lines:\n"
        "ANSWER: <your succinct, exact final answer -- no explanation, no units unless asked>\n"
        "CONFIDENCE: <integer 0-100>"
    ),
    "critic_system": (
        "You are a rigorous expert reviewer across all academic fields. Given a question and a "
        "candidate answer, check the reasoning and the exact final answer for errors. If fully "
        "correct reply 'APPROVED' on the first line; otherwise reply 'REJECTED' then the errors."
    ),
}


def _is_approved(critique):
    return critique.strip().upper().startswith("APPROVED")


def solve_layer(model, problem_text, spec, prior_answer=None, temperature=0.2, log=None):
    """Run the solver<->critic loop for one model layer. Returns final answer text."""
    log = log or (lambda *a, **k: None)

    solver_msgs = [{"role": "system", "content": spec["solver_system"]}]
    if prior_answer:
        user = (
            f"{problem_text}\n\n"
            "A smaller model produced the following candidate solution. Treat it as a "
            "hint only: verify it, fix any mistakes, and produce YOUR OWN best solution.\n\n"
            f"--- candidate ---\n{prior_answer}\n--- end candidate ---"
        )
    else:
        user = problem_text
    solver_msgs.append({"role": "user", "content": user})

    answer = chat(model, solver_msgs, temperature=temperature)
    solver_msgs.append({"role": "assistant", "content": answer})

    for rnd in range(MAX_ROUNDS):
        critique = chat(
            model,
            [
                {"role": "system", "content": spec["critic_system"]},
                {
                    "role": "user",
                    "content": f"PROBLEM:\n{problem_text}\n\nCANDIDATE SOLUTION:\n{answer}",
                },
            ],
            temperature=0.1,
        )
        log(f"    [{model}] round {rnd + 1} critic: {'APPROVED' if _is_approved(critique) else 'rejected'}")
        if _is_approved(critique):
            break
        # solver revises given the critique
        solver_msgs.append(
            {
                "role": "user",
                "content": (
                    "A reviewer raised the following issues. Fix them and output the full "
                    f"corrected solution again in the required format.\n\n{critique}"
                ),
            }
        )
        answer = chat(model, solver_msgs, temperature=temperature)
        solver_msgs.append({"role": "assistant", "content": answer})

    return answer


def escalate(problem_text, spec, ladder=None, log=None, status_out=None, tests=None):
    """Run the full ladder small->large, threading each layer's answer upward."""
    log = log or (lambda *a, **k: None)
    ladder = ladder or LADDER
    prior = None
    for model in ladder:
        log(f"  layer {model}")
        prior = solve_layer(model, problem_text, spec, prior_answer=prior, log=log)
    if status_out is not None:
        status_out.setdefault("finish_reason", None)
    return prior
