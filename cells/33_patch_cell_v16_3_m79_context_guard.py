# Cell 33 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v16.3 M79 (context guard))
# ===== v16.3/v16.4 = M79: NEVER SEND A PROMPT THE SERVER WILL REJECT, AND ALWAYS LEAVE ROOM FOR THE REPLY =====
"""v16.2 v2 (real run, 2026-09-23): lp85 froze at action 22. The server rejected its prompt 374 times with
    400 "The input (32815 tokens) is longer than the model's context length (32768 tokens)."
and the harness re-sent the same prompt every time. Two harness defects, both latent since the move to SGLang:

1. The harness HAS an overflow recovery (tool_agent.py analyze: on a context-length error, trim harder / drop the oldest
   block and retry), but `_is_context_length_error` only knows vLLM/OpenAI wording ("maximum context length", "reduce the
   length of the input prompt", "input_tokens"). SGLang says "longer than the model's context length", so the recovery
   never fired. FIX A: recognise SGLang's wording.

2. The trim budget is enforced with `_estimate_tokens` = len(json) / 3. Measured with the real Flash-Next tokenizer on 75
   real prompt snapshots: tool results run 1.01-1.64 chars/token (digits/coordinates tokenize ~1 per char), tool-call code
   ~2.2-2.7, English 3.5-4.2. So the estimate undercounts exactly the content that fills a long history; lp85's failing
   prompt was estimated at 25.6K and was 32.8K. The same undercount also decided how much room the reply got: replies
   average 1,767 tokens (v16.0 v2) with no output cap. FIX B: after the harness's own estimate, count the prompt EXACTLY
   with the server's /v1/tokenize (the same chat template the request goes through; images stripped and charged at 66
   tokens each, M74's measured cost) and make the trim continue until the real count is <= M79_HARD - (the M24 ledger
   reserve already in extra_safety_tokens). M79_HARD = 32,768 - 4,096 keeps >= 4,096 tokens for the reply; v16.0 v2's
   largest real prompts were ~28.4K (p99), so the guard only binds beyond what v16.0 v2 ever sent.
   If /v1/tokenize is unavailable, a conservative local count is used (worst measured chars/token per part), so the
   guard never fails open; FIX A remains as the last line.

Nothing else changes: prompts under the guard are trimmed exactly as before (the harness estimate still decides first).

READ: m79_summary() -> exact_counts, count_errors (tokenize failures -> local fallback), cache_hits, guard_binds (estimates
the exact count raised above the harness estimate), max_real, context_errors_recognised, fallback_counts, ms_total.
KILL: exact_counts == 0 and fallback_counts == 0 -> not active.
"""
from __future__ import annotations
import json as _json79
import threading as _th79
import time as _time79
import urllib.request as _ur79

M79_LIMIT = 69632   # v16.7: context 49,152 (M84 watermarks keep prompts <= ~36.9k)
M79_REPLY_ROOM = 4096
M79_HARD = M79_LIMIT - M79_REPLY_ROOM
M79_IMAGE_TOKENS = 66
# worst chars/token per part over 75 real snapshots (v16.0, v16.0 v2, v16.2 v2), used only if /v1/tokenize fails
M79_FALLBACK_CPT = {"tool": 1.0, "assistant": 1.7, "reasoning": 1.85, "user": 2.9, "system": 3.5}
M79_FALLBACK_PER_MSG = 30

_STAT79 = {"exact_counts": 0, "count_errors": 0, "cache_hits": 0, "guard_binds": 0, "max_real": 0, "skipped_over_budget": 0, "turn_protected": 0,
           "context_errors_recognised": 0, "fallback_counts": 0, "ms_total": 0.0, "last_error": ""}
_LK79 = _th79.Lock()


def m79_summary():
    with _LK79:
        s = dict(_STAT79)
    s["ms_total"] = round(s["ms_total"])
    return s


def _prep(messages):
    """Messages as the server's tokenizer should see them: text only (images counted separately), prior reasoning as
    reasoning_content (what the v14.58 patch sends), tool-call arguments as objects (what the template renders)."""
    out, images = [], 0
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        m2 = dict(m)
        c = m2.get("content")
        if isinstance(c, list):
            texts = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    images += 1
                elif isinstance(p, dict):
                    texts.append(str(p.get("text", "")))
            m2["content"] = "\n".join(texts)
        if m2.get("role") == "assistant":
            if m2.get("reasoning") and not m2.get("reasoning_content"):
                m2["reasoning_content"] = m2["reasoning"]
            m2.pop("reasoning", None)
            tcs = []
            for tc in m2.get("tool_calls") or []:
                tc = _json79.loads(_json79.dumps(tc))
                f = tc.get("function", {}) if isinstance(tc, dict) else {}
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = _json79.loads(f["arguments"])
                    except Exception:
                        pass
                tcs.append(tc)
            if tcs:
                m2["tool_calls"] = tcs
        out.append(m2)
    return out, images


def _fallback_count(prepped, images, tools):
    n = M79_FALLBACK_PER_MSG * len(prepped) + M79_IMAGE_TOKENS * images
    n += int(len(_json79.dumps(tools or [])) / M79_FALLBACK_CPT["system"])
    for m in prepped:
        role = m.get("role")
        text = str(m.get("content") or "")
        if role == "tool":
            n += int(len(text) / M79_FALLBACK_CPT["tool"])
        elif role == "assistant":
            n += int(len(text) / M79_FALLBACK_CPT["assistant"])
            n += int(len(str(m.get("reasoning_content") or "")) / M79_FALLBACK_CPT["reasoning"])
            n += int(len(_json79.dumps(m.get("tool_calls") or [])) / M79_FALLBACK_CPT["assistant"])
        elif role == "system":
            n += int(len(text) / M79_FALLBACK_CPT["system"])
        else:
            n += int(len(text) / M79_FALLBACK_CPT["user"])
    return n


def m79_real_tokens(agent, messages, tools):
    """Exact prompt tokens via the server's /v1/tokenize (cached per content), else the conservative local count."""
    prepped, images = _prep(messages)
    key = _json79.dumps([prepped, images, tools], sort_keys=True, default=str)
    cache = getattr(agent, "_m79_cache", None)
    if cache is None:
        cache = agent._m79_cache = {}
    if key in cache:
        with _LK79:
            _STAT79["cache_hits"] += 1
        return cache[key]
    t0 = _time79.monotonic()
    try:
        base = str(agent._model.base_url).rstrip("/")
        body = _json79.dumps({"model": agent._model.model_id, "messages": prepped, "tools": tools or None,
                              "add_generation_prompt": True}).encode()
        req = _ur79.Request(base + "/tokenize", data=body, headers={"Content-Type": "application/json"})
        with _ur79.urlopen(req, timeout=30) as r:
            n = int(_json79.loads(r.read())["count"]) + M79_IMAGE_TOKENS * images
        with _LK79:
            _STAT79["exact_counts"] += 1
    except Exception as exc:
        n = _fallback_count(prepped, images, tools)
        with _LK79:
            _STAT79["count_errors"] += 1
            _STAT79["fallback_counts"] += 1
            _STAT79["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
    with _LK79:
        _STAT79["ms_total"] += 1000 * (_time79.monotonic() - t0)
        _STAT79["max_real"] = max(_STAT79["max_real"], n)
    if len(cache) > 256:
        cache.clear()
    cache[key] = n
    return n


def m79_probe():
    """One tiny /v1/tokenize call against the live server at install time, so the log says which mode M79 runs in."""
    import os
    class _P:
        pass
    p = _P(); p._model = _P()
    p._model.base_url = os.environ.get("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:1234/v1")
    p._model.model_id = os.environ.get("LOCAL_ANALYZER_MODEL_ID", "Qwen/Qwen3.8-Flash-Next-NVFP4")
    before = m79_summary()["exact_counts"]
    n = m79_real_tokens(p, [{"role": "user", "content": "M79 probe: count these tokens."}], None)
    ok = m79_summary()["exact_counts"] > before
    return {"mode": "exact /v1/tokenize" if ok else "FALLBACK (conservative local count)", "probe_tokens": n,
            "last_error": m79_summary()["last_error"]}


def install79():
    import inference.agent.tool_agent as T
    A = T.ToolAgent
    if getattr(T._is_context_length_error, "_m79_wrapped", False) or getattr(A._estimate_request_input_tokens, "_m79_wrapped", False):
        raise RuntimeError("M79 installed twice")

    # FIX A: recognise SGLang's context-length rejection so the harness's own overflow recovery runs
    orig_is = T._is_context_length_error

    def _is_context_length_error(exc):
        if orig_is(exc):
            return True
        msg = str(exc).lower()
        hit = "longer than the model's context length" in msg or "longer than the model\\'s context length" in msg
        if hit:
            with _LK79:
                _STAT79["context_errors_recognised"] += 1
        return hit

    _is_context_length_error._m79_wrapped = True
    T._is_context_length_error = _is_context_length_error

    # FIX B: the trim continues until the REAL count fits M79_HARD (minus whatever reserve the caller asked for)
    orig_est = A._estimate_request_input_tokens

    def _estimate_request_input_tokens(self, messages, *, tools=None):
        est = orig_est(self, messages, tools=tools)
        if est > int(self._context_budget_tokens):
            # the trim loop drops a block whenever est > budget (budget <= _context_budget_tokens for every caller),
            # so the exact count cannot change this decision -- skip the /v1/tokenize call
            with _LK79:
                _STAT79["skipped_over_budget"] += 1
            return est
        try:
            real = m79_real_tokens(self, messages, tools)
            # trim loop condition is `est > budget` with budget = _context_budget_tokens - extra_safety_tokens;
            # returning real + (budget_full - HARD) makes it also require real <= HARD - extra_safety_tokens
            guarded = real + (int(self._context_budget_tokens) - M79_HARD)
            if guarded > est:
                with _LK79:
                    _STAT79["guard_binds"] += 1
                return guarded
        except Exception as exc:
            with _LK79:
                _STAT79["count_errors"] += 1
                _STAT79["last_error"] = f"guard: {type(exc).__name__}: {exc}"[:160]
        return est

    _estimate_request_input_tokens._m79_wrapped = True
    A._estimate_request_input_tokens = _estimate_request_input_tokens

    # FIX C (v16.4): never trim away the current turn. v16.3 re86: a turn's own user message was dropped by the
    # stricter trim after a tool call, and the next request carried no user query -> 400 "No user query found in
    # messages". Protect everything from the last non-tool user message to the end (the current turn); only older
    # history can be dropped. If the current turn alone is over budget it is sent as is -- tool outputs are capped by
    # the harness, and FIX A handles a real overflow.
    orig_trim = A._trim_messages_for_context

    def _trim_messages_for_context(self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0):
        try:
            msgs = list(messages or [])
            last_user = max((i for i, m in enumerate(msgs) if i > 0 and isinstance(m, dict) and m.get("role") == "user"),
                            default=None)
            if last_user is not None:
                need = len(msgs) - last_user
                if need > preserve_recent:
                    with _LK79:
                        _STAT79["turn_protected"] += 1
                    preserve_recent = need
        except Exception as exc:
            with _LK79:
                _STAT79["last_error"] = f"protect: {type(exc).__name__}: {exc}"[:160]
        return orig_trim(self, messages, tools=tools, preserve_recent=preserve_recent, extra_safety_tokens=extra_safety_tokens)

    for f in dir(orig_trim):
        if f.endswith("_wrapped"):
            try:
                setattr(_trim_messages_for_context, f, getattr(orig_trim, f))
            except Exception:
                pass
    _trim_messages_for_context._m79_wrapped = True
    A._trim_messages_for_context = _trim_messages_for_context
    return True

install79()
print('V1630_M79 installed', __import__('json').dumps(m79_summary()), flush=True)
print('V1630_M79_PROBE', __import__('json').dumps(m79_probe()), flush=True)
