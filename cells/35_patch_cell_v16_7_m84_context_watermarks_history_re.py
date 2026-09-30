# Cell 35 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v16.7 M84 (context watermarks + history retention))
# ===== v16.7 = M84: CONTEXT WATERMARKS (rank-6 fork, commit 64cda76 "Trim context by steps with watermarks") =====
"""History oscillates between 26,624 and 36,864 REAL tokens instead of sitting at the budget and dropping one turn per
request.

WHY: their Sep 17 -> Sep 18 change set (6.18 -> 11.49 on the board) included context 32,768 -> 49,152 with a high/low
watermark of 36,864 / 26,624 (configs/inference.flash-next.json at 6420079; notes/context/context_oscillation_0918.md:
"32k +/- 4k", "oscillating context window seems to perform better than control", ~10% faster on Kaggle from prefix-cache
reuse). Our own trims at the budget re-process the whole prompt: 86% of all fresh prompt tokens (QUEUE_ANALYSIS_2026-09-23
section 4).

HOW: wrapper on ToolAgent._trim_messages_for_context installed AFTER M24 and M79 but BEFORE the context repair, so the
repair (outermost) still checks only the real hard input limit with the caller's own reserve. Review 2026-09-25 found
that with M84 outermost, the repair read M84's watermark reserve as a hard limit and emergency-shortened a protected
current turn (38,614 -> 19,385 tokens) that fitted under 45,056.
  real = M79's exact server token count of the request (m79_real_tokens, cached per request).
  real <= 36,864 -> pass through unchanged (the inner trim only fires at the 49k budget).
  real >  36,864 -> ask the inner trim for extra_safety_tokens = M79_HARD - 26,624. M79's guard makes the inner loop
                    keep dropping old blocks until real + (budget - M79_HARD) <= budget - extra, i.e. real <= 26,624.
The inner trim only drops whole old blocks (bundle tool_agent.py _trim_messages_for_context); M79's current-turn
protection keeps the current turn whole even above 26,624; M24 records dropped reasoning in its ledger.
ALSO: the inherited 30-assistant-message retention cap (_PERSISTENT_HISTORY_ASSISTANT_TURNS) is lifted. With unlimited
tool calls one turn can exceed 30 assistant messages; the cap then cut the turn's user message and
_drop_until_first_user_message persisted NOTHING (review repro: 31 messages, 1,860 tokens -> 0 kept). The rank-6 fork
deleted the cap; history stays bounded by these token watermarks.
READ: m84_summary() -> calls, high_hits (trims to the low watermark), max_real_seen, count_errors, history_turn_cap.
"""
import threading as _th84

M84_HIGH = 57_344
M84_LOW = 45_056
M84_HISTORY_TURN_CAP = 1_000_000_000
_LK84 = _th84.Lock()
_STAT84 = {"calls": 0, "high_hits": 0, "max_real_seen": 0, "count_errors": 0, "last_error": "", "history_turn_cap": None}


def install84():
    import inference.agent.tool_agent as _T84
    _A84 = _T84.ToolAgent
    _inner84 = _A84._trim_messages_for_context
    if getattr(_inner84, "_m84_wrapped", False):
        raise RuntimeError("M84 installed twice")
    if not getattr(_A84._estimate_request_input_tokens, "_m79_wrapped", False):
        raise RuntimeError("M84 needs M79's exact-count guard installed first")
    if getattr(_A84._drop_oldest_history_block, "_deadline_repair_wrapped", False):
        raise RuntimeError("M84 must be installed BEFORE the context repair")
    _T84._PERSISTENT_HISTORY_ASSISTANT_TURNS = M84_HISTORY_TURN_CAP
    _STAT84["history_turn_cap"] = M84_HISTORY_TURN_CAP
    extra84 = int(M79_HARD) - M84_LOW
    if extra84 <= 0:
        raise RuntimeError(f"M84: M79_HARD {M79_HARD} must exceed the low watermark {M84_LOW}")

    def _trim_messages_for_context(self, messages, *, tools=None, preserve_recent=1, extra_safety_tokens=0):
        with _LK84:
            _STAT84["calls"] += 1
        try:
            real = int(m79_real_tokens(self, list(messages or []), tools))
        except Exception as exc:
            with _LK84:
                _STAT84["count_errors"] += 1
                _STAT84["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
            real = int(self._estimate_request_input_tokens(list(messages or []), tools=tools))
        with _LK84:
            _STAT84["max_real_seen"] = max(_STAT84["max_real_seen"], real)
        if real > M84_HIGH:
            with _LK84:
                _STAT84["high_hits"] += 1
            extra_safety_tokens = max(int(extra_safety_tokens or 0), extra84)
        return _inner84(self, messages, tools=tools, preserve_recent=preserve_recent,
                        extra_safety_tokens=extra_safety_tokens)

    for _f in dir(_inner84):
        if _f.endswith("_wrapped"):
            setattr(_trim_messages_for_context, _f, getattr(_inner84, _f))
    _trim_messages_for_context._m84_wrapped = True
    _A84._trim_messages_for_context = _trim_messages_for_context
    return extra84


def m84_summary():
    with _LK84:
        return dict(_STAT84, high=M84_HIGH, low=M84_LOW)

_extra84 = install84()
print('V1670_M84 installed extra_safety', _extra84, __import__('json').dumps(m84_summary()), flush=True)
