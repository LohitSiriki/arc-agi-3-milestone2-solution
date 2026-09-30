# Cell 31 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v14.58 reasoning history)
# ===== v14.58 = past reasoning sent as reasoning_content =====
import inference.agent.tool_agent as _ta58
_STAT58 = {"requests": 0, "assistant_msgs": 0, "copied": 0}
_orig_cc58 = _ta58.ToolAgent._chat_completion
if getattr(_orig_cc58, "_v1458_wrapped", False):
    raise RuntimeError("v14.58 reasoning patch applied twice")
def _cc58(self, messages, **kw):
    out = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant":
            _STAT58["assistant_msgs"] += 1
            r = m.get("reasoning")
            if isinstance(r, str) and r and not m.get("reasoning_content"):
                m = dict(m); m["reasoning_content"] = r
                _STAT58["copied"] += 1
        out.append(m)
    _STAT58["requests"] += 1
    return _orig_cc58(self, out, **kw)
_cc58._v1458_wrapped = True
_ta58.ToolAgent._chat_completion = _cc58
def m58_summary():
    return dict(_STAT58)
print("V1458_REASONING_PATCH installed", flush=True)
