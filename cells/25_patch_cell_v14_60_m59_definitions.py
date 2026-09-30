# Cell 25 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v14.60 M59 definitions)
# ===== v14.44 = M44: MULTI-REQUEST TURNS + STICKY SANDBOX =====
# Measured across 10 Flash-Next runs (13,126 model calls, results 54-69): 49.1% of calls execute
# no action; 88% of those ran one investigative python call and were then cut off by
# LOCAL_ANALYZER_YIELD_SECONDS=60 (< 128-177 s per request), so every turn = one request.
# 6.5% of no-action calls were NameErrors: the sandbox is a fresh subprocess per call.
#
# M44a: yield 60 -> 0 (off) and TOOL_STEPS 0 -> 4: a turn may run up to 4 python calls, then
#       acts or yields. M43's permit check runs before EVERY request (should_stop), so permits
#       still gate requests, not turns.
# M44b: plain-data top-level variables persist between python calls within a game (pickle in
#       the game's runtime dir, cleared on level change). Functions/classes do not persist.
import os as _os, sys as _s, threading as _th, json as _json, pickle as _pk

_STAT44 = {"yield_seconds": None, "tool_steps": None, "persist_saves": 0, "persist_loads": 0,
           "persist_names_max": 0, "persist_errors": 0, "level_resets": 0, "prompt_lines": 0,
           "persist_dropped": 0, "persist_bytes_max": 0, "last_error": ""}
_TL = _th.local()

# ---------------------------------------------------------------- M44a: turn shape
def install44a(yield_seconds=0.0, tool_steps=4):
    _os.environ["LOCAL_ANALYZER_YIELD_SECONDS"] = str(yield_seconds)
    _os.environ["LOCAL_ANALYZER_TOOL_STEPS"] = str(tool_steps)
    n = 0
    for mod in list(_s.modules.values()):
        if mod is None or not hasattr(mod, "_LOCAL_ANALYZER_YIELD_SECONDS"):
            continue
        mod._LOCAL_ANALYZER_YIELD_SECONDS = float(yield_seconds)
        mod._LOCAL_ANALYZER_TOOL_STEPS = int(tool_steps)
        n += 1
    _STAT44["yield_seconds"] = float(yield_seconds); _STAT44["tool_steps"] = int(tool_steps)
    return n

# ---------------------------------------------------------------- M44b: sticky sandbox
_LOAD_BODY_M44_ORIG = '''
_m44_path = str((initial.get("state") or {}).get("_m44_persist_path") or "")
_m44_level = (initial.get("state") or {}).get("_m44_level")
_m44_base = set(runtime_globals)
_m44_loaded = []
if _m44_path and os.path.exists(_m44_path):
    try:
        import pickle as _m44_pk
        with open(_m44_path, "rb") as _m44_f:
            _m44_saved = _m44_pk.load(_m44_f)
        if _m44_saved.get("level") == _m44_level:
            for _k, _v in (_m44_saved.get("vars") or {}).items():
                if _k not in _m44_base:
                    runtime_globals[_k] = _v; _m44_loaded.append(_k)
    except Exception:
        pass
'''
_SAVE_BODY_M44_ORIG = '''
try:
    if _m44_path:
        import pickle as _m44_pk, json as _m44_js
        _m44_out = {}; _m44_toobig = []
        for _k, _v in list(runtime_globals.items()):
            if _k in _m44_base or _k.startswith("_") or callable(_v) or isinstance(_v, type):
                continue
            try:
                _m44_blob = _m44_pk.dumps(_v)
                if len(_m44_blob) <= 200_000:
                    _m44_out[_k] = (_v, len(_m44_blob))
                else:
                    _m44_toobig.append(_k)
            except Exception:
                _m44_toobig.append(_k)
        _m44_dropped = []
        while True:
            _m44_data = _m44_pk.dumps({"level": _m44_level, "vars": {_k: _v for _k, (_v, _n) in _m44_out.items()}})
            if len(_m44_data) <= 900_000 or not _m44_out:
                break
            _m44_big = max(_m44_out, key=lambda _k: _m44_out[_k][1])   # drop the largest, keep the rest
            _m44_dropped.append(_m44_big); _m44_out.pop(_m44_big)
        with open(_m44_path + ".tmp", "wb") as _m44_f:
            _m44_f.write(_m44_data)
        os.replace(_m44_path + ".tmp", _m44_path)
        with open(_m44_path + ".names.tmp", "w") as _m44_f:
            _m44_js.dump({"level": _m44_level, "names": sorted(_m44_out), "bytes": len(_m44_data),
                          "dropped": _m44_dropped + _m44_toobig, "loaded": sorted(_m44_loaded)}, _m44_f)
        os.replace(_m44_path + ".names.tmp", _m44_path + ".names")
except Exception:
    pass
'''
# v14.60 M59d: robust per-variable persistence + updated prompt line
_LOAD_BODY = M59_LOAD_BODY
_SAVE_BODY = M59_SAVE_BODY
_PROMPT_LINE = M59_PROMPT_LINE
def _ind(body, n):
    return "".join((" " * n + ln if ln.strip() else ln) + "\n" for ln in body.strip("\n").splitlines())

def _patch_bootstrap(S):
    src = S._SANDBOX_BOOTSTRAP
    if "_m44_persist_path" in src:
        return False
    a1 = '    _refresh_state(initial.get("state") or {})\n'
    a2 = '        _send(\n            {\n                "type": "final",'
    a3 = '        _send(\n            {\n                "type": "error",'
    for a in (a1, a2, a3):
        if src.count(a) != 1:
            raise RuntimeError("v14.44: bootstrap anchor not found exactly once: %r" % a[:40])
    src = src.replace(a1, a1 + _ind(_LOAD_BODY, 4))
    src = src.replace(a2, _ind(_SAVE_BODY, 8) + a2)
    src = src.replace(a3, _ind(_SAVE_BODY, 8) + a3)
    compile(src, "<m44_bootstrap>", "exec")      # must still be valid python
    S._SANDBOX_BOOTSTRAP = src
    return True

def _wrap_run(S, T):
    orig = S.run_sandboxed_python
    if getattr(orig, "_m44_wrapped", False):
        return orig
    def run(*a, **kw):
        # signature-agnostic: the Kaggle bundle's run_sandboxed_python lacks animation_handler
        st = dict(kw.get("initial_state") or {})
        pp = getattr(_TL, "persist_path", None)
        if pp:
            st["_m44_persist_path"] = pp
            st["_m44_level"] = getattr(_TL, "level", None)
        if "initial_state" in kw:
            kw["initial_state"] = st
        return orig(*a, **kw)
    run._m44_wrapped = True
    S.run_sandboxed_python = run
    T.run_sandboxed_python = run          # tool_agent imported the name directly
    return run

def _level_of(state_path):
    try:
        st = _json.loads(open(state_path).read())
        fr = st.get("current_frame") or {}
        lv = fr.get("level")
        if lv is None:
            lv = st.get("level")
        return lv
    except Exception:
        return None

def _wrap_tool(T):
    A = T.ToolAgent
    orig = A._run_python_tool
    if getattr(orig, "_m44_wrapped", False):
        return
    def _run_python_tool(self, state_path, arguments, *a, **k):
        try:
            self._m44_game = state_path.name.split("_tool_runtime_state")[0]; pp = str(state_path.parent / (self._m44_game + "_m44_vars.pkl"))
            lv = _level_of(state_path)
            prev = getattr(self, "_m44_level", None)
            if prev is not None and lv is not None and lv != prev and _os.path.exists(pp):
                _os.remove(pp); _STAT44["level_resets"] += 1
            self._m44_level = lv
            _TL.persist_path = pp; _TL.level = lv
        except Exception as exc:
            _STAT44["persist_errors"] += 1; _STAT44["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
            _TL.persist_path = None
        try:
            return orig(self, state_path, arguments, *a, **k)
        finally:
            try:
                info = _sidecar(_TL.persist_path) if getattr(_TL, "persist_path", None) else None
                if info: _STAT44["persist_saves"] += 1; _STAT44["persist_bytes_max"] = max(_STAT44["persist_bytes_max"], int(info.get("bytes") or 0))
            except Exception:
                pass
            _TL.persist_path = None
    _run_python_tool._m44_wrapped = True
    A._run_python_tool = _run_python_tool

_PROMPT_LINE = ("Persistence: top-level plain-data variables (lists, dicts, tuples, sets, numbers, strings) "
                "that you assign in `python` are saved and available again in your next `python` call on this "
                "level; functions and classes are NOT saved -- redefine them, and file I/O (`open`) is unavailable. "
                "Saved variables so far: ")

def _sidecar(path):
    try:
        import json as _js
        with open(path + ".names") as f:
            return _js.load(f)
    except Exception:
        return None

def _wrap_prompt(T):
    A = T.ToolAgent
    orig = A._build_user_prompt
    if getattr(orig, "_m45_wrapped", False) or getattr(orig, "_m44_wrapped", False):
        return
    def _build_user_prompt(self, *a, **k):
        out = orig(self, *a, **k)
        try:
            names = []
            d = getattr(self, "_session_runtime_dir", None)
            if d is not None:
                info = _sidecar(_os.path.join(str(d), getattr(self, "_m44_game", "") + "_m44_vars.pkl"))
                if info and info.get("level") == getattr(self, "_m44_level", info.get("level")):
                    names = list(info.get("names") or [])
                    _STAT44["persist_saves"] = max(_STAT44["persist_saves"], 0) + 0
                    if info.get("dropped"): _STAT44["persist_dropped"] += len(info["dropped"])
                    if info.get("loaded"): _STAT44["persist_loads"] += len(info["loaded"])
            _STAT44["persist_names_max"] = max(_STAT44["persist_names_max"], len(names))
            _STAT44["prompt_lines"] += 1
            return out + "\n" + _PROMPT_LINE + (", ".join(names[:40]) if names else "none") + "."
        except Exception as exc:
            _STAT44["persist_errors"] += 1
            _STAT44.setdefault("last_error", "")[:0]; _STAT44["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
            return out
    _build_user_prompt._m44_wrapped = True
    A._build_user_prompt = _build_user_prompt

def install44b():
    S = _s.modules.get("inference.agent.python_tool_sandbox")
    T = _s.modules.get("inference.agent.tool_agent")
    if S is None or T is None:
        raise RuntimeError("v14.44: sandbox/tool_agent modules not imported yet")
    changed = _patch_bootstrap(S)
    _wrap_run(S, T); _wrap_tool(T); _wrap_prompt(T)
    return changed

def m44_summary():
    return dict(_STAT44)

# ---------------------------------------------------------------- M44 install + REAL self-check gate
_n44 = install44a(0.0, 0)   # v16.7: 0 = unlimited python calls per turn (rank-6 max_tool_calls 0)
if _n44 < 1:
    raise RuntimeError("v14.44: no imported module carried _LOCAL_ANALYZER_YIELD_SECONDS -- M44a would be INERT")
import inference.agent.tool_agent as _T44
if _T44._LOCAL_ANALYZER_YIELD_SECONDS != 0.0 or _T44._LOCAL_ANALYZER_TOOL_STEPS != 0:
    raise RuntimeError("v14.44: tool_agent globals not patched: yield=%r steps=%r" % (_T44._LOCAL_ANALYZER_YIELD_SECONDS, _T44._LOCAL_ANALYZER_TOOL_STEPS))

_changed44 = install44b()
if not _changed44:
    raise RuntimeError("v14.44: sandbox bootstrap was not patched")
import tempfile as _tf44, os as _os44
import inference.agent.python_tool_sandbox as _S44
_d44 = _tf44.mkdtemp(); _TL.persist_path = _os44.path.join(_d44, "m44_vars.pkl"); _TL.level = 1
_r1 = _S44.run_sandboxed_python(code="R=[4,12]; print('set')", timeout_seconds=15, initial_state={}, action_handler=lambda a: {})
_r2 = _S44.run_sandboxed_python(code="print(R)", timeout_seconds=15, initial_state={}, action_handler=lambda a: {})
_info44 = _sidecar(_TL.persist_path)
_TL.persist_path = None
if _r1.get("error") or "[4, 12]" not in str(_r2.get("stdout", "")):
    raise RuntimeError("v14.44: sticky sandbox round-trip FAILED: %r / %r" % (_r1, _r2))
if not _info44 or _info44.get("names") != ["R"]:
    raise RuntimeError("v14.44: sidecar names file missing or wrong: %r" % (_info44,))
print("M44b sticky sandbox round-trip OK", flush=True)

print("M44 installed:", m44_summary(), flush=True)
