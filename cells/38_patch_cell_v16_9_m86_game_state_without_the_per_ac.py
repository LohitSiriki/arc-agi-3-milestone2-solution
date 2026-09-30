# Cell 38 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v16.9 M86 (game state without the per-action O(history) rewrite))
# ===== v16.9 = M86: GAME STATE WITHOUT AN O(HISTORY) REWRITE AND RE-PARSE ON EVERY ACTION =====
"""WHY (v16.7/v16.8 transcripts + bundle code, 2026-09-26):
  * solver._execute_action rewrites the whole game history to tool_runtime_state.json after EVERY action, with
    json.dumps(indent=2) (one line per grid cell: 50 MB at 800 actions, 0.8 s per write on an idle core).
  * every python tool call, and every action() inside it, re-reads and re-parses that file (0.36 s at 800), rebuilds
    ascii + grid payloads for every past frame (0.61 s) and ships them to the sandbox; M44's _level_of parses the
    whole file again just to read the current level.
  * all of this runs in the one notebook process shared by every game. Measured effect: turns +4.5 s per 100 actions
    of history (v16.8 regression, se 0.63); action() calls killed by the 30 s tool timeout rose from 0/328 (first
    20 min) to 36/150 (last 12 min) -- a killed call can leave a batch half executed.
HOW (same information reaches the model and the sandbox; only the cost changes):
  1. write: each history entry is serialized once and cached; the file is written compactly (same JSON content);
     an unchanged state is not rewritten; inside one action batch (step_env) the per-action writes are deferred to
     one write when the batch ends.
  2. read: tool_agent.load_runtime_state returns the in-memory objects the writer produced -- built by the same
     payload round trip a file read performs -- instead of parsing the file. Unknown paths fall back to the file.
  3. sandbox payload: tool_agent._ascii_history_view_payload reuses each entry's view payload (built once).
  4. M44's _level_of reads the level from the in-memory copy.
READ: m86_summary()
"""
import json as _j86
import threading as _th86
import time as _t86
from pathlib import Path as _P86

_LK86 = _th86.Lock()
_REG86 = {}      # str(path) -> dict: src_len, src_first, src_last, jsons, entries, cf, written
_VIEW86 = {}     # id(round-tripped entry) -> (entry, view payload or None)
_STAT86 = {"writes": 0, "writes_skipped_unchanged": 0, "writes_deferred": 0, "batches": 0, "entries_serialized": 0,
           "rebuilds": 0, "loads_cached": 0, "loads_fallback": 0, "view_hits": 0, "view_misses": 0,
           "level_cached": 0, "write_s_max": 0.0, "errors": 0, "last_error": ""}


def _m86_err(exc):
    with _LK86:
        _STAT86["errors"] += 1
        _STAT86["last_error"] = f"{type(exc).__name__}: {exc}"[:200]


def install86():
    import inference.framework.solver as _S86
    import inference.agent.tool_agent as _T86
    import inference.agent.runtime_state as _R86
    if getattr(_S86.write_runtime_state, "_m86", False):
        raise RuntimeError("M86 installed twice")
    orig_write = _S86.write_runtime_state
    orig_load = _T86.load_runtime_state
    frame_view = _T86._ascii_frame_view_payload
    Sess = _S86._HarnessGameSession
    orig_step = Sess.step_env
    orig_wrs = Sess.write_runtime_state

    def _view_entry(rt):
        fp = frame_view(rt.frame)
        return None if fp is None else {"action": rt.action, "frame": fp}

    def write_runtime_state(path, *, current_frame, history):
        t0 = _t86.monotonic()
        key = str(path)
        try:
            hist = list(history)
            ent = _REG86.get(key)
            if ent is not None and (ent["src_len"] > len(hist) or (ent["src_len"] and (
                    hist[ent["src_len"] - 1] is not ent["src_last"] or hist[0] is not ent["src_first"]))):
                ent = None
                with _LK86:
                    _STAT86["rebuilds"] += 1
            if ent is None:
                ent = {"src_len": 0, "src_first": None, "src_last": None, "jsons": [], "entries": [], "cf": None,
                       "written": None}
            new = 0
            for e in hist[ent["src_len"]:]:
                p = _R86.history_entry_to_payload(e)
                ent["jsons"].append(_j86.dumps(p))
                rt = _R86.history_entry_from_payload(p)
                if rt is not None:
                    ent["entries"].append(rt)
                    _VIEW86[id(rt)] = (rt, _view_entry(rt))
                new += 1
            ent["src_len"] = len(hist)
            ent["src_first"] = hist[0] if hist else None
            ent["src_last"] = hist[-1] if hist else None
            cfp = _R86.frame_to_payload(current_frame)
            cf_json = _j86.dumps(cfp)
            sig = (len(ent["jsons"]), cf_json)
            p = _P86(path)
            if ent["written"] == sig and p.exists():
                with _LK86:
                    _REG86[key] = ent
                    _STAT86["writes_skipped_unchanged"] += 1
                    _STAT86["entries_serialized"] += new
                return
            p.parent.mkdir(parents=True, exist_ok=True)
            text = '{"current_frame": ' + cf_json + ', "history": [' + ", ".join(ent["jsons"]) + "]}"
            tmp = p.with_suffix(f"{p.suffix}.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(p)
            ent["cf"] = _R86.frame_from_payload(cfp)
            ent["written"] = sig
            with _LK86:
                _REG86[key] = ent
                _STAT86["writes"] += 1
                _STAT86["entries_serialized"] += new
                _STAT86["write_s_max"] = max(_STAT86["write_s_max"], round(_t86.monotonic() - t0, 3))
        except Exception as exc:                 # never lose the state: drop the cache, use the original writer
            _m86_err(exc)
            with _LK86:
                _REG86.pop(key, None)
            return orig_write(path, current_frame=current_frame, history=history)

    def load_runtime_state(path):
        p = _P86(path)
        ent = _REG86.get(str(p))
        if ent is None or ent.get("written") is None:
            with _LK86:
                _STAT86["loads_fallback"] += 1
            return orig_load(p)
        if not p.exists():
            return None, []
        with _LK86:
            _STAT86["loads_cached"] += 1
        return ent["cf"], list(ent["entries"])

    def _ascii_history_view_payload(history_entries):
        out, hits, misses = [], 0, 0
        for e in history_entries:
            c = _VIEW86.get(id(e))
            if c is not None and c[0] is e:
                hits += 1
                if c[1] is not None:
                    out.append({"action": c[1]["action"], "frame": c[1]["frame"]})
                continue
            misses += 1
            v = _view_entry(e)
            if v is not None:
                out.append(v)
        with _LK86:
            _STAT86["view_hits"] += hits
            _STAT86["view_misses"] += misses
        return out

    def _sess_write_runtime_state(self):
        if getattr(self, "_m86_defer", 0):
            self._m86_dirty = True
            with _LK86:
                _STAT86["writes_deferred"] += 1
            return None
        return orig_wrs(self)

    def step_env(self, arguments):
        self._m86_defer = getattr(self, "_m86_defer", 0) + 1
        try:
            return orig_step(self, arguments)
        finally:
            self._m86_defer -= 1
            if not self._m86_defer:
                with _LK86:
                    _STAT86["batches"] += 1
                if getattr(self, "_m86_dirty", False):
                    self._m86_dirty = False
                    orig_wrs(self)

    write_runtime_state._m86 = True
    _S86.write_runtime_state = write_runtime_state
    _T86.load_runtime_state = load_runtime_state
    _T86._ascii_history_view_payload = _ascii_history_view_payload
    Sess.write_runtime_state = _sess_write_runtime_state
    Sess.step_env = step_env

    # M44 (sticky sandbox) parses the whole state file per python call just to read the level.
    g = globals()
    if callable(g.get("_level_of")) and not getattr(g["_level_of"], "_m86", False):
        orig_level_of = g["_level_of"]

        def _level_of(state_path):
            ent = _REG86.get(str(state_path))
            if ent is not None and ent.get("cf") is not None and _P86(state_path).exists():
                with _LK86:
                    _STAT86["level_cached"] += 1
                return ent["cf"].level
            return orig_level_of(state_path)
        _level_of._m86 = True
        g["_level_of"] = _level_of
    return True


def m86_summary():
    with _LK86:
        return dict(_STAT86)

install86()
print('V1690_M86 installed', __import__('json').dumps(m86_summary()), flush=True)
