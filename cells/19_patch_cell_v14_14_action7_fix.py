# Cell 19 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v14.14 ACTION7 FIX)
# ============================================== v14.14 ACTION7 FIX =========
"""Restore ACTION7 to the action-name mapping.

THE BUG (verified at four levels, 2026-09-04):
  1. The ENGINE supports it: arcengine/enums.py has `ACTION7 = (7, SimpleAction)`.
  2. The HARNESS omits it: inference/agent/action_names.py ENGINE_TO_MODEL_ACTION
     lists ACTION1-6 and RESET only.
  3. So `to_engine_action("ACTION7")` returns **None** (verified by running it),
     while every other action round-trips correctly.
  4. solver.py:627-632 then rejects it before it ever reaches the engine:
         action_name = to_engine_action(raw_action.get("action"))
         if not action_name:
             return None, f"Unknown action at index {index}: ..."

MEASURED CONSEQUENCE: ACTION7 executed **0 times in 25 games / 3,566 actions**,
while being advertised to the model in `valid_actions` in 6 of those games on
every single turn. The model is told it is legal and cannot use it.

WHY OUR ARCHIVE SAYS OTHERWISE: module M7 added ACTION7 in the patched 27B
lineage, so "ACTION7 confirmed working" was true WITH M7 installed. The raw
Flash-Next fork did not inherit the M-modules, so the bug returned. Both the old
retraction and the new zero are true.

WHY IT MIGHT MATTER: 4 of the 6 games that offer ACTION7 are perpetual-zero
(g50t, sk48, sp80, tn36).

HONEST LIMIT: this proves the action CANNOT work. It does not prove the model
would choose it. Both failure modes may be present.
"""
import sys

STAT = {"modules_patched": 0, "errors": 0}


def install(stat=None):
    """Add ACTION7 to every loaded copy of the mapping. Returns modules patched."""
    st = stat if stat is not None else STAT
    done = 0
    for name in list(sys.modules):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        e2m = getattr(mod, "ENGINE_TO_MODEL_ACTION", None)
        if not isinstance(e2m, dict):
            continue
        try:
            if "ACTION7" not in e2m:
                # identity label: the engine name IS the model-facing name, as
                # there is no semantic alias for it (unlike UP/DOWN/MOUSE...).
                e2m["ACTION7"] = "ACTION7"
                done += 1
            m2e = getattr(mod, "MODEL_TO_ENGINE_ACTION", None)
            if isinstance(m2e, dict) and "ACTION7" not in m2e:
                m2e["ACTION7"] = "ACTION7"
        except Exception:
            st["errors"] += 1
    st["modules_patched"] = done
    return done


_n_a7 = install()
# Verify against the REAL mapping, not the installer's return value.
import sys as _s7
def _a7_ok():
    for _n in list(_s7.modules):
        _m = _s7.modules.get(_n)
        _f = getattr(_m, 'to_engine_action', None) if _m is not None else None
        if _f is not None:
            return _f('ACTION7') == 'ACTION7'
    return False
if not _a7_ok():
    raise RuntimeError('v14.14: ACTION7 still does not map -- fix before the run.')
# Pre-existing mappings must be untouched.
for _mdl in list(_s7.modules):
    _mm = _s7.modules.get(_mdl)
    _tf = getattr(_mm, 'to_engine_action', None) if _mm is not None else None
    if _tf is None:
        continue
    for _a, _e in [('UP','ACTION1'),('DOWN','ACTION2'),('LEFT','ACTION3'),
                   ('RIGHT','ACTION4'),('SPACE','ACTION5'),('MOUSE','ACTION6'),
                   ('RESET','RESET')]:
        if _tf(_a) != _e:
            raise RuntimeError(f'v14.14: mapping regression {_a} -> {_tf(_a)}')
    break
print(f'V14_14_ACTION7 patched={_n_a7} maps_ok=True', flush=True)
