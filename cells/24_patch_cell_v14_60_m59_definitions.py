# Cell 24 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v14.60 M59 definitions)
# ===== v14.60 = M59: WORLD-MODEL CAPTURE + CODE MEMORY =====
# Measured on v14.59 (25 games, 2,188 model replies):
#  (a) 496 of 1,377 assistant texts label the note "World model (revised):"; the harness label matcher only
#      accepts "World model:", so those updates were dropped. 811 replies had no visible text at all; 170 of
#      them carried a labeled world model inside the reasoning. Zero-level games froze the carried model for
#      19/45 (bp35) and 7/29 (tr87) turns.
#  (b) Functions/classes/imports do not survive between python calls (M44 keeps plain data only): tr87 lost
#      12 of 30 turns to no-action, 10 NameErrors (WHEEL, T, r, ...).
# M59a: label variants "World model (...):" etc. are accepted; when the visible text carries no world model,
#       the LAST labeled block of the reasoning is used (earlier drafts in the reasoning are ignored).
# M59b: top-level def/class/import/lambda-assignments from earlier python calls on the SAME level are
#       prepended to each new call (skipping names the new code redefines); cleared on level change.
#       v14.63 fix (v14.62: 8/25 games, 6 dead for the whole run): definitions were remembered BEFORE the call ran,
#       so a sandbox-rejected `import difflib` was stored and prepended to every later call on the level, which then
#       failed at "line 2" forever (su15/wa30: 500+ ImportErrors, 0 actions). Now definitions are remembered only
#       after a successful call; a failure inside the prepended block drops the offending entry and reruns the call
#       in the same turn; traceback line numbers are shifted back to the model's own code (M60 quotes them).
import ast as _ast59
import json as _js59
import re as _re59
import sys as _s59
import threading as _th59

_STAT59 = {"variant_labels": 0, "reasoning_notes": 0, "code_prepends": 0, "prepended_defs_max": 0,
           "level_resets": 0, "prepend_failures": 0, "line_shifts": 0, "errors": 0, "last_error": ""}
_TB_LINE = _re59.compile(r'(File "<python_tool>", line )(\d+)')
_LABEL_VARIANT = _re59.compile(r"^([A-Za-z][A-Za-z -]{2,30}?)\s*\([^)]{0,40}\)\s*:")
_WM_LABELS = ("World model", "Goal model", "Action model", "Recent findings", "Open questions", "Plan",
              "Cross-level notes", "Hypothesis", "History check", "Next test")


def _normalize_label_variants(content: str) -> str:
    """'World model (revised): x' -> 'World model: x' for the known labels only."""
    known = {l.lower() for l in _WM_LABELS}
    out = []
    for line in content.splitlines():
        stripped = line.lstrip()
        bullet = ""
        while stripped.startswith(("-", "*")):
            bullet += stripped[0]
            stripped = stripped[1:].lstrip()
        m = _LABEL_VARIANT.match(stripped)
        if m and m.group(1).strip().lower() in known:
            _STAT59["variant_labels"] += 1
            line = (bullet + " " if bullet else "") + m.group(1).strip() + ":" + stripped[m.end():]
        out.append(line)
    return "\n".join(out)


def _last_labeled_block(reasoning: str) -> str:
    """Text from the last line that starts with a world-model label to the end of the reasoning."""
    lines = _normalize_label_variants(reasoning).splitlines()
    start = None
    for i, line in enumerate(lines):
        cand = line.strip().lstrip("-* ").lower()
        if cand.startswith("world model:"):
            start = i
    return "\n".join(lines[start:]) if start is not None else ""


def install59():
    T = _s59.modules.get("inference.agent.tool_agent")
    if T is None:
        raise RuntimeError("v14.60: tool_agent not imported yet")
    A = T.ToolAgent

    # ---- M59a: label variants + reasoning fallback ----
    orig_extract = T._extract_scientist_note
    if not getattr(orig_extract, "_m59_wrapped", False):
        def _extract_scientist_note(content):
            return orig_extract(_normalize_label_variants(content or ""))
        _extract_scientist_note._m59_wrapped = True
        T._extract_scientist_note = _extract_scientist_note

    orig_cc = A._chat_completion
    if not getattr(orig_cc, "_m59_wrapped", False):
        def _chat_completion(self, messages, **kw):
            result = orig_cc(self, messages, **kw)
            try:
                msg = result.message or {}
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                if not T._extract_scientist_note(content).get("world_model"):
                    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                    block = _last_labeled_block(reasoning if isinstance(reasoning, str) else "")
                    if block and T._extract_scientist_note(block).get("world_model"):
                        self._update_summarized_knowledge_from_assistant(block)
                        _STAT59["reasoning_notes"] += 1
            except Exception as exc:
                _STAT59["errors"] += 1
                _STAT59["last_error"] = f"a: {type(exc).__name__}: {exc}"[:160]
            return result
        _chat_completion._m59_wrapped = True
        A._chat_completion = _chat_completion

    # ---- M59c: a GAME_OVER that restarts the same level keeps the carried notes ----
    orig_step = A._update_summarized_knowledge_from_step_summary
    if not getattr(orig_step, "_m59_wrapped", False):
        def _update_summarized_knowledge_from_step_summary(self):
            summary = self._last_step_summary or {}
            if summary.get("game_over") and not summary.get("level_transition") and not summary.get("run_complete"):
                _STAT59["game_over_kept"] = _STAT59.get("game_over_kept", 0) + 1
                return
            return orig_step(self)
        _update_summarized_knowledge_from_step_summary._m59_wrapped = True
        A._update_summarized_knowledge_from_step_summary = _update_summarized_knowledge_from_step_summary

    # ---- M59b: code memory ----
    orig_run = A._run_python_tool
    if not getattr(orig_run, "_m59_wrapped", False):
        def _run_python_tool(self, state_path, arguments, *a, **k):
            code, new_code, n = "", "", 0
            try:
                level = _level_of(state_path)
                if getattr(self, "_m59_level", None) != level:
                    if getattr(self, "_m59_defs", None):
                        _STAT59["level_resets"] += 1
                    self._m59_defs = {}
                    self._m59_level = level
                code = str((arguments or {}).get("code", ""))
                new_code, n = prepend_definitions(self._m59_defs, code)
                if n:
                    arguments = dict(arguments, code=new_code)
                    _STAT59["code_prepends"] += 1
                    _STAT59["prepended_defs_max"] = max(_STAT59["prepended_defs_max"], n)
            except Exception as exc:
                _STAT59["errors"] += 1
                _STAT59["last_error"] = f"b: {type(exc).__name__}: {exc}"[:160]
            result = orig_run(self, state_path, arguments, *a, **k)
            try:
                # up to 3 rounds: a failure inside the prepended block (the model's code never ran) drops that
                # entry and reruns; a failure in the model's own code gets its line numbers shifted back.
                for _ in range(3):
                    text = getattr(result, "content", None) or ""
                    failed = '"error"' in text[:400] or "Traceback" in text
                    if not n or not failed:
                        break
                    header_lines = new_code.count("\n") - code.count("\n")
                    try:
                        payload = _js59.loads(text)
                    except Exception:
                        payload = None
                    err = payload.get("error") if isinstance(payload, dict) else None
                    err = str(err) if err else text          # the traceback lives in payload["error"] (JSON-escaped in text)
                    m = _TB_LINE.search(err)
                    line = int(m.group(2)) if m else None
                    if line is None or line > header_lines:
                        if line is not None:
                            shifted = _TB_LINE.sub(lambda mm: mm.group(1) + str(int(mm.group(2)) - header_lines), err)
                            if isinstance(payload, dict) and payload.get("error"):
                                payload["error"] = shifted
                                text = _js59.dumps(payload, indent=2)
                            else:
                                text = shifted
                            result = T._ToolDispatchResult(text, step_executed=getattr(result, "step_executed", False))
                            _STAT59["line_shifts"] += 1
                        break
                    bad = new_code.splitlines()[line - 1]
                    dropped = [name for name, src in self._m59_defs.items() if bad in src.splitlines()]
                    for name in dropped:
                        self._m59_defs.pop(name, None)
                    if not dropped:
                        self._m59_defs = {}
                    _STAT59["prepend_failures"] += 1
                    _STAT59["last_error"] = f"b: dropped {dropped or 'all'} after {bad[:60]!r}"[:160]
                    new_code, n = prepend_definitions(self._m59_defs, code)
                    result = orig_run(self, state_path, dict(arguments, code=new_code if n else code), *a, **k)
                text = getattr(result, "content", None) or ""
                if not ('"error"' in text[:400] or "Traceback" in text):
                    remember_definitions(self._m59_defs, code)
            except Exception as exc:
                _STAT59["errors"] += 1
                _STAT59["last_error"] = f"b2: {type(exc).__name__}: {exc}"[:160]
            return result
        _run_python_tool._m59_wrapped = True
        A._run_python_tool = _run_python_tool
    return True


def _def_names(node):
    if isinstance(node, (_ast59.FunctionDef, _ast59.AsyncFunctionDef, _ast59.ClassDef)):
        return [node.name]
    if isinstance(node, (_ast59.Import, _ast59.ImportFrom)):
        return [(al.asname or al.name).split(".")[0] for al in node.names]
    if isinstance(node, _ast59.Assign) and isinstance(node.value, _ast59.Lambda):
        return [t.id for t in node.targets if isinstance(t, _ast59.Name)]
    return []


def remember_definitions(store: dict, code: str, max_chars: int = 20000) -> None:
    try:
        tree = _ast59.parse(code)
    except SyntaxError:
        return
    for node in tree.body:
        names = _def_names(node)
        if not names:
            continue
        src = _ast59.get_source_segment(code, node) or _ast59.unparse(node)
        for name in names:
            store.pop(name, None)
            store[name] = src
    while sum(len(v) for v in store.values()) > max_chars and store:
        store.pop(next(iter(store)))


def prepend_definitions(store: dict, code: str):
    """Return (code with earlier definitions prepended, number of definitions prepended)."""
    if not store:
        return code, 0
    try:
        tree = _ast59.parse(code)
    except SyntaxError:
        return code, 0
    redefined = {n for node in tree.body for n in _def_names(node)}
    chunks, seen = [], set()
    for name, src in store.items():
        if name in redefined or src in seen:
            continue
        seen.add(src)
        chunks.append(src)
    if not chunks:
        return code, 0
    header = "# --- definitions from your earlier python calls on this level (auto) ---\n"
    candidate = header + "\n".join(chunks) + "\n# --- your code ---\n" + code
    try:
        compile(candidate, "<m59>", "exec")
    except SyntaxError:
        return code, 0
    return candidate, len(chunks)


def m59_summary():
    return dict(_STAT59)


# ---- M59d: M44 persistence made robust: per-variable blobs, no all-or-nothing load, keep blobs that failed to load,
# load errors recorded in the sidecar. (v14.59: 23 NameErrors for names the prompt listed as saved; cause unconfirmed.)
M59_LOAD_BODY = '''
_m44_path = str((initial.get("state") or {}).get("_m44_persist_path") or "")
_m44_level = (initial.get("state") or {}).get("_m44_level")
_m44_base = set(runtime_globals)
_m44_loaded = []
_m44_load_errors = []
_m44_failed_blobs = {}
if _m44_path and os.path.exists(_m44_path):
    try:
        import pickle as _m44_pk
        with open(_m44_path, "rb") as _m44_f:
            _m44_saved = _m44_pk.load(_m44_f)
        for _k, _b in (_m44_saved.get("blobs") or {}).items():
            if _k in _m44_base:
                continue
            try:
                runtime_globals[_k] = _m44_pk.loads(_b); _m44_loaded.append(_k)
            except Exception as _m44_e:
                _m44_failed_blobs[_k] = _b
                _m44_load_errors.append("%s: %s" % (_k, type(_m44_e).__name__))
    except Exception as _m44_e:
        _m44_load_errors.append("file: %s: %s" % (type(_m44_e).__name__, str(_m44_e)[:120]))
'''
M59_SAVE_BODY = '''
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
                    _m44_out[_k] = _m44_blob
                else:
                    _m44_toobig.append(_k)
            except Exception:
                _m44_toobig.append(_k)
        for _k, _b in _m44_failed_blobs.items():
            if _k not in _m44_out and _k not in runtime_globals:
                _m44_out[_k] = _b
        _m44_dropped = []
        while sum(len(_b) for _b in _m44_out.values()) > 900_000 and _m44_out:
            _m44_big = max(_m44_out, key=lambda _k: len(_m44_out[_k]))
            _m44_dropped.append(_m44_big); _m44_out.pop(_m44_big)
        _m44_data = _m44_pk.dumps({"level": _m44_level, "blobs": _m44_out})
        with open(_m44_path + ".tmp", "wb") as _m44_f:
            _m44_f.write(_m44_data)
        os.replace(_m44_path + ".tmp", _m44_path)
        with open(_m44_path + ".names.tmp", "w") as _m44_f:
            _m44_js.dump({"level": _m44_level, "names": sorted(_m44_out), "bytes": len(_m44_data),
                          "dropped": _m44_dropped + _m44_toobig, "loaded": sorted(_m44_loaded),
                          "load_errors": _m44_load_errors[:20]}, _m44_f)
        os.replace(_m44_path + ".names.tmp", _m44_path + ".names")
except Exception:
    pass
'''

M59_PROMPT_LINE = ("Persistence: top-level plain-data variables (lists, dicts, tuples, sets, numbers, strings) that you "
                   "assign in `python` are saved and available again in your next `python` call on this level; your "
                   "top-level `def` functions, lambdas and imports from earlier calls on this level are re-defined "
                   "automatically at the top of each new call. File I/O (`open`) is unavailable. Saved variables so far: ")
