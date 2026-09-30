# Cell 42 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v16.8 M85 (solved-level memory))
# ===== v16.8 = M85: SOLVED-LEVEL MEMORY (the rule that cleared a level is carried into every later level) =====
"""WHY (v16.6 vs v16.7 local transcripts, 2026-09-25): in all 4 levels v16.7 lost (cd82 L2, dc22 L2, s5i5 L3, sb26 L4)
it entered the level without the rule that solved the previous one and spent 60-100+ minutes re-deriving it
(cd82 L2: 122 actions vs 8; sb26 L4: 189 without clearing vs 37), while v16.6 carried a verified rule and cleared near
baseline. The harness has no working carrier: the world model is wiped at every level change by design, and the
"Cross-level notes" field that survives the wipe was filled in 0 of 989 / 0 of 1,189 level>=2 prompts (the model writes
that label in visible text 4 / 2 times per run; capture reads visible text only). M24's ledger at dc22's end held
working fragments, not the L1 rule.

HOW (per agent = per game; adds, never removes, history):
  1. At the first prompt after a level transition, record FACTS the harness already has: the level number, how many
     actions it took, its last M85_ACTIONS_KEPT actions (the winning tail), and the tail of the model's own reasoning
     from the turn that cleared it (last assistant message(s) in the kept history).
  2. The same prompt asks the model to write, in its reply, a line starting "Rule for level N:" with the goal, what
     each control did and the plan that won (<= 5 lines). No extra request.
  3. The rule is captured from the reply's visible text OR its reasoning on the next prompt(s) (asked on up to
     M85_ASK_TURNS prompts, then left "not stated").
  4. Everything is pinned as one block at the END of the agent's system prompt (after M77's block), replaced in place
     on updates. The newest M85_FULL_LEVELS levels keep actions + reasoning; older ones keep the rule only.
READ: m85_summary() -> levels_recorded, rules_captured, rules_missing, asks, block_chars_max, errors.
"""
import re as _re85
import threading as _th85

M85_ACTIONS_KEPT = 30
M85_REASONING_CHARS = 900
M85_RULE_CHARS = 600
M85_FULL_LEVELS = 3
M85_ASK_TURNS = 2
M85_HEADER = "=== SOLVED LEVELS (facts from this game + your own conclusions; kept for the whole game) ==="
M85_FOOTER = "=== END SOLVED LEVELS ==="
_LK85 = _th85.Lock()
_STAT85 = {"levels_recorded": 0, "rules_captured": 0, "rules_missing": 0, "asks": 0, "block_chars_max": 0,
           "errors": 0, "last_error": ""}


def _m85_text(msg):
    parts = []
    for k in ("content", "reasoning_content", "reasoning"):
        v = msg.get(k) if isinstance(msg, dict) else None
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts += [p.get("text", "") for p in v if isinstance(p, dict) and isinstance(p.get("text"), str)]
    return "\n".join(p for p in parts if p)


def m85_level_actions(history_entries, level):
    """Actions taken while on `level`: history[i].frame is the frame AFTER history[i].action, so an action belongs to
    the level of the frame before it. The seeded first entry has an empty action and is skipped."""
    out, prev = [], None
    for e in history_entries or []:
        act = str(getattr(e, "action", "") or "").strip()
        if act and prev == level:
            out.append(act)
        lv = getattr(getattr(e, "frame", None), "level", None)
        if isinstance(lv, int):
            prev = lv
    return out


def m85_find_rule(text, level):
    """The model's 'Rule for level N:' statement. Its own line if that line carries >= 40 chars; otherwise up to 5
    following non-empty lines (a bulleted rule), stopping at a blank line. Blank lines cannot be relied on as the end
    marker (the stored reasoning in history can lose them), so the capture never runs past one line / five lines."""
    m = _re85.search(r"Rule for level\s*%d\s*:[ \t]*" % int(level), text or "", _re85.I)
    if not m:
        return None
    lines = (text[m.end():]).split("\n")
    first = lines[0].strip()
    if len(first) >= 40:
        rule = first
    else:
        out = [first] if first else []
        for ln in lines[1:6]:
            if not ln.strip():
                if out:
                    break
                continue
            out.append(ln.strip())
        rule = " ".join(out)
    rule = " ".join(rule.split())[:M85_RULE_CHARS]
    return rule if len(rule) >= 20 else None


def m85_block(levels):
    if not levels:
        return ""
    lines = [M85_HEADER]
    full_from = max(0, len(levels) - M85_FULL_LEVELS)
    for i, rec in enumerate(levels):
        lines.append(f"Level {rec['level']} - cleared after {rec['n_actions']} actions.")
        lines.append(f"  Your rule: {rec['rule'] if rec.get('rule') else 'not stated'}")
        if i >= full_from:
            tail = rec["actions"]
            skipped = rec["n_actions"] - len(tail)
            lines.append("  Last actions before it cleared" + (f" ({skipped} earlier omitted)" if skipped > 0 else "")
                         + ": " + (", ".join(tail) if tail else "none recorded"))
            if rec.get("reasoning"):
                lines.append("  Your reasoning in the turn that cleared it: " + rec["reasoning"])
    lines.append("Reuse these where the new level has the same controls; verify with one probe before relying on them.")
    lines.append(M85_FOOTER)
    return "\n".join(lines)


def install85():
    import inference.agent.tool_agent as _T85
    _A85 = _T85.ToolAgent
    _inner85 = _A85._build_user_prompt
    if getattr(_inner85, "_m85_wrapped", False):
        raise RuntimeError("M85 installed twice")

    def _set_block(self):
        block = m85_block(getattr(self, "_m85_levels", []))
        sp = getattr(self, "_system_prompt", "") or ""
        old = getattr(self, "_m85_block_text", "")
        if old and old in sp:
            sp = sp.replace(old, block)
        elif block:
            sp = sp.rstrip("\n") + "\n\n" + block
        self._system_prompt = sp
        self._m85_block_text = block
        with _LK85:
            _STAT85["block_chars_max"] = max(_STAT85["block_chars_max"], len(block))

    def _capture_rules(self):
        """Look for 'Rule for level N:' in assistant messages written since the ask (reply text or reasoning)."""
        pending = [r for r in getattr(self, "_m85_levels", []) if not r.get("rule") and r.get("asks", 0) > r.get("checks", 0)]
        if not pending:
            return False
        texts = [_m85_text(m) for m in (getattr(self, "_history_messages", []) or [])
                 if isinstance(m, dict) and m.get("role") == "assistant"]
        changed = False
        for rec in pending:
            rec["checks"] = rec.get("checks", 0) + 1
            for t in reversed(texts):
                rule = m85_find_rule(t, rec["level"])
                if rule:
                    rec["rule"] = rule
                    changed = True
                    with _LK85:
                        _STAT85["rules_captured"] += 1
                    break
            if not rec.get("rule") and rec["checks"] >= M85_ASK_TURNS:
                with _LK85:
                    _STAT85["rules_missing"] += 1
        return changed

    def _build_user_prompt(self, action_num, *args, **kwargs):
        out = _inner85(self, action_num, *args, **kwargs)
        try:
            if not hasattr(self, "_m85_levels"):
                self._m85_levels = []
                self._m85_block_text = ""
            changed = _capture_rules(self)
            summary = kwargs.get("previous_step_summary") or {}
            level_now = getattr(kwargs.get("current_frame"), "level", None)
            if summary.get("level_transition") and not summary.get("run_complete") and isinstance(level_now, int):
                solved = level_now - 1
                if solved >= 1 and all(r["level"] != solved for r in self._m85_levels):
                    acts = m85_level_actions(kwargs.get("history_entries") or [], solved)
                    reasoning = ""
                    for m in reversed(getattr(self, "_history_messages", []) or []):
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            t = " ".join(_m85_text(m).split())
                            if t:
                                reasoning = t[-M85_REASONING_CHARS:]
                                break
                    self._m85_levels.append(dict(level=solved, n_actions=len(acts), actions=acts[-M85_ACTIONS_KEPT:],
                                                 reasoning=reasoning, rule=None, asks=0, checks=0))
                    changed = True
                    with _LK85:
                        _STAT85["levels_recorded"] += 1
            last = self._m85_levels[-1] if self._m85_levels else None
            if last is not None and not last.get("rule") and last["asks"] < M85_ASK_TURNS:
                last["asks"] += 1
                with _LK85:
                    _STAT85["asks"] += 1
                out += ("\n\nLEVEL %d SOLVED. In your reply this turn, before acting, write one short paragraph that "
                        "starts exactly with 'Rule for level %d:' -- the goal, what each control/action did, and the "
                        "plan that won (at most 5 lines). It is pinned for every later level." % (last["level"], last["level"]))
            if changed:
                _set_block(self)
        except Exception as exc:
            with _LK85:
                _STAT85["errors"] += 1
                _STAT85["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out

    for _f in dir(_inner85):
        if _f.endswith("_wrapped"):
            setattr(_build_user_prompt, _f, getattr(_inner85, _f))
    _build_user_prompt._m85_wrapped = True
    _A85._build_user_prompt = _build_user_prompt
    return True


def m85_summary():
    with _LK85:
        return dict(_STAT85)

install85()
print('V1680_M85 installed', __import__('json').dumps(m85_summary()), flush=True)
