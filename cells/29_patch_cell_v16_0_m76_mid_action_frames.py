# Cell 29 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v16.0 M76 (mid-action frames))
# ===== v16.0 = M76: SHOW THE INTERMEDIATE ANIMATION FRAMES, NOT JUST COUNT THEM =====
"""The harness shows the model only the FINAL frame of an action. When an action animates and the board then
returns to (near) its starting appearance, that final frame contains no evidence the action did anything -- so
"press the button, nothing changed, the button is inert" is a CORRECT inference from what the model was shown.

Measured, 860 actions over nine games of the v14.75 run: seven games had intermediate changes absent from the
final frame -- sp80 (22 frames, 704 transiently changed cells -> 2 changed pixels in the final frame), cd82
(15 frames, 158 cells -> no change), ft09 (5 frames, 80 cells -> no change), ls20 (6 frames, 76 cells -> no
change), plus tn36, sc25, s5i5. Independently, WALL_ANALYSIS_2026-09-22.md recorded the same thing for sp80
from the transcript side ("a failed pour is wiped before the next frame") and found that the deterministic
walls are exactly the levels where a new element is probed once, seen to do nothing, and written off.

M66 (v14.68) read `taaf.game.GameState.all_frames` and reported three SCALARS -- animation_frames,
animation_changed_cells, animation_only_changed. v14.68 scored 11.55 local vs v14.64's 12.04: telling the model
"12 cells changed" is a number it cannot form a hypothesis from. NOTE: M66 is NOT present in this lineage --
v14.74 branched from v14.64, which predates M62/M64/M66/M70/M72 -- so M76 reads all_frames itself and is the only
module doing so. Consequence to watch: the action result still reports `board_changed: False` with no flag
contradicting it, so the model must infer from the images alone that the action reached the game. M76 adds the
CONTENT:
up to M76_MAX_IMAGES intermediate boards, rendered with the harness's own renderer, labelled with action and
frame index, appended to the next user message beside the current-grid image.

Cost: M74 measured one board image at 66 prompt tokens (64 visual + 2 markers) at MULTIMODAL_UPSCALE=4.
Four extra images ~= 264 tokens on a ~24,000-token prompt, about 1%.

ONLY fires when information was actually lost: some frame differs from the pre-action board in strictly more
cells than the final board does. An action whose effect is already visible in the final frame adds nothing.

Per-game state lives on the session/agent instance (the pattern M64/M66 use, audited 2026-09-22: M64's handle is
thread-local and cleared in a finally, its sqlite is keyed `<game>_p<pass>`, and every _STATnn dict is an
aggregate counter). Nothing here is shared between games.

READ: m76_summary() -> actions, lossy_actions, images_attached, messages_with_images, skipped_no_loss, errors.
KILL: lossy_actions == 0 -> either no animations in this run, or all_frames is not populated.
"""
from __future__ import annotations

_STAT76 = {"actions": 0, "lossy_actions": 0, "images_attached": 0, "messages_with_images": 0,
           "skipped_no_loss": 0, "render_failures": 0, "errors": 0, "last_error": ""}

M76_MAX_IMAGES = 4


def m76_summary():
    return dict(_STAT76)


def _grid(frame):
    """A frame as a tuple-of-tuples of ints, or () if it is not one.

    `GameState.all_frames` returns taaf `Frame` objects: a dataclass whose ONLY field is `data`, a 2-D int8 ndarray
    with values 0-15 (taaf/game.py:73). It has no `.grid`. v16.0 v1 read `.grid`, got None for every real frame, and
    was inert for the whole run -- its tests passed only because the fixtures were dicts with a "grid" key, i.e. they
    encoded the same wrong assumption. This follows M66's extractor (which fired on 969 actions in v14.70): prefer
    `.data`, fall back to `.grid` / dict keys / the object itself, and convert via .tolist() so a numpy array is
    never truth-tested (`if not array` raises)."""
    data = getattr(frame, "data", None)
    if data is None:
        data = getattr(frame, "grid", None)
    if data is None and isinstance(frame, dict):
        data = frame.get("data", frame.get("grid"))
    if data is None:
        data = frame
    rows = data.tolist() if hasattr(data, "tolist") else data
    try:
        out = tuple(tuple(int(v) for v in row) for row in rows)
    except (TypeError, ValueError):
        return ()
    return out if out and all(len(r) for r in out) else ()


def _delta(a, b):
    """Number of cells that differ between two grids, tolerating ragged/absent rows."""
    n = 0
    for r in range(max(len(a), len(b))):
        ra = a[r] if r < len(a) else ()
        rb = b[r] if r < len(b) else ()
        for c in range(max(len(ra), len(rb))):
            if (ra[c] if c < len(ra) else None) != (rb[c] if c < len(rb) else None):
                n += 1
    return n


class _FrameShim:
    """frame_to_png_data_url only reads .grid."""

    __slots__ = ("grid",)

    def __init__(self, grid):
        self.grid = [list(r) for r in grid]


def select_frames(previous_grid, frames, max_images=M76_MAX_IMAGES):
    """Intermediate boards worth showing, oldest-first, or [] when the final frame already shows the effect.

    Returns [(frame_index, changed_cells, grid)]. A frame is a candidate when it differs from the pre-action
    board; it is only worth sending when it moved strictly more cells than the final board did, which is the
    definition of 'the final frame lost information'.
    """
    grids = [g for g in (_grid(f) for f in frames) if g]
    if len(grids) <= 1:
        return []
    final_delta = _delta(previous_grid, grids[-1])
    cands = []
    for i, g in enumerate(grids[:-1]):                 # the final frame is already shown by the harness
        d = _delta(previous_grid, g)
        if d > final_delta:
            cands.append((i, d, g))
    if not cands:
        return []
    # Whole-board flashes (>= 90% of cells changed) are death/reset or level-change animations. The harness
    # already reports those via game_over / level_completed, so rank them BELOW localised frames: they are kept
    # only when fewer than max_images localised frames exist. (Replay of v16.0 v1: ls20 had a 4,096-cell frame
    # -- the whole 64x64 board -- that would otherwise always win selection by raw delta.)
    n_cells = sum(len(r) for r in previous_grid) or max((sum(len(r) for r in c[2]) for c in cands), default=0)
    full = lambda d: n_cells > 0 and d >= 0.9 * n_cells
    keep = sorted(sorted(cands, key=lambda x: (full(x[1]), -x[1]))[:max_images], key=lambda x: x[0])
    # drop consecutive duplicates so several identical frames of a hold do not spend images
    out = []
    for item in keep:
        if out and item[2] == out[-1][2]:
            continue
        out.append(item)
    return out


def install76():
    """Wrap _execute_action (capture) and ToolAgent._build_user_message (attach). Idempotent by assertion."""
    import inference.framework.solver as S
    import inference.agent.tool_agent as T
    from inference.agent import vision_context as V

    sess = S._HarnessGameSession
    if getattr(sess._execute_action, "_m76_wrapped", False):
        raise RuntimeError("M76 installed twice")
    _orig_exec = sess._execute_action

    def _execute_action(self, action, **kw):
        try:
            previous_grid = _grid_from_session(S, self)
        except Exception:
            previous_grid = ()
        payload = _orig_exec(self, action, **kw)
        try:
            _STAT76["actions"] += 1
            frames = list(getattr(self.game.current_state, "all_frames", ()) or ())
            picked = select_frames(previous_grid, frames)
            if picked:
                _STAT76["lossy_actions"] += 1
                label = getattr(action, "id", None) or getattr(action, "name", None) or str(action)
                _stash(self, [(label, idx, g) for idx, _d, g in picked])
            elif len(frames) > 1:
                _STAT76["skipped_no_loss"] += 1
        except Exception as exc:
            _STAT76["errors"] += 1
            _STAT76["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
        return payload

    _execute_action._m76_wrapped = True
    sess._execute_action = _execute_action

    agent = T.ToolAgent if hasattr(T, "ToolAgent") else None
    if agent is None or not hasattr(agent, "_build_user_message"):
        raise RuntimeError("M76: ToolAgent._build_user_message not found")
    if getattr(agent._build_user_message, "_m76_wrapped", False):
        raise RuntimeError("M76 installed twice (user message)")
    _orig_msg = agent._build_user_message

    def _build_user_message(self, user_prompt, current_frame):
        msg = _orig_msg(self, user_prompt, current_frame)
        try:
            pend = getattr(self, "_m76_pending", None) or []
            self._m76_pending = []                      # show once, on the very next turn
            if not pend:
                return msg
            parts = []
            for label, idx, grid in pend:
                try:
                    url = V.frame_to_png_data_url(_FrameShim(grid))
                except Exception:
                    _STAT76["render_failures"] += 1
                    continue
                parts.append({"type": "text",
                              "text": f"Mid-action frame {idx} of {label} (the board during the animation, "
                                      f"before it settled into the current grid):"})
                parts.append({"type": "image_url", "image_url": {"url": url}})
            if not parts:
                return msg
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, list):
                return msg
            _STAT76["images_attached"] += sum(1 for p in parts if p.get("type") == "image_url")
            _STAT76["messages_with_images"] += 1
            return {**msg, "content": list(content) + parts}
        except Exception as exc:
            _STAT76["errors"] += 1
            _STAT76["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
            return msg

    _build_user_message._m76_wrapped = True
    agent._build_user_message = _build_user_message
    return True


def _grid_from_session(S, session):
    return tuple(tuple(int(v) for v in row) for row in S._grid_from_state(session.game.current_state))


def _stash(session, items):
    """Hand the frames to THIS game's agent; per-game by construction (each session owns its analyzer)."""
    target = getattr(session, "analyzer", None) or session
    prev = getattr(target, "_m76_pending", None) or []
    target._m76_pending = (prev + items)[-M76_MAX_IMAGES:]

install76()
print('V1600_M76 installed', __import__('json').dumps(m76_summary()), flush=True)
