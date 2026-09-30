# Cell 44 of arc-agi-3-duck-18-1gc-submit.ipynb (section: Patch cell - v16.7 M72 (one-pool UCB scheduler))
# ===== v14.72 = M72: ONE-POOL UCB SCHEDULER (club E1) =====
"""Replace the fixed per-game clock with one shared time pool, handed out in 5-minute slices by UCB.

WHY: every game used to get a fixed 7,920 s whether it was clearing levels or dead after ten minutes. Our local score
(25 games, all running at once) is ~2x our board score (110 games, 4 waves of 28): 12.04 local vs 5.02/5.50 board for
v14.64. The rank-3 team (board 11.49-11.64) runs exactly this scheduler on the same GPU and reports local ~= board.
SOURCE: github.com/Wang-Zhongwei/duck-harness main, inference/framework/allocation.py (ucb_score, Progress) and
solver.py `_run_allocated_games` (policy "ucb"), config ucb_c=0.0002, quantum 300 s. Ported, not re-invented.
Their own note: no score gain was measured for the scheduler in isolation; it is what their notebook runs.

HOW (same concurrency as before -- 28 slices at once -- and the same total budget):
  budget   = min(ceil(games / concurrency) * max_runtime_s_per_game, soft time left)   (local 25 games: 7,920 s)
  warm-up  : every game plays one 300 s slice, in order, 28 at a time.
  extension: until the budget ends, each free slot goes to the playable game with the highest
             UCB = priority + c * sqrt(log(1 + minutes elapsed) / actions taken),
             priority = 0.5 * (levels-per-token over the game's life + over its last 32k generated tokens),
             levels weighted like the score (triangular: L*(L+1)/2).
  A slice ends at the first request boundary after its end: the slice end is part of should_stop, so the agent's own
  yield (tool_agent.py control_yield_reason, checked before each request and after each tool result) stops the turn
  with its history kept and the same analysis step resumes next slice. v16.7 change after review 2026-09-25: with
  unlimited tool calls a no-action turn otherwise held its slot until the shared deadline (simulated 300 s slice ->
  1,200 s, 20 requests, 0 actions). The rank-6 fork switches only between whole turns; they are not bounded here.
  Each request's timeout is re-capped to the session's remaining time (they pass a callable; our analyze() took one
  value per turn, so every request of a long turn used 900 s even with 60 s left).
  A game keeps its agent, history, sandbox variables and runtime files between slices. Won / game-over /
  out-of-actions games are never resumed.
  At the end every game is finished exactly once.
M43 (permit scheduler) is switched OFF: its permits are held per game until the game ends, so paused games would
hold all 28 and block the rest.
Model-visible change: `time_remaining_seconds` in action results now counts down the shared budget.
READ: m72_summary() -> warmup_slices, extension_slices, games_resumed, per-game slices/levels/priority, errors.
"""
from __future__ import annotations

import asyncio as _aio72
import functools as _ft72
import json as _js72
import math as _m72
import time as _t72

_M72_QUANTUM_S = 2400.0  # v18.1n: was 300 s; resume re-prefill costs ~10% of the GPU on 110 games
_M72_UCB_C = 0.0002
_M72_RECENT_TOKENS = 32_000
_STAT72 = {"games": 0, "concurrency": 0, "budget_s": 0.0, "warmup_slices": 0, "extension_slices": 0,
           "games_resumed": 0, "finalized": 0, "m43_unwrapped": False, "errors": 0, "last_error": "", "per_game": {},
           "slice_yields": 0, "timeouts_capped": 0, "timeout_errors": 0}


def m72_ucb_score(priority: float, actions: int, elapsed_minutes: float, c: float) -> float:
    """Rank-3 allocation.ucb_score: unvisited games first, then priority + exploration bonus."""
    if actions == 0:
        return _m72.inf
    return priority + c * _m72.sqrt(_m72.log1p(max(0.0, elapsed_minutes)) / actions)


class M72Progress:
    """Rank-3 allocation.Progress: (generated tokens, weighted levels) samples -> lifetime and recent rates."""

    def __init__(self):
        self.samples = []

    def observe(self, tokens: int, weighted_levels: float) -> None:
        s = (max(0, int(tokens)), float(weighted_levels))
        if self.samples and s[0] == self.samples[-1][0]:
            self.samples[-1] = s
        else:
            self.samples.append(s)

    def priority(self) -> float:
        if not self.samples:
            return 0.0
        tokens, weighted = self.samples[-1]
        past = self.samples[0][1]
        cutoff = tokens - _M72_RECENT_TOKENS
        for st, sw in self.samples:
            if st > cutoff:
                break
            past = sw
        lifetime = weighted / tokens if tokens else 0.0
        span = min(tokens, _M72_RECENT_TOKENS)
        recent = max(0.0, weighted - past) / span if span > 0 else 0.0
        return 0.5 * (lifetime + recent)


def install72():
    import inference.framework.solver as S

    sess = S._HarnessGameSession
    HS = S.HarnessSolver
    if getattr(HS._run_games, "_m72_wrapped", False):
        raise RuntimeError("M72 installed twice")

    # --- M43 off: restore the unwrapped should_stop (M43 keeps it in its wrapper's closure)
    cur = sess.should_stop
    if getattr(cur, "_m43_wrapped", False):
        inner = [c.cell_contents for c in (cur.__closure__ or ()) if callable(getattr(c, "cell_contents", None))
                 and getattr(c.cell_contents, "__name__", "") == "should_stop"]
        if len(inner) != 1:
            raise RuntimeError(f"M72: cannot find M43's inner should_stop ({len(inner)} candidates)")
        sess.should_stop = inner[0]
        _STAT72["m43_unwrapped"] = True

    # --- one shared deadline instead of the per-game clock
    orig_limit = sess.runtime_limit_reached
    orig_timing = sess.timing_payload

    def runtime_limit_reached(self):
        dl = getattr(self.solver, "_m72_deadline", None)
        if dl is None:
            return orig_limit(self)
        return _t72.monotonic() >= dl

    def timing_payload(self):
        dl = getattr(self.solver, "_m72_deadline", None)
        if dl is None:
            return orig_timing(self)
        now = _t72.monotonic()
        return {"run_elapsed_seconds": max(0.0, now - self.started_at), "time_remaining_seconds": max(0.0, dl - now)}

    sess.runtime_limit_reached = runtime_limit_reached
    sess.timing_payload = timing_payload

    # --- every request's timeout re-capped to the session's remaining time (analyze() passes one value per turn)
    import inference.agent.tool_agent as TA
    orig_cc = TA.ToolAgent._chat_completion
    if getattr(orig_cc, "_m72_wrapped", False):
        raise RuntimeError("M72 request wrapper installed twice")

    def _chat_completion(self, messages, **kw):
        s = getattr(self, "_m72_session", None)
        if s is not None:
            try:
                t = s.request_timeout_seconds()
                cur = kw.get("request_timeout_seconds")
                if t is not None and (cur is None or float(t) < float(cur)):
                    kw["request_timeout_seconds"] = float(t)
                    _STAT72["timeouts_capped"] += 1
            except Exception as exc:
                _STAT72["timeout_errors"] += 1
                _STAT72["last_error"] = f"timeout: {type(exc).__name__}: {exc}"[:200]
        return orig_cc(self, messages, **kw)

    for f in dir(orig_cc):
        if f.endswith("_wrapped"):
            try:
                setattr(_chat_completion, f, getattr(orig_cc, f))
            except Exception:
                pass
    _chat_completion._m72_wrapped = True
    TA.ToolAgent._chat_completion = _chat_completion

    # --- a resumable slice: the body of the stock play() loop, stopping between turns at the slice end
    def _m72_record(self):
        run = self.game.game_run
        state = getattr(self.game, "current_state", None)
        lv = max(0, int(getattr(state, "levels_completed", 0) or 0)) if state is not None else 0
        toks = max(0, S._analyzer_reported_tokens(self.analyzer) - self._m72_token_origin)
        self._m72_progress.observe(toks, lv * (lv + 1) / 2.0)
        self._m72_levels = lv

    def play_slice(self, quantum_s):
        run = self.game.game_run
        assert run is not None
        if not getattr(self, "_m72_init", False):
            run.solver_analysis_html = self.analysis_html_relpath
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.touch(exist_ok=True)
            self.token_baseline = S._analyzer_reported_tokens(self.analyzer)
            self._m72_token_origin = self.token_baseline
            self._m72_progress = M72Progress()
            self._m72_retry = None
            self.seed_initial_history()
            self.write_runtime_state()
            self._append_initial_viewer_event()
            self.write_viewer_payload()
            self._m72_init = True
            _m72_record(self)
        end = _t72.monotonic() + float(quantum_s)
        self.analyzer._m72_session = self                  # lets the request wrapper re-cap each timeout

        def slice_should_stop():
            return self.should_stop() or _t72.monotonic() >= end
        try:
            while not self.should_stop():
                if _t72.monotonic() >= end:
                    break                                   # slice over; resume later from the same state
                if S._is_engine_game_over(self.game) and self.last_engine_action != "RESET":
                    self._execute_auto_reset()
                    continue
                if self._m72_retry is None:
                    self.analysis_step += 1
                    step = self.analysis_step
                else:
                    step = self._m72_retry
                self.write_runtime_state()
                before = self._read_transcript_bytes()
                try:
                    result = self.analyzer.analyze(
                        self.state_path, self.action_count,
                        valid_actions=S._engine_action_names(self.game), step_env=self.step_env,
                        transcript_path=self.transcript_path, analysis_step=step,
                        request_timeout_seconds=self.request_timeout_seconds(), should_stop=slice_should_stop)
                finally:
                    delta = self._transcript_delta_since(before)
                    if delta.strip():
                        self._append_analysis_viewer_event(step, delta)
                        self.write_viewer_payload()
                    _m72_record(self)
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    self._m72_retry = step
                    if self.should_stop():
                        break
                    _t72.sleep(S.ANALYZER_RETRY_BACKOFF_SECONDS)
                    continue
                self._m72_retry = None
                if getattr(result, "yielded_control", False):
                    self._m72_retry = step
                    if _t72.monotonic() >= end:
                        _STAT72["slice_yields"] += 1
                    continue
        except Exception as exc:
            if run.final_score is None:
                run.solver_note = f"error: {type(exc).__name__}: {exc}"
                if run.state == "playing":
                    run.state = "crashed"
            self._m72_dead = True

    def playable(self):
        run = self.game.game_run
        if getattr(self, "_m72_dead", False) or run is None or run.final_score is not None:
            return False
        if run.state != "playing" or S._is_run_complete(self.game):
            return False
        mx = self.solver.max_actions_per_game
        return mx is None or self.action_count < mx

    def finalize(self):
        run = self.game.game_run
        if getattr(self, "_m72_final", False):
            return
        self._m72_final = True
        try:
            if run is not None and run.solver_note is None:
                run.solver_note = f"tokens={S._analyzer_reported_tokens(self.analyzer)}"
            self._finish_if_needed()
        finally:
            self.state_path.unlink(missing_ok=True)
            try:
                self._write_analysis_html()
                self.write_viewer_payload()
            except Exception:
                pass
            _STAT72["finalized"] += 1

    sess._m72_play_slice = play_slice
    sess._m72_playable = playable
    sess._m72_finalize = finalize

    def _m72_session(self, game, index, pass_index):
        """_play_one's construction, without playing."""
        run = game.game_run
        run_stem = self._run_stem(run.game_id, pass_index)
        return S._HarnessGameSession(
            solver=self, game=game, analyzer=self._make_analyzer(game, index, self._local_server_for_game_index(index)),
            game_index=index, pass_index=pass_index,
            state_path=self._artifacts_dir() / f"{run_stem}_{S.RUNTIME_STATE_FILENAME}",
            transcript_path=self._transcripts_dir() / f"{run_stem}.txt",
            analysis_html_relpath=f"solver_analysis/{run_stem}.html",
            stop_event=self._stop_event,
            viewer_data_path=self._artifacts_dir() / f"{run_stem}_viewer_data.json")

    def _event(self, name, **kw):
        try:
            if self.job_dir is not None:
                with open(self.job_dir / "m72_allocation_events.jsonl", "a") as fh:
                    fh.write(_js72.dumps({"event": name, "t": round(_t72.monotonic() - self._m72_t0, 1), **kw}) + "\n")
        except Exception:
            pass

    async def _run_games(self, games):
        self._stop_event.clear()
        conc = max(1, int(self.concurrency))
        per_game = float(self.max_runtime_s_per_game or 7920.0)
        budget = _m72.ceil(len(games) / conc) * per_game
        soft = self.soft_time_remaining_seconds()
        if soft is not None:
            budget = min(budget, soft)
        self._m72_t0 = _t72.monotonic()
        self._m72_deadline = self._m72_t0 + budget
        _STAT72.update(games=len(games), concurrency=conc, budget_s=round(budget, 1))
        loop = _aio72.get_running_loop()
        pool = self._worker_pool
        sessions, running, service, slices = {}, {}, {}, {}
        tasks = set()

        async def execute(fn):
            if pool is not None:
                return await loop.run_in_executor(pool, fn)
            return await _aio72.to_thread(fn)

        def start(i, s, kind):
            q = min(_M72_QUANTUM_S, max(0.0, self._m72_deadline - _t72.monotonic()))
            slices[i] = slices.get(i, 0) + 1
            if slices[i] > 1:
                _STAT72["games_resumed"] += 1 if slices[i] == 2 else 0
            _STAT72["warmup_slices" if kind == "warmup" else "extension_slices"] += 1

            def run_slice():
                t = _t72.monotonic()
                s._m72_play_slice(q)
                return _t72.monotonic() - t

            async def go():
                try:
                    dt = await execute(run_slice)
                    service[i] = service.get(i, 0.0) + dt
                except Exception as exc:
                    _STAT72["errors"] += 1
                    _STAT72["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
                    s._m72_dead = True
                finally:
                    running.pop(i, None)
                _event(self, "slice_done", game=i, kind=kind, service_s=round(service.get(i, 0.0), 1),
                       levels=getattr(s, "_m72_levels", 0), actions=s.action_count,
                       priority=round(s._m72_progress.priority(), 8) if hasattr(s, "_m72_progress") else 0)
            running[i] = kind
            tasks.add(_aio72.create_task(go()))

        async def wait_one():
            remaining = max(0.0, self._m72_deadline - _t72.monotonic())
            done, _ = await _aio72.wait(tasks, timeout=remaining + 60.0, return_when=_aio72.FIRST_COMPLETED)
            for t in done:
                tasks.discard(t)
            return done

        try:
            _event(self, "start", games=len(games), concurrency=conc, budget_s=budget, quantum_s=_M72_QUANTUM_S,
                   ucb_c=_M72_UCB_C)
            passes = {}
            order = []
            for index, game in enumerate(games):
                gid = game.game_run.game_id if game.game_run is not None else str(index)
                p = passes.get(gid, 0)
                passes[gid] = p + 1
                order.append((index, p, game))
            # warm-up: one slice per game, in order
            pending = iter(order)
            exhausted = False
            while _t72.monotonic() < self._m72_deadline and not self._stop_event.is_set():
                while len(tasks) < conc and not exhausted:
                    nxt = next(pending, None)
                    if nxt is None:
                        exhausted = True
                        break
                    index, p, game = nxt
                    try:
                        sessions[index] = _m72_session(self, game, index, p)
                    except Exception as exc:
                        _STAT72["errors"] += 1
                        _STAT72["last_error"] = f"session: {type(exc).__name__}: {exc}"[:200]
                        self._finish_after_error(game, exc)
                        continue
                    start(index, sessions[index], "warmup")
                if not tasks:
                    break
                await wait_one()
            _event(self, "warmup_done", slices=_STAT72["warmup_slices"])
            # extensions: highest UCB among playable, not-running games
            while _t72.monotonic() < self._m72_deadline and not self._stop_event.is_set():
                minutes = (_t72.monotonic() - self._m72_t0) / 60.0
                cands = {i: s for i, s in sessions.items() if i not in running and s._m72_playable()}
                while cands and len(tasks) < conc and _t72.monotonic() < self._m72_deadline - 1.0:
                    scores = {i: m72_ucb_score(s._m72_progress.priority() if hasattr(s, "_m72_progress") else 0.0,
                                               s.action_count, minutes, _M72_UCB_C) for i, s in cands.items()}
                    pick = min(cands, key=lambda i: (-scores[i], service.get(i, 0.0), i))
                    start(pick, cands.pop(pick), "ucb")
                if not tasks:
                    break
                await wait_one()
        except _aio72.CancelledError:
            self._stop_event.set()
            raise
        finally:
            if tasks:
                # Past the deadline every slice stops by itself after its current turn (should_stop sees the shared
                # deadline), so wait WITHOUT the stop flag: games then end "gave_up" exactly like the stock per-game
                # clock. The dry run's first version set the flag here and every game ended "cancelled".
                done, still = await _aio72.wait(list(tasks), timeout=max(1.0, float(self.cancel_drain_timeout_s)))
                if still:
                    self._stop_event.set()
                    await self._drain_game_tasks(list(still))
            for i, s in sessions.items():
                try:
                    s._m72_finalize()
                except Exception as exc:
                    _STAT72["errors"] += 1
                    _STAT72["last_error"] = f"finalize: {type(exc).__name__}: {exc}"[:200]
                    self._finish_after_error(s.game, exc)
                _STAT72["per_game"][str(i)] = {"slices": slices.get(i, 0), "service_s": round(service.get(i, 0.0)),
                                               "levels": getattr(s, "_m72_levels", 0), "actions": s.action_count}
            self._finish_remaining(games)
            _event(self, "finish", warmup=_STAT72["warmup_slices"], extension=_STAT72["extension_slices"])
            self._m72_deadline = None

    _run_games._m72_wrapped = True
    HS._run_games = _run_games
    return True


def m72_summary():
    st = {k: v for k, v in _STAT72.items() if k != "per_game"}
    pg = _STAT72["per_game"]
    if pg:
        sl = sorted(v["slices"] for v in pg.values())
        st["slices_min_median_max"] = [sl[0], sl[len(sl) // 2], sl[-1]]
        st["top_service_games"] = sorted(((v["service_s"], k, v["levels"]) for k, v in pg.items()), reverse=True)[:8]
    return st

install72()
print('V1670_M72 installed', __import__('json').dumps(m72_summary()), flush=True)
