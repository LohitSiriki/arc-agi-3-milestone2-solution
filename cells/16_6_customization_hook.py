# Cell 16 of arc-agi-3-duck-18-1gc-submit.ipynb (section: 6. Customization hook)
# TIMING IS CALIBRATED -- DO NOT RAISE THESE TO "USE THE WHOLE SESSION".
# Competition gives each of 110 games 31,800 s x 3.11 lanes / 110 = 899 GPU-seconds.
# The stock 7,920 s per-game cap gives each of 25 local games 985 -- within 10%. A local
# run that ends at ~2.2 h is therefore PARITY, not waste. Raising it to 31,000 s would
# give each game 3,856 GPU-s = 4.3x competition and inflate every local score.
# concurrency 28 (> 25) means all 25 local games get threads with no batching, and at
# 110 games it batches 4 x 28 = 112 inside 4 x 7,920 = 31,680 s.
bm.solver.max_runtime_s_per_game = 7920.0
bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = 28
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = False
if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
    raise RuntimeError(f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.')
print(f'V1458_SGLANG=1 V1444_SETTINGS budget_s={bm.solver.max_runtime_s_per_game} conc={bm.solver.concurrency} permits=28 grant=20actions kv=10GiB mtp=1 head=nvfp4', flush=True)
