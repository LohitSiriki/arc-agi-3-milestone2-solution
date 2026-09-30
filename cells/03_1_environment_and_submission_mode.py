# Cell 3 of arc-agi-3-duck-18-1gc-submit.ipynb (section: 1. Environment and submission mode)
import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

# True only inside a real competition rerun; switches diagnostics + soft deadline.
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

# Non-interactive matplotlib backend: diagnostics render plots with no display attached.
os.environ["MPLBACKEND"] = "Agg"
# Marks the run as a (real or emulated) submission so the framework + solver can adjust.
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
# Skip periodic JSON/HTML diagnostics and per-frame logging for every run.
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"

# Apply the measured vLLM winner before any serving setup command runs.
# KV RAISED 5.0 -> 6.5 GiB. MEASURED at 5 GiB: 105,202 KV tokens =
# 'Maximum concurrency 3.21x' at 32,768 tok/request, and the GPU served
# 3.11 concurrent requests -- decode-bound at 3 lanes, 243 tok/s aggregate.
# 6.5 GiB should give ~136,800 tokens = ~4.17 lanes = ~+30% actions.
# vLLM SKIPS memory profiling when kv_cache_memory_bytes is set, so real
# headroom has never been measured -- that is what this MINI is for.
# ---- MEASURED ON THE v14.42 MINI (2026-09-10), read this before changing any of it ----
# KV 6.5 GiB OOM'd 6 min into serving. The full accounting at the moment of death:
#   weights 81.80 + KV 6.50 + cudagraphs 0.36 + activations ~3.46 + PyTorch
#   reserved-but-unallocated 1.00 + cuda ctx ~0.87 + PleOffloadWorker(pid 301) 0.71
#   = 94.70 of 94.97 GiB -> 278 MB free -> died on a 308 MiB conv activation.
# NOTE vLLM reserves the KV pool against "Initial free memory 94.43 GiB" READ BEFORE the
# 81.8 GiB of weights load, and `skipped memory profiling` because the cap is manual --
# so the cap is applied BLIND and would accept any value. 5.0 GiB was never validated,
# it merely fit. 6.0 GiB leaves ~790 MB, ~2.5x the observed 308 MiB spike.
# KV measured: 5.0 GiB -> 105,202 tokens / 3.21x (3.11 running); 6.5 -> 136,245 / 4.16x
# (4.83 running). Tokens scale linearly with bytes, so 6.0 should give ~125,800 / 3.84x.
#
# TAAF_VLLM_MAX_NUM_BATCHED_TOKENS IS DELIBERATELY ABSENT. serving_setup.py
# resolve_vllm_tuning() only reads it as `requested`, and its default is ALSO 8192, so
# the initial launch is byte-identical. But the WATCHDOG RESTART path passes
# enable_chunked_prefill=False, and there an OVERRIDDEN value below ANALYZER_CONTEXT
# (32768) raises "must be at least 32768 when chunked prefill is disabled" -- which is
# why the MINI's two restart attempts BOTH failed and the run spent 30 of its 45 minutes
# with no model server. Unset, the restart falls through to 32768 and succeeds.
# PREFIX CACHING IS ON IN THIS BUILD (requested). What we already measured about it,
# so the result is read correctly rather than re-litigated:
#   * KV pool is IDENTICAL on/off -- 342,400 tokens either way on the 27B probe, and the
#     1600-token attention block size does not change. So it costs no KV.
#   * measured hit rate was 12.3% median / 47.6% max / 0.0% min over 790 samples.
#   * vLLM issue #45238: for hybrid GDN models, align mode keeps only the mamba
#     checkpoint at the last block boundary before the prompt ends; if that lands in
#     request-unique tokens ALL reuse silently drops to 0%. Our arithmetic -- checkpoint
#     at 17,600 vs a 3,761-token stable prefix -- puts us in that regime.
#   * BUT the M24 ledger now pins a long stable prefix, which is exactly the condition
#     the note said was needed to make align-mode caching work. That is what this run
#     tests. Read `prefix cache hit rate` out of the vLLM log to judge it.
PUBLIC25_VLLM_PROFILE_NAME = 'kv10.0-nvfp4mtp-mtp1-c8-cg32'
PUBLIC25_VLLM_PROFILE_ENV = {
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "10737418240",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_MAX_NUM_SEQS": "8",
    "TAAF_VLLM_MTP_TOKENS": "1",
    "TAAF_VLLM_OMP_THREADS": "1"
}
for key, value in PUBLIC25_VLLM_PROFILE_ENV.items():
    os.environ[key] = value
print(
    f'PUBLIC25_VLLM_PROFILE name={PUBLIC25_VLLM_PROFILE_NAME} '
    f'env={json.dumps(PUBLIC25_VLLM_PROFILE_ENV, sort_keys=True)}',
    flush=True,
)
# Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level).
os.environ["ONLY_RESET_LEVELS"] = "true"

# (expandable_segments removed: serving_setup.py forbids it for the PLE CPU-offload IPC -- fatal on 2026-09-12 run)

# Prepend the CUDA toolkit to the linker path (it is off it on Kaggle GPU images) so the
# solver's GPU libraries (e.g. vllm / torch) can link against libcuda.
cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    entry for entry in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if entry
)

# Everything the run produces is written here.
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")
# ---- v14.57 HARD CAP (local runs only): a stock local run ends ~2.4 h; kill at 3 h. Never armed in a real rerun. ----
if not TRUE_SUBMISSION:
    import threading as _cap_t
    def _cap_kill():
        print("V1462_HARD_CAP_REACHED 10800s -> os._exit", flush=True)
        os._exit(3)
    _cap_timer = _cap_t.Timer(10800.0 - (time.time() - NOTEBOOK_START_EPOCH), _cap_kill)
    _cap_timer.daemon = True
    _cap_timer.start()
    print("V1462_HARD_CAP armed 10800s", flush=True)
