# Cell 11 of arc-agi-3-duck-18-1gc-submit.ipynb (section: 4. Import the bundled source and boot SGLang)
# ---------------- 2. launch Pennyroyal SGLang on our RadixArk NVFP4 checkpoint ----------------
SITE = Path("/tmp/sgl/sgl-site")
PORT = 1234   # the harness's LOCAL_ANALYZER_BASE_URL port
BASE = f"http://127.0.0.1:{PORT}"
SERVED = "Qwen/Qwen3.8-Flash-Next-NVFP4"
BOOT = Path("/tmp/sgl/boot")
BOOT.mkdir(parents=True, exist_ok=True)
# PYTHONPATH alone skips .pth files; sitecustomize adds the runtime as a site dir AHEAD of Kaggle's own torch.
(BOOT / "sitecustomize.py").write_text(
    "import site, sys\n"
    f"_s = {str(SITE)!r}\n"
    "site.addsitedir(_s)\n"
    "_new = [p for p in sys.path if p.startswith(_s)]\n"
    "sys.path[:] = _new + [p for p in sys.path if not p.startswith(_s)]\n"
)
# The borrowed toolkit (vLLM image layers) has no lib64/ and its libcudart.so target was whited out.
# JIT kernels link with -L$CUDA_HOME/lib64 -lcudart, so repair both before launch.
def _repair_cuda_libs():
    info = {}
    tgt = CUDA_HOME / "targets/x86_64-linux/lib"
    lib64 = CUDA_HOME / "lib64"
    if not lib64.exists() and tgt.is_dir():
        lib64.symlink_to(tgt); info["lib64"] = f"symlink -> {tgt}"
    cudart = lib64 / "libcudart.so"
    if not cudart.exists():                      # missing or dangling symlink
        cands = sorted(glob.glob(str(SITE / "nvidia/cu13/lib/libcudart.so*"))) + sorted(glob.glob(str(SITE / "nvidia/*/lib/libcudart.so*")))
        if not cands:
            raise RuntimeError("no libcudart.so* in the SGLang runtime to link against")
        if cudart.is_symlink(): cudart.unlink()
        cudart.symlink_to(cands[0]); info["libcudart"] = f"symlink -> {cands[0]}"
    info["libcudart_resolves"] = str(cudart.resolve()) if cudart.exists() else None
    if not cudart.exists():
        raise RuntimeError("libcudart.so still does not resolve")
    return info
R["cuda_lib_repair"] = _repair_cuda_libs()
log("CUDA_LIB_REPAIR", R["cuda_lib_repair"])
save()
CACHE = Path("/tmp/sglcache")
for d in ("huggingface", "torch", "torchinductor", "triton", "cuda", "flashinfer", "sglang/jit", "tilelang"):
    (CACHE / d).mkdir(parents=True, exist_ok=True)

# v16.9 M87: compiled-kernel caches. The W4A16 boot JIT-compiles Triton/TileLang kernels (target-verify graph capture
# 142.6 s vs 6.0 s on NVFP4) into an empty /tmp/sglcache every session. _m87_restore() unpacks an earlier local run's
# sglcache_w4a16.tar from any attached input before the server starts (none attached -> no-op); _m87_save() writes that
# archive after a local run (called in the play cell's finally, after the server stopped).
M87 = {"restored_from": None, "files": 0, "seconds": 0.0, "error": ""}
M87_NAME = "sglcache_w4a16.tar"

def _m87_rel_roots(home):
    h = home.strip("/")
    return ("tmp/sglcache", h + "/.tilelang", h + "/.triton")

def _m87_restore(inputs="/kaggle/input", root="/", home=None):
    import glob as _g87, tarfile as _tar87
    t0 = time.time()
    cands = []
    for depth in range(1, 5):
        cands += _g87.glob(inputs.rstrip("/") + "/*" * depth + "/" + M87_NAME)
    if not cands:
        return None
    src_tar = sorted(cands)[0]
    allowed = tuple(r + "/" for r in _m87_rel_roots(home or os.path.expanduser("~")))
    with _tar87.open(src_tar) as tf:
        members = [m for m in tf.getmembers() if (m.isfile() or m.isdir()) and not m.name.startswith("/")
                   and ".." not in m.name.split("/") and m.name.startswith(allowed)]
        try:
            tf.extractall(root, members=members, filter="tar")
        except TypeError:
            tf.extractall(root, members=members)
    M87.update(restored_from=src_tar, files=sum(1 for m in members if m.isfile()), seconds=round(time.time() - t0, 1))
    return src_tar

def _m87_save(out_dir, root="/", home=None, max_bytes=3 * 1024**3):
    import tarfile as _tar87
    t0 = time.time()
    out = Path(out_dir) / M87_NAME
    roots = [Path(root) / r for r in _m87_rel_roots(home or os.path.expanduser("~"))]
    files = [f for r in roots if r.exists() for f in sorted(r.rglob("*"))
             if f.is_file() and not f.is_symlink() and "huggingface" not in f.parts]
    nbytes = sum(f.stat().st_size for f in files)
    if files and nbytes < max_bytes:
        with _tar87.open(out, "w") as tf:
            for f in files:
                tf.add(str(f), arcname=str(f.relative_to(root)), recursive=False)
    return {"files": len(files), "bytes": nbytes, "written": out.exists(), "seconds": round(time.time() - t0, 1)}

try:
    _m87_restore()
except Exception as exc:
    M87["error"] = f"{type(exc).__name__}: {exc}"[:200]
log("V1690_M87_RESTORE", M87)

def server_env():
    env = os.environ.copy()
    libs = [str(CUDA_HOME / "targets/x86_64-linux/lib"), str(CUDA_HOME / "lib64"), str(SITE / "torch/lib")]
    libs += sorted(glob.glob(str(SITE / "nvidia/*/lib")))
    env.update({
        "PYTHONPATH": f"{BOOT}:{SITE}", "PYTHONNOUSERSITE": "1",
        "PATH": f"{CUDA_HOME}/bin:{SITE}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": ":".join(libs + [env.get("LD_LIBRARY_PATH", "")]), "LIBRARY_PATH": ":".join(libs + [env.get("LIBRARY_PATH", "")]),
        "CUDA_HOME": str(CUDA_HOME), "CUDACXX": str(CUDA_HOME / "bin/nvcc"),
        "CC": "gcc", "CXX": "g++", "CUDAHOSTCXX": "g++", "TORCH_CUDA_ARCH_LIST": "12.0",
        "MAX_JOBS": "4", "FLASHINFER_NINJA_JOBS": "4", "FLASHINFER_NVCC_THREADS": "1",
        "HF_HOME": str(CACHE / "huggingface"), "XDG_CACHE_HOME": str(CACHE), "TORCH_HOME": str(CACHE / "torch"),
        "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "torchinductor"), "TRITON_CACHE_DIR": str(CACHE / "triton"),
        "CUDA_CACHE_PATH": str(CACHE / "cuda"), "FLASHINFER_WORKSPACE_BASE": str(CACHE / "flashinfer"),
        "SGLANG_CACHE_DIR": str(CACHE / "sglang"), "SGLANG_JIT_CACHE_DIR": str(CACHE / "sglang/jit"),
        "TILELANG_CACHE_DIR": str(CACHE / "tilelang"),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SGLANG_NUMA_BIND_V2": "false",
        "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_MM_PREPROCESS_DEVICE": "cpu", "NUMPY_MADVISE_HUGEPAGE": "0",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
    })
    return env

# v14.64: partial KV reclaim. Pennyroyal subtracts the per-draft SSM snapshot scratch (stage_per_req x (reqs+1) x
# draft_tokens = 2.9 GB at 12 reqs) from the KV budget although memory_pool.py never allocates it under
# --gdn-mtp-cache-mode none. v14.64 v1 removed the reservation entirely and ran 15 requests: the server died ~10-15 min
# into play (no server log survived; most likely OOM in post-sizing allocations). v2 keeps HALF of it as headroom
# (~1.5 GB) and runs 13 requests. Both anchors must match exactly once or the boot raises.
_KVC = SITE / "sglang/srt/mem_cache/kv_cache_configurator.py"
_kvc_src = _KVC.read_text().replace("\r\n", "\n")
_KVC_ANCHOR = ("                intermediate_size = (\n"
               "                    stage_per_req\n"
               "                    * (capped_reqs + 1)\n"
               "                    * get_spec().speculative_num_draft_tokens\n"
               "                )")
_KVC_NEW = ("                intermediate_size = (\n"
            "                    stage_per_req\n"
            "                    * (capped_reqs + 1)\n"
            "                    * (get_spec().speculative_num_draft_tokens if get_exec().mamba.gdn_mtp_cache_mode != \"none\" else 0)  # v14.64 v3: reservation removed (never allocated in none-mode)\n"
            "                )")
if _kvc_src.count(_KVC_ANCHOR) != 1:
    raise RuntimeError(f"V1464 GUARD: kv_cache_configurator anchor found {_kvc_src.count(_KVC_ANCHOR)} times (expected 1); runtime differs from the bundle tree")
if "get_exec,\n" not in _kvc_src and "get_exec\n" not in _kvc_src:
    raise RuntimeError("V1464 GUARD: kv_cache_configurator does not import get_exec")
_KVC.write_text(_kvc_src.replace(_KVC_ANCHOR, _KVC_NEW))
compile(_KVC.read_text(), str(_KVC), "exec")
for _pyc in (_KVC.parent / "__pycache__").glob("kv_cache_configurator*.pyc"):
    _pyc.unlink()
print("V1464_KVC_PATCHED", _KVC, flush=True)
# Pennyroyal serve-flash-next.sh shape, adapted: our 32K context (no YaRN override), our checkpoint chat
# template, 16 running requests (production vLLM allows 8 but KV held it near 4), no NIXL persistence, same native NEXTN depth 3 as our vLLM.
# ===== M108: draft-only checkpoint folder for the NVFP4 MTP draft (boot -5 min) =====
# The draft loader keeps only tensors whose name contains "mtp" (qwen3_5_mtp.py load_weights: `if "mtp" not in name:
# continue`; embeddings and LM head come from the main model), yet it opened all 206 shards of the RadixArk checkpoint to
# find them: 365-459 s in v18.14/v18.15 vs 135-153 s for the BF16 draft. The 31 mtp.* tensors live in 3 files
# (model-bf16-00010/11/12, 14.7 GB of ~135 GB), so the draft reads from a folder holding only those 3 files, the same
# config/quant files, and an index listing only the mtp.* tensors. Same tensors, same config -> same draft.
# Any surprise (different tensor count, missing file) -> falls back to the full RadixArk folder (old speed, same draft).
import json as _j108, shutil as _sh108
from pathlib import Path as _P108
DRAFT = NV
try:
    _d108 = _P108("/tmp/mtpdraft")
    _sh108.rmtree(_d108, ignore_errors=True)
    _d108.mkdir(parents=True)
    _nvi108 = _j108.loads((NV / "model.safetensors.index.json").read_text())
    _mtp108 = {k: f for k, f in _nvi108["weight_map"].items() if "mtp" in k}
    _files108 = sorted(set(_mtp108.values()))
    if len(_mtp108) != 31 or not all(k.startswith("mtp.") for k in _mtp108) or len(_files108) > 4:
        raise RuntimeError(f"unexpected mtp tensors: {len(_mtp108)} in {_files108}")
    for _f108 in NV.iterdir():
        if _f108.is_file() and not _f108.name.endswith(".safetensors") and _f108.name != "model.safetensors.index.json":
            (_d108 / _f108.name).symlink_to(_f108)
    for _n108 in _files108:
        if not (NV / _n108).is_file():
            raise RuntimeError(f"missing {_n108}")
        (_d108 / _n108).symlink_to(NV / _n108)
    (_d108 / "model.safetensors.index.json").write_text(_j108.dumps({**_nvi108, "weight_map": _mtp108}))
    DRAFT = _d108
    print("M108_DRAFT_DIR", _j108.dumps({"dir": str(DRAFT), "tensors": len(_mtp108), "files": _files108,
                                          "gb": round(sum((NV / n).stat().st_size for n in _files108) / 1e9, 2)}), flush=True)
except Exception as _e108:
    DRAFT = NV
    print("M108_FALLBACK full checkpoint folder:", type(_e108).__name__, str(_e108)[:200], flush=True)
BASE_ARGS = [
    "--model-path", str(MODEL), "--load-format", "safetensors", "--served-model-name", SERVED,
    "--host", "127.0.0.1", "--port", str(PORT), "--tp", "1",
    "--dtype", "bfloat16", "--quantization", "auto-round", "--kv-cache-dtype", "fp8_e4m3",
    "--mem-fraction-static", "0.97", "--context-length", "69632", "--page-size", "64",
    "--max-running-requests", "16", "--chunked-prefill-size", "4096",
    "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-ssm-dtype", "bfloat16",
    "--max-mamba-cache-size", "80", "--gdn-mtp-cache-mode", "none",
    "--linear-attn-decode-backend", "flashinfer", "--linear-attn-prefill-backend", "flashinfer",
    "--mamba-track-interval", "64", "--ple-offload-embedding", "--trust-remote-code",
    "--chat-template", str(MODEL / "chat_template.jinja"), "--image-processor-backend", "pil",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
    "--enable-metrics", "--enable-cache-report", "--enable-request-time-stats-logging",
    "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "4", "--speculative-draft-model-path", str(DRAFT), "--speculative-draft-model-quantization", "modelopt_fp4", "--speculative-moe-runner-backend", "flashinfer_cutlass",  # v18.15: NVFP4 MTP draft (3.79 GB) instead of BF16 (7.15 GB); v2: draft MoE on flashinfer_cutlass (v16.3 used it; auto picked FLASHINFER_TRTLLM, SM100-only, and v1 died at draft CUDA-graph capture)
    "--watchdog-timeout", "1800",
    "--enable-hierarchical-cache", "--hicache-size", "48", "--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel",
    # v4 startup fix: safetensors get_tensor is lazy mmap, so the single consumer thread did all NFS reads.
    "--weight-loader-prefetch-checkpoints", "--weight-loader-prefetch-num-threads", "16",
    "--model-loader-extra-config", json.dumps({"enable_multithread_load": True, "num_threads": 8}),
]
SERVER = {"proc": None, "log": None}

def http(path, body=None, timeout=60):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

def stop_server():
    p = SERVER["proc"]
    if p and p.poll() is None:
        try: os.killpg(p.pid, signal.SIGTERM)
        except Exception: pass
        try: p.wait(60)
        except Exception:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
    sh("pkill -9 -f sglang.launch_server || true")
    for _ in range(60):
        used = gpumem().split(",")[0].strip()
        if used.isdigit() and int(used) < 2000: break
        time.sleep(2)

def boot_facts(text):
    keys = ("max_total_num_tokens", "KV Cache", "Mamba Cache", "mamba", "avail mem", "Load weight", "Capture",
            "cuda graph", "speculative", "fired up", "ple", "PLE", "backend", "Backend", "hicache", "HiCache", "memory", "Memory")
    lines = [l[-400:] for l in text.splitlines() if any(k in l for k in keys)]
    num = lambda pat: [float(x) for x in re.findall(pat, text)]
    return {"max_total_num_tokens": num(r"max_total_num_tokens=(\d+)"), "kv_tokens": num(r"#tokens: (\d+)"),
            "avail_mem_GB": num(r"avail mem=([\d.]+) GB"), "lines": lines[:160]}

def launch(tag, extra=(), boot_timeout=75 * 60):
    stop_server()
    logp = WORK / f"sglang-{tag}.log"
    fh = open(logp, "w")
    cmd = [sys.executable, "-m", "sglang.launch_server", *BASE_ARGS, *extra]
    log("LAUNCH", tag, " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=server_env(), start_new_session=True)
    SERVER.update(proc=p, log=logp)
    t = time.time()
    ready = False
    limit = min(boot_timeout, left() - 60)   # computed once (v16.2 fix)
    while time.time() - t < limit:
        if p.poll() is not None: break
        try:
            http("/health", timeout=5); json.loads(http("/v1/models", timeout=5)); ready = True; break
        except Exception:
            time.sleep(10)
    text = logp.read_text(errors="replace")
    info = {"tag": tag, "ready": ready, "ready_seconds": round(time.time() - t, 1), "exit_code": p.poll(),
            "gpu_mem_used_total_MiB": gpumem(), "host_mem": meminfo(), "facts": boot_facts(text)}
    if not ready:
        info["log_tail"] = text[-12000:]
    log("BOOT", tag, json.dumps({k: info[k] for k in ("ready", "ready_seconds", "exit_code", "gpu_mem_used_total_MiB")}))
    return info

# ---------------- 2b. patch: parallel PLE row copy (fixes the CPU/page-fault-bound PLE shard tail) ----------------
# sglang-boot-v4 log: expert shards 1-194 loaded in 89 s with prefetch, but the 12 PLE-tail shards took 17.5 -> 38.0 s
# each even after prefetch finished. safetensors get_tensor() is lazy mmap, so copy_ does the reads in ONE thread.
# Split each PLE row copy into chunks copied by parallel threads (torch copy_ releases the GIL).
PLE_PATCH_OLD = '''                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )
'''
PLE_PATCH_NEW = '''                _dst = emb.weight.data[local_start : local_start + n_rows]
                _src = loaded_weight[src_start : src_start + n_rows]
                if (
                    _dst.device.type == "cpu"
                    and _src.device.type == "cpu"
                    and _src.dtype == _dst.dtype
                    and n_rows >= 65536
                ):
                    import concurrent.futures as _penny_cf
                    import os as _penny_os
                    _nt = max(1, int(_penny_os.environ.get("PENNY_PLE_COPY_THREADS", "16")))
                    _step = (n_rows + _nt - 1) // _nt
                    _spans = [(_i, min(n_rows, _i + _step)) for _i in range(0, n_rows, _step)]
                    with _penny_cf.ThreadPoolExecutor(len(_spans)) as _ex:
                        list(_ex.map(lambda _s: _dst[_s[0]:_s[1]].copy_(_src[_s[0]:_s[1]]), _spans))
                    if not getattr(emb, "_penny_parallel_logged", False):
                        emb._penny_parallel_logged = True
                        print(f"PENNY_PLE_PARALLEL_COPY threads={_nt} rows={n_rows}", flush=True)
                else:
                    _dst.copy_(_src.to(device=emb.weight.device, dtype=emb.weight.dtype))
'''
LOADER_PATCH_OLD = '            start_iterator_prefetch = (\n                configured_prefetch and not startup_prefetch_started\n            )\n'
LOADER_PATCH_NEW = '            start_iterator_prefetch = (\n                configured_prefetch\n                and not startup_prefetch_started\n                and not globals().get("_PENNY_PREFETCH_DONE", False)\n            )\n            if start_iterator_prefetch:\n                globals()["_PENNY_PREFETCH_DONE"] = True\n                print(f"PENNY_PREFETCH_ONCE started for {source.model_or_path}", flush=True)\n            elif configured_prefetch and not startup_prefetch_started:\n                print(f"PENNY_PREFETCH_ONCE skipped repeat prefetch for {source.model_or_path}", flush=True)\n'
@phase("patch_loader_prefetch_once")
def _loader_patch():
    target = SITE / "sglang/srt/model_loader/loader.py"
    txt = target.read_text(encoding="utf-8")
    n = txt.count(LOADER_PATCH_OLD)
    if n != 1:
        raise RuntimeError(f"loader prefetch anchor found {n} times in {target}")
    new = txt.replace(LOADER_PATCH_OLD, LOADER_PATCH_NEW)
    compile(new, str(target), "exec")
    target.write_text(new, encoding="utf-8")
    for pyc in (target.parent / "__pycache__").glob("loader.*.pyc"):
        pyc.unlink()
    return {"file": str(target), "sha_after": hashlib.sha256(new.encode()).hexdigest()[:16]}
if "patch_loader_prefetch_once" in R["errors"]:
    SETUP_OK = False
@phase("patch_ple_parallel_copy")
def _ple_patch():
    target = SITE / "sglang/srt/models/qwen4_exp.py"
    txt = target.read_text(encoding="utf-8")
    before = hashlib.sha256(txt.encode()).hexdigest()
    n = txt.count(PLE_PATCH_OLD)
    if n != 1:
        raise RuntimeError(f"PLE copy anchor found {n} times in {target}; runtime source differs from pennyroyal-v2.5.0")
    new = txt.replace(PLE_PATCH_OLD, PLE_PATCH_NEW)
    compile(new, str(target), "exec")
    target.write_text(new, encoding="utf-8")
    for pyc in (target.parent / "__pycache__").glob("qwen4_exp.*.pyc"):
        pyc.unlink()
    return {"file": str(target), "sha_before": before[:16], "sha_after": hashlib.sha256(new.encode()).hexdigest()[:16]}
if "patch_ple_parallel_copy" in R["errors"]:
    SETUP_OK = False

# ---------------- 2c. M82 patch: GPTQ-Marlin MoE scales in the model dtype (v16.5 v1 died here) ----------------
# v1: AssertionError "moe_wna16_marlin_gemm assumes hidden_states.dtype (torch.bfloat16) == w1_scale.dtype
# (torch.float16)" on the first expert forward. gptq_moe.py GPTQMarlinMoEScheme.create_weights hardcodes the
# w13/w2 scale tensors as torch.half while the model runs bf16. Create them in params_dtype instead: the loader's
# copy_ casts the checkpoint's fp16 scales to bf16 (what vLLM does for this checkpoint), marlin_moe_permute_scales
# keeps the dtype, and the JIT Marlin kernel is instantiated for bf16 activations.
M82_SCALE_OLD = (
    "                2 * intermediate_size_per_partition,\n                dtype=torch.half,\n",
    "            torch.empty(num_experts, scales_size2, hidden_size, dtype=torch.half),\n",
)
M82_SCALE_NEW = (
    "                2 * intermediate_size_per_partition,\n                dtype=params_dtype,\n",
    "            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),\n",
)
@phase("patch_gptq_moe_scale_dtype")
def _m82_scale_patch():
    target = SITE / "sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py"
    txt = target.read_text(encoding="utf-8")
    for old in M82_SCALE_OLD:
        n = txt.count(old)
        if n != 1:
            raise RuntimeError(f"GPTQ MoE scale anchor found {n} times in {target}")
    new = txt
    for old, rep_ in zip(M82_SCALE_OLD, M82_SCALE_NEW):
        new = new.replace(old, rep_)
    if "dtype=torch.half" in new:
        raise RuntimeError("GPTQ MoE scales still hardcoded to torch.half after patch")
    compile(new, str(target), "exec")
    target.write_text(new, encoding="utf-8")
    for pyc in (target.parent / "__pycache__").glob("gptq_moe.*.pyc"):
        pyc.unlink()
    return {"file": str(target), "sha_after": hashlib.sha256(new.encode()).hexdigest()[:16]}
if "patch_gptq_moe_scale_dtype" in R["errors"]:
    SETUP_OK = False
save()

BOOT_OK = False
if SETUP_OK:
    @phase("boot_main")
    def _boot():
        global BOOT_OK
        info = launch("main", boot_timeout=1500)
        BOOT_OK = info["ready"]
        if BOOT_OK:
            try:
                info["server_info"] = json.loads(http("/get_server_info", timeout=30))
            except Exception as exc:
                info["server_info_error"] = str(exc)
        return info

if not BOOT_OK:
    _bm = R["phases"].get("boot_main", {}).get("result") or {}
    print((_bm.get("log_tail") or "")[-8000:], flush=True)
    raise RuntimeError(f"V1458 SGLang did not become ready: errors={sorted(R['errors'])}")
_models = json.loads(http("/v1/models", timeout=30))
if SERVED not in [m.get("id") for m in _models.get("data", [])]:
    raise RuntimeError(f"served model id mismatch: {_models}")
_boot_txt = (WORK / "sglang-main.log").read_text(errors="replace")
_m = re.findall(r"max_total_num_tokens=(\d+), chunked_prefill_size=\d+, max_prefill_tokens=\d+, max_running_requests=(\d+)", _boot_txt)
if not _m:
    raise RuntimeError("V1458 could not read max_total_num_tokens/max_running_requests from the SGLang log")
V1458_KV_TOKENS, V1458_RUNNING = int(_m[-1][0]), int(_m[-1][1])
print(f"V1458_POOL kv_tokens={V1458_KV_TOKENS} max_running_requests={V1458_RUNNING} fp8_unscaled_warning={'no scaling factors provided' in _boot_txt}", flush=True)
_hic = re.findall(r"hicache[^\\n]{0,160}", _boot_txt, re.I)[:6]
print("V1462_HICACHE_LINES", json.dumps(_hic), flush=True)
_flags_ok = all(k in _boot_txt for k in ("\'chunked_prefill_size\': 4096", "\'enable_hierarchical_cache\': True", "hicache_attached=True"))
print(f"V1462_FLAGS chunk4096+hicache_flag+hicache_attached={_flags_ok}", flush=True)
if not _flags_ok:
    stop_server()
    raise RuntimeError("V1462 GUARD: server_args/log do not show chunk 4096 + hicache attached")
_alloc = re.findall(r"(KV Cache is allocated[^\n]{0,120}|Mamba Cache is allocated[^\n]{0,240}|Memory pool end[^\n]{0,60})", _boot_txt)
print("V1464_ALLOC_LINES", json.dumps(_alloc), flush=True)
if "# v14.64" not in _KVC.read_text():
    stop_server()
    raise RuntimeError("V1464 GUARD: runtime patch marker missing")
if V1458_RUNNING < 15 or V1458_KV_TOKENS < 330000:
    stop_server()
    raise RuntimeError(f"V1464 GUARD: running={V1458_RUNNING} (<15) or kv_tokens={V1458_KV_TOKENS} (<330000)")
print(f"V181GC_HISTORY running={V1458_RUNNING} kv_tokens={V1458_KV_TOKENS} fp8_kv=1 ctx=69632 trims=57344->45056", flush=True)
_dm = re.findall(r"Load weight end\. .*?type=Qwen4ExpForCausalLMMTP, quant=([\w-]+).*?mem usage=([\d.]+) GB", _boot_txt)
print(f"V1815_DRAFT quant={_dm[-1][0] if _dm else None} mem_gb={_dm[-1][1] if _dm else None} kv_tokens={V1458_KV_TOKENS} max_running={V1458_RUNNING}", flush=True)
if not _dm or _dm[-1][0] != "modelopt_fp4":
    raise RuntimeError(f"V1815 GUARD: MTP draft did not load as modelopt_fp4: {_dm}")
_dt108 = re.findall(r"Load weight end\. elapsed=([\d.]+) s, type=Qwen4ExpForCausalLMMTP", _boot_txt)
print(f"M108_DRAFT_LOAD_S {_dt108[-1] if _dt108 else None} dir={DRAFT}", flush=True)

# v14.64 v2: server-alive watchdog + device memory log. v1 lost the server mid-run and the harness spun on
# "Connection refused" for the rest of the session. Three failed polls -> dump the server log tail and stop the notebook.
import threading as _wd64_th, subprocess as _wd64_sp
# GPU memory sampler runs in its OWN process: importing torch in the notebook process broke the v14.14 ACTION7 check
# (it scans sys.modules for `to_engine_action`, and torch.ops answers any attribute with an _OpNamespace) -- v14.64 v2.
_WD64_SAMPLER = _wd64_sp.Popen([sys.executable, "-c", (
    "import time,sys\n"
    "t0=time.time()\n"
    "try:\n"
    "    import torch\n"
    "except Exception as e:\n"
    "    open(sys.argv[1],'a').write(f'torch import failed: {e}\\n'); raise SystemExit(0)\n"
    "while True:\n"
    "    try:\n"
    "        f,t=torch.cuda.mem_get_info(0); open(sys.argv[1],'a').write(f'{time.time()-t0:.0f}s used_gb={(t-f)/1024**3:.2f}\\n')\n"
    "    except Exception as e:\n"
    "        open(sys.argv[1],'a').write(f'err {e}\\n')\n"
    "    time.sleep(30)\n"), str(WORK / "gpu_mem.log")], env=server_env(), stdout=_wd64_sp.DEVNULL, stderr=_wd64_sp.DEVNULL)
_WD64 = {"fails": 0, "polls": 0}
def _wd64_peak():
    try:
        vals = [float(l.split("used_gb=")[1]) for l in (WORK / "gpu_mem.log").read_text().splitlines() if "used_gb=" in l]
        return max(vals) if vals else -1.0
    except Exception:
        return -1.0
def _wd64_loop():
    while True:
        time.sleep(30)
        try:
            http("/v1/models", timeout=20)
            _WD64["fails"] = 0
        except Exception as exc:
            _WD64["fails"] += 1
            print(f"V1464_WATCHDOG server poll failed ({_WD64['fails']}/3): {str(exc)[:160]}", flush=True)
            if _WD64["fails"] >= 3:
                try:
                    tail = (WORK / "sglang-main.log").read_text(errors="replace")[-20000:]
                    (WORK / "sglang-main-tail-at-death.log").write_text(tail)
                    print("V1464_SERVER_DEAD peak_used_gb=%.2f last log lines:\n%s" % (_wd64_peak(), tail[-3000:]), flush=True)
                except Exception as exc:
                    print("V1464_SERVER_DEAD (log tail unavailable)", exc, flush=True)
                os._exit(4)
        _WD64["polls"] += 1
_wd64_t = _wd64_th.Thread(target=_wd64_loop, daemon=True); _wd64_t.start()
print("V1464_WATCHDOG armed (30 s polls, 3 strikes); sampler pid", _WD64_SAMPLER.pid, flush=True)

def _prompt_tokens(with_rc):
    asst = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})}}]}
    secret = " ".join(["the lever at row three toggles the blue gate"] * 40)
    asst["reasoning"] = secret
    if with_rc:
        asst["reasoning_content"] = secret
    body = {"model": SERVED, "max_tokens": 1, "temperature": 0.0,
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}, asst,
                         {"role": "tool", "tool_call_id": "c1", "content": "1"}],
            "chat_template_kwargs": {"enable_thinking": True}}
    return json.loads(http("/v1/chat/completions", body, timeout=120))["usage"]["prompt_tokens"]
_pt_plain, _pt_rc = _prompt_tokens(False), _prompt_tokens(True)
print(f"V1458_REASONING_RENDER prompt_tokens reasoning_only={_pt_plain} with_reasoning_content={_pt_rc}", flush=True)
if _pt_rc - _pt_plain < 200:
    stop_server()
    raise RuntimeError("V1458 GUARD: reasoning_content is not rendered into the prompt by this server/template")
print(f"V1458_SGLANG_READY notebook_elapsed_s={time.time() - NOTEBOOK_START_EPOCH:.1f} boot_s={R['phases']['boot_main']['result']['ready_seconds']}", flush=True)
