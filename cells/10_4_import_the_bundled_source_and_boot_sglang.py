# Cell 10 of arc-agi-3-duck-18-1gc-submit.ipynb (section: 4. Import the bundled source and boot SGLang)
# ---------------- 1. inventory + unpack SGLang runtime + borrow CUDA 13.0 toolkit ----------------
import base64, concurrent.futures, glob, hashlib, io, json, os, re, shutil, signal, statistics, subprocess, sys, tarfile, threading, time, traceback, urllib.request
from pathlib import Path

T0 = time.time()
DEADLINE = T0 + 1500   # SGLang boot budget (v6 measured ready at 546 s)
WORK = Path("/kaggle/working")
R = {"errors": {}, "phases": {}}

def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

def save():
    (WORK / "sglang_boot_result.json").write_text(json.dumps(R, indent=2, default=str))

def left():
    return DEADLINE - time.time()

def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as exc:
        return f"ERR {exc}"

def phase(name):
    def deco(fn):
        t = time.time()
        log("PHASE_START", name)
        try:
            R["phases"][name] = {"result": fn()}
        except Exception:
            R["errors"][name] = traceback.format_exc()[-4000:]
            log("PHASE_ERROR", name, R["errors"][name][-1500:])
        R["phases"].setdefault(name, {})["seconds"] = round(time.time() - t, 1)
        save()
        return fn
    return deco

def meminfo():
    m = dict(re.findall(r"^(\w+):\s+(\d+) kB", open("/proc/meminfo").read(), flags=re.M))
    return {"MemTotal_GiB": round(int(m["MemTotal"]) / 1024**2, 1), "MemAvailable_GiB": round(int(m["MemAvailable"]) / 1024**2, 1)}

def gpumem():
    return sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")

R["env"] = {
    "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
    "mem": meminfo(), "nproc": os.cpu_count(), "python": sys.version,
    "gcc": sh("gcc --version | head -1"), "disk": sh("df -h /tmp /kaggle/working"),
}
log("ENV", json.dumps(R["env"]))
if "RTX PRO 6000" not in R["env"]["gpu"]:
    save()
    raise RuntimeError(f"wrong accelerator, expected RTX PRO 6000: {R['env']['gpu']}")

def _one(pattern, root="/kaggle/input"):
    hits = sorted({p.parent for p in Path(root).rglob(pattern)})
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {pattern} under {root}, found {hits}")
    return hits[0]

RT = _one("RUNTIME_MANIFEST.json")
VRT = _one("runtime-manifest.json")
MODEL = _one("MODEL_MANIFEST.json", "/kaggle/input/models")
R["paths"] = {"sglang_runtime": str(RT), "vllm_runtime_layers": str(VRT), "model": str(MODEL)}
R["model_config_sha256"] = hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest()
log("PATHS", R["paths"], "config sha", R["model_config_sha256"][:12])
save()

# ===== v16.5 = M82: serve Intel AutoRound W4A16 experts (BF16 activations) instead of NVFP4 W4A4 experts =====
# Merged model dir: every Intel file except its BF16 PLE tables (model-00016-of-00017, 102.4 GB), plus the served
# NVFP4 checkpoint's FP8 PLE tables (same tables, checked offline), so host RAM and PLE numerics match v16.0 v2.
NV = MODEL
W4 = _one("config.json.autoround.bak", "/kaggle/input/models")      # woochangsim mirror of Intel @ 4c67bf68
MIX = Path("/tmp/w4a16mix")
shutil.rmtree(MIX, ignore_errors=True)
MIX.mkdir(parents=True)
_PLE_BF16 = "model-00016-of-00017.safetensors"
_w4_idx = json.loads((W4 / "model.safetensors.index.json").read_text())
_nv_idx = json.loads((NV / "model.safetensors.index.json").read_text())
_drop = {k for k, f in _w4_idx["weight_map"].items() if f == _PLE_BF16}
if len(_drop) != 128 or not all(".ple.ple_embedding.ngram_embedding.shard_" in k for k in _drop):
    raise RuntimeError(f"M82: {_PLE_BF16} is not exactly the 128 PLE tables ({len(_drop)} tensors)")
_ple = {k: f for k, f in _nv_idx["weight_map"].items() if ".ple.ple_embedding.ngram_embedding." in k}
if len(_ple) != 129 or not all(f.startswith("model-plefp8-") for f in _ple.values()) or set(_drop) - set(_ple):
    raise RuntimeError(f"M82: served checkpoint's FP8 PLE tables not as expected ({len(_ple)} tensors)")
_wm = {k: f for k, f in _w4_idx["weight_map"].items() if k not in _drop}
_wm.update(_ple)
_skip = {_PLE_BF16, "model.safetensors.index.json", "config.json", "config.json.autoround.bak"}
for _f in sorted(W4.iterdir()):
    if _f.is_file() and _f.name not in _skip:
        (MIX / _f.name).symlink_to(_f)
for _name in sorted(set(_ple.values())):
    if not (NV / _name).is_file():
        raise RuntimeError(f"M82: missing {_name} in {NV}")
    (MIX / _name).symlink_to(NV / _name)
(MIX / "model.safetensors.index.json").write_text(json.dumps({**_w4_idx, "weight_map": _wm}))
_cfg = json.loads((W4 / "config.json").read_text())
if _cfg.get("quantization_config", {}).get("quant_method") != "auto-round":
    raise RuntimeError(f"M82: unexpected quantization_config {str(_cfg.get('quantization_config'))[:200]}")
_cfg["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
(MIX / "config.json").write_text(json.dumps(_cfg, indent=2))
MODEL = MIX
R["paths"].update(model=str(MODEL), w4a16_source=str(W4), ple_source=str(NV))
log("V1650_M82", json.dumps({"files": len(list(MIX.iterdir())), "tensors": len(_wm), "ple_fp8_files": len(set(_ple.values())),
                             "w4": str(W4), "nv": str(NV)}))
save()

@phase("unpack_sglang")
def _unpack():
    man = json.loads((RT / "RUNTIME_MANIFEST.json").read_text())
    for name, meta in man["parts"].items():
        if (RT / name).stat().st_size != meta["bytes"]:
            raise RuntimeError(f"part size mismatch {name}")
    whl = sorted(RT.glob("zstandard-*.whl"))
    zdir = Path("/tmp/zst")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps", "--target", str(zdir), str(whl[-1])], check=True)
    sys.path.insert(0, str(zdir))
    import zstandard

    class Cat(io.RawIOBase):
        def __init__(self, files):
            self.files, self.f = list(files), None
        def readable(self):
            return True
        def readinto(self, b):
            while True:
                if self.f is None:
                    if not self.files:
                        return 0
                    self.f = open(self.files.pop(0), "rb")
                n = self.f.readinto(b)
                if n:
                    return n
                self.f.close(); self.f = None

    dest = Path("/tmp/sgl")
    if dest.exists(): shutil.rmtree(dest)
    dest.mkdir()
    src = Cat(RT / n for n in sorted(man["parts"]))
    with zstandard.ZstdDecompressor().stream_reader(src) as zr, tarfile.open(fileobj=zr, mode="r|") as tf:
        tf.extractall(dest, filter="fully_trusted")
    return {"manifest": {k: man[k] for k in ("tag", "source_rev", "rust_extensions", "versions", "flashinfer_jit_cache") if k in man},
            "site_GiB": sh("du -sh /tmp/sgl/sgl-site | cut -f1")}

@phase("unpack_cuda_toolkit")
def _cuda():
    man = json.loads((VRT / "runtime-manifest.json").read_text())
    root = Path("/tmp/cuda-root")
    if root.exists(): shutil.rmtree(root)
    root.mkdir()
    n_files = 0
    skipped = []
    _ordered = sorted(man["selected_layers"], key=lambda l: (l["index"] != 8, l["index"]))
    _scanned = []
    for layer in _ordered:
        if layer["index"] != 8 and list(root.glob("usr/local/cuda-*/bin/nvcc")):
            break   # toolkit found in layer 8 (verified from image build history); skip the pip layers
        _scanned.append(layer["index"])
        path = VRT / layer["file"]
        if not path.exists():
            path = VRT / (layer["file"] + ".blob")
        with tarfile.open(path, mode="r|gz") as tf:
            for m in tf:
                name = m.name.lstrip("./")
                if not name.startswith("usr/local/cuda"):
                    continue
                base = os.path.basename(name)
                target = root / os.path.dirname(name)
                if base == ".wh..wh..opq":
                    if target.is_dir():
                        for c in target.iterdir():
                            shutil.rmtree(c) if c.is_dir() and not c.is_symlink() else c.unlink()
                    continue
                if base.startswith(".wh."):
                    victim = target / base[4:]
                    if victim.is_symlink() or victim.is_file(): victim.unlink()
                    elif victim.is_dir(): shutil.rmtree(victim)
                    continue
                m.name = name
                try:
                    tf.extract(m, root, filter="fully_trusted")
                    n_files += 1
                except Exception as exc:
                    skipped.append(f"{name}: {type(exc).__name__}")
    homes = sorted(p.parent.parent for p in root.glob("usr/local/cuda-*/bin/nvcc"))
    if not homes:
        raise RuntimeError("no nvcc found in the vLLM runtime layers")
    return {"layers_scanned": _scanned, "files": n_files, "skipped": skipped[:20], "skipped_count": len(skipped), "cuda_home": str(homes[-1]), "nvcc": sh(f"{homes[-1]}/bin/nvcc --version | tail -2")}

CUDA_HOME = Path(R["phases"].get("unpack_cuda_toolkit", {}).get("result", {}).get("cuda_home", "/usr/local/cuda"))
SETUP_OK = not R["errors"]
log("SETUP_OK", SETUP_OK, "CUDA_HOME", CUDA_HOME)
save()
