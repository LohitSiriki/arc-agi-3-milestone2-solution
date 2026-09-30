# Cell 12 of arc-agi-3-duck-18-1gc-submit.ipynb (section: 4. Import the bundled source and boot SGLang)

# ---------------- boot timing summary ----------------
def _phase_times(text):
    out = {"load_weight_end_s": [float(x) for x in re.findall(r"Load weight end\. elapsed=([\d.]+) s", text)],
           "prefetch_lines": [l[-200:] for l in text.splitlines() if ("prefetching" in l.lower() or "PENNY_PLE_PARALLEL_COPY" in l)][-4:],
           "fired_up": "fired up" in text}
    return out
try:
    _txt = (WORK / "sglang-main.log").read_text(errors="replace") if (WORK / "sglang-main.log").exists() else ""
    R["boot_timing"] = {"notebook_elapsed_s": round(time.time() - T0, 1), **_phase_times(_txt),
                        "boot": R["phases"].get("boot_main", {}).get("result", {}).get("ready_seconds") if isinstance(R["phases"].get("boot_main", {}).get("result"), dict) else None}
    log("SGLBOOT_TIMING", json.dumps(R["boot_timing"]))
except Exception as exc:
    R["errors"]["boot_timing"] = str(exc)
save()
