# Cell 5 of arc-agi-3-duck-18-1gc-submit.ipynb (section: 2. Install the ARC runtime)
# Install the ARC runtime from the bundled competition wheels.
# Quiet: stdout is discarded; stderr (and a non-zero exit) still surface real failures.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links",
        str(next(iter(sorted(Path("/kaggle/input").glob("**/arc_agi_3_wheels"))), Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"))),
        "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)
