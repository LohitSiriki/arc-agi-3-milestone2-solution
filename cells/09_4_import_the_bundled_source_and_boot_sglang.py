# Cell 9 of arc-agi-3-duck-18-1gc-submit.ipynb (section: 4. Import the bundled source and boot SGLang)
# Harness repos on the path. The bundle's own sglang-rtxpro6000 tree is left OFF the path so it can never shadow
# the Pennyroyal runtime the server imports (the harness itself never imports sglang).
def _source_path_entries(bundle_dir: Path) -> list:
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        if repo.name.startswith("sglang"):
            continue
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries

source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    sys.path.insert(0, str(entry))
pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
pth_path.write_text("".join(f"{entry}\n" for entry in source_entries))
print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")

HARNESS_ENV = {"INFERENCE_ANALYZER_MODEL": "Qwen/Qwen3.8-Flash-Next-NVFP4", "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness", "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:1234/v1", "LOCAL_ANALYZER_CONTEXT_WINDOW": "69632", "LOCAL_ANALYZER_ENABLE_THINKING": "true", "LOCAL_ANALYZER_MAX_OUTPUT": "0", "LOCAL_ANALYZER_MODEL_ID": "Qwen/Qwen3.8-Flash-Next-NVFP4", "LOCAL_ANALYZER_PROVIDER": "vllm", "LOCAL_ANALYZER_TEMPERATURE": "0.6", "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024", "LOCAL_ANALYZER_TOOL_STEPS": "0", "LOCAL_ANALYZER_TOOL_TIMEOUT": "30", "LOCAL_ANALYZER_TOP_K": "20", "LOCAL_ANALYZER_TOP_P": "0.95", "LOCAL_ANALYZER_YIELD_SECONDS": "60", "MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "4", "OPENAI_API_KEY": "offline-kaggle-local-server", "OPENAI_BASE_URL": "http://127.0.0.1:1234/v1", "OPENAI_PROVIDER": "vllm"}
os.environ.update(HARNESS_ENV)
SETUP_ENV_PATH.write_text(json.dumps({**json.loads(SETUP_ENV_PATH.read_text()), **HARNESS_ENV}, indent=2, sort_keys=True) + "\n")
print("V1458_HARNESS_ENV", json.dumps(HARNESS_ENV, sort_keys=True), flush=True)
