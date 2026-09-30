# ARC Prize 2026, ARC-AGI-3: rellik13's Milestone 2 notebook (public LB 22.53)

rellik13's (solo) submission for ARC Prize 2026 ARC-AGI-3, Milestone 2 (September 30, 2026). The notebook scored
**22.53** on the public leaderboard (submission 56707217, submitted 2026-09-30 12:22 UTC).

Kaggle notebook: `sirikilohit/arc-agi-3-duck-18-1gc-submit` (version 1). Thought process and decisions: [`WRITEUP.md`](WRITEUP.md).

## What it is

An agent that plays ARC-AGI-3 games offline on one RTX PRO 6000, built on Tufa Labs' Duck/TAAF harness with a port of
Wang's harness design, serving Qwen3.8-Flash-Next through the Pennyroyal SGLang fork.

| Part | Setting |
|---|---|
| Model | Qwen3.8-Flash-Next: Intel AutoRound W4A16 checkpoint with RadixArk's FP8 PLE tables swapped in (mixed at load time) |
| Draft | RadixArk NVFP4 MTP head (NEXTN, 3 steps / 4 draft tokens); only the draft shards are loaded |
| Server | SGLang (Pennyroyal build), FP8 e4m3 KV cache (1,004,288 tokens), 16 concurrent requests, context 69,632 |
| History | Trimmed at 57,344 prompt tokens back to 45,056 (`cells/35_...`) |
| Scheduler | One-pool UCB scheduler over all games with 2,400 s slices |
| Harness patches | Solved-level rule memory, context guard and watermarks, mid-action animation frames, `UNDO` label for ACTION7, accurate "Level restarted." line with auto-reset retry, action-budget-bar prompt text removed |
| Sampling | Thinking mode, temperature 0.6, top_p 0.95, top_k 20 |

The key choice is memory: the FP8 KV cache makes room for a much longer per-game history. Prompts and harness logic
are otherwise unchanged. See [`WRITEUP.md`](WRITEUP.md) for why.

## Repository layout

| Path | Contents |
|---|---|
| `notebook/arc-agi-3-duck-18-1gc-submit.ipynb` | The submitted notebook (competition rerun plays all 110 games; an interactive Save & Run plays the 25 public games for 20 minutes) |
| `notebook/arc-agi-3-duck-18-1gc-history.ipynb` | The same notebook in practice form (full 25-game public run) |
| `notebook/kernel-metadata.json` | Kaggle inputs: datasets, models, accelerator |
| `cells/` | Every code cell of the submitted notebook as a Python file, named after its section |
| `run/` | Outputs of the Kaggle Save & Run (25 public games, 20 minutes): kernel log, score, benchmark, SGLang log, boot result, scheduler events, transcripts and prompts |
| `analysis/` | `tps_by_running.py` (decode speed from the SGLang log), `sawtooth.py` (how much of a level's history is in view vs clearing it), and the long-prompt one-liner; see `analysis/README.md` |

## Running it on Kaggle

Copy the notebook, keep all inputs attached, and select the RTX PRO 6000 accelerator. Inputs (see
`notebook/kernel-metadata.json`):

- Models: `woochangsim/qwen38-flash-next-w4a16-autoround-4c67bf6` (Intel AutoRound W4A16) and
  `keithtyser/qwen3-8-flash-next-nvfp4` (RadixArk NVFP4: BF16 core, FP8 PLE, MTP draft).
- Datasets: `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` (harness bundle), `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`
  (CUDA runtime layers), `sirikilohit/sglang-penny-build-qwen` (SGLang build).

## Credits

Tufa Labs (Duck/TAAF harness and notebook), Wang's harness design, John Pezzulli's Pennyroyal SGLang fork, Intel
(AutoRound quantization), RadixArk (NVFP4 checkpoint and MTP head), and keithtyser (Kaggle mirrors of the harness bundle,
runtime and model). Third-party code and weights remain under their own licenses.

## License

Apache License 2.0 for the code in this repository (see `LICENSE`).
