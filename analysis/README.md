# analysis

- `tps_by_running.py`: decode throughput by number of running requests, from an SGLang log.
  `python3 analysis/tps_by_running.py 20 run/sglang-main.log`
- `sawtooth.py`: for every turn, how much of the current level's history is still in the prompt, and how often the
  level is cleared within the next W turns when more versus less of it is visible (same point of the same level,
  stratified by turns into the level and turn size). `python3 analysis/sawtooth.py 3 RUN_DIR [RUN_DIR ...]`

Share of model calls with long prompts, from the SGLang log:

```bash
grep -o 'new-seq: 1, #new-token: [0-9]*, #cached-token: [0-9]*' run/sglang-main.log \
  | awk '{gsub(",","",$4); t=$4+$6; n++; if(t>36864)b++; if(t>45056)c++} END{printf "calls %d: >36,864 %.0f%%, >45,056 %.0f%%\n",n,100*b/n,100*c/n}'
```
