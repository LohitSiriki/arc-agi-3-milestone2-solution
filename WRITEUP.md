# rellik13: ARC-AGI-3 Milestone 2 write-up (public LB 22.53)

## TL;DR

Solo entry. Tufa Labs' Duck harness, Wang's sandbox and prompt design, Qwen3.8-Flash-Next served with SGLang on one
RTX PRO 6000. 44 submissions: 0.99 on Aug 11, 22.53 on Sep 30.

The last jump, 14.49 to 22.53, changed no prompts. I switched the KV cache to FP8 and used the freed memory to keep more
of each game's history in the prompt. At this context budget, the agent's own recent history is worth more than any
instruction I could add.

## How I worked

Solo, limited compute. Claude (Claude Code) did most of the implementation and the grunt work: reading run logs and game
transcripts, comparing runs, building notebook variants. I made the decisions and chose what to spend GPU time on.

Variance was the main obstacle. The same notebook submitted three times scored 5.02, 5.50 and 6.42. A notebook that
scored 37 on my 25-game practice run got 9.98 on the leaderboard; one that scored 32 in practice got 13.40. The same agent
clears a level in one run and is stuck on it for an hour in the next, depending on the first idea it commits to. The
first 30 minutes of a run say little about the result. With one submission a day and little GPU time, most decisions
were made on one or two runs. To cope, I compared runs at the same number of generated tokens per game instead of the
same wall time, preferred full-length runs, and read transcripts instead of trusting scores alone.

## Score history

| Date | Public score | Change |
|---|---|---|
| Aug 11 | 0.99 | First submission, Duck harness |
| Sep 2 | 3.81 | Qwen3.8-Flash-Next |
| Sep 18 | 6.42 | SGLang, more games at once |
| Sep 25 | 7.14 | Bigger context |
| Sep 26 | 9.96 | Memory of solved levels |
| Sep 27 | 13.40 | Wang's harness design |
| Sep 29 | 14.49 | Harness fixes |
| Sep 30 | **22.53** | FP8 KV cache, longer history |

## Stages

### August (0.99 to about 2.5)

Started from Tufa Labs' Duck harness, the Milestone 1 winner: the model reasons, writes Python in a sandbox to inspect the
game, and sends actions. I tested what the model should see: text-only boards, images, different board descriptions.
Scores stayed between 1 and 2.5. Run-to-run noise was as large as the effects I was testing.

### Model (3.81)

Qwen3.8-Flash-Next was the first clear jump. It is the strongest model that fits on one RTX PRO 6000: mostly quantized
experts, a few billion active parameters per token, fast, with a multi-token-prediction head for speculative decoding.

### SGLang instead of vLLM (up to 6.42)

Most public notebooks used vLLM. I used SGLang (the Pennyroyal build for this GPU). One reason: I knew SGLang better than
vLLM, so I could debug and tune it faster. The technical reasons: this model mixes full-attention and linear-attention
layers, SGLang's prefix cache handles that mix, so a game's long, mostly unchanged prompt isn't re-read on every call;
and it runs the model's own speculative-decoding head. Result: more games at once and more prompt room per game.

### Why more prompt text hurt

Adding text to every prompt dropped the score from 6.42 to 4.16. Later, a level-start advice block dropped it from 13.40
to 9.98. Normally more guidance helps a model. My assumption for why it didn't here: the context was capped at about 37k
tokens and old turns were trimmed to fit, so every repeated line of instructions pushed out some of the agent's own
recent turns, its only evidence about the current game. The agent already knew how to play; it lacked memory of what it
had tried. Advice also made it doubt rules it had right. After that I only kept changes that added no text or corrected
existing text.

### Memory (7.14, 9.96, 13.40)

- Bigger context: 49k tokens, trimmed back in large chunks so the server's prefix cache survives. 7.14.
- Memory of solved levels: the rule the agent wrote for a cleared level stays pinned in the prompt for the rest of the
  game, so it doesn't re-derive the basics on each new level. The only text addition that helped (+2.8), because it
  returns evidence the agent would otherwise lose rather than adding advice. 9.96.
- Wang's harness design: stricter Python sandbox, object segmentation (the board as objects with colour, size and
  position instead of raw cells), and prompts that push the agent to test ideas with code. Cheaper, more precise turns.
  13.40.

### Harness fixes (14.49)

Transcripts showed the harness misleading the agent:

- ACTION7 is undo, but the agent only saw "ACTION7" and wasted turns guessing. Renamed it `UNDO`.
- After a game over, the harness had already reset the level but told the agent "The game is over." Some turns
  reasoned as if the game had ended. Changed to "Level restarted."
- The prompt told the agent to ignore the bar along the board edge. That bar is the action budget, and running out of
  it ends many attempts. Deleted the line.
- Some actions only show their effect in intermediate animation frames. Made those frames available.
- 40-minute time slices per game in the scheduler, so games are paused and resumed less often.

No instructions added; wrong ones removed.

### FP8 KV and longer history (22.53)

Transcripts of stuck levels showed the agent often had the right goal but lost track of what it had tried, because
trims left a handful of turns on levels that take dozens. The final notebook spends GPU memory on history and changes
nothing else:

| Setting | Before | Now | Why |
|---|---|---|---|
| KV cache | BF16 | FP8 | Half the bytes per token; pool of about 1.0M tokens |
| History trims | 36,864 back to 26,624 tokens | 57,344 back to 45,056 | Far more of the level's own turns stay in view |
| Context length | 49,152 | 69,632 | Same 12,288 tokens above the trim point, so long replies aren't cut off |
| Games at once | 15 | 16 | What the larger pool supports |

Prompts, harness logic, model and sampling unchanged, so the effect comes from memory. Trims happen in one large chunk,
not every turn: each trim changes the start of the prompt and forces the server to re-read all of it, so fewer, larger
trims keep the prefix cache useful.

**Why FP8 came late.** It is a one-line server setting. I used it on the last day because:

- My mid-September test ran FP8 KV together with an MXFP8 re-quantization of the model's non-expert layers. The score
  dropped and reasoning got shorter at long context. I blamed FP8 KV and dropped it.
- The flags didn't match the intent. That test never raised the cap on concurrent requests, so the freed memory went
  unused. A later FP8 build lost two server flags to comments in the launch arguments. I caught it before it ran and
  added a check that every server argument matches the intended list, but I had already written FP8 off, so it never
  ran.
- I believed throughput was the limit and more concurrency didn't help, so freed memory had no use.
- The server logs showed otherwise: the cache pool was 91-97% full all run, fewer requests ran than allowed, and the rest
  queued for cache space. Memory was a bottleneck.
- The transcripts then gave the memory a use: history. Rechecking the old test, the drop came from the MXFP8 change
  (about 25% slower decoding); play per token was unchanged. FP8 KV alone had never been tested, so the final notebook
  tested exactly that.

**Why the final check was 20 minutes.** I had about 40 minutes of Kaggle GPU left for the week. The Save & Run required
before submitting doubled as the check: all 25 public games for 20 minutes. It booted cleanly with the larger pool,
decoded faster than with BF16, and cleared more levels than my earlier full runs at the same point. It couldn't show the
history effect: most games hadn't reached the trim point in 20 minutes. When my quota refreshes I'll run the notebook for
its full ~2.5 hours on the 25 public games and publish the scores.

### Held back: priority scheduling

I planned priority-based scheduling: more GPU time for games still making progress, instead of equal slices. I held it
back because games stagnate in their later stages (level clears per unit of time fall off), so moving time between games
would mostly move stagnant time; and my priority scheduler was basic and I had no compute to validate it.

## What didn't work

*Disclaimer: none of these were researched properly. Compute was the limit: most got one or two runs, some were cut
short, and with this variance one run settles little. "Didn't work" means "didn't work in the runs I could afford". The
same applies to most conclusions in this write-up.*

- Adding advice or summaries to the prompt, including per-action "what changed" notes and a level-start comparison
  block.
- Restarting a stuck level with a fresh context. The chance of clearing a level didn't fall with time spent on it, so a
  fresh start bought nothing.
- Temperature 1.0 instead of 0.6.
- A ledger for proven and refuted assumptions. The agent never used it.
- Pruning a quarter of the model's experts to free memory. More room, worse play.
- Higher-resolution board images.
- Swift 1.5, a public fine-tune of the same model. No better per token and about 35% slower to decode on my setup.

## Resources

- Kaggle's weekly GPU quota (RTX PRO 6000), one submission a day.
- About 12 hours of a rented RTX PRO 6000 after the quota ran out.
- A small fine-tuning trial: LoRA on Qwen3.8-27B on Kaggle's TPU, trained on games from an open game pool. A trial run
  within my resources; it scored badly.
- Claude, for implementation and reading logs and transcripts.

## Limits and next steps

I couldn't reach firm conclusions on many questions: most runs were single runs and several were cut short. I want to
research this properly after the competition. Whether I continue to November depends on the compute I can get. If I do:
longer history per game, and a validated priority scheduler.

## Credits

- **Tufa Labs**, the largest contribution: the Duck harness, solver and notebook this builds on.
- **Wang**, for the harness design (sandbox, segmentation, prompts), from his public repository until it was made
  private (I assume after he joined a team).
- **John Pezzulli** for the Pennyroyal SGLang build for the RTX PRO 6000, and the SGLang project.
- **Qwen** (Qwen3.8-Flash-Next), **Intel** (AutoRound quantization), **RadixArk** (NVFP4 checkpoint and
  speculative-decoding head), **UkisAI** (Swift 1.5), **keithtyser** (Kaggle mirrors of the harness bundle, runtime and
  model).
