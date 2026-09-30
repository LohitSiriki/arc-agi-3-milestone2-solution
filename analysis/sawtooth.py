"""Does more of the current level's history in context help? Natural experiment from M84's sawtooth: context grows
~26.6k -> 36.9k tokens, then a trim drops the oldest turns. At the same point of a level (k = turns already spent on
it), how many of those k turns the model can still see (h) depends on where in the sawtooth the level happens to be.
Per turn: k, h (from the transcript's history_messages, same window rebuild as analysis/2026-09-29-stuck), the
level's mean turn size so far (reasoning chars, a proxy for how much each turn eats of the window), and whether the
level is cleared within the next W turns. Compare turns with h >= median vs below, within strata of k bin x turn-size
tercile, pooled (Mantel-Haenszel style weighted difference).
Usage: sawtooth.py W RUN_DIR [RUN_DIR ...]"""
import re, sys, glob, collections, statistics as st

W = int(sys.argv[1])
HDR = re.compile(r'^--- analysis_step=(\d+) \|', re.M)
LV = re.compile(r'Current state: step \d+, level (\d+)')
META = re.compile(r'reasoning_chars: (\d+)')


def windows(t):
    win, dq, order = {}, collections.deque(), []
    for block in re.split(r'\n--- analysis_step=', t)[1:]:
        step = int(block.split(' ', 1)[0])
        hm = re.findall(r'history_messages: (\d+)', block)
        reqs = block.count('[MODEL RESPONSE META]')
        dq.append([step, 1 + 2 * reqs])
        if hm:
            h = int(hm[-1])
            while dq and sum(m for _, m in dq) > h:
                if len(dq) == 1:
                    break
                over = sum(m for _, m in dq) - h
                if dq[0][1] <= over:
                    dq.popleft()
                else:
                    dq[0][1] -= over
        win[step] = {s for s, _ in dq}
        lv = LV.search(block)
        order.append((step, int(lv.group(1)) if lv else None, sum(int(x) for x in META.findall(block))))
    return win, order


rows = []
for run in sys.argv[2:]:
    for f in glob.glob(run + '/transcripts/*_p0.txt'):
        win, order = windows(open(f, errors='replace').read())
        # fill missing levels forward
        last = None; o2 = []
        for s, L, rc in order:
            L = L if L is not None else last; last = L; o2.append((s, L, rc))
        for i, (s, L, rc) in enumerate(o2):
            if L is None or i == 0:
                continue
            j = i; k_steps = []
            while j - 1 >= 0 and o2[j - 1][1] == L:
                j -= 1; k_steps.append(o2[j][0])
            k = len(k_steps)
            if k < 2:
                continue
            vis = win.get(o2[i - 1][0], set())
            h = sum(1 for x in k_steps if x in vis)
            size = st.mean(o2[m][2] for m in range(j, i))
            fut = [o2[m][1] for m in range(i + 1, min(len(o2), i + 1 + W))]
            cleared = any(x is not None and x > L for x in fut)
            rows.append(dict(run=run.split('/')[-1], k=k, h=h, size=size, cleared=cleared))

print(f'turns analysed: {len(rows)} from {len(set(r["run"] for r in rows))} runs; outcome = level cleared within next {W} turns')
KB = [(2, 3), (4, 7), (8, 15), (16, 31), (32, 999)]
sizes = sorted(r['size'] for r in rows)
t1, t2 = sizes[len(sizes) // 3], sizes[2 * len(sizes) // 3]
num = den = 0.0
print('k bin   | turns | h median | frac seen | clear rate: h>=median  vs  h<median   (n)')
for lo, hi in KB:
    sub = [r for r in rows if lo <= r['k'] <= hi]
    if not sub:
        continue
    print(f'{lo:2d}-{hi:<3d}  | {len(sub):5d} | {st.median(r["h"] for r in sub):6.1f} | {st.mean(r["h"] / r["k"] for r in sub):8.2f} |', end='')
    tot = []
    for tl, th in ((-1, t1), (t1, t2), (t2, 1e18)):
        s2 = [r for r in sub if tl < r['size'] <= th]
        if len(s2) < 20:
            continue
        med = st.median(r['h'] for r in s2)
        hi_ = [r for r in s2 if r['h'] >= med and r['h'] > 0]; lo_ = [r for r in s2 if r['h'] < med]
        if len(hi_) < 10 or len(lo_) < 10:
            continue
        a = sum(r['cleared'] for r in hi_) / len(hi_); b = sum(r['cleared'] for r in lo_) / len(lo_)
        w = len(hi_) * len(lo_) / (len(hi_) + len(lo_))
        num += w * (a - b); den += w
        tot.append((a, b, len(hi_), len(lo_)))
    if tot:
        A = sum(x[0] * x[2] for x in tot) / sum(x[2] for x in tot); B = sum(x[1] * x[3] for x in tot) / sum(x[3] for x in tot)
        print(f'   {A:.3f}  vs  {B:.3f}   ({sum(x[2] for x in tot)} / {sum(x[3] for x in tot)})')
    else:
        print('   (too few)')
print(f'pooled difference (more history minus less), stratified by k bin x turn-size tercile: {num / den:+.4f}' if den else '')
base = sum(r['cleared'] for r in rows) / len(rows)
print(f'base clear rate: {base:.3f}')
