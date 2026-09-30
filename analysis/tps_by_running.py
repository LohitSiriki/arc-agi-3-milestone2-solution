"""Decode throughput by #running-req from SGLang logs, first N minutes of decoding only (so a 4-minute Kaggle save and
the box run are compared over the same early phase). Usage: tps_by_running.py MINUTES LOG [LOG ...]"""
import re, sys, statistics as st
from datetime import datetime
PAT = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, #running-req: (\d+), #full token: (\d+).*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)')
mins = float(sys.argv[1])
for f in sys.argv[2:]:
    rows = []
    for l in open(f, errors='replace'):
        m = PAT.match(l)
        if m:
            rows.append((datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'), int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5))))
    if not rows:
        print(f, 'no decode lines'); continue
    t0 = rows[0][0]
    rows = [r for r in rows if (r[0] - t0).total_seconds() <= mins * 60]
    print(f'== {f.split("/")[-3] if "/x/" in f else f.split("/")[-2]}: {len(rows)} decode lines in first {mins:g} min')
    by = {}
    for r in rows:
        by.setdefault(min(r[1] // 4 * 4, 28), []).append(r)
    for k in sorted(by):
        v = by[k]
        print(f'   running {k:2d}-{k+3:2d}: n={len(v):4d}  gen tok/s median {st.median(x[4] for x in v):7.1f}  accept {st.median(x[3] for x in v):.2f}  ctx tokens median {st.median(x[2] for x in v):8.0f}')
    print(f'   all: median gen tok/s {st.median(r[4] for r in rows):.1f}, mean running {st.mean(r[1] for r in rows):.1f}')
