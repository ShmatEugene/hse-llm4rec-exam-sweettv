"""Validation accounting and the two baselines (popularity, personal re-watch) before any learned model.
Run: python3 -I src/explore_val.py  -> results/explore_val_output.txt"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import RESULTS
from pipeline import Split, top5, mapk
import methods as M

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
t0 = time.time()
V = Split("val", 18, (19, 30), quiet=True)
p(f"[val] history weeks 11-18 (log clipped at Monday of week 19), target weeks 19-30; split built in {time.time() - t0:.1f}s")
p("[val] history join counts:", {k: v for k, v in V.join_counts.items()})
tu, tshow, tsec = V.target_raw
p(f"[val] target watched airings {len(tu):,}; users with >=1 {len(np.unique(tu)):,}; users with >=5 programmes (cohort) {len(V.target):,}; "
  f"cohort users with a frequency tie between 5th and 6th place {V.target_ties:,}")
ev = V.eval_users()
hist_users = np.unique(V.h_u)
has_h = np.isin(ev, hist_users)
p(f"[val] cohort {len(ev):,} = with history {has_h.sum():,} + without history (cold start) {(~has_h).sum():,}")
p(f"[val] pool: {len(V.pool):,} programmes airing in weeks 19-30; items in index space {len(V.items):,}; "
  f"history programmes {len(np.unique(V.h_show)):,}; target top-5 ids outside pool {sum(i not in set(V.pool.tolist()) for u in ev for i in V.target[u])}")
# H4: how many of the target top-5 were already watched (>=1 airing at >=0.8) in history
H = V.matrix().tocsr()
seen = sum(H[u, V.item_ix[i]] > 0 for u in ev for i in V.target[u])
p(f"[val] H4 repeats: {seen:,} of {5 * len(ev):,} target slots ({seen / (5 * len(ev)):.4f}) are programmes the user watched in history")
actual = [V.target[u] for u in ev]
fb = M.fallback_order(V)
pop = M.popularity(V)
P = top5(np.tile(pop, (len(ev), 1)), V.pool_ix, fb)
p(f"[val] popularity MAP@5 {mapk(actual, [V.items[r].tolist() for r in P]):.5f}  (n={len(ev)})")
p(f"[val] popularity top-5 ids: {V.items[P[0]].tolist()}")
for hl in [None, 8, 4, 2, 1, 0.5]:
    S_ = M.personal(V, ev, hl)
    P = top5(S_, V.pool_ix, fb)
    p(f"[val] personal re-watch half-life={hl} MAP@5 {mapk(actual, [V.items[r].tolist() for r in P]):.5f}")
# schedule-aware variant: user count x number of airings of the programme in the target window
air = np.zeros(len(V.items)); air[V.pool_ix] = V.pool_airings
for b in [0.5, 1.0]:
    S_ = M.personal(V, ev, 2) * air[None, :] ** b
    P = top5(S_, V.pool_ix, fb)
    p(f"[val] personal(hl=2) x airings^{b} MAP@5 {mapk(actual, [V.items[r].tolist() for r in P]):.5f}")
p(f"total {time.time() - t0:.1f}s")
(RESULTS / "explore_val_output.txt").write_text("\n".join(out) + "\n")
