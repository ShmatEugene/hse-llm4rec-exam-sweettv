"""Validation grid for one method (selection on validation only). Every configuration is ranked into exactly five
pool ids by pipeline.top5 with the shared popularity fallback. Optional schedule-aware re-weighting: score x
(number of airings of the programme in the target window)^b; b = 0 is the method alone.
Run: python3 -I src/run_val.py <method>   -> results/val_<method>.txt, cache/val_<method>_best.npz"""
import sys, pathlib, time, json, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import RESULTS, CACHE, load_meta
from pipeline import Split, top5, apk
import methods as M

method = sys.argv[1]
GRIDS = {
    "personal": [dict(half_life=h) for h in [None, 16, 8]],
    "ppr": [dict(alpha=a, half_life=h) for a in [0.3, 0.5, 0.7, 0.85] for h in [None, 8]],
    # first grid (linear alpha 10/40/100, factors 32) fell monotonically with alpha: results/val_als_grid1_partial.txt
    "als": [dict(factors=32, alpha=a, reg=1.0, conf=c, iters=8) for c, alist in
            [("linear", [1, 3]), ("log", [1, 3, 10])] for a in alist],
    "content": [dict(half_life=h) for h in [None, 8]],
    "formula": [dict(half_life=h, rating=r) for h in [None, 4, 8] for r in ["soft", "hard"]],
}
BS = [0.0, 0.25, 0.5]

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

V = Split("val", 18, (19, 30), quiet=True)
ev = V.eval_users(); actual = [V.target[u] for u in ev]
fb = M.fallback_order(V)
air = np.zeros(len(V.items)); air[V.pool_ix] = V.pool_airings
meta = load_meta() if method == "content" else None
X = M.content_vectors(V, meta)[0] if method == "content" else None
best = (-1, None, None, None)
p(f"[val:{method}] cohort n={len(ev)}, pool {len(V.pool)}; grid {len(GRIDS[method])} configs x b in {BS}")
for cfg in GRIDS[method]:
    t0 = time.time()
    if method == "personal":
        S = M.personal(V, ev, **cfg)
    elif method == "ppr":
        S = M.ppr(V, ev, **cfg)
    elif method == "als":
        S = M.als(V, ev, **cfg)
    elif method == "formula":
        S = M.formula(V, ev, **cfg)
    elif method == "content":
        S = M.content(V, ev, meta, X=X, **cfg)
    fit = time.time() - t0
    for b in BS:
        Sb = np.clip(S, 0, None) * (air[None, :] ** b if b else 1.0)
        P = top5(Sb, V.pool_ix, fb)
        ap = np.array([apk(a, V.items[r].tolist()) for a, r in zip(actual, P)])
        extra = f" iters={V.ppr_last['iters']} resid={V.ppr_last['max_l1_residual']:.2e}" if method == "ppr" else ""
        p(f"[val:{method}] {json.dumps(cfg)} b={b} MAP@5 {ap.mean():.5f} fit {fit:.1f}s{extra}")
        if ap.mean() > best[0]:
            best = (ap.mean(), cfg, b, ap)
            np.savez(CACHE / f"val_{method}_best.npz", S=S.astype(np.float32), ev=ev, ap=ap)
p(f"[val:{method}] BEST {json.dumps(best[1])} b={best[2]} MAP@5 {best[0]:.5f}")
json.dump({"cfg": best[1], "b": best[2], "map": best[0]}, open(RESULTS / f"val_{method}_best.json", "w"))
(RESULTS / f"val_{method}.txt").write_text("\n".join(out) + "\n")
