"""Out-of-fold validation of the re-ranking hybrid and leave-one-group-out ablations.
Uses the best validation configuration of PPR, ALS and content (cache/val_<m>_best.npz written by run_val.py).
Run: python3 -I src/run_rerank_val.py  -> results/rerank_val.txt, cache/rerank_val.npz"""
import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import GroupKFold
from common import RESULTS, CACHE, load_meta
from pipeline import Split, apk
import rerank as R
import methods as M

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
t0 = time.time()
V = Split("val", 18, (19, 30), quiet=True)
ev = V.eval_users()
SMOKE = int(__import__("os").environ.get("SMOKE", "0"))   # smoke test on the first N cohort users
sel = slice(0, SMOKE) if SMOKE else slice(None)
ev = ev[sel]; actual = [V.target[u] for u in ev]
scores = {}
meta = load_meta()
for m in ["ppr", "als", "content"]:
    sel_m = json.load(open(RESULTS / f"val_{m}_best.json"))
    scores[m] = (M.content(V, ev, meta, **sel_m["cfg"]) if m == "content" else getattr(M, m)(V, ev, **sel_m["cfg"])).astype(np.float32)
    p(f"[rerank] {m} scores from {sel_m}")
r, i, X, ncand = R.build(V, ev, scores, meta)
y = R.labels(V, ev, r, i)
cov = y.sum() / (5 * len(ev))
p(f"[rerank] candidate pairs {len(r):,} ({ncand:.1f} per user); positives {y.sum():,} of {5 * len(ev):,} target slots "
  f"(candidate recall {cov:.4f}); build {time.time() - t0:.1f}s")
folds = list(GroupKFold(n_splits=5).split(X, y, groups=r))


def oof(cols):
    prob = np.zeros(len(y))
    for tr, te in folds:
        mdl = R.model().fit(X[tr][:, cols], y[tr])
        prob[te] = mdl.predict_proba(X[te][:, cols])[:, 1]
    P = R.rank_lists(r, i, prob, len(ev), V)
    ap = np.array([apk(a, V.items[row].tolist()) for a, row in zip(actual, P)])
    return ap, prob, P

allc = list(range(len(R.FEATS)))
ap, prob, P = oof(allc)
p(f"[rerank] ALL features ({len(allc)}): out-of-fold MAP@5 {ap.mean():.5f} (n={len(ev)}) [{time.time() - t0:.0f}s]")
if not SMOKE: np.savez(CACHE / "rerank_val.npz", ap=ap, P=P, ev=ev)
for g, fs in R.GROUPS.items():
    cols = [j for j, f in enumerate(R.FEATS) if f not in fs]
    a2, _, _ = oof(cols)
    p(f"[rerank] without {g:<10} ({len(cols)} feats): MAP@5 {a2.mean():.5f}  delta {a2.mean() - ap.mean():+.5f} [{time.time() - t0:.0f}s]")
(RESULTS / ("rerank_val_smoke.txt" if SMOKE else "rerank_val.txt")).write_text("\n".join(out) + "\n")
