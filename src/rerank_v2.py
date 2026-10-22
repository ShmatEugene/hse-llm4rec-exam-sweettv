"""Second re-ranking round: variants scored out-of-fold on validation (5 folds over users); the best variant BY
VALIDATION is refitted on all validation pairs and applied to the test (submissions/f_rerank_v2.csv).
Variants: candidate depth k (30 per source as in round 1, or 60), trees (300 or 600), objective (binary "in target
top-5" classifier, or Poisson regression on the user's target-window frequency of the programme).
Run: python3 -I src/rerank_v2.py  -> results/rerank_v2.txt, results/val_top5_rerank_v2.csv, results/rerank_v2.json"""
import sys, pathlib, time, json, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from common import RESULTS, load_meta, load_test_users
from pipeline import Split, apk
import methods as M
import rerank as R
from submission import write

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
t0 = time.time()
meta = load_meta()
V = Split("val", 18, (19, 30), quiet=True); T = Split("test", 30, (31, 42), quiet=True)
ev = V.eval_users(); actual = [V.target[u] for u in ev]
tu, _ = load_test_users(); tix = np.searchsorted(T.users, tu)


def method_scores(split, uix):
    sc = {}
    for m in ["ppr", "als", "content"]:
        sel = json.load(open(RESULTS / f"val_{m}_best.json"))
        sc[m] = (M.content(split, uix, meta, **sel["cfg"]) if m == "content" else getattr(M, m)(split, uix, **sel["cfg"])).astype(np.float32)
    return sc

sv, st = method_scores(V, ev), method_scores(T, tix)
# target-window frequency of every (user, programme) for the Poisson objective
tu_, ts_, _ = V.target_raw
key = tu_.astype(np.int64) * 10**9 + ts_
k_, c_ = np.unique(key, return_counts=True); freq = dict(zip(k_.tolist(), c_.tolist()))


def make(kind, trees):
    common = dict(max_iter=trees, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
                  early_stopping=False, random_state=42)
    return HistGradientBoostingClassifier(**common) if kind == "clf" else HistGradientBoostingRegressor(loss="poisson", **common)


def predict(mdl, X, kind):
    return mdl.predict_proba(X)[:, 1] if kind == "clf" else mdl.predict(X)

results = {}
for k in [30, 60]:
    rv, iv, Xv, ncv = R.build(V, ev, sv, meta, k_ppr=k, k_pop=k, k_ch=k)
    y = R.labels(V, ev, rv, iv)
    yf = np.array([freq.get(int(ev[a]) * 10**9 + int(V.items[b]), 0) for a, b in zip(rv, iv)], float)
    folds = list(GroupKFold(n_splits=5).split(Xv, y, groups=rv))
    p(f"[v2] k={k}: candidate pairs {len(y):,} ({ncv:.1f}/user), candidate recall {y.sum() / (5 * len(ev)):.4f} [{time.time() - t0:.0f}s]")
    for kind, trees in [("clf", 300), ("clf", 600), ("poisson", 300)]:
        if k == 30 and (kind, trees) != ("clf", 300) and (kind, trees) != ("poisson", 300):
            continue
        prob = np.zeros(len(y))
        for tr, te in folds:
            mdl = make(kind, trees).fit(Xv[tr], y[tr] if kind == "clf" else yf[tr])
            prob[te] = predict(mdl, Xv[te], kind)
        P = R.rank_lists(rv, iv, prob, len(ev), V)
        ap = np.array([apk(a, V.items[row].tolist()) for a, row in zip(actual, P)])
        results[(k, kind, trees)] = (ap.mean(), P, ap)
        p(f"[v2] k={k} {kind} trees={trees}: out-of-fold MAP@5 {ap.mean():.5f} (n={len(ev)}) [{time.time() - t0:.0f}s]")

best = max(results, key=lambda c: results[c][0])
k, kind, trees = best
p(f"[v2] selected by validation: k={k} {kind} trees={trees} MAP@5 {results[best][0]:.5f}")
rng = np.random.default_rng(0); idx = rng.integers(0, len(ev), (1000, len(ev)))
base = results[(30, "clf", 300)][2]; d = (results[best][2] - base)[idx].mean(axis=1)
p(f"[v2] selected - round-1 configuration (k=30 clf 300): {results[best][0] - base.mean():+.5f}, 95% CI [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}]")
with open(RESULTS / "val_top5_rerank_v2.csv", "w") as f:
    f.write("user_id,top5\n")
    for u, row in zip(ev, results[best][1]):
        f.write(f"{V.users[u]},{' '.join(str(x) for x in V.items[row])}\n")
rv, iv, Xv, _ = R.build(V, ev, sv, meta, k_ppr=k, k_pop=k, k_ch=k)
y = R.labels(V, ev, rv, iv)
yf = np.array([freq.get(int(ev[a]) * 10**9 + int(V.items[b]), 0) for a, b in zip(rv, iv)], float)
mdl = make(kind, trees).fit(Xv, y if kind == "clf" else yf)
rt, it, Xt, nct = R.build(T, tix, st, meta, k_ppr=k, k_pop=k, k_ch=k)
Pt = R.rank_lists(rt, it, predict(mdl, Xt, kind), len(tix), T)
f = write("f_rerank_v2", T, tix, Pt)
H = T.matrix().tocsr()[tix]
fam = np.mean([H[r, j] > 0 for r, row in enumerate(Pt) for j in row])
p(f"[v2] test: candidates {nct:.1f}/user; distinct ids {len(np.unique(Pt))}; familiar share {fam:.4f}; {f.name} sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")
json.dump({"selected": {"k": k, "objective": kind, "trees": trees}, "map": results[best][0],
           "all": {f"k={a} {b} trees={c}": v[0] for (a, b, c), v in results.items()}}, open(RESULTS / "rerank_v2.json", "w"), indent=1)
p(f"total {time.time() - t0:.0f}s")
(RESULTS / "rerank_v2.txt").write_text("\n".join(out) + "\n")
