"""Single source of truth for the report: validation Top-5 of every method at its validation-selected configuration,
bootstrap intervals, the two example users, the test predictions and the Kaggle submission files.
Reads the selected configurations from results/val_<method>_best.json (written by run_val.py).
Run: python3 -I src/final.py  -> results/dump.json, results/final_output.txt, results/val_top5_<m>.csv,
     results/val_target.csv, results/examples.txt, submissions/NN_<method>.csv"""
import sys, pathlib, time, json, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from sklearn.model_selection import GroupKFold
from common import RESULTS, load_meta, load_test_users
from pipeline import Split, top5, apk
import methods as M
import rerank as R
from submission import write

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
t0 = time.time()
meta = load_meta()
V = Split("val", 18, (19, 30), quiet=True)
T = Split("test", 30, (31, 42), quiet=True)
ev = V.eval_users(); actual = [V.target[u] for u in ev]
tu, _ = load_test_users(); tix = np.searchsorted(T.users, tu)
dump = {"val": {}, "test": {}, "join_val_history": V.join_counts, "join_test_history": T.join_counts}


def accounting(split, uix, tag):
    """Disjoint categories: >=1 watched airing (screen time >= 0.8) | labelled contact only below 0.8 | log rows but
    no labelled airing touched | not in the log at all."""
    watched = np.isin(uix, np.unique(split.h_u)); contact = np.isin(uix, np.unique(split.a_u))
    inlog = np.isin(split.users[uix], np.unique(split.L_hist["user"]))
    c = {"users": int(len(uix)), "watched>=1": int(watched.sum()), "contact_below_0.8_only": int((contact & ~watched).sum()),
         "log_but_no_labelled_contact": int((inlog & ~contact).sum()), "not_in_history_log": int((~inlog).sum())}
    assert c["watched>=1"] + c["contact_below_0.8_only"] + c["log_but_no_labelled_contact"] + c["not_in_history_log"] == c["users"]
    p(f"[{tag}] cohort accounting: {c}")
    return c

dump["val_accounting"] = accounting(V, ev, "val"); dump["test_accounting"] = accounting(T, tix, "test")
dump["val_target_ties"] = V.target_ties
p(f"[val] cohort {len(ev):,} users with >= 5 target programmes; 5th/6th-place frequency ties {V.target_ties:,}; "
  f"pool {len(V.pool):,}; test pool {len(T.pool):,}")


def scores(m, split, uix, cfg):
    if m == "popularity":
        return np.tile(M.popularity(split), (len(uix), 1))
    if m == "content":
        return M.content(split, uix, meta, **cfg)
    return getattr(M, m)(split, uix, **cfg)


def finish(split, S, b):
    air = np.zeros(len(split.items)); air[split.pool_ix] = split.pool_airings
    return top5(np.clip(S, 0, None) * (air[None, :] ** b if b else 1.0), split.pool_ix, M.fallback_order(split))


def save_val_lists(name, P):
    with open(RESULTS / f"val_top5_{name}.csv", "w") as f:
        f.write("user_id,top5\n")
        for u, row in zip(ev, P):
            f.write(f"{V.users[u]},{' '.join(str(x) for x in V.items[row])}\n")

with open(RESULTS / "val_target.csv", "w") as f:
    f.write("user_id,top5\n")
    for u in ev:
        f.write(f"{V.users[u]},{' '.join(str(x) for x in V.target[u])}\n")

ORDER = ["popularity", "personal", "formula", "content", "als", "ppr"]
lists_v, lists_t, keep_v, keep_t, aps = {}, {}, {}, {}, {}
for m in ORDER:
    sel = {"cfg": {}, "b": 0.0} if m == "popularity" else json.load(open(RESULTS / f"val_{m}_best.json"))
    t1 = time.time(); Sv = scores(m, V, ev, sel["cfg"]); fit_v = time.time() - t1
    Pv = finish(V, Sv, sel["b"]); lists_v[m] = Pv
    ap = np.array([apk(a, V.items[row].tolist()) for a, row in zip(actual, Pv)]); aps[m] = ap
    save_val_lists(m, Pv)
    t1 = time.time(); St = scores(m, T, tix, sel["cfg"]); fit_t = time.time() - t1
    Pt = finish(T, St, sel["b"]); lists_t[m] = Pt
    f = write(f"f_{m}", T, tix, Pt)
    if m in ("ppr", "als", "content"):
        keep_v[m], keep_t[m] = Sv.astype(np.float32), St.astype(np.float32)
    dump["val"][m] = {"cfg": sel["cfg"], "b": sel["b"], "map": float(ap.mean()), "n": int(len(ap)), "score_seconds": round(fit_v, 1)}
    dump["test"][m] = {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "score_seconds": round(fit_t, 1)}
    p(f"[val] {m:<10} cfg={json.dumps(sel['cfg'])} b={sel['b']} MAP@5 {ap.mean():.5f} (n={len(ap)}) | val {fit_v:.1f}s, test {fit_t:.1f}s -> {f.name}")

# hybrid: candidate generation + learned re-ranking (out-of-fold on validation, refit on all validation pairs for test)
t1 = time.time()
rv, iv, Xv, ncv = R.build(V, ev, keep_v, meta)
y = R.labels(V, ev, rv, iv)
prob = np.zeros(len(y))
for tr, te in GroupKFold(n_splits=5).split(Xv, y, groups=rv):
    prob[te] = R.model().fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
Pv = R.rank_lists(rv, iv, prob, len(ev), V); lists_v["rerank"] = Pv
ap = np.array([apk(a, V.items[row].tolist()) for a, row in zip(actual, Pv)]); aps["rerank"] = ap
save_val_lists("rerank", Pv)
mdl = R.model().fit(Xv, y)
rt, it, Xt, nct = R.build(T, tix, keep_t, meta)
Pt = R.rank_lists(rt, it, mdl.predict_proba(Xt)[:, 1], len(tix), T); lists_t["rerank"] = Pt
f = write("f_rerank", T, tix, Pt)
dump["val"]["rerank"] = {"cfg": {"model": "HistGradientBoostingClassifier", "max_iter": 300, "learning_rate": 0.05,
                                 "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0, "seed": 42,
                                 "features": R.FEATS, "folds": "GroupKFold(5) by user"},
                         "map": float(ap.mean()), "n": int(len(ap)), "candidates_per_user": float(ncv),
                         "candidate_pairs": int(len(y)), "candidate_positives": int(y.sum()),
                         "candidate_recall": float(y.sum() / (5 * len(ev))), "seconds": round(time.time() - t1, 1)}
dump["test"]["rerank"] = {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                          "candidates_per_user": float(nct), "candidate_pairs": int(len(rt))}
p(f"[val] rerank     out-of-fold MAP@5 {ap.mean():.5f} (n={len(ap)}); candidates {ncv:.1f}/user, recall {y.sum() / (5 * len(ev)):.4f}; "
  f"test candidates {nct:.1f}/user -> {f.name} [{time.time() - t1:.0f}s]")

# bootstrap over users (1,000 resamples, seed 0): 95% interval of each MAP@5 and of the paired difference to rerank
rng = np.random.default_rng(0); idx = rng.integers(0, len(ev), (1000, len(ev)))
for m, ap in aps.items():
    bs = ap[idx].mean(axis=1); d = (aps["rerank"] - ap)[idx].mean(axis=1)
    dump["val"][m]["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    dump["val"][m]["rerank_minus_this_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    dump["val"][m]["users_with_hit"] = int((ap > 0).sum())
    p(f"[boot] {m:<10} MAP@5 {ap.mean():.5f} 95% CI [{np.percentile(bs, 2.5):.5f}, {np.percentile(bs, 97.5):.5f}]; "
      f"rerank - {m}: [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}]; users with >= 1 hit {int((ap > 0).sum()):,}")

# offline platform quantities on the test lists: catalogue coverage, share of slots already watched (familiar)
H = T.matrix().tocsr()[tix]
for m, Pt in lists_t.items():
    fam = np.mean([H[r, j] > 0 for r, row in enumerate(Pt) for j in row])
    dump["test"][m]["distinct_ids"] = int(len(np.unique(Pt))); dump["test"][m]["familiar_share"] = float(fam)
    p(f"[test] {m:<10} distinct ids in 6,300 slots {len(np.unique(Pt)):,}; share of slots already watched by that user {fam:.4f}")

# two example users (validation setting, so their actual weeks 19-30 top-5 can be shown)
hist_n = np.bincount(V.h_u, minlength=len(V.users))
heavy = int(ev[np.argmax(hist_n[ev])])
light_cands = ev[(hist_n[ev] > 0)]
light = int(light_cands[np.lexsort((V.users[light_cands], hist_n[light_cands]))][0])
title = meta.groupby("tv_show_id").tv_show_title.agg(lambda s: s.value_counts().index[0])
catg = meta.groupby("tv_show_id").tv_show_category.first()
ex_lines, dump["examples"] = [], {}
for tag, u in [("heavy", heavy), ("light", light)]:
    r = int(np.flatnonzero(ev == u)[0])
    hu = V.h_show[V.h_u == u]; hs, hc = np.unique(hu, return_counts=True)
    ex = {"user_id": str(V.users[u]), "history_watched_airings": int(hist_n[u]), "history_programmes": int(len(hs)),
          "actual_val_top5": V.target[u], "lists": {}}
    ex_lines.append(f"== {tag} user {V.users[u]}: {hist_n[u]} watched airings of {len(hs)} programmes in weeks 11-18 ==")
    ex_lines.append("  actual weeks 19-30 top-5: " + "; ".join(f"{i} {title.get(str(i), '?')[:40]} [{catg.get(str(i), '?')}]" for i in V.target[u]))
    for m, Pv in lists_v.items():
        ids = V.items[Pv[r]].tolist(); ex["lists"][m] = ids
        hits = [i in set(V.target[u]) for i in ids]
        ex_lines.append(f"  {m:<10} AP@5 {apk(V.target[u], ids):.4f}: " + "; ".join(
            f"{'*' if h else ''}{i} {title.get(str(i), '?')[:32]} [{catg.get(str(i), '?')}]" for i, h in zip(ids, hits)))
    dump["examples"][tag] = ex
(RESULTS / "examples.txt").write_text("\n".join(ex_lines) + "\n")
for l in ex_lines:
    p(l)
dump["seconds_total"] = round(time.time() - t0, 1)
json.dump(dump, open(RESULTS / "dump.json", "w"), indent=1, default=int)
p(f"total {time.time() - t0:.0f}s")
(RESULTS / "final_output.txt").write_text("\n".join(out) + "\n")
