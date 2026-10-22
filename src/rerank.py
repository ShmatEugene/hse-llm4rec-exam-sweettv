"""Hybrid: candidate generation + learned re-ranking.

Candidates per user (pool programmes only): every programme the user touched in history (any screen time > 0),
plus the top-30 by PPR, the top-30 by popularity and the top-30 by channel exposure. Features (grouped for ablation) are computed from history
and from the target-window schedule only. A gradient-boosted classifier (scikit-learn HistGradientBoosting,
fixed hyper-parameters, seed 42) predicts "programme is in the user's target top-5"; the list is the top-5 by
predicted probability, padded by the shared popularity fallback.

Fitting protocol. Labels come from the validation target (weeks 19-30, built from history weeks 11-18). On
validation the re-ranker is scored out-of-fold: 5 folds over USERS, so no user is scored by a model that saw that
user's labels. For the test, the model is fitted on all validation pairs (weeks <= 30 only, i.e. training data
from the test's point of view) and applied to test features (history 11-30, schedule 31-42). Counts are divided by
the number of history weeks (8 vs 20) so the two settings share a scale."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.sparse as sp
from common import week_start, load_meta
import methods as M

WEEK = 7 * 86400
GROUPS = {
    "counts": ["c80_pw", "rate80", "exp80"],
    "quality": ["cpart_pw", "cany_pw", "sst_pw", "c100_pw", "sec_pw"],
    "recency": ["c80_d4_pw", "c80_last2", "age_last"],
    "popularity": ["pop_users", "pop_air"],
    "schedule": ["tgt_air", "hist_air", "tgt_dur", "tgt_nch"],
    "habits": ["aff_ch", "aff_hr", "ucat", "expo"],
    "user": ["u_c80_pw", "u_nshow"],
    "methods": ["ppr", "ppr_rank", "als", "content"],
}
FEATS = [f for g in GROUPS.values() for f in g]


def _um(split, mask, vals):
    return sp.csr_matrix((vals[mask], (split.a_u[mask], np.searchsorted(split.items, split.a_show[mask]))),
                         shape=(len(split.users), len(split.items)))


def build(split, uix, scores, meta, k_ppr=30, k_pop=30, k_ch=30):
    """Returns (rows, items, X[n, len(FEATS)]) for candidate pairs of users uix. scores: dict with dense
    (len(uix), n_items) arrays 'ppr', 'als', 'content'."""
    S = split.S
    H = (split.hist_end - week_start(11)) / WEEK                       # history length in weeks
    st, t = split.a_st, split.a_t
    one = np.ones(len(st))
    age = (split.hist_end - t) / WEEK
    m80 = st >= 0.8
    mats = {
        "c80": _um(split, m80, one), "cpart": _um(split, (st >= 0.2) & ~m80, one), "cany": _um(split, st > 0, one),
        "sst": _um(split, st > 0, st), "c100": _um(split, st >= 0.99, one), "sec": _um(split, st > 0, split.a_sec.astype(float)),
        "c80_d4": _um(split, m80, 0.5 ** (age / 4)), "c80_last2": _um(split, m80 & (age <= 2), one),
    }
    mats = {k: v.tocsr()[uix] for k, v in mats.items()}
    # most recent watched airing per (user, item): age in weeks (H + 1 if never)
    nI = len(split.items)
    ii = np.searchsorted(split.items, split.a_show)
    key = split.a_u.astype(np.int64) * nI + ii
    km = key[m80]; am = age[m80]
    oo = np.lexsort((am, km)); km, am = km[oo], am[oo]
    first = np.r_[True, km[1:] != km[:-1]]
    lastage = dict(zip(km[first].tolist(), am[first].tolist()))
    # item features
    hist_rows = (S["week"] >= 11) & (S["start"] < split.hist_end) & (S["show"] != 0)
    hist_air = np.zeros(nI); hi_ = np.searchsorted(split.items, S["show"][hist_rows])
    ok = split.items[np.minimum(hi_, nI - 1)] == S["show"][hist_rows]
    np.add.at(hist_air, hi_[ok], 1)
    tw = (S["week"] >= split.target_weeks[0]) & (S["week"] <= split.target_weeks[1]) & (S["show"] != 0)
    ti = np.searchsorted(split.items, S["show"][tw])
    tgt_air = np.zeros(nI); np.add.at(tgt_air, ti, 1)
    tgt_dur = np.zeros(nI); np.add.at(tgt_dur, ti, S["dur"][tw]); tgt_dur = tgt_dur / np.maximum(tgt_air, 1)
    chs = np.unique(S["ch"]); ci = np.searchsorted(chs, S["ch"][tw])
    IC = sp.csr_matrix((np.ones(tw.sum()), (ti, ci)), shape=(nI, len(chs)))   # target airings per (item, channel)
    tgt_nch = np.asarray((IC > 0).sum(axis=1)).ravel()
    IH = sp.csr_matrix((np.ones(tw.sum()), (ti, (S["start"][tw] % 86400) // 3600)), shape=(nI, 24))
    B = split.matrix(binary=True)
    pop_users = np.asarray(B.sum(axis=0)).ravel() / max(1, len(np.unique(split.h_u)))
    pop_air = np.asarray(split.matrix().sum(axis=0)).ravel() / H
    # user habits from the raw log of the history window (all airtime, labelled or not)
    Lh = split.L_hist
    lu = np.searchsorted(split.users, Lh["user"])
    lc = np.searchsorted(chs, Lh["ch"]); lok = chs[np.minimum(lc, len(chs) - 1)] == Lh["ch"]
    UC = sp.csr_matrix((Lh["dur"][lok].astype(float), (lu[lok], lc[lok])), shape=(len(split.users), len(chs))).tocsr()[uix]
    UC = sp.diags(1 / np.maximum(np.asarray(UC.sum(axis=1)).ravel(), 1)) @ UC
    UH = sp.csr_matrix((np.minimum(Lh["dur"], 3600).astype(float), (lu, (Lh["start"] % 86400) // 3600)),
                       shape=(len(split.users), 24)).tocsr()[uix]
    UH = sp.diags(1 / np.maximum(np.asarray(UH.sum(axis=1)).ravel(), 1)) @ UH
    cat = meta.drop_duplicates("tv_show_id").set_index("tv_show_id").tv_show_category
    cats = sorted(cat.unique()); cmap = {c: j for j, c in enumerate(cats)}
    icat = np.array([cmap.get(cat.get(str(s), ""), -1) for s in split.items])
    UCat = (mats["c80"] @ sp.csr_matrix((np.ones((icat >= 0).sum()), (np.flatnonzero(icat >= 0), icat[icat >= 0])),
                                         shape=(nI, len(cats)))).toarray()
    UCat = UCat / np.maximum(UCat.sum(axis=1, keepdims=True), 1)
    # candidates
    pool = np.zeros(nI, bool); pool[split.pool_ix] = True
    cand = (mats["cany"].toarray() > 0) & pool[None, :]
    pp = np.where(pool[None, :], scores["ppr"], -np.inf)
    top_ppr = np.argpartition(-pp, k_ppr, axis=1)[:, :k_ppr]
    cand[np.repeat(np.arange(len(uix)), k_ppr), top_ppr.ravel()] = True
    fb = M.fallback_order(split)[:k_pop]
    cand[:, fb] = True
    # channel-exposure candidates: target-window airings weighted by the user's channel shares (new programmes on
    # the channels the user already watches)
    expo = np.where(pool[None, :], (UC @ IC.T).toarray(), -np.inf)
    top_ch = np.argpartition(-expo, k_ch, axis=1)[:, :k_ch]
    cand[np.repeat(np.arange(len(uix)), k_ch), top_ch.ravel()] = True
    r, i = np.nonzero(cand)
    g = lambda Mx: np.asarray(Mx[r, i]).ravel()
    ppr_rank = np.argsort(np.argsort(-pp, axis=1), axis=1)
    F = {
        "c80_pw": g(mats["c80"]) / H, "rate80": g(mats["c80"]) / np.maximum(hist_air[i], 1),
        "cpart_pw": g(mats["cpart"]) / H, "cany_pw": g(mats["cany"]) / H, "sst_pw": g(mats["sst"]) / H,
        "c100_pw": g(mats["c100"]) / H, "sec_pw": g(mats["sec"]) / H,
        "c80_d4_pw": g(mats["c80_d4"]), "c80_last2": g(mats["c80_last2"]),
        "age_last": np.array([lastage.get(int(uix[a]) * nI + int(b), H + 1) for a, b in zip(r, i)]),
        "pop_users": pop_users[i], "pop_air": pop_air[i],
        "tgt_air": tgt_air[i], "hist_air": hist_air[i] / H, "tgt_dur": tgt_dur[i], "tgt_nch": tgt_nch[i],
        "ucat": UCat[r, np.maximum(icat[i], 0)] * (icat[i] >= 0),
        "u_c80_pw": np.asarray(mats["c80"].sum(axis=1)).ravel()[r] / H,
        "u_nshow": np.asarray((mats["c80"] > 0).sum(axis=1)).ravel()[r],
        "ppr": scores["ppr"][r, i], "ppr_rank": ppr_rank[r, i], "als": scores["als"][r, i], "content": scores["content"][r, i],
    }
    F["exp80"] = F["rate80"] * tgt_air[i]
    F["expo"] = expo[r, i]
    # channel affinity: expected share of the programme's target airings on channels the user watches
    ICn = sp.diags(1 / np.maximum(tgt_air, 1)) @ IC
    F["aff_ch"] = np.asarray(UC[r].multiply(ICn.tocsr()[i]).sum(axis=1)).ravel()
    IHn = sp.diags(1 / np.maximum(tgt_air, 1)) @ IH
    F["aff_hr"] = np.einsum("ij,ij->i", UH[r].toarray(), IHn[i].toarray())
    X = np.column_stack([F[f] for f in FEATS]).astype(np.float32)
    return r, i, X, cand.sum() / len(uix)


def labels(split, uix, r, i):
    tg = {u: set(v) for u, v in split.target.items()}
    return np.array([split.items[b] in tg[int(uix[a])] for a, b in zip(r, i)], dtype=np.int8)


def rank_lists(r, i, prob, n_users, split):
    """Top-5 item indices per user by predicted probability, padded with the shared fallback."""
    fb = M.fallback_order(split)
    o = np.lexsort((-prob, r))
    out = [[] for _ in range(n_users)]
    for a in o:
        if len(out[r[a]]) < 5:
            out[r[a]].append(i[a])
    for lst in out:
        for j in fb:
            if len(lst) == 5:
                break
            if j not in lst:
                lst.append(j)
    return np.array(out)


def model(seed=42):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                          l2_regularization=1.0, early_stopping=False, random_state=seed)
