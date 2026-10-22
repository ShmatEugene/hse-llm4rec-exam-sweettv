"""Scoring methods. Each returns a dense (n_eval_users, n_items) score array over the Split's item space;
pipeline.top5 turns any of them into exactly five pool ids with the same popularity fallback."""
import numpy as np
import scipy.sparse as sp


def popularity(split):
    """Number of distinct history users with >= 1 watched airing of the programme (same list for everyone)."""
    B = split.matrix(binary=True)
    return np.asarray(B.sum(axis=0)).ravel()


def fallback_order(split):
    """Pool item indices by popularity DESC, ties by programme id ASC. Shared fallback for every method."""
    pop = popularity(split)[split.pool_ix]
    o = np.lexsort((split.pool, -pop))
    return split.pool_ix[o]


def personal(split, uix, half_life=None):
    """Re-watch heuristic: the user's own (optionally recency-decayed) watched-airing counts."""
    return split.matrix(half_life).tocsr()[uix].toarray()


def als(split, uix, factors=32, reg=0.1, alpha=10.0, iters=15, half_life=None, seed=42, conf="linear"):
    """Implicit-feedback MF (Hu, Koren, Volinsky 2008): preference p_ui = 1[r_ui > 0], confidence
    c_ui = 1 + alpha * r_ui (or 1 + alpha * log1p(r_ui) with conf='log'); alternating exact least squares with the
    YtY + Yt(C_u - I)Y trick. r_ui = (decayed) watched-airing count. Returns X[uix] @ Y.T."""
    R = split.matrix(half_life).tocsr()
    R.eliminate_zeros()
    Rt = R.T.tocsr()
    rng = np.random.default_rng(seed)
    nU, nI = R.shape
    X = rng.normal(0, 0.01, (nU, factors)); Y = rng.normal(0, 0.01, (nI, factors))
    cf = (lambda r: alpha * r) if conf == "linear" else (lambda r: alpha * np.log1p(r))
    I = reg * np.eye(factors)

    def solve(A, Fixed):
        G = Fixed.T @ Fixed
        out = np.zeros((A.shape[0], factors))
        for u in range(A.shape[0]):
            s, e = A.indptr[u], A.indptr[u + 1]
            if s == e:
                continue
            idx = A.indices[s:e]; cm1 = cf(A.data[s:e])      # c - 1
            Yu = Fixed[idx]
            M = G + (Yu.T * cm1) @ Yu + I
            out[u] = np.linalg.solve(M, Yu.T @ (cm1 + 1.0))  # p = 1 on observed entries
        return out

    for _ in range(iters):
        X = solve(R, Y)
        Y = solve(Rt, X)
    return X[uix] @ Y.T


def ppr(split, uix, alpha=0.5, iters=30, half_life=None, tol=1e-9):
    """Personalized PageRank on the user-programme bipartite graph (edge weight = (decayed) watched-airing count),
    restart at the target user's node. Written in item space: an even-length walk item -> user -> item has transition
    T = D_I^-1 R^T D_U^-1 R; one bipartite step pair is p <- (1-alpha) r + alpha * p T with r = the user's first
    step (R row normalised). Returns item-node mass (users x items). Dangling items (no history edge) keep no mass;
    the mass that reaches them is returned to the restart vector so every row sums to 1."""
    R = split.matrix(half_life).tocsr()
    du = np.asarray(R.sum(axis=1)).ravel(); di = np.asarray(R.sum(axis=0)).ravel()
    Du = sp.diags(np.where(du > 0, 1 / np.maximum(du, 1e-300), 0)); Di = sp.diags(np.where(di > 0, 1 / np.maximum(di, 1e-300), 0))
    Pu = (Du @ R).tocsr()          # user -> item
    Pi = (Di @ R.T).tocsr()        # item -> user
    r = Pu[uix].toarray()
    p = r.copy()
    for it in range(iters):
        q = (Pi.T @ p.T).T         # item mass -> users
        q = (Pu.T @ q.T).T         # users -> items
        lost = 1.0 - q.sum(axis=1, keepdims=True)  # mass that sat on dangling items
        nxt = (1 - alpha) * r + alpha * (q + lost * r)
        d = np.abs(nxt - p).sum(axis=1).max()
        p = nxt
        if d < tol:
            break
    split.ppr_last = {"iters": it + 1, "max_l1_residual": float(d), "row_sum_min": float(p.sum(axis=1).min()),
                      "row_sum_max": float(p.sum(axis=1).max())}
    return p


def content_vectors(split, meta):
    """TF-IDF item vectors over category, genres, director tokens, actor tokens and production decade, L2-normalised.
    Built from the schedule metadata of each programme (most frequent value per field)."""
    from collections import Counter
    m = meta[meta.tv_show_id.astype(np.int64).isin(split.items)]
    feats = {}
    for sid, g in m.groupby("tv_show_id"):
        toks = set()
        row = {c: g[c].mode().iat[0] for c in ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
                                               "tv_show_genre_3", "year_of_production", "director", "actors"]}
        toks.add("cat=" + row["tv_show_category"])
        for c in ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]:
            if row[c]:
                toks.add("gen=" + row[c])
        y = row["year_of_production"][:4]
        if y.isdigit():
            toks.add("dec=" + y[:3])
        for c, pre in [("director", "dir="), ("actors", "act=")]:
            for t in row[c].split(","):
                t = t.strip()
                if t:
                    toks.add(pre + t)
        feats[int(sid)] = toks
    vocab = {t: j for j, t in enumerate(sorted({t for s in feats.values() for t in s}))}
    rows, cols = [], []
    for sid, toks in feats.items():
        i = split.item_ix[sid]
        for t in toks:
            rows.append(i); cols.append(vocab[t])
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(split.items), len(vocab)))
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    X = X @ sp.diags(np.log(len(split.items) / np.maximum(df, 1)) + 1.0)
    n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sp.diags(np.where(n > 0, 1 / np.maximum(n, 1e-300), 0)) @ X
    return X.tocsr(), vocab


def content(split, uix, meta, half_life=None, X=None):
    """Profile = (decayed) count-weighted sum of the user's item vectors; score = cosine(profile, item)."""
    if X is None:
        X, _ = content_vectors(split, meta)
    R = split.matrix(half_life).tocsr()[uix]
    P = (R @ X).toarray()
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    return (X @ P.T).T


def formula(split, uix, half_life=None, rating="soft"):
    """Hand-built personal score suggested by the user: count(>=0.8) x mean view-rating x recency x log1p(popularity).
    view_rating = 5 if screen time >= 0.99 else 2 + 2 * st^2 ('soft') or 2 + 9 * st^2 ('hard'), averaged over all of the
    user's views of the programme (partial ones included); recency = 0.5 ** (weeks since last watched airing / half_life)."""
    st = split.a_st
    vr = np.where(st >= 0.99, 5.0, 2 + (2 if rating == "soft" else 9) * st ** 2)
    ii = np.searchsorted(split.items, split.a_show)
    shape = (len(split.users), len(split.items))
    n_all = sp.csr_matrix((np.ones(len(st)), (split.a_u, ii)), shape=shape).tocsr()[uix].toarray()
    vr_sum = sp.csr_matrix((vr, (split.a_u, ii)), shape=shape).tocsr()[uix].toarray()
    c80 = split.matrix().tocsr()[uix].toarray()
    S = c80 * (vr_sum / np.maximum(n_all, 1))
    if half_life:
        age = (split.hist_end - split.a_t) / (7 * 86400)
        m = st >= 0.8
        rec = sp.csr_matrix((0.5 ** (age[m] / half_life), (split.a_u[m], ii[m])), shape=shape).tocsr()[uix]
        rec = rec.tocsr(); rec.data = np.minimum(rec.data, 1.0)  # duplicates summed; keep the max-like cap at 1
        S = S * rec.toarray()
    pop = np.asarray(split.matrix(binary=True).sum(axis=0)).ravel()
    return S * np.log1p(pop)[None, :]
