"""Shared protocol: splits, interaction matrices, candidate pool, ground truth, MAP@5, Top-5 extraction.

Validation split (chosen to mirror the test setting, 12 target weeks right after the history):
  history = view log clipped at Monday of week 19 (weeks 11-18), target = airings starting in weeks 19-30.
Test: history = whole log (weeks 11-30), target airings = schedule weeks 31-42 (used only for the pool).
Ground truth (validation), built exactly as the organisers describe theirs: per user, programmes ranked by the
number of airings with screen time >= 0.8 in the target window; the top 5 form the relevant set; only users
with >= 5 such programmes are evaluated (the competition says every test user has strictly 5)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.sparse as sp
from common import load_log, week_start, CACHE
from join import schedule, join, THRESH

K = 5


def apk(actual, predicted, k=K):
    """Kaggle / benhamner Metrics average precision at k (duplicates in predicted are not re-counted)."""
    predicted = list(predicted)[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0


def mapk(actual_lists, predicted_lists, k=K):
    return float(np.mean([apk(a, p, k) for a, p in zip(actual_lists, predicted_lists)]))


class Split:
    """Everything a method needs: history interactions, candidate pool, target (validation only)."""

    def __init__(self, name, hist_end_week, target_weeks, L=None, S=None, quiet=False):
        self.name = name
        L = L if L is not None else load_log()
        S = S if S is not None else schedule()
        self.S = S
        clip = week_start(hist_end_week + 1) if hist_end_week < 30 else None
        log = (lambda *a: None) if quiet else print
        users, ui, ai, w, self.join_counts = join(L, S, clip_stop=clip, log=log)
        good = w >= THRESH * S["dur"][ai]
        # all labelled (user, airing) pairs of the history, including partial views (re-ranker features)
        am = S["week"][ai] <= hist_end_week
        self.a_u, self.a_show, self.a_t = ui[am], S["show"][ai[am]], S["start"][ai[am]]
        self.a_st = np.minimum(w[am] / S["dur"][ai[am]], 1.0); self.a_sec = w[am]
        lm = L["start"] < (clip if clip is not None else np.iinfo(np.int64).max)
        self.L_hist = {k: L[k][lm] for k in ["user", "ch", "start"]}
        self.L_hist["dur"] = (np.minimum(L["stop"], clip) - L["start"])[lm] if clip is not None else L["dur"][lm]
        self.target_weeks = target_weeks
        # history: one row per watched airing (user, programme, airing start time, watched seconds)
        self.users = users                      # all log users (uint64), index space for ui
        self.h_u = ui[good]; self.h_show = S["show"][ai[good]]; self.h_t = S["start"][ai[good]]
        hist_mask = S["week"][ai[good]] <= hist_end_week
        self.h_u, self.h_show, self.h_t = self.h_u[hist_mask], self.h_show[hist_mask], self.h_t[hist_mask]
        self.h_ch = S["ch"][ai[good]][hist_mask]
        self.hist_end = week_start(hist_end_week + 1)
        # candidate pool: labelled programmes with >= 1 airing in the target window (schedule only)
        tw = (S["week"] >= target_weeks[0]) & (S["week"] <= target_weeks[1]) & (S["show"] != 0)
        self.pool, self.pool_airings = np.unique(S["show"][tw], return_counts=True)
        # item index space = programmes seen in history or in the pool
        self.items = np.unique(np.concatenate([self.h_show, self.pool]))
        self.item_ix = {int(s): i for i, s in enumerate(self.items)}
        self.h_i = np.searchsorted(self.items, self.h_show)
        self.pool_ix = np.searchsorted(self.items, self.pool)
        self.target = None
        if hist_end_week < 30:  # validation target from the full (unclipped) join, airings in the target window
            fu, fui, fai, fw, _ = join(L, S, log=lambda *a: None)
            g = (fw >= THRESH * S["dur"][fai]) & (S["week"][fai] >= target_weeks[0]) & (S["week"][fai] <= target_weeks[1])
            self.target_raw = (fui[g], S["show"][fai[g]], fw[g])
            self.target, self.target_ties = self._top5(*self.target_raw)

    def _top5(self, u, show, sec):
        """Per user: programmes by watched-airing frequency DESC; ties broken by total watched seconds DESC, then id ASC.
        Returns {user_idx: [5 ids]} for users with >= 5 programmes, and the count of users whose 5th/6th place tie on frequency."""
        key = u.astype(np.int64) * 10**9 + show
        k, inv = np.unique(key, return_inverse=True)
        f = np.bincount(inv); s = np.bincount(inv, weights=sec)
        uu, ss = k // 10**9, k % 10**9
        o = np.lexsort((ss, -s, -f, uu))
        out, ties = {}, 0
        b = np.flatnonzero(np.r_[True, uu[o][1:] != uu[o][:-1], True])
        for a, z in zip(b[:-1], b[1:]):
            idx = o[a:z]
            if len(idx) >= K:
                out[int(uu[idx[0]])] = [int(x) for x in ss[idx[:K]]]
                ties += int(len(idx) > K and f[idx[K - 1]] == f[idx[K]])
        return out, ties

    def matrix(self, half_life_weeks=None, binary=False):
        """User x item matrix of watched-airing counts, optionally exponentially decayed by age (in weeks)."""
        if half_life_weeks is None:
            v = np.ones(len(self.h_u))
        else:
            age = (self.hist_end - self.h_t) / (7 * 86400)
            v = 0.5 ** (age / half_life_weeks)
        M = sp.csr_matrix((v, (self.h_u, self.h_i)), shape=(len(self.users), len(self.items)))
        if binary:
            M.data[:] = 1.0
        return M

    def eval_users(self):
        return np.array(sorted(self.target)) if self.target is not None else None


def top5(scores, pool_ix, fallback_ix):
    """scores: (n_users, n_items) dense array. Ranks pool items only, DESC by score, ties by fallback order
    (fallback_ix = pool items in popularity order). Items with score <= 0 / NaN are never preferred over the fallback
    order: every list is padded from the fallback ranking, so each user gets exactly 5 distinct pool ids."""
    rank_fb = np.empty(scores.shape[1]); rank_fb[:] = np.inf
    rank_fb[fallback_ix] = np.arange(len(fallback_ix))
    sub = scores[:, pool_ix]
    sub = np.where(np.isfinite(sub) & (sub > 0), sub, 0.0)
    out = []
    for r in range(sub.shape[0]):
        o = np.lexsort((rank_fb[pool_ix], -sub[r]))
        chosen = [pool_ix[j] for j in o if sub[r, j] > 0][:K]
        for j in fallback_ix:
            if len(chosen) == K:
                break
            if j not in chosen:
                chosen.append(j)
        out.append(chosen)
    return np.array(out)
