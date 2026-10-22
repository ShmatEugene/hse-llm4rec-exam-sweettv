"""Interval join of the view log with the broadcast schedule (vectorised NumPy).

Rule (follows the organisers' description of how the hidden test answer was built, see
results/kaggle_pages.txt): a view row (user, channel, [start, stop)) is matched to every airing
(channel, [start, start+duration)) on the same channel whose interval overlaps it; overlap seconds are
summed per (user, airing); screen time = watched seconds / airing duration; an airing counts as
"watched" when screen time >= 0.8. Only labelled airings (tv_show_id != 0) become items. The strength of
a (user, programme) pair is the number of watched airings of that programme.

Run: python3 -I src/join.py   (prints step counts, writes results/join_output.txt)"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import load_log, load_sched, week_of, SCHED_FILES, RESULTS, CACHE

THRESH = 0.8
K = np.int64(10**10)  # channel * K + seconds: one sortable key per (channel, time)


def schedule():
    """All three files concatenated and sorted by (channel, start). Returns dict of arrays."""
    parts = [load_sched(p) for p in SCHED_FILES]
    S = {k: np.concatenate([s[k] for s in parts]) for k in ["ch", "start", "dur", "show"]}
    S["part"] = np.concatenate([np.full(len(s["ch"]), i, np.int8) for i, s in enumerate(parts)])
    o = np.lexsort((S["start"], S["ch"]))
    S = {k: v[o] for k, v in S.items()}
    S["end"] = S["start"] + S["dur"]
    S["week"] = week_of(S["start"])
    return S


def join(L, S, clip_stop=None, log=print):
    """Returns (user_idx, airing_idx, watched_seconds) aggregated per (user, labelled airing), plus counts.
    clip_stop: views are cut at this naive-seconds time (views starting at/after it are dropped)."""
    start, stop = L["start"], L["stop"]
    keep = np.ones(len(start), bool)
    if clip_stop is not None:
        keep = start < clip_stop
        stop = np.minimum(stop, clip_stop)
    c = {}
    c["log rows"] = int(len(start))
    c["log rows used (before clip boundary)"] = int(keep.sum())
    ks = S["ch"].astype(np.int64) * K + S["start"]
    ke = S["ch"].astype(np.int64) * K + S["end"]
    # contiguous candidate range requires ends to be non-decreasing in (channel, start) order
    nondec = np.all(np.diff(ke) >= 0)
    if not nondec:  # the schedule has a nested airing; fall back to a running max of ends (still exact for range start)
        ke_run = np.maximum.accumulate(ke)
    else:
        ke_run = ke
    vs = L["ch"].astype(np.int64) * K + start
    ve = L["ch"].astype(np.int64) * K + stop
    chan_ok = np.isin(L["ch"], np.unique(S["ch"])) & keep
    c["log rows on a schedule channel"] = int(chan_ok.sum())
    lo = np.searchsorted(ke_run, vs, side="right")          # first airing whose end > view start
    hi = np.searchsorted(ks, ve, side="left") - 1           # last airing whose start < view stop
    cnt = np.where(chan_ok, np.maximum(hi - lo + 1, 0), 0)
    c["log rows overlapping >=1 airing"] = int((cnt > 0).sum())
    tot = int(cnt.sum())
    c["(view, airing) candidate pairs"] = tot
    rows = np.repeat(np.arange(len(start)), cnt)
    off = np.repeat(np.cumsum(cnt) - cnt, cnt)
    air = lo[rows] + (np.arange(tot) - off)
    ov = np.minimum(stop[rows], S["end"][air]) - np.maximum(start[rows], S["start"][air])
    pos = ov > 0  # false only for nested-airing fallback candidates
    rows, air, ov = rows[pos], air[pos], ov[pos]
    c["(view, airing) pairs with overlap > 0"] = int(len(ov))
    lab = S["show"][air] != 0
    c["  of which on a labelled airing (tv_show_id != 0)"] = int(lab.sum())
    c["  of which on an unlabelled airing (tv_show_id = 0)"] = int((~lab).sum())
    c["log rows touching >=1 labelled airing"] = int(len(np.unique(rows[lab])))
    rows, air, ov = rows[lab], air[lab], ov[lab]
    users, uinv = np.unique(L["user"], return_inverse=True)
    key = uinv[rows].astype(np.int64) * len(S["ch"]) + air
    ukey, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, weights=ov).astype(np.int64)
    ui, ai = (ukey // len(S["ch"])).astype(np.int32), (ukey % len(S["ch"])).astype(np.int64)
    c["(user, labelled airing) pairs"] = int(len(ukey))
    st = w / S["dur"][ai]
    c[f"(user, labelled airing) pairs with screen time >= {THRESH}"] = int((st >= THRESH).sum())
    c["  of which screen time > 1 (overlapping views of one user)"] = int((st > 1).sum())
    for k, v in c.items():
        log(f"{k}: {v:,}")
    return users, ui, ai, w, c


if __name__ == "__main__":
    out = []
    def p(*a):
        s = " ".join(str(x) for x in a); print(s); out.append(s)
    t0 = time.time()
    L = load_log(); S = schedule()
    p(f"schedule airings {len(S['ch']):,}; labelled {int((S['show'] != 0).sum()):,}; ends non-decreasing within channel: "
      f"{bool(np.all(np.diff(S['ch'].astype(np.int64) * K + S['end']) >= 0))}")
    users, ui, ai, w, c = join(L, S, log=p)
    st = w / S["dur"][ai]
    good = st >= THRESH
    pairs = np.unique(ui[good].astype(np.int64) * 10**8 + S["show"][ai[good]])
    p(f"distinct (user, programme) pairs with >=1 watched airing: {len(pairs):,}; users with >=1: {len(np.unique(ui[good])):,} of {len(users):,}; "
      f"programmes: {len(np.unique(S['show'][ai[good]])):,}")
    # distribution evidence for the threshold choice: screen-time histogram over (user, labelled airing) pairs
    h, e = np.histogram(np.minimum(st, 1.0), bins=10, range=(0, 1.0000001))
    p("screen-time histogram (deciles, capped at 1):", h.tolist())
    p("watched seconds per (user, labelled airing) p50/p90:", np.percentile(w, [50, 90]).astype(int).tolist(),
      "; airing duration of labelled airings p10/p50/p90 s:", np.percentile(S["dur"][S["show"] != 0], [10, 50, 90]).astype(int).tolist())
    np.savez(CACHE / "join_full.npz", users=users, ui=ui, ai=ai, w=w)
    p(f"join time {time.time() - t0:.1f}s")
    (RESULTS / "join_output.txt").write_text("\n".join(out) + "\n")
