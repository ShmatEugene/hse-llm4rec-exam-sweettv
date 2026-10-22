"""Data audit before any model (hypotheses H1-H8 of the plan). Writes results/audit_output.txt.
Run: python3 -I src/audit.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import load_log, load_sched, load_meta, load_test_users, week_of, week_start, SCHED_FILES, RESULTS

out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

L = load_log()
n = len(L["user"])
p("== view log ==")
p(f"rows {n:,}; distinct users {len(np.unique(L['user'])):,}; distinct vsetv_id {len(np.unique(L['ch']))}")
p(f"duraton == stop-start: {(L['stop'] - L['start'] == L['dur']).sum():,} of {n:,}; dur<=0: {(L['dur'] <= 0).sum():,}")
q = np.percentile(L["dur"], [50, 90, 99, 99.9, 100])
p("duraton s p50/p90/p99/p99.9/max:", [int(x) for x in q], f"; rows >4h: {(L['dur'] > 4*3600).sum():,}, >1d: {(L['dur'] > 86400).sum():,}")
p("start range:", pd.to_datetime(L["start"].min(), unit="s"), "->", pd.to_datetime(L["start"].max(), unit="s"),
  "; stop max:", pd.to_datetime(L["stop"].max(), unit="s"))
w = week_of(L["start"]); wk, wc = np.unique(w, return_counts=True)
p("log rows by ISO week:", dict(zip(wk.tolist(), wc.tolist())))
dup = pd.DataFrame({"u": L["user"], "c": L["ch"], "s": L["start"], "e": L["stop"]}).duplicated().sum()
p(f"exact duplicate log rows (user,ch,start,stop): {dup:,}")
h = (L["start"] % 86400) // 3600
p("log view starts by hour of day:", np.bincount(h, minlength=24).tolist())

S = {k: load_sched(k) for k in SCHED_FILES}
p("\n== schedule ==")
for k, s in S.items():
    lab = s["show"] != 0
    p(f"[{k}] rows {len(s['ch']):,}; labelled {lab.sum():,} ({lab.mean():.4f}); unlabelled {(~lab).sum():,}; "
      f"distinct labelled ids {len(np.unique(s['show'][lab])):,}; channels {len(np.unique(s['ch']))}")
    wk = week_of(s["start"])
    p(f"   start {pd.to_datetime(s['start'].min(), unit='s')} -> {pd.to_datetime(s['start'].max(), unit='s')}; "
      f"ISO weeks {wk.min()}-{wk.max()}; rows outside nominal weeks: {((wk < int(k[:2])) | (wk > int(k[3:]))).sum():,}")
    p(f"   duration p50/p99/max {np.percentile(s['dur'], [50, 99, 100]).astype(int).tolist()}; dur<=0 {(s['dur'] <= 0).sum():,}")
    o = np.lexsort((s["start"], s["ch"]))
    ch, st, du = s["ch"][o], s["start"][o], s["dur"][o]
    same = ch[1:] == ch[:-1]
    gap = st[1:][same] - (st[:-1][same] + du[:-1][same])
    gapm = st[1:][same] - (st[:-1][same] + 60 * du[:-1][same])
    p(f"   consecutive airings on a channel: {same.sum():,}; next_start - (start+duration [s]): ==0 {(gap == 0).mean():.4f}, "
      f">0 (hole) {(gap > 0).mean():.4f}, <0 (overlap) {(gap < 0).mean():.4f}; if duration were minutes: ==0 {(gapm == 0).mean():.4f}")
    p(f"   duplicate (channel,start) rows: {(same & (st[1:] == st[:-1])).sum():,}")

p("\n== cross-file ==")
keys = {k: set(zip(s["ch"].tolist(), s["start"].tolist())) for k, s in S.items()}
p("shared (channel,start) 11-20 & 21-30:", len(keys["11-20"] & keys["21-30"]), "; 21-30 & 31-42:", len(keys["21-30"] & keys["31-42"]))
ids = {k: set(np.unique(s["show"][s["show"] != 0]).tolist()) for k, s in S.items()}
p(f"labelled ids: train(11-30) {len(ids['11-20'] | ids['21-30']):,}; test-file {len(ids['31-42']):,}; "
  f"test-file ids also in train files {len(ids['31-42'] & (ids['11-20'] | ids['21-30'])):,}; "
  f"21-30 ids also in 11-20 {len(ids['21-30'] & ids['11-20']):,} of {len(ids['21-30']):,}")

p("\n== H1 channel key ==")
logch = set(np.unique(L["ch"]).tolist()); sch = set().union(*[set(np.unique(s["ch"]).tolist()) for s in S.values()])
p(f"log vsetv_id {len(logch)}; schedule channel_id {len(sch)}; intersection {len(logch & sch)}; "
  f"log rows whose vsetv_id is a schedule channel: {np.isin(L['ch'], list(sch)).sum():,} of {n:,}")
# Timezone / key test: if the clocks agree, views should start disproportionately near programme starts
# on the SAME channel. Count log starts within [0, 120) s after a programme start on that channel, for clock
# shifts of -3..+3 h, and for a control that pairs each view with a wrong channel (channel id + 1).
sall = {k: np.concatenate([S[f][k] for f in ["11-20", "21-30"]]) for k in ["ch", "start"]}
key = np.sort(sall["ch"].astype(np.int64) * 10**10 + sall["start"])
def near_start(ch, t):
    kk = ch.astype(np.int64) * 10**10 + t
    i = np.searchsorted(key, kk, side="right") - 1
    ok = i >= 0
    d = kk - key[np.maximum(i, 0)]
    return (ok & (d >= 0) & (d < 120)).mean()
for sh in [-3, -2, -1, 0, 1, 2, 3]:
    p(f"shift {sh:+d} h: share of views starting <120 s after a programme start on its channel: {near_start(L['ch'], L['start'] + sh*3600):.4f}")
p(f"control (channel id + 1, shift 0): {near_start(L['ch'] + 1, L['start']):.4f}")

p("\n== H8 programme id consistency (labelled rows, all three files) ==")
M = load_meta()
g = M.groupby("tv_show_id")
p(f"labelled rows {len(M):,}; ids {g.ngroups:,}; ids with >1 title {(g.tv_show_title.nunique() > 1).sum():,}; "
  f">1 category {(g.tv_show_category.nunique() > 1).sum():,}; >1 channel {(g.channel_id.nunique() > 1).sum():,}")
p("category counts (rows):", M.tv_show_category.value_counts().to_dict())
for c in ["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3", "year_of_production", "director", "actors"]:
    p(f"  non-empty {c}: {(M[c] != '').mean():.4f}")
ex = g.tv_show_title.nunique().sort_values(ascending=False).index[:3]
for i in ex:
    p(f"  example id {i}: titles {M[M.tv_show_id == i].tv_show_title.value_counts().head(3).to_dict()}")

p("\n== H6 test users / submission template ==")
tu, sub = load_test_users()
p(f"template rows {len(tu):,}; distinct {len(np.unique(tu)):,}; all rows '0 0 0 0 0': {(sub.tv_show_id == '0 0 0 0 0').all()}")
inlog = np.isin(tu, L["user"])
p(f"test users present in view log 11-30: {inlog.sum():,}; absent: {(~inlog).sum():,}")
m = np.isin(L["user"], tu)
p(f"log rows of test users {m.sum():,}; of all log users, share that are test users {len(np.unique(L['user'][m])) / len(np.unique(L['user'])):.4f}")
wk, wc = np.unique(w[m], return_counts=True); p("test-user log rows by week:", dict(zip(wk.tolist(), wc.tolist())))
last = pd.Series(w[m]).groupby(L["user"][m]).max()
p("test users by last active week:", last.value_counts().sort_index().to_dict())
(RESULTS / "audit_output.txt").write_text("\n".join(out) + "\n")
