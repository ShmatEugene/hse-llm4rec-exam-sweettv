"""Locates the (view, airing) pairs counted by the NumPy join but skipped by the stdlib walk, which starts at the
airing with the latest start <= view start. Run: python3 -I eval/explain_join_diff.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd
from common import load_log
from join import schedule, K
L = load_log(); S = schedule()
ks = S["ch"].astype(np.int64) * K + S["start"]; ke = S["ch"].astype(np.int64) * K + S["end"]
vs = L["ch"].astype(np.int64) * K + L["start"]; ve = L["ch"].astype(np.int64) * K + L["stop"]
ok = np.isin(L["ch"], np.unique(S["ch"]))
lo = np.searchsorted(ke, vs, side="right"); first_std = np.searchsorted(ks, vs, side="right") - 1
hi = np.searchsorted(ks, ve, side="left") - 1
m = ok & (lo < first_std) & (lo <= hi)
extra = np.minimum(first_std, hi + 1) - lo
print("views where the NumPy range starts before the stdlib start:", int(m.sum()), "; skipped pairs:", int(extra[m].sum()))
for v in np.flatnonzero(m)[:12]:
    a = np.arange(lo[v], first_std[v] + 1)
    print("view ch", L["ch"][v], pd.to_datetime(L["start"][v], unit="s"), "->", pd.to_datetime(L["stop"][v], unit="s"), "| airings:",
          [(str(pd.to_datetime(S["start"][i], unit="s")), int(S["dur"][i]), int(S["show"][i]), int(S["part"][i])) for i in a])
