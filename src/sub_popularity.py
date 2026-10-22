"""Submission 1: popularity baseline on the test setting (history weeks 11-30, pool = programmes airing weeks 31-42).
Run: python3 -I src/sub_popularity.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import load_test_users
from pipeline import Split, top5
import methods as M
from submission import write
T = Split("test", 30, (31, 42), quiet=True)
tu, _ = load_test_users()
uix = np.searchsorted(T.users, tu)
fb = M.fallback_order(T)
P = top5(np.tile(M.popularity(T), (len(uix), 1)), T.pool_ix, fb)
f = write("01_popularity", T, uix, P)
print(f"[test] pool {len(T.pool):,} programmes airing weeks 31-42; history programmes {len(np.unique(T.h_show)):,}; "
      f"test users with history {np.isin(uix, np.unique(T.h_u)).sum():,} of {len(uix):,}")
print("[test] popularity top-5:", T.items[P[0]].tolist(), "->", f.name)
