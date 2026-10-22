"""Quick test submission for one validation-selected method (personal or ppr), used before final.py ran on Kaggle.
Run: python3 -I src/sub_quick.py <method> <NN>"""
import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import load_test_users, RESULTS
from pipeline import Split, top5
import methods as M
from submission import write
m, nn = sys.argv[1], sys.argv[2]
sel = json.load(open(RESULTS / f"val_{m}_best.json"))
T = Split("test", 30, (31, 42), quiet=True)
tu, _ = load_test_users(); uix = np.searchsorted(T.users, tu)
S = getattr(M, m)(T, uix, **sel["cfg"])
air = np.zeros(len(T.items)); air[T.pool_ix] = T.pool_airings
P = top5(np.clip(S, 0, None) * air[None, :] ** sel["b"], T.pool_ix, M.fallback_order(T))
f = write(f"{nn}_{m}", T, uix, P)
print(f"[test] {m} cfg={sel['cfg']} b={sel['b']} (validation MAP@5 {sel['map']:.5f}) -> {f.name}")
