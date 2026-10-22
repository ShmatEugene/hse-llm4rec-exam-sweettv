"""Inputs for the student's three-rung hand check on one validation user (personal re-watch method, half-life 16 weeks,
b = 0.25). Prints inputs only; `--answers` prints the code's values afterwards for comparison.
 (a) join of one raw view-log row with one airing: overlap seconds, screen time, >= 0.8 decision;
 (b) the personal score of that programme: sum over watched airings of 0.5^(age/16), times (target airings)^0.25;
 (c) AP@5 of the user's printed Top-5 against the actual weeks 19-30 top-5 set.
Run: python3 -I eval/hand.py [--answers]   -> results/hand_inputs.txt / results/hand_answers.txt"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd
from common import RESULTS, load_log
from pipeline import Split, top5, apk
import methods as M

ans = "--answers" in sys.argv
L = load_log()
V = Split("val", 18, (19, 30), L=L, quiet=True)
ev = V.eval_users()
sel = json.load(open(RESULTS / "val_personal_best.json"))
S = M.personal(V, ev, **sel["cfg"])
air = np.zeros(len(V.items)); air[V.pool_ix] = V.pool_airings
Sb = np.clip(S, 0, None) * air[None, :] ** sel["b"]
P = top5(Sb, V.pool_ix, M.fallback_order(V))
ts = lambda t: str(pd.to_datetime(int(t), unit="s"))
Sd = V.S
for r in np.argsort(V.users[ev]):           # deterministic scan by user id
    u = int(ev[r]); ids = V.items[P[r]].tolist(); a = apk(V.target[u], ids)
    hits = sum(i in set(V.target[u]) for i in ids)
    if not (2 <= hits <= 4):
        continue
    top = ids[0]
    m = (V.h_u == u) & (V.h_show == top)
    if not (1 <= m.sum() <= 3):
        continue
    # one log row of this user that alone covers >= 80% of one watched airing of `top`
    t0 = V.h_t[m][0]; ch = V.h_ch[m][0]
    k = np.flatnonzero((Sd["ch"] == ch) & (Sd["start"] == t0))[0]
    a0, d = int(Sd["start"][k]), int(Sd["dur"][k])
    lr = np.flatnonzero((L["user"] == V.users[u]) & (L["ch"] == ch) & (L["start"] < a0 + d) & (L["stop"] > a0))
    if len(lr) != 1:
        continue
    v = lr[0]
    break
ages = (V.hist_end - V.h_t[m]) / (7 * 86400)
lines = [f"User {V.users[u]} (validation cohort; history = weeks 11-18, target = weeks 19-30)",
         f"(a) view-log row: channel {ch}, start {ts(L['start'][v])}, stop {ts(L['stop'][v])}; "
         f"airing of programme {top} on channel {ch}: start {ts(a0)}, duration {d} s. "
         f"Compute overlap seconds = min(stops) - max(starts), screen time = overlap / duration, and whether it is >= 0.8.",
         f"(b) programme {top}: the user's watched airings in history have ages (weeks before Monday of week 19) "
         f"{[round(float(x), 4) for x in ages]}; half-life 16 weeks; airings of {top} in weeks 19-30: {int(air[V.item_ix[top]])}. "
         f"Compute score = (sum of 0.5^(age/16)) x airings^0.25.",
         f"(c) printed Top-5 (in order): {ids}; actual weeks 19-30 top-5 set: {sorted(V.target[u])}. "
         f"Compute AP@5 = (sum of precision@k at each hit position k) / 5."]
for l in lines:
    print(l)
(RESULTS / "hand_inputs.txt").write_text("\n".join(lines) + "\n")
if ans:
    ov = min(int(L["stop"][v]), a0 + d) - max(int(L["start"][v]), a0)
    a_lines = [f"(a) overlap {ov} s, screen time {ov / d:.4f}, watched (>= 0.8): {ov / d >= 0.8}",
               f"(b) sum of decays {float((0.5 ** (ages / 16)).sum()):.4f}; airings^0.25 {air[V.item_ix[top]] ** 0.25:.4f}; "
               f"score {float(Sb[r, V.item_ix[top]]):.4f}",
               f"(c) hits at positions {[k + 1 for k, i in enumerate(ids) if i in set(V.target[u])]}; AP@5 {apk(V.target[u], ids):.4f}"]
    for l in a_lines:
        print(l)
    (RESULTS / "hand_answers.txt").write_text("\n".join(a_lines) + "\n")
