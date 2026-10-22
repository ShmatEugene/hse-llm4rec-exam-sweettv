"""Independent checks of the metric and of the validation ground truth (standard library only, raw CSVs):
 1. MAP@5 written from the formula (sum over hit positions of precision@k, divided by min(|actual|, 5)),
    checked on toy cases with hand-known answers, including the all-zero template;
 2. the validation target (weeks 19-30, per-user top-5 by number of airings with screen time >= 0.8, ties by
    watched seconds then id) recomputed by a csv/bisect join and compared user by user with results/val_target.csv;
 3. every method's validation MAP@5 recomputed from its exported Top-5 file (results/val_top5_<method>.csv) and
    compared with the value in results/dump.json.
Run: python3 -I eval/verify.py   -> results/verify_output.txt"""
import csv, bisect, calendar, json, os, pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("SWEETTV_DATA", ROOT.parent / "data"))
RES = ROOT / "results"
out = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def ap5(actual, pred):
    actual = set(actual); hits = 0; total = 0.0; seen = set()
    for k, x in enumerate(pred[:5], start=1):
        if x in actual and x not in seen:
            hits += 1; total += hits / k
        seen.add(x)
    return total / min(len(actual), 5) if actual else 0.0

# 1. toy cases
toys = [([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
        ([1, 2, 3, 4, 5], [9, 1, 9, 2, 9], (1 / 2 + 2 / 4) / 5),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], 1.0),           # order inside the actual set does not matter
        ([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], 0.0),           # the template
        ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1], 1 / 5),          # duplicates are not re-counted
        ([7], [9, 7, 9, 9, 9], 1 / 2)]                      # min(|actual|, 5) = 1
for a, pr, want in toys:
    got = ap5(a, pr)
    p(f"[toy] actual={a} pred={pr} AP@5={got:.6f} expected={want:.6f} {'OK' if abs(got - want) < 1e-12 else 'MISMATCH'}")

# 2. validation target from raw CSVs
def ts(s, sched):
    if sched:
        return calendar.timegm((int(s[6:10]), int(s[3:5]), int(s[0:2]), int(s[11:13]), int(s[14:16]), int(s[17:19])))
    return calendar.timegm((int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]), int(s[17:19])))
W19 = calendar.timegm((2020, 5, 4, 0, 0, 0)); W31 = calendar.timegm((2020, 7, 27, 0, 0, 0))  # Mondays of weeks 19 and 31
chan = defaultdict(list)
for fn in ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv"]:
    with open(DATA / fn, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = ts(row["start_time"], True)
            chan[int(row["channel_id"])].append((s, s + int(row["duration"]), int(row["tv_show_id"])))
starts = {}
for c in chan:
    chan[c].sort(); starts[c] = [a[0] for a in chan[c]]
sec = defaultdict(int)
with open(DATA / "dataset11-30.csv", newline="") as f:
    for row in csv.DictReader(f):
        c = int(row["vsetv_id"])
        if c not in chan:
            continue
        s, e = ts(row["start_time"], False), ts(row["stop_time"], False)
        al, st = chan[c], starts[c]
        # every airing with start < e and end > s: scan back over airings that may still be running at s
        i = bisect.bisect_left(st, e) - 1
        while i >= 0 and al[i][1] > s - 86400 * 2:  # airings are <= 1 day long; scan a safe margin back
            a0, a1, show = al[i]
            if show and a1 > s and a0 < e and W19 <= a0 < W31:
                sec[(row["user_id"], c, a0)] += min(e, a1) - max(s, a0)
            if a1 <= s and a0 < s - 86400:
                break
            i -= 1
dur = {(c, a[0]): (a[1] - a[0], a[2]) for c in chan for a in chan[c]}
freq, tsec = defaultdict(int), defaultdict(int)
for (u, c, a0), w in sec.items():
    d, show = dur[(c, a0)]
    if w >= 0.8 * d:
        freq[(u, show)] += 1; tsec[(u, show)] += w
byu = defaultdict(list)
for (u, show), n in freq.items():
    byu[u].append((-n, -tsec[(u, show)], show))
target = {u: [x[2] for x in sorted(v)[:5]] for u, v in byu.items() if len(v) >= 5}
mine = {r["user_id"]: [int(x) for x in r["top5"].split()] for r in csv.DictReader(open(RES / "val_target.csv"))}
same = sum(1 for u in mine if set(mine[u]) == set(target.get(u, [])))
p(f"[target] stdlib cohort {len(target):,} vs pipeline {len(mine):,}; users only in one: {len(set(target) ^ set(mine))}; "
  f"identical top-5 sets: {same:,}; identical ordered lists: {sum(1 for u in mine if mine[u] == target.get(u))}")

# 3. MAP@5 of every exported validation Top-5 list, with the stdlib target and with this file's ap5
dump = json.load(open(RES / "dump.json"))
if (RES / "rerank_v2.json").exists():  # the designated final's configuration (round 2, src/rerank_v2.py)
    dump["val"]["rerank_v2"] = {"map": json.load(open(RES / "rerank_v2.json"))["map"]}
for m, v in dump["val"].items():
    f = RES / f"val_top5_{m}.csv"
    pred = {r["user_id"]: [int(x) for x in r["top5"].split()] for r in csv.DictReader(open(f))}
    aps = [ap5(target[u], pred[u]) for u in sorted(target)]
    val = sum(aps) / len(aps)
    p(f"[map] {m}: recomputed MAP@5 {val:.5f} (n={len(aps)}) vs dump {v['map']:.5f} -> {'OK' if abs(val - v['map']) < 5e-6 else 'MISMATCH'}")
(RES / "verify_output.txt").write_text("\n".join(out) + "\n")
