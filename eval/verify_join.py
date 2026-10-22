"""Independent re-implementation of the join: standard-library csv + bisect, own time parsing, no NumPy/pandas,
reads the raw CSVs (not the .npz cache). Compares against the NumPy join (src/join.py, cache/join_full.npz):
step counters and every (user, programme) watched-airing frequency.
Run: python3 -I eval/verify_join.py   (about 1-2 min)"""
import csv, bisect, calendar, os, pathlib, sys, time
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("SWEETTV_DATA", ROOT.parent / "data"))
CACHE = pathlib.Path(os.environ.get("SWEETTV_CACHE", ROOT.parent / "cache"))
t0 = time.time()

def ts_sched(s):  # "09.03.2020 06:00:00"
    return calendar.timegm((int(s[6:10]), int(s[3:5]), int(s[0:2]), int(s[11:13]), int(s[14:16]), int(s[17:19])))

def ts_log(s):  # "2020-03-09 00:00:02"
    return calendar.timegm((int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]), int(s[17:19])))

chan = defaultdict(list)  # channel -> [(start, end, show)]
for fn in ["export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"]:
    with open(DATA / fn, newline="", encoding="utf-8") as f:
        r = csv.reader(f); h = next(r)
        ic, ist, idu, ish = h.index("channel_id"), h.index("start_time"), h.index("duration"), h.index("tv_show_id")
        for row in r:
            s = ts_sched(row[ist])
            chan[int(row[ic])].append((s, s + int(row[idu]), int(row[ish])))
starts, airs = {}, {}
for c, lst in chan.items():
    lst.sort()
    airs[c] = lst
    starts[c] = [a[0] for a in lst]

cnt = defaultdict(int)
watched = defaultdict(int)  # (user, channel, airing position) -> seconds
with open(DATA / "dataset11-30.csv", newline="") as f:
    r = csv.reader(f); next(r)
    for u, c, a, b, _ in r:
        cnt["log rows"] += 1
        c = int(c)
        if c not in airs:
            continue
        cnt["log rows on a schedule channel"] += 1
        s, e = ts_log(a), ts_log(b)
        st, al = starts[c], airs[c]
        i = max(bisect.bisect_right(st, s) - 1, 0)  # airing containing s (or the first one)
        hit = lab = False
        while i < len(al) and al[i][0] < e:
            ov = min(e, al[i][1]) - max(s, al[i][0])
            if ov > 0:
                hit = True
                cnt["(view, airing) pairs with overlap > 0"] += 1
                if al[i][2] != 0:
                    lab = True
                    watched[(u, c, i)] += ov
            i += 1
        cnt["log rows overlapping >=1 airing"] += hit
        cnt["log rows touching >=1 labelled airing"] += lab

cnt["(user, labelled airing) pairs"] = len(watched)
freq = defaultdict(int)
for (u, c, i), sec in watched.items():
    a = airs[c][i]
    if sec >= 0.8 * (a[1] - a[0]):
        cnt["(user, labelled airing) pairs with screen time >= 0.8"] += 1
        freq[(u, a[2])] += 1
cnt["distinct (user, programme) pairs with >=1 watched airing"] = len(freq)
print(f"[stdlib] done in {time.time() - t0:.1f}s")
for k, v in cnt.items():
    print(f"[stdlib] {k}: {v:,}")

# compare with the NumPy join output
import numpy as np  # only to read the other implementation's saved arrays
sys.path.insert(0, str(ROOT / "src"))
from join import schedule
z = np.load(CACHE / "join_full.npz"); S = schedule()
users, ui, ai, w = z["users"], z["ui"], z["ai"], z["w"]
good = w >= 0.8 * S["dur"][ai]
other = defaultdict(int)
for uu, sh in zip(users[ui[good]].tolist(), S["show"][ai[good]].tolist()):
    other[(str(uu), sh)] += 1
diff_keys = set(freq) ^ set(other)
diff_val = sum(1 for k in set(freq) & set(other) if freq[k] != other[k])
print(f"[compare] (user, programme) pairs: stdlib {len(freq):,} vs numpy {len(other):,}; keys in only one: {len(diff_keys)}; "
      f"same key, different frequency: {diff_val}; total watched airings stdlib {sum(freq.values()):,} vs numpy {sum(other.values()):,}")
