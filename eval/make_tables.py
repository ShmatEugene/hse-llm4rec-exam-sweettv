"""Generates report/tables.typ from results (no number is typed by hand): the Kaggle submission table (refs and scores
parsed from results/kaggle_submissions.txt, validation MAP@5 from results/dump.json / rerank_v2.json) and the two
example users' Top-5 lists with id, title and category from the schedule.
Run: python3 -I eval/make_tables.py"""
import json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import load_meta

R = ROOT / "results"
dump = json.load(open(R / "dump.json"))
v2 = json.load(open(R / "rerank_v2.json")) if (R / "rerank_v2.json").exists() else None
val = {m: d["map"] for m, d in dump["val"].items()}
if v2:
    val["rerank_v2"] = v2["map"]
rows = []
for line in (R / "kaggle_submissions.txt").read_text().splitlines():
    m = re.match(r"\s*(\d{8})\s+(\S+\.csv)\s+.*?SubmissionStatus\.(\w+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
    if m:
        rows.append(m.groups())
rows.sort()
name = lambda f: re.sub(r"^\d+_", "", f[:-4])
esc = lambda s: s.replace("\\", "").replace("*", "\\*").replace("#", "\\#").replace("@", "\\@").replace("$", "\\$").replace("_", "\\_").replace("[", "(").replace("]", ")").replace("<", "").replace(">", "")
t = ["#let subtable = table(columns: 6, table.header([Ref], [File], [Method], [Val MAP\\@5], [Public], [Private]),"]
for ref, f, st, pub, prv in rows:
    m = name(f)
    t.append(f"  [{ref}], [{esc(f)}], [{esc(m)}], [{val[m]:.5f}], [{pub}], [{prv}],")
t.append(")")
meta = load_meta()
title = meta.groupby("tv_show_id").tv_show_title.agg(lambda s: s.value_counts().index[0])
cat = meta.groupby("tv_show_id").tv_show_category.first()
short = lambda i: esc(title.get(str(i), "?").strip(".").replace("М/с ", "").replace("Т/с ", "").replace("Х/ф ", "")[:26])
import csv
v2lists = {r["user_id"]: [int(x) for x in r["top5"].split()] for r in csv.DictReader(open(R / "val_top5_rerank_v2.csv"))} \
    if (R / "val_top5_rerank_v2.csv").exists() else {}
sys.path.insert(0, str(ROOT / "src")); from pipeline import apk
ex_log = []
for tag, ex in dump["examples"].items():
    act = set(ex["actual_val_top5"])
    if ex["user_id"] in v2lists:
        ex["lists"]["rerank_v2"] = v2lists[ex["user_id"]]
    ex_log.append(f"{tag} user {ex['user_id']}: " + ", ".join(f"{m} AP@5 {apk(ex['actual_val_top5'], ids):.4f}" for m, ids in ex["lists"].items()))
    t.append(f"#let ex_{tag} = table(columns: 6, table.header([Method], [\\#1], [\\#2], [\\#3], [\\#4], [\\#5]),")
    t.append("  [*actual 19--30*], " + ", ".join(f"[{i} {short(i)} ({cat.get(str(i), '?')})]" for i in ex["actual_val_top5"]) + ",")
    for m, ids in ex["lists"].items():
        t.append(f"  [{esc(m)}], " + ", ".join(f"[{'*' if i in act else ''}{i}{'*' if i in act else ''} {short(i)} ({cat.get(str(i), '?')})]" for i in ids) + ",")
    t.append(")")
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report" / "tables.typ").write_text("\n".join(t) + "\n")
(R / "examples_ap.txt").write_text("\n".join(ex_log) + "\n")
print(f"wrote report/tables.typ: {len(rows)} submissions, examples {list(dump['examples'])}")
