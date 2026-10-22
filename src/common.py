"""Shared loaders. Raw Kaggle CSVs are parsed once into compact NumPy arrays and cached as .npz
(cache dir is outside the repository). Times are kept as naive local wall-clock seconds
(int64, seconds since 1970-01-01 of the naive timestamp): both files use local time strings,
so no timezone conversion is applied (H1 in audit.py checks this on the data).
"""
import os, pathlib, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(os.environ.get("SWEETTV_DATA", ROOT.parent / "data"))
CACHE = pathlib.Path(os.environ.get("SWEETTV_CACHE", ROOT.parent / "cache"))
RESULTS = ROOT / "results"
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

SCHED_FILES = {"11-20": "export_arh_11-20-final.csv", "21-30": "export_arh_21-30-final.csv",
               "31-42": "export_arh_31-42-final.csv"}
WEEK11_MONDAY = pd.Timestamp("2020-03-09")  # ISO week 11 of 2020 starts on Monday 2020-03-09


def week_start(w):
    """Naive-seconds timestamp of Monday 00:00 of ISO week w (2020)."""
    return int((WEEK11_MONDAY + pd.Timedelta(weeks=w - 11)).value // 10**9)


def week_of(t):
    """ISO week number (2020) of naive-seconds timestamps t."""
    return 11 + (np.asarray(t) - week_start(11)) // (7 * 86400)


def _savez(f, **arrays):
    """Atomic cache write: parallel processes must never read a half-written file (EOFError seen on Kaggle v1)."""
    tmp = f.with_name(f.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays); os.replace(tmp, f)


def _secs(s, fmt):
    return (pd.to_datetime(s, format=fmt).astype("int64") // 10**9).to_numpy()


def load_log():
    f = CACHE / "log.npz"
    if not f.exists():
        t0 = time.time()
        d = pd.read_csv(DATA / "dataset11-30.csv", dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64"})
        _savez(f, user=d.user_id.to_numpy(), ch=d.vsetv_id.to_numpy(),
                 start=_secs(d.start_time, "%Y-%m-%d %H:%M:%S"), stop=_secs(d.stop_time, "%Y-%m-%d %H:%M:%S"),
                 dur=d.duraton.to_numpy())
        print(f"[cache] log parsed in {time.time() - t0:.1f}s")
    z = np.load(f)
    return {k: z[k] for k in z.files}


def load_sched(part):
    """part in SCHED_FILES. Returns arrays ch, start, dur, show (tv_show_id) plus row index in file."""
    f = CACHE / f"sched_{part}.npz"
    if not f.exists():
        t0 = time.time()
        d = pd.read_csv(DATA / SCHED_FILES[part], usecols=["channel_id", "start_time", "duration", "tv_show_id"],
                        dtype={"channel_id": "int32", "duration": "int64", "tv_show_id": "int64"})
        _savez(f, ch=d.channel_id.to_numpy(), start=_secs(d.start_time, "%d.%m.%Y %H:%M:%S"),
                 dur=d.duration.to_numpy(), show=d.tv_show_id.to_numpy())
        print(f"[cache] schedule {part} parsed in {time.time() - t0:.1f}s")
    z = np.load(f)
    return {k: z[k] for k in z.files}


def load_meta():
    """Programme metadata: one row per (tv_show_id, file) with the most frequent title/category/genres."""
    f = CACHE / "meta.pkl"
    if not f.exists():
        cols = ["tv_show_id", "tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
                "tv_show_genre_3", "year_of_production", "director", "actors", "channel_id"]
        parts = []
        for p, fn in SCHED_FILES.items():
            d = pd.read_csv(DATA / fn, usecols=cols, dtype=str, keep_default_na=False)
            d = d[d.tv_show_id != "0"]
            d["part"] = p
            parts.append(d)
        tmp = f.with_name(f.name + f".tmp{os.getpid()}")
        pd.concat(parts, ignore_index=True).to_pickle(tmp); os.replace(tmp, f)
    return pd.read_pickle(f)


def load_test_users():
    s = pd.read_csv(DATA / "submission.csv", dtype={"user_id": "uint64", "tv_show_id": str})
    return s.user_id.to_numpy(), s
