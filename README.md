# sweet.tv TV-programme recommender — LLM4Rec final exam (HSE, AVITO cohort)

Kaggle community competition [`sweettv-tv-program-recommender`](https://www.kaggle.com/competitions/sweettv-tv-program-recommender):
predict, for each of 1,260 users, the five TV programmes (`tv_show_id`) they watch most often in weeks 31-42, scored by MAP@5.
Author: Eugene Shmat. Report: `eugene_shmat_exam_report.pdf` (submitted separately).

## Data (not redistributed)

Accept the competition rules with your own account, then

    kaggle competitions download -c sweettv-tv-program-recommender -p ../data && (cd ../data && unzip sweettv-tv-program-recommender.zip)

The code expects the CSVs in `../data` (override with `SWEETTV_DATA`) and caches parsed arrays in `../cache`
(`SWEETTV_CACHE`). Checksums of the files used: `results/data_sha256.txt`.

## Layout

| Path | What |
|---|---|
| `src/common.py` | loaders, `.npz` cache (atomic writes), ISO-week helpers |
| `src/audit.py` | data audit before any model (channel key, units, seams, ids, test users) |
| `src/join.py` | vectorised interval join: same channel, overlap summed per (user, airing), screen time >= 0.8 |
| `src/pipeline.py` | validation split (history weeks 11-18, target 19-30), pool, ground truth, MAP@5, Top-5 with shared fallback |
| `src/methods.py` | popularity, personal re-watch, user-suggested formula, content-based, implicit ALS, Personalized PageRank |
| `src/run_val.py` | validation grid per method (selection on validation only) |
| `src/rerank.py`, `src/run_rerank_val.py` | hybrid: candidate generation + gradient-boosted re-ranking, out-of-fold by user, ablations |
| `src/final.py` | single source of truth: all validation lists, bootstrap, examples, test predictions -> `results/dump.json` |
| `src/submission.py` | writes and validates Kaggle files (1,260 rows, 5 distinct pool ids) |
| `eval/verify_join.py` | independent join (stdlib `csv` + `bisect`) compared pair by pair with the NumPy join |
| `eval/verify.py` | independent MAP@5 (toy cases) and validation target recomputed from raw CSVs |
| `eval/hand.py` | inputs of the hand check (answers only with `--answers`) |
| `eval/check_numbers.py` | every number of the report PDF must occur in `results/` |
| `kaggle/make_kernel.py` | bundles `src/` into a private Kaggle script kernel (heavy runs were executed there) |
| `submissions/` | every file sent to Kaggle; `results/kaggle_submissions.txt` holds refs and returned scores |

## Reproduce

    python3 -I src/audit.py && python3 -I src/join.py && python3 -I eval/verify_join.py
    for m in personal ppr als content formula; do python3 -I src/run_val.py $m; done   # warm the cache first
    python3 -I src/run_rerank_val.py && python3 -I src/final.py && python3 -I eval/verify.py

The validation grids, the re-ranker and `final.py` need more than 8 GB of RAM; they were run as the private Kaggle
kernel `shmate/sweettv-exam-val` (4 CPUs) built by `kaggle/make_kernel.py`; its log is in `results/kaggle_val_*/`.
