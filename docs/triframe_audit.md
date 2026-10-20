# TriFrame audit (2026-10-21): the s120 / v3 run, before v4

Scope: a source-grounded review of the v3 pipeline and its 120-stock run (`s120`), written before any v4 change, as section 0 of
the external review brief asks. Every statement points to code or to a run artifact; anything that cannot be established is marked
**unknown**.

## 1. Pipeline map
| Step | Code | What it does |
|---|---|---|
| Build the notebook | `make_notebook.py` | Assembles `TriFrame_CV_EndToEnd.ipynb` from `cells/*.py` and the modules below |
| Data access | `scripts/triframe_loader.py` (`ImageStore`, in the private dataset) | The only reader of charts, features and labels; seals rows dated 2023-01-03 onward (`holdout_guard`) |
| Shared inference | `tfcv_core.py` | Live data, the 33 features (`feat_visible`), the renderer (`tf_render`), the bundle, the NO TRADE rule, the Monte Carlo |
| Training | `tfcv_train.py` | Fast chart reader, embeddings, fine-tuning, PCA, heads, metrics, month-block bootstrap |
| Stages | `cells/10..99` | 1 load, 2 split, 3 numeric baselines, 4 embeddings, 5 PCA + heads, 5F fine-tuning, 6 select + freeze, 6L leaderboard, 7 test, 7M Monte Carlo, 8 bundle, 9 dashboard files, 10 holdout, 11 report |
| Dashboard | `streamlit_app.py`, `dashboard_logic.py`, `dashboard_charts.py` | Loads the frozen bundle from the private model repo |

## 2. Targets (exact formulas)
| Target | Formula (code) | Notes |
|---|---|---|
| `logrv20` (V) | `ln sqrt( sum_{k=t+1..t+20} r_k^2 )`, `r_k = close_k / close_{k-1} - 1` (`scripts/tf_data.py`, `labels`) | Log of 20-day realised volatility, close-to-close, split-adjusted, **not** dividend-adjusted, not annualised. Annualised volatility = `exp(logrv20) * sqrt(252/20)` (`tfcv_core.ann_vol`). |
| `dd20_le_10` (D10) | `min(low_{t+1..t+20}) / open_{t+1} - 1 <= -0.10` (`scripts/derive_labels.py`) | Intraday lows; reference price = next session's open. |
| `y_out5` (O5) | `ret5 > median(ret5)` over the **point-in-time eligible stocks of the whole 979-stock universe** on that date (needs >= 100 eligible) (`DATA_CONTRACT.md` 2.2) | `ret5 = (close_{t+5} + dividends with ex-dates t+2..t+5) / open_{t+1} - 1`. Not the median of the 120 sampled stocks. |
| `ret20` | as `ret5` with h = 20 | Used only by the Monte Carlo check (Stage 7M). |
Features: 33 `x_*` columns computed from the last 126 daily bars ending at the decision close `t` (`feat_visible`, spec v1.2); entry is at
the open of `t+1`. Stage 8 checks that the dashboard's own code reproduces the training features and charts exactly (0 mismatches on 40 test rows).

## 3. Splits and sample (s120)
A row joins a split only if its decision date **and** its 20-session label end (`end20`) fall inside the split (`cells/20_stage2.py`); one
purge serves all targets because 20 >= 5. Splits are by date, so the **same stocks** appear in every split (no stock-held-out test).
| Split | Dates | Chart sets | Stocks | D10 rate | O5 rate |
|---|---|---|---|---|---|
| train | 2001-10-22 .. 2016-12-01 | 38,937 | 117 | 0.219 | 0.500 |
| validation | 2017-01-03 .. 2019-12-02 | 8,496 | 118 | 0.176 | 0.502 |
| test | 2020-01-02 .. 2022-12-01 | 8,666 | 120 | 0.293 | 0.506 |
| purged | | 465 | | | |
Rows per date: median 13 stocks (max 20) with 120 stocks; median 57 per calendar week. Data: 979 US stocks that exist today
(survivor universe), one vendor, Yahoo used to fill gaps (`final/` source column); live dashboard inputs come from Yahoo.

## 4. Holdout and test-set status
`governance/holdout_access_log.csv` (hash-chained) shows **two openings** of 2023-01-03 onward, each returning 89,542 rows:
| UTC | Who | Frozen config SHA-256 |
|---|---|---|
| 2026-10-19 20:08 | `tfcv_holdout_quick` (60-stock pilot) | `d0d6c71a...` |
| 2026-10-20 12:42 | `tfcv_holdout_s120` | `a3a2e66e...` |
Both openings came after their run's choices were frozen, so no choice in either run was tuned on the holdout. But the period has now
been viewed twice; from v4 on it is **exposed** and is treated as development data, not as an untouched test. The 2020-2022 test years
were read by both runs (including two 15-model leaderboards), so they are a **historical diagnostic set**, not a fresh selection set.
The pilot's notebook computed the unlock key itself instead of the owner pasting it (a deviation recorded earlier).

## 5. Selection logic (v3)
Preprocessing (median fill, standardisation, PCA, image scalers) is fitted on training rows only (`cells/30_stage3.py`, `T.pca_fit_transform`,
`img_matrix`). Ridge alpha (17 values), logistic C (6 values) and 4 tree settings with early stopping are chosen on validation
(`fit_one`). The pre-registered rule (`cells/60_stage6.py`) picks the best validation score per target; an image model wins only if its
gain over the best numbers-only model has a month-block 95% CI above zero. Isotonic calibration and the NO TRADE thresholds are fitted on validation.
**Finding:** the fine-tuning in Stage 5F used the same 2017-2019 validation years for early stopping that Stage 6 then used to choose
between frozen and fine-tuned models. That makes the fine-tuned candidates' validation scores optimistic, and is a plausible part of
why the chosen model's validation gain (+0.041 R2, CI +0.025..+0.061) shrank to +0.003 (CI -0.016..+0.027) on the test years
(the 2020 crash is the other obvious candidate). v4 early-stops fine-tuning on an inner validation slice (2015-2016) instead.

## 6. Metric provenance and the discrepancies in the review brief
| Metric | Summary | Leaderboard | Cause (from code) | v4 |
|---|---|---|---|---|
| V QLIKE | 0.5398 | 0.5055 | The summary (`eval_target`) uses the bias-correction variance `s2` from **validation** residuals (correct, pre-registered). The leaderboard (`T.board_metrics`, `s2=None`) used `var(y - p)` of the **test** residuals: a test-label quantity inside the metric. **Bug** (metric only, not the model). | `s2` always from the model's validation residuals |
| O5 / D10 AUC, ECE, BSS, log loss | 0.5075 / 0.6625 / ... | 0.5090 / 0.6637 / ... | Same model, same 8,666 rows. The summary scores **isotonic-calibrated** probabilities (calibrator fitted on validation); the leaderboard scores the **uncalibrated** refit. Isotonic maps are step functions, so ties change AUC slightly, and calibration changes ECE / BSS / log loss. Both BSS use the training base rate. | Every metric row carries `probability_calibrated` |
| D10 accuracy ~0.71 | | 0.71 | Accuracy at a 0.5 threshold. Test prevalence is 0.293, so "never an event" scores 0.707: accuracy is uninformative here. AP 0.448 = 1.53x prevalence. | Precision / recall / F1 at thresholds chosen on validation, with prevalence |
| O5 rank IC | NaN | | `rank_ic` groups by exact date and needs >= 20 stocks; the 120-stock sample has a median of 13 per date. | Weekly cross-sections, HAC t-statistic |
| CIs | | | Month-block bootstrap: calendar months resampled with replacement, all their rows kept; 1,000 resamples for test summaries, 400 for the validation leaderboard; percentile 2.5-97.5; paired (same resamples) for differences. | Adds a quarter-block sensitivity for headline deltas |
| "Survivorship sensitivity" | | | Refit without 2001-2009 training rows: a **training-history** sensitivity, not a survivorship test (no delisted stocks exist in the data). | Renamed |
| Monte Carlo low_k | 1.0 | | The chosen intraday-range factor sat at the edge of its grid (0.25..1.0). | Grid extended to 2.0 |

## 7. Ranked issues
1. **Exposed holdout and inspected test years** (section 4): no further selection may use them.
2. **Fine-tuning early stopping on the selection years** (section 5): optimistic validation scores for fine-tuned models.
3. **Leaderboard QLIKE on test residual variance** (section 6): a metric bug.
4. **Unlabelled calibration status** across tables (section 6).
5. **Rank IC not computable** at 120 stocks per date (section 6).
6. **Selection among many candidates** (1,690 configurations, 87 test rows): test leaderboards are descriptive only.
7. **Survivor universe and single vendor:** risk may be understated for stocks that later fail; not fixable without delisted data.
8. **Shared multi-task encoder** for all three targets: one representation may not suit all (v4 tests a risk-only variant).
9. **In-sample image features for the heads:** a fine-tuned encoder has seen its own training rows, so heads trained on those rows may
   over-trust image features; validation measures this honestly, but cross-fitting would be needed to remove it (not done: cost).
10. **Renderer:** each chart window is min-max scaled in log price, so pictures cannot show the volatility level (by design).
    The hourly chart ends at the 15:30 bar (closes 16:00), i.e. at the decision close, so it uses no later information.

## 8. Reproduce
Open `TriFrame_CV_EndToEnd.ipynb` (v3, commit `fd2a054`) in Colab, set `QUICK_TICKERS = 120`, Run all. Artifacts of the s120 run are in
the private runs repo under `results_s120/`, `frozen/frozen_config_s120.json`, `ft_s120/` and the model repo `triframe-cv-model-s120`.
**Unknown from the repository alone:** the original vendor's corporate-action handling beyond "split-adjusted, not dividend-adjusted",
and point-in-time index membership (not in the data).
