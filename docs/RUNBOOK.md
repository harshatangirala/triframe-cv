# TriFrame CV: how to run the end-to-end notebook (v4)

**File:** `TriFrame_CV_EndToEnd.ipynb`. It is self-contained: it writes its own code modules and needs no other files.

## 1. One-time setup (about 5 minutes)
1. Open https://colab.research.google.com, then File → Upload notebook → `TriFrame_CV_EndToEnd.ipynb`.
2. Runtime → Change runtime type → **A100 GPU** and **High-RAM**. A T4 would take about three times as long (too long for 500 stocks).
3. Click the key icon (Secrets) and add `HF_TOKEN`: a Hugging Face token with **write** access; switch on "Notebook access".

## 2. The v4 run: QUICK = True, 500 stocks (about 6-8 hours on an A100)
- Leave the config cell as it is (`QUICK_TICKERS = 500`) and choose Runtime → Run all. Accept the Google Drive pop-up.
- The 500 stocks contain the pilot's 60 and v3's 120 (same seeds, same draw order), so Stage 7 compares all three runs on the same rows.
  Results go to `results_s500/` in the runs repo and the model to its own repo, `triframe-cv-model-s500`; the live dashboard keeps its
  current model until you switch it (`docs/DEPLOY.md`).
- To save about 1.5 hours, set `FT_STUDY = False` (skips the controlled fine-tuning study). `LANDSCAPE = False` saves a few minutes.
- Everything long is resumable: embeddings are saved chunk by chunk and every fine-tuning run is saved when it finishes, so after a
  disconnect just Run all again. The heads (Stage 5) are refitted after a restart (about an hour for 500 stocks).

### Lines to check
| Stage | What to look for |
|---|---|
| 0 | `Metric unit tests passed`; the holdout log lists the two earlier openings and says **EXPOSED**. |
| 1 | `500 used in this run, containing the 60-stock and 120-stock samples of earlier runs`. |
| 3 | The baselines table (HAR, ROLL22, EWMA, GARCH, NUM) and `GARCH(1,1) forecasts for ...% of rows`. |
| 5F | One line per epoch per network; the study table; the seeds line; frozen vs fine-tuned with CIs; learning-curve figures. |
| 6 | For each target the chosen model, and for image models the gain **by validation year** (`not consistent across years` = rejected). |
| 6D | `GLOBAL minimum reached (certified)` for the ridge and logistic heads; boosting curves; learning curves; the loss landscape. |
| 6L | The validation leaderboard with precision, recall and F1. |
| 7 | Test metrics (historical diagnostic), the quarter-block CI check, and `each earlier run's model vs this ... model` on the same rows. |
| 7E | ROC / precision-recall / confusion / reliability figures and the metrics table. |
| 7D | The direction ranking verdict. |
| 8 | `Bundle vs notebook ...: rows > 1e-3: 0.00%`, `feature mismatches 0, chart mismatches 0`, cosine similarity above 0.995. |
| 11 | `PDF report: ...` (also downloaded to your computer and saved in Google Drive under `My Drive/triframe_cv/`). |

## 3. What v4 does differently, and why
- **Selection years stay clean.** Fine-tuning now early-stops on 2015-2016 (inside the training years) instead of the 2017-2019
  validation years, which v3 used both to stop fine-tuning and to choose between models.
- **Stricter image rule.** An image model must beat the numbers-only model on validation with a CI above zero *and* in each of the
  three validation years.
- **Test years and 2023+ are labelled honestly.** Both were already looked at by earlier runs, so they are a historical diagnostic and
  an exposed period. Nothing in v4 is chosen with them; Stage 10 stays off by default.
- **Global minimum.** Certified for the convex heads; for the networks the notebook shows convergence, the held-out minimum, the loss
  landscape and seed stability. No method can certify a global minimum for a neural network.

## 4. Decisions that are yours
| Decision | Default | Notes |
|---|---|---|
| Open the 2023+ period (Stage 10) | off | It has been opened twice already; any result is a "previously viewed period", not a holdout. |
| Fine-tuning study | on | `FT_STUDY = False` saves about 1.5 hours. |
| Live data source | Yahoo Finance | Yahoo's terms limit commercial use. |

## 5. After the run, send back
`REPORT_s500.md`, `TriFrame_CV_report_s500.pdf` and `metrics_provenance.csv` (Drive: `My Drive/triframe_cv/`, or the runs repo),
and the notebook with its outputs, for review.
