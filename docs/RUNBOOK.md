# TriFrame CV: how to run the end-to-end notebook (v3)

**File:** `TriFrame_CV_EndToEnd.ipynb`. It is self-contained: it writes its own code modules and needs no other files.

## 1. One-time setup (about 5 minutes)
1. Open https://colab.research.google.com, then File → Upload notebook → `TriFrame_CV_EndToEnd.ipynb`.
2. Runtime → Change runtime type → **A100 GPU**, and switch on **High-RAM**. A T4 also works, about 3× slower.
3. Click the key icon (Secrets) in the left bar and add:
   - `HF_TOKEN`: a Hugging Face token with **write** access. Use a new one; the two old tokens were leaked.
   - Switch on "Notebook access" for it.

## 2. The v3 run: QUICK = True, 120 stocks (about 2-3 hours on an A100)
- Leave the config cell as it is (`QUICK = True`, `QUICK_TICKERS = 120`) and choose Runtime → Run all. Accept the Google Drive pop-up.
- The 120 stocks are the pilot's 60 plus 60 more, so the comparison with the pilot is like for like. Its run tag is `s120`: results go
  to `results_s120/` in the runs repo and the model to its own repo, `triframe-cv-model-s120`, so the live dashboard keeps its current
  model until you switch it (see `docs/DEPLOY.md`).
- Lines to check in the output:
  - Stage 1: `Universe: 979 US stocks in the data (120 used in this run, including all 60 pilot stocks)`.
  - Stage 4: `Fast chart reader matches ImageStore.triplet ...` (the run stops if it does not) and the table of model sizes and speeds.
  - Stage 5F: one line per epoch per backbone, then the **frozen vs fine-tuned** table and a one-sentence verdict.
  - Stage 6L: the model cards and the **leaderboard** (six metrics per target) with three charts.
  - Stage 7: test results, the test leaderboard and `More data: 60-stock pilot model vs this 120-stock model ... (identical test rows: 4,358)`.
  - Stage 7M: the Monte Carlo check (a 90% band should hold about 90% of real outcomes).
  - Stage 8: `Bundle vs notebook test predictions ...`, `feature mismatches 0, chart mismatches 0` and the dashboard-embedding check.
  - Stage 9: **which models the dashboard uses**, and an example of its plain-language explanation.
  - Stage 11: `PDF report: ...`. The PDF downloads to your computer and is also saved in Google Drive under `My Drive/triframe_cv/`.
- Fine-tuning is resumable per backbone (weights are kept on Drive and in the private runs repo), and embeddings are saved chunk by
  chunk, so after a disconnect just Run all again.

## 3. Full run: QUICK = False (all 979 stocks, several hours)
- Set `QUICK = False` and Run all. The model goes to `triframe-cv-model-full`. Fine-tuning uses at most `FT_MAX_TRAIN_ROWS` training
  chart sets (60,000 by default); raise it if you have the GPU time.
- The full run stores about 20 GB of frozen embeddings plus about 14 GB of fine-tuned ones on Hugging Face. If that is too much, set
  `UPLOAD_EMBEDDINGS = False`; they then go to Google Drive (which then needs that space free).
- The test years (2020-2022) are read once per run. A second evaluation needs `os.environ['TFCV_TEST_REUSE_REASON'] = '<why>'`.
- The sealed holdout (Stage 10) stays off until you set `RUN_HOLDOUT = True` and paste the frozen config's SHA-256. Each run has its
  own frozen config; note that every opening is another look at the same 2023-2026 period.

## 4. Decisions that are yours
| Decision | Default | Notes |
|---|---|---|
| Open the 2023+ holdout (Stage 10) | off | Only when you are happy with the frozen config. Compute its SHA-256 yourself (`!sha256sum <path>`) and paste it when asked. Every opening is logged. |
| Fine-tuning | on, all 7 backbones, up to 4 epochs | `FINETUNE = False` skips it; `FT_BACKBONES = ["resnet18", ...]` limits it. |
| Live data source | Yahoo Finance (yfinance) | Yahoo's terms limit commercial use. For a public product, consider a licensed data API. |
| Directional calls | shown only if both validation and the test years show an edge | Otherwise the dashboard gives risk outputs and NO TRADE. |

## 5. Put the dashboard on krishnaharsha.com
1. Deploy the dashboard on Streamlit Community Cloud (`docs/DEPLOY.md`), which gives a public address such as `https://triframe-cv.streamlit.app`.
2. The project page on the site embeds it with `<iframe src=".../?embed=true">`.

## 6. What the dashboard does
1. You choose 1 to 4 companies and upload three TradingView screenshots of each (1W, 1D, 1H), or type the tickers.
2. It reads the ticker, timeframe and last-bar O/H/L/C from each screenshot (tested on 66 real screenshots: timeframe 100%, legend
   100%, true ticker among the candidates 66/66) and confirms the ticker by matching the legend to Yahoo bars, which also pins the
   analysis to the screenshot's date.
3. It rebuilds the exact training inputs (33 numbers and the three charts) and runs the frozen models.
4. For each company: expected volatility (gauge), the chance of a 10% drawdown (pie), the chance of beating the median stock (bar),
   the risk tier and why, a detailed plain-language explanation of every number, a summary table and a primary takeaway, and a Monte
   Carlo fan of 20-day price paths with the distribution of outcomes.
5. For several companies: a ranking from riskiest to safest, a risk map, side-by-side bars and side-by-side Monte Carlo bands.

## 7. After the run, send back
`REPORT_s120.md` and `TriFrame_CV_report_s120.pdf` (Drive: `My Drive/triframe_cv/`), and the notebook with its outputs, for review.
