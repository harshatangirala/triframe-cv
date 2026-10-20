# TriFrame CV: do stock chart images add anything to the numbers?

TriFrame tests whether pictures of price charts at three time scales (weekly, daily and hourly) tell a model anything about
a US stock's next 20 days of volatility, its drawdown risk, or its 5-day relative direction **beyond 33 ordinary numbers
computed from the same price history**. Seven frozen, ImageNet-pretrained vision backbones are compared with that numerical
baseline under chronological splits, month-block bootstrap confidence intervals, label-shuffle and random-weight controls,
and a sealed final holdout.

The short answer from the pilot run: **the numbers carry the signal.** Volatility and drawdown risk are predictable; chart
images do not add a significant improvement on the test years; and there is no detectable 5-day directional edge, so the
decision layer defaults to **NO TRADE**.

> **Scope of the results below.** They come from the 60-stock pilot: 60 randomly chosen stocks out of the 979-stock universe,
> 27,313 chart sets, with **frozen** backbones. Notebook **v3** (this version) repeats everything on 120 stocks (the pilot's 60 + 60
> more), adds fine-tuning, a model leaderboard, a Monte Carlo check and a PDF report; its results will replace these once it has run.

## What is new in v3

| | |
|---|---|
| 120 stocks | The sample contains the pilot's 60 stocks, and Stage 7 scores the pilot's model and the new one on the same 60 stocks and the same test rows, so "does more data help?" gets a like-for-like answer. |
| Fine-tuning | Until v2 every backbone was **frozen** (ImageNet features, only the small heads trained). Stage 5F fine-tunes all seven pretrained backbones end to end (one network over the three views + the 33 numbers, multi-task loss, training years only, early stopping on validation, no augmentation) and lets them compete with their frozen versions under the same selection rule, with confidence intervals and GPU time. |
| Leaderboard | Stage 6L: model cards (what each network is, pretraining data, size, speed) and every model on six metrics per target (R², within-month R², QLIKE, MAE, RMSE, Spearman; AUC, average precision, accuracy, log loss, Brier skill, calibration error) with bars, a heatmap and accuracy-vs-size charts; the same on the test years in Stage 7. |
| Monte Carlo | Stage 7M calibrates the dashboard's price simulation on validation (tail shape, intraday range) and checks it once on the test years. |
| PDF report | Stage 11 writes a 5-page plain-language report (steps, models, architecture, results, fine-tuning, simulation, dashboard) to Drive, Hugging Face and your Downloads. |
| Dashboard | Up to four companies at once; per company a gauge, a pie, bars, the risk tier, a detailed plain-language explanation, a summary table, a primary takeaway and a Monte Carlo fan; across companies a risk ranking, a risk map and side-by-side simulations; the models in use and (with a v3 bundle) how every candidate scored. |

## Results (pilot, 60 stocks)

| Target | Chosen model (pre-registered rule) | Validation 2017-2019 | Test 2020-2022 | Holdout 2023-2026 |
|---|---|---|---|---|
| 20-day volatility (log realised vol) | numbers + Swin-T chart features, boosted trees | R² 0.553 | R² 0.496 | R² 0.550 (95% CI 0.508-0.588) |
| Drawdown of 10% or more within 20 days | numbers + DINOv2-S chart features, boosted trees | AUC 0.711, Brier skill 0.095 | AUC 0.656, Brier skill 0.062 | not scored (no labels returned) |
| Beats the median stock over 5 days | numbers only, boosted trees | AUC 0.516 | AUC 0.510 (95% CI 0.498-0.522) | AUC 0.490 (95% CI 0.474-0.505) |

**Hypotheses** (rules written before the test years were read):

- **H1, chart images add to the numbers: not supported.** On the test years, numbers + all three chart views vs numbers
  alone: volatility ΔR² +0.004 (95% CI −0.004 to +0.012); direction ΔAUC −0.007 (−0.026 to +0.012). The volatility and
  drawdown models picked image features because they cleared the validation gate, but the gain did not hold up on the test
  years.
- **H2, backbones differ: supported**, weakly. 3 of 21 backbone pairs differ in image-only volatility R² after Holm
  correction.
- **H3, abstention improves accepted direction calls: not supported.** Accuracy gain at 10% coverage −0.005 (−0.047 to +0.037).

**Controls.** With shuffled labels the pipeline finds nothing: volatility R² −0.115 (within-month R² 0.005) and direction AUC
0.507. Randomly initialised ResNet-18s score volatility R² −0.027 to −0.038 from images alone.

**NO TRADE rule on the test years.** The risk flag covers 56% of cases. Realised volatility is 50.2% a year in flagged cases
vs 27.2% otherwise, and 10% drawdowns happen 37.1% vs 18.0% of the time. Directional calls are never shown: the direction
model's test AUC interval includes 0.5.

**Why images struggle here.** The chart renderer scales every window so its lowest low and highest high fill the panel. Two
price paths that differ only by a constant volatility factor therefore draw identical pictures, so a chart cannot show the
volatility *level*, and that level is most of what the 20-day target measures. The images keep shape and relative movement,
which is a much weaker signal.

## How it works

1. **Data.** 979 US-listed stocks, daily and hourly bars from 2001-01-02 to 2026-10-09 from one vendor. Each decision date
   becomes a chart set: a 104-bar weekly chart, a 126-bar daily chart, a 140-bar hourly chart, 33 numeric features from the
   last 126 daily bars, and forward labels. The data is private and is not in this repository.
2. **Splits.** Train 2001-2016, validation 2017-2019, test 2020-2022, each with a 20-day purge so no label window crosses a
   boundary. 2023 onward is a sealed holdout behind a hash-checked, logged, single-use guard.
3. **Backbones.** ResNet-18, EfficientNet-B0, ViT-B/16, ConvNeXt-Tiny, Swin-T, DINOv2-S and MobileNetV3 (all frozen, via
   `timm`), plus three randomly initialised ResNet-18s as controls. Ten backbones in total, three views each.
4. **Heads and tuning.** PCA to 128 dimensions per view (fitted on training rows), then ridge / logistic paths and a boosted
   tree grid, tuned on validation only. Every configuration is logged; the pilot tried 1,200.
5. **Selection, calibration, freeze.** A pre-registered rule picks one model per target; an image model wins only if its
   validation gain over the best numbers-only model has a 95% CI that excludes zero. Probabilities are calibrated with
   isotonic regression, and the NO TRADE thresholds are fixed on validation, before any test data is read.
6. **Checks.** The saved model bundle reproduces the notebook's own test predictions (max difference below 1e-4). The
   dashboard's input code reproduces the training features and charts exactly.

## TradingView screenshot input and the dashboard

The dashboard (`streamlit_app.py`) takes three TradingView screenshots (1W, 1D, 1H) for each of up to four stocks. `screen_reader.py` reads the
ticker, the timeframe and the last bar's O/H/L/C from each image with RapidOCR. On 66 real TradingView screenshots it read the
timeframe and the legend correctly every time, and the true ticker was among its candidates in all 66. The app confirms the
ticker by matching the legend against Yahoo Finance bars, which also pins the analysis to the screenshot's own date. It then
rebuilds the exact model inputs and returns expected volatility with a range, the chance of a 10% drawdown, the chance of
beating the median stock, and a NO TRADE / risk decision with reasons, plus a risk tier, a plain-language explanation of every
number and a Monte Carlo simulation of the next 20 trading days; several stocks are ranked from riskiest to safest. Typing a
ticker works too.

**The Monte Carlo simulation.** Each path draws a volatility from the model's forecast and its measured validation error, then
fat-tailed daily moves (Student-t) with exactly that volatility, and intraday lows from a Brownian-bridge minimum. There is no
drift, because the direction model has no edge. Optionally, and by default, the paths are reweighted as little as possible (mean
return unchanged) so that the share touching a 10% drawdown equals the calibrated drawdown model's probability.

The trained model bundle stays in a private Hugging Face repo and is loaded at start-up with a read-only token. See
`docs/DEPLOY.md` to deploy it for free on Streamlit Community Cloud. `app.py` is an equivalent Gradio front end; both use
`dashboard_logic.py`.

## Repository layout

| Path | What it is |
|---|---|
| `TriFrame_CV_EndToEnd.ipynb` | The end-to-end Colab notebook (clean, no outputs). Stages 1-11, from loading data to the report. |
| `notebooks/TriFrame_CV_pilot_run_60_stocks.ipynb` | The executed pilot run with all outputs, as the record of the results above. |
| `cells/` | The notebook's code cells as plain Python files, one per stage. |
| `tfcv_core.py` | Shared inference code: live data, features, chart rendering, model bundle, NO TRADE rule, Monte Carlo engine. |
| `tfcv_train.py` | Training code: fast chart reader, embeddings, fine-tuning, PCA, heads, metrics, month-block bootstrap, Monte Carlo calibration. |
| `screen_reader.py` | Screenshot reader (ticker, timeframe, last-bar legend). |
| `streamlit_app.py` | The dashboard (Streamlit): up to four companies, charts, explanations, Monte Carlo. |
| `dashboard_charts.py` | The dashboard's plotly charts. |
| `tfcv_report.py` | The plain-language PDF report and its figures. |
| `dashboard_logic.py` | Dashboard logic shared by the Streamlit and Gradio front ends: screenshot checks, market data, analysis, risk tiers, explanations, comparison. |
| `app.py` | Gradio version of the dashboard. |
| `tf_render.py`, `feat_visible.py` | The project's chart renderer and numeric-feature code (needed by the modules above). |
| `make_notebook.py`, `run_local.py` | Build the notebook from `cells/` and run it locally on CPU with stand-in backbones. |
| `tests/` | Parity, OCR, dashboard-logic, Streamlit, fine-tuned-weights round-trip tests (most need the private data, bundle or screenshots). |
| `results/REPORT_pilot_60_stocks.md` | The report the pilot run generated. |
| `docs/RUNBOOK.md` | How to run the notebook on Colab. |
| `docs/DEPLOY.md` | How to deploy the dashboard on Streamlit Community Cloud. |

## Running it

The notebook expects a Colab GPU runtime (an A100 is fastest) and a Hugging Face token with access to the private
`triframe-us-equities-chart-ohlcv` dataset. See `docs/RUNBOOK.md`. Without that dataset the notebook cannot run.
`requirements.txt` lists the dashboard's packages (Streamlit Cloud reads it) and `requirements-notebook.txt` the notebook's.

## Limits

- Pilot results only: 60 of 979 stocks, frozen backbones (no fine-tuning in the pilot).
- Survivor-biased universe (stocks that still exist), one data vendor. Risk may be understated for stocks that later fail.
- The 2023-2026 holdout was opened once, for the pilot's frozen configuration. In that run the notebook computed the unlock
  key itself rather than the owner pasting it. A full-universe run needs its own frozen configuration.
- Live inputs come from Yahoo Finance.
- Research output, not investment advice.
