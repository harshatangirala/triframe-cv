# TriFrame CV: do stock chart images add anything to the numbers?

TriFrame tests whether pictures of price charts at three time scales (weekly, daily and hourly) tell a model anything about
a US stock's next 20 days of volatility, its drawdown risk, or its 5-day relative direction **beyond 33 ordinary numbers
computed from the same price history**. Seven frozen, ImageNet-pretrained vision backbones are compared with that numerical
baseline under chronological splits, month-block bootstrap confidence intervals, label-shuffle and random-weight controls,
and a sealed final holdout.

The short answer from the pilot run: **the numbers carry the signal.** Volatility and drawdown risk are predictable; chart
images do not add a significant improvement on the test years; and there is no detectable 5-day directional edge, so the
decision layer defaults to **NO TRADE**.

> **Scope of the results below.** They come from the notebook's pilot mode (`QUICK = True`): 60 randomly chosen stocks out of
> the 979-stock universe, 27,313 chart sets. The full 979-stock run uses the same notebook with `QUICK = False` and has not
> been run yet.

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

## TradingView screenshot input

The dashboard (`app.py`, Gradio) takes three TradingView screenshots of one stock (1W, 1D, 1H). `screen_reader.py` reads the
ticker, the timeframe and the last bar's O/H/L/C from each image with RapidOCR. On 66 real TradingView screenshots it read the
timeframe and the legend correctly every time, and the true ticker was among its candidates in all 66. The app confirms the
ticker by matching the legend against Yahoo Finance bars, which also pins the analysis to the screenshot's own date. It then
rebuilds the exact model inputs and returns expected volatility with a range, the chance of a 10% drawdown, the chance of
beating the median stock, and a NO TRADE / risk decision with reasons.

**Hosting status:** the app is built and tested locally but is not online yet. Hugging Face now requires a paid plan for
Gradio Spaces on its free CPU tier. `web/krishnaharsha_cv_dashboard.html` is the embed block for when it is hosted.

## Repository layout

| Path | What it is |
|---|---|
| `TriFrame_CV_EndToEnd.ipynb` | The end-to-end Colab notebook (clean, no outputs). Stages 1-11, from loading data to the report. |
| `notebooks/TriFrame_CV_pilot_run_60_stocks.ipynb` | The executed pilot run with all outputs, as the record of the results above. |
| `cells/` | The notebook's code cells as plain Python files, one per stage. |
| `tfcv_core.py` | Shared inference code: live data, features, chart rendering, model bundle, NO TRADE rule. |
| `tfcv_train.py` | Training code: fast chart reader, embeddings, PCA, heads, metrics, month-block bootstrap. |
| `screen_reader.py` | Screenshot reader (ticker, timeframe, last-bar legend). |
| `app.py` | Gradio dashboard. |
| `tf_render.py`, `feat_visible.py` | The project's chart renderer and numeric-feature code (needed by the modules above). |
| `make_notebook.py`, `run_local.py` | Build the notebook from `cells/` and run it locally on CPU with stand-in backbones. |
| `tests/` | Parity, OCR and app tests (they need the private data and screenshots). |
| `results/REPORT_pilot_60_stocks.md` | The report the pilot run generated. |
| `docs/RUNBOOK.md` | How to run the notebook on Colab. |

## Running it

The notebook expects a Colab GPU runtime (an A100 is fastest) and a Hugging Face token with access to the private
`triframe-us-equities-chart-ohlcv` dataset. See `docs/RUNBOOK.md`. Without that dataset the notebook cannot run.
`requirements.txt` lists the notebook's packages and `requirements-app.txt` the dashboard's.

## Limits

- Pilot results only: 60 of 979 stocks.
- Survivor-biased universe (stocks that still exist), one data vendor. Risk may be understated for stocks that later fail.
- The 2023-2026 holdout was opened once, for the pilot's frozen configuration. In that run the notebook computed the unlock
  key itself rather than the owner pasting it. A full-universe run needs its own frozen configuration.
- Live inputs come from Yahoo Finance.
- Research output, not investment advice.
