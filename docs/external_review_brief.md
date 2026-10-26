# TriFrame — Quantitative Research Audit, Refinement Plan, and Implementation Brief

**Prepared for:** Claude Code  
**Project:** TriFrame computer-vision/deep-learning research for stock risk and relative-return forecasting  
**Current report:** `TriFrame_CV_report_s120.pdf`, notebook/project version `s120 / v3`, report generated 2026-10-20  
**Purpose of this document:** Give Claude Code the full research context, current results, concerns, recommended experiments, evaluation framework, and implementation constraints needed to improve the project without introducing leakage, overfitting, or unsupported claims.

---

## 0. Instructions to Claude Code

Act as a **senior quantitative researcher, ML research engineer, and time-series validation auditor**. Do not start by rewriting the pipeline. First inspect the repository, notebooks, configuration files, feature-generation code, target-generation code, model-selection code, evaluation code, saved predictions, and artifacts. Produce an audit report before changing anything.

### Non-negotiable rules

1. **Do not use the existing 2020–2022 test results to select the next model.** The leaderboard and metrics below have already been inspected. Treat 2020–2022 as a historical diagnostic set for this iteration, not as a fresh model-selection set.
2. **Do not assume 2023–2026 is still an untouched holdout.** The report describes it as sealed but also contains holdout score entries and an “opened” status. Establish the chronology from files, logs, timestamps, and code. Until proven otherwise, treat it as potentially exposed and do not use it for iterative model selection.
3. **Do not modify the existing baseline in place.** Preserve the current pipeline, predictions, metrics, configuration and environment. Create versioned experimental branches/configurations and retain a reproducible baseline.
4. **Do not chase test-set leaders.** Select all future models using validation or walk-forward validation only. Report test performance only after model selection and lock-down.
5. **No leakage:** use only data available at prediction time; fit scalers, imputers, PCA, feature selection, calibration, probability thresholds and every learned preprocessing step on training/validation data according to the appropriate protocol. Do not fit anything on test labels.
6. **Respect temporal and cross-sectional dependence.** Rows are stock-day observations with overlapping historical windows and forward labels. Do not assume 8,666 rows are 8,666 independent observations.
7. **No unsupported claims of a global minimum.** The goal is not to find the global minimum of a non-convex neural-network loss. The goal is robust generalisation and incremental economic utility.
8. **Do not add epochs blindly.** First inspect training/validation curves and diagnose underfitting, overfitting, optimisation instability, representation limitations or lack of target signal.
9. **Do not optimise all metrics at once without a declared selection rule.** Specify a primary validation metric per task, secondary diagnostics, model-selection tie-break rules, and any practical thresholds before running experiments.
10. **Do not silently reconcile conflicting metrics.** Investigate discrepancies described in Section 4. Label outputs with exact target, model, feature set, probability calibration status, data split, target scale, metric implementation and artifact/version ID.
11. **Do not claim the images improve prediction unless a paired incremental comparison demonstrates this on suitable out-of-sample data with uncertainty intervals.** Better raw test scores among 15 candidate models are descriptive and do not validate a model chosen after seeing those scores.
12. **Do not claim a trading strategy is profitable until an executable portfolio backtest includes realistic costs, turnover, capacity/exposure controls and an untouched evaluation period.**

### Expected first deliverable before code changes

Create an audit note containing:

- a map of the repository and the main pipeline entry points;
- exact code paths and formulas for each target;
- exact train/validation/test/holdout row and date boundaries;
- source data vendors and corporate-action handling;
- the identities and overlap of securities across splits;
- current model, hyperparameter and preprocessing-selection logic;
- the exact source of every reported metric;
- a discrepancy table for all conflicting numbers;
- a ranked list of bugs/data-quality risks versus research hypotheses;
- a reproducibility command and baseline output manifest.

If a fact cannot be established from the repository, record it as unknown instead of guessing.

---

## 1. Project question and current scope

TriFrame asks whether visual representations of three stock-chart timeframes — weekly, daily, and hourly — contribute predictive information about future stock behaviour beyond a set of numerical features derived from recent price data.

The project currently predicts three distinct outcomes:

### V — Forecast future volatility

- Target name shown in outputs: `logrv20`.
- Horizon: next 20 trading days.
- Model family in the currently selected pipeline: numerical features + three chart-image embeddings, fine-tuned ViT-B/16, and a tree-based prediction head.
- Main reported test performance: R² approximately 0.503.
- Interpretation: some variation in the volatility target is predictable. This does **not** mean the model predicts return direction or generates trading alpha.

### D10 — Predict a 10% drawdown event

- Target name shown in outputs: `dd20_le_10`.
- Event: price reaches at least 10% below the next day’s opening price at some point within the subsequent 20 trading days. Verify the precise implementation (intraday low versus close, adjustment rules, and exact reference-price alignment) in code.
- Selected model reported in the summary: numerical features only + tree-based model.
- Main reported test AUC: approximately 0.663.
- Interpretation: modest but potentially useful risk ranking; probability calibration and decision utility still need to be assessed.

### O5 — Predict five-day relative outperformance

- Target name shown in outputs: `y_out5`.
- Horizon: next five trading days.
- Label: whether a stock beats the median stock over the horizon. Verify whether the median is computed across the selected 120 stocks, the available point-in-time universe, or another benchmark. Verify precise return calculation, missing-value treatment, and alignment.
- Selected model reported in the summary: numerical features only + tree-based model.
- Main reported test AUC: approximately 0.508.
- Interpretation: no reliable out-of-sample direction/ranking edge has been demonstrated. Do not force a positive conclusion by adding model complexity.

### Broader research hypothesis

The project has two separate hypotheses that must never be conflated:

1. Historical price and market features contain information about future volatility and downside risk.
2. Three-timeframe chart images provide **incremental** information beyond the 33 numerical features.

The first hypothesis has encouraging preliminary evidence for volatility and drawdown risk. The second is currently **not supported** by the reported formal incremental comparisons. The O5 directional hypothesis is also currently unsupported.

---

## 2. Data, architecture, and experiment design reported in the PDF

The report states the following. Verify each item against source code and artifacts.

- 120 randomly selected US stocks, including the 60 stocks from the previous pilot; the broader source universe is described as 979 stocks.
- 56,099 chart sets in total, dated 2001–2022.
- Each observation represents one stock on one date.
- Three black-and-white chart images per observation:
  - weekly chart: 104 bars;
  - daily chart: 126 bars;
  - hourly chart: 140 bars.
- Images are rendered with height of 128 pixels and scaled to fill each panel. The rendering may preserve shape but removes absolute price-scale information.
- 33 numerical features derived from recent daily price history, described as the last 126 daily bars; verify exact feature definitions and whether any features encode information not available at the decision time.
- Vision backbones compared include ResNet-18, EfficientNet-B0, ViT-B/16, ConvNeXt-Tiny, Swin-T, DINOv2 ViT-S/14 and MobileNetV3-Large, in frozen and fine-tuned configurations. Random-weight ResNet controls were also reported.
- Embeddings are compressed using PCA to 128 dimensions per image branch, with PCA fitted on training data according to the report.
- The model head may be ridge/logistic regression or tree-based boosting, depending on the configuration. Selected models use a tree-based head.
- The report says 1,690 model configurations were tried and selected on validation data.
- The report describes preprocessing, PCA and scaling as fitted on training years only; audit this implementation in the code.
- The reported chronology is:
  - train: 2001–2016, 38,937 chart sets, 117 stocks;
  - validation: 2017–2019, 8,496 chart sets, 118 stocks;
  - test: 2020–2022, 8,666 chart sets, 120 stocks;
  - holdout: 2023–2026, status unclear because the report both describes it as sealed and reports scores/statuses associated with it.
- A 20-trading-day gap is reported at split boundaries to prevent forward-label overlap across adjacent splits. Verify boundary construction for each target and ensure it is sufficient for the exact target windows.
- The model dashboard accepts user-uploaded TradingView screenshots. The report states live data comes from Yahoo Finance while training data come from another vendor. The discrepancy between training charts and live screenshot/rendering conventions is a potential deployment distribution-shift problem.

### Important limits already stated in the report

- Only 120 of the stated 979 stocks were used in this run.
- The universe is survivorship-biased because it includes stocks that still exist; historical failures and delistings may be underrepresented.
- Chart images are scaled to fill panels and therefore do not show absolute price/volatility scale in a consistent way.
- Live dashboard inputs may differ slightly from the training data vendor.
- The project is research output, not investment advice; the default directional decision is NO TRADE.

---

## 3. Consolidated current reported results

This section transcribes the main results supplied in the conversation. It is an input to the audit, **not proof that every number came from the same exact evaluation path**. Conflicts are listed in Section 4 and must be resolved in the implementation.

### 3.1 Summary lines supplied by the user

**V — selected numerical + image model:**

```text
n = 8666
R2 = 0.5029
wmR2 = 0.5720
QLIKE = 0.5398
MAE = 0.2722
R2_CI = [0.2664811717056162, 0.6389736566747707]
wmR2_CI = [0.48951559780292525, 0.6395942829406952]
```

**O5 — numerical-only tree model:**

```text
n = 8666
AUC = 0.5075
logloss = 0.6937
brier = 0.2503
ECE = 0.0149
BSS = -0.0011
rankIC = NaN
rankIC_t = NaN
AUC_CI = [0.4956562233382843, 0.5195710752552606]
BSS_CI = [-0.0033023779405646435, 0.001208184934099621]
```

**D10 — numerical-only tree model:**

```text
n = 8666
AUC = 0.6625
logloss = 0.5788
brier = 0.1939
ECE = 0.0279
BSS = 0.0886
AUC_CI = [0.6013711721041067, 0.7088707682022746]
BSS_CI = [0.047268313666808864, 0.12430597973208621]
```

### 3.2 Formal incremental-image tests supplied by the user

**H1 — do three timeframes of chart images add information beyond numerical features?** Reported result: **NOT SUPPORTED**.

- V: numerical + images, fine-tuned ViT-B/16/tree versus numerical-only/tree: `delta R2 = +0.0030`, 95% CI `[-0.0156, +0.0265]`.
- O5: numerical + images, fine-tuned ResNet-18/tree versus numerical-only/tree: `delta AUC = +0.0001`, 95% CI `[-0.0136, +0.0154]`.

Both intervals include zero. This is no statistically convincing evidence of incremental value on the reported test period.

### 3.3 Abstention experiment supplied by the user

**H3 — does abstaining on low-confidence calls improve accuracy?** Reported result: **NOT SUPPORTED**.

- Accuracy gain at nominal 10% coverage: `+0.0045`.
- 95% CI: `[-0.0192, +0.0291]`.

| Coverage target | Realised coverage | n | Accuracy | Wilson lower bound |
|---:|---:|---:|---:|---:|
| 1.00 | 1.0000 | 8666 | 0.5051 | 0.4946 |
| 0.50 | 0.5500 | 4766 | 0.5063 | 0.4921 |
| 0.25 | 0.3067 | 2658 | 0.5090 | 0.4900 |
| 0.10 | 0.1388 | 1203 | 0.5096 | 0.4813 |

Interpretation: selective predictions have not demonstrated a statistically reliable accuracy gain. This could be revisited only with a pre-specified acceptance rule and fresh validation/test data; do not tune thresholds on the existing test period.

### 3.4 NO TRADE risk-flag result supplied by the user

- Flags cover 49.7% of test cases.
- Flagged cases: realised annualised volatility 54.0%; 10% drawdowns occur in 39.5% of cases.
- Unflagged cases: realised annualised volatility 28.4%; 10% drawdowns occur in 19.2% of cases.

This is potentially useful risk stratification. It does not prove that acting on the flag improves a portfolio; a portfolio-level intervention/backtest is required.

### 3.5 Historical-data sensitivity supplied by the user

The supplied section is labelled “Survivorship sensitivity,” but the actual described comparison is test performance with versus without the 2001–2009 rows in training:

| Target | Including older training rows | Excluding 2001–2009 rows from training |
|---|---:|---:|
| V R² | 0.5029 | 0.4959 |
| O5 AUC | 0.5075 | 0.5009 |
| D10 AUC | 0.6625 | 0.6597 |

This is a **training-history sensitivity test, not by itself a survivorship-bias test**. A valid survivorship-bias evaluation must use point-in-time constituents, handle delisted securities and returns, and avoid selecting only current survivors.

---

## 4. Metric and reporting discrepancies — resolve before new experiments

The supplied summaries contain values that do not exactly match the detailed test leaderboard. Do not assume these are rounding differences. They may arise from different prediction files, calibrated versus uncalibrated probabilities, different subsets, different target scales, different run artifacts or metric implementations.

| Metric/target | Summary line | Detailed leaderboard value | Required action |
|---|---:|---:|---|
| V QLIKE | 0.5398 | Selected fine-tuned ViT leaderboard row: 0.5055 | Identify exact target/prediction scale, formula, subset and prediction artifact. Check whether QLIKE is correctly applied to positive variance rather than log-volatility. |
| O5 AUC | 0.5075 | Detailed numerical-only reference row: 0.5090 | Identify model artifact, calibration status, thresholding, target rows and exact evaluator. |
| D10 AUC | 0.6625 | Detailed numerical-only reference row: 0.6637 | Identify model artifact, calibration status, target rows and exact evaluator. |
| D10 ECE | 0.0279 | Detailed numerical-only row: 0.0321 | Check whether one report uses calibrated predictions and the other uncalibrated predictions, plus bin definitions. |
| D10 BSS | 0.0886 | Detailed numerical-only row: 0.0894 | Check reference probability used for the Brier skill calculation and sample rows. |
| O5 log-loss/Brier/ECE/BSS | 0.6937 / 0.2503 / 0.0149 / -0.0011 | Detailed numerical-only row: 0.6931 / 0.25-ish / 0.0092 / 0.0000 | Resolve same-model, same-rows and calibration-status provenance. |

Also inspect all reported confidence intervals. The summary volatility CI is approximately `[0.2665, 0.6390]`, while the detailed leaderboard uses a rounded interval displayed as `0.2665–0.6390`; that looks consistent, but the interval method, number of resamples and unit of resampling still need to be documented.

### Required reconciliation output

For each final metric, emit a machine-readable row with at least:

```text
run_id
code_commit_or_hash
config_hash
prediction_file_hash
target_name
target_formula_version
feature_set
vision_model
frozen_or_finetuned
head_type
split_name
split_start
split_end
n_rows
n_stocks
probability_calibrated
prediction_scale
target_scale
metric_name
metric_value
ci_method
ci_lower
ci_upper
```

A metric should not appear in a report without enough provenance to reconstruct it.

---

## 5. Interpretation of the existing leaderboard

### 5.1 Volatility leaderboard (selected entries supplied by the user)

Metrics are on the supplied test years; most candidate comparisons are descriptive. Do not select a new model from the test leaderboard.

| Model | Mode | R² | wmR² | QLIKE | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | frozen | 0.5116 | 0.5657 | 0.5101 | 0.2704 | 0.3845 | 0.7408 |
| ConvNeXt-Tiny | frozen | 0.5113 | 0.5686 | 0.4942 | 0.2709 | 0.3846 | 0.7404 |
| DINOv2 ViT-S/14 | frozen | 0.5103 | 0.5694 | 0.5073 | 0.2702 | 0.3850 | 0.7410 |
| MobileNetV3-Large | frozen | 0.5101 | 0.5642 | 0.4971 | 0.2721 | 0.3851 | 0.7405 |
| ResNet-18 | frozen | 0.5090 | 0.5662 | 0.5029 | 0.2717 | 0.3855 | 0.7411 |
| MobileNetV3-Large | fine-tuned | 0.5086 | 0.5669 | 0.5021 | 0.2712 | 0.3856 | 0.7379 |
| Swin-T | fine-tuned | 0.5085 | 0.5677 | 0.5138 | 0.2701 | 0.3856 | 0.7419 |
| EfficientNet-B0 | fine-tuned | 0.5084 | 0.5654 | 0.5015 | 0.2722 | 0.3857 | 0.7377 |
| ResNet-18 | fine-tuned | 0.5080 | 0.5671 | 0.5040 | 0.2721 | 0.3859 | 0.7392 |
| EfficientNet-B0 | frozen | 0.5079 | 0.5651 | 0.5046 | 0.2720 | 0.3859 | 0.7419 |
| ConvNeXt-Tiny | fine-tuned | 0.5062 | 0.5671 | 0.5097 | 0.2709 | 0.3866 | 0.7416 |
| Swin-T | frozen | 0.5040 | 0.5607 | 0.5132 | 0.2724 | 0.3874 | 0.7363 |
| ViT-B/16 | fine-tuned — selected on validation | 0.5029 | 0.5720 | 0.5055 | 0.2722 | 0.3879 | 0.7486 |
| Numerical only (33 features) | reference | 0.4998 | 0.5542 | 0.5059 | 0.2763 | 0.3891 | 0.7281 |
| DINOv2 ViT-S/14 | fine-tuned | 0.4980 | 0.5565 | 0.5147 | 0.2748 | 0.3898 | 0.7308 |

Key readings:

- Several models have slightly higher R² on this test leaderboard than the model selected from validation. This is not a reason to switch to the test winner. It is a reason to define fresh walk-forward evaluation folds and test hypotheses that were not chosen by inspecting the test results.
- The selected fine-tuned ViT has the best listed wmR² and Spearman value among these rows, while it does not have the best R², MAE, RMSE or QLIKE. This demonstrates why the primary objective and tie-break rules must be declared beforehand.
- The numerical-only baseline has R² 0.4998, compared with 0.5029 for the selected image+numerical model. The observed difference is small and its supplied CI includes zero.
- Do not assume the QLIKE values in the table are correct until the discrepancy in Section 4 is resolved.

### 5.2 O5 five-day relative-outperformance leaderboard (all rows supplied by user)

| Model | Mode | AUC | AP | Accuracy | Log-loss | BSS | ECE | CI low | CI high |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | fine-tuned | 0.5144 | 0.5180 | 0.5142 | 0.6930 | 0.0004 | 0.0083 | 0.5012 | 0.5261 |
| ConvNeXt-Tiny | frozen | 0.5126 | 0.5194 | 0.5087 | 0.6929 | 0.0005 | 0.0076 | 0.5005 | 0.5250 |
| ViT-B/16 | frozen | 0.5118 | 0.5145 | 0.5023 | 0.6929 | 0.0005 | 0.0086 | 0.4999 | 0.5243 |
| DINOv2 ViT-S/14 | fine-tuned | 0.5102 | 0.5124 | 0.5022 | 0.6931 | 0.0001 | 0.0066 | 0.5023 | 0.5184 |
| ResNet-18 | fine-tuned | 0.5092 | 0.5196 | 0.5065 | 0.6931 | 0.0001 | 0.0095 | 0.4959 | 0.5219 |
| Numerical only (33 features) — selected reference | — | 0.5090 | 0.5152 | 0.5050 | 0.6931 | 0.0000 | 0.0092 | 0.4960 | 0.5223 |
| Swin-T | frozen | 0.5066 | 0.5219 | 0.5055 | 0.6931 | 0.0001 | 0.0087 | 0.4940 | 0.5186 |
| ResNet-18 | frozen | 0.5066 | 0.5101 | 0.5017 | 0.6934 | -0.0004 | 0.0099 | 0.4941 | 0.5172 |
| EfficientNet-B0 | fine-tuned | 0.5053 | 0.5124 | 0.5015 | 0.6931 | 0.0001 | 0.0067 | 0.4920 | 0.5176 |
| MobileNetV3-Large | fine-tuned | 0.5016 | 0.5131 | 0.5020 | 0.6933 | -0.0003 | 0.0080 | 0.4856 | 0.5164 |
| Swin-T | fine-tuned | 0.5010 | 0.5068 | 0.5002 | 0.6938 | -0.0013 | 0.0148 | 0.4899 | 0.5125 |
| DINOv2 ViT-S/14 | frozen | 0.5002 | 0.5130 | 0.4957 | 0.6935 | -0.0006 | 0.0137 | 0.4886 | 0.5129 |
| EfficientNet-B0 | frozen | 0.4962 | 0.5031 | 0.4939 | 0.6936 | -0.0008 | 0.0147 | 0.4826 | 0.5098 |
| MobileNetV3-Large | frozen | 0.4926 | 0.5029 | 0.4893 | 0.6933 | -0.0004 | 0.0153 | 0.4764 | 0.5093 |
| ViT-B/16 | fine-tuned | 0.4909 | 0.4979 | 0.4940 | 0.6938 | -0.0013 | 0.0156 | 0.4762 | 0.5060 |

Interpretation:

- Most O5 AUC values are close to 0.50, log-loss is around 0.693, and BSS is close to zero or negative. This is consistent with little or no reliable predictive signal.
- A few candidate test AUC values are above 0.51, but the confidence intervals and repeated model comparison mean they should be treated as descriptive. Do not cherry-pick the highest row.
- `rankIC` and `rankIC_t` are NaN in the summary. Determine why: missing implementation, invalid sample, no cross-sectional rank calculation, or no valid predictions. Do not report them as meaningful until resolved.
- The binary outperformance label discards the magnitude of excess return. A continuous excess-return or cross-sectional ranking formulation is a reasonable separate research direction, not a guaranteed improvement.

### 5.3 D10 drawdown leaderboard (all rows supplied by user)

| Model | Mode | AUC | AP | Accuracy | Log-loss | BSS | ECE | CI low | CI high |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | fine-tuned | 0.6736 | 0.4478 | 0.7124 | 0.5806 | 0.0849 | 0.0569 | 0.6199 | 0.7178 |
| ResNet-18 | fine-tuned | 0.6656 | 0.4461 | 0.7169 | 0.5771 | 0.0886 | 0.0319 | 0.6063 | 0.7108 |
| ConvNeXt-Tiny | fine-tuned | 0.6654 | 0.4455 | 0.7112 | 0.5814 | 0.0839 | 0.0473 | 0.6077 | 0.7098 |
| ResNet-18 | frozen | 0.6653 | 0.4479 | 0.7147 | 0.5763 | 0.0887 | 0.0339 | 0.6046 | 0.7104 |
| EfficientNet-B0 | frozen | 0.6644 | 0.4448 | 0.7161 | 0.5792 | 0.0851 | 0.0376 | 0.6046 | 0.7094 |
| Numerical only (33 features) | reference | 0.6637 | 0.4477 | 0.7142 | 0.5756 | 0.0894 | 0.0321 | 0.6023 | 0.7104 |
| MobileNetV3-Large | fine-tuned | 0.6635 | 0.4435 | 0.7134 | 0.5773 | 0.0870 | 0.0337 | 0.6010 | 0.7100 |
| DINOv2 ViT-S/14 | frozen | 0.6635 | 0.4456 | 0.7156 | 0.5780 | 0.0865 | 0.0348 | 0.6012 | 0.7104 |
| Swin-T | frozen | 0.6627 | 0.4472 | 0.7167 | 0.5790 | 0.0856 | 0.0360 | 0.6001 | 0.7092 |
| EfficientNet-B0 | fine-tuned | 0.6615 | 0.4453 | 0.7147 | 0.5800 | 0.0841 | 0.0355 | 0.6000 | 0.7081 |
| MobileNetV3-Large | frozen | 0.6613 | 0.4445 | 0.7130 | 0.5794 | 0.0848 | 0.0350 | 0.5984 | 0.7075 |
| ViT-B/16 | frozen | 0.6609 | 0.4447 | 0.7157 | 0.5791 | 0.0849 | 0.0310 | 0.5992 | 0.7072 |
| ConvNeXt-Tiny | frozen | 0.6608 | 0.4416 | 0.7166 | 0.5803 | 0.0832 | 0.0390 | 0.6004 | 0.7059 |
| Swin-T | fine-tuned | 0.6571 | 0.4403 | 0.7147 | 0.5827 | 0.0814 | 0.0332 | 0.5950 | 0.7041 |
| DINOv2 ViT-S/14 | fine-tuned | 0.6570 | 0.4398 | 0.7146 | 0.5825 | 0.0817 | 0.0324 | 0.5926 | 0.7063 |

Interpretation:

- The fine-tuned ViT row has the highest listed AUC in this table, but its BSS and ECE are worse than the numerical-only reference. This is a real example of why one metric must not dominate model selection without considering probability quality.
- The table’s selected numerical-only AUC (0.6637) does not match the summary line (0.6625). Audit that first.
- The binary accuracy values around 0.71 cannot be interpreted in isolation; determine the event prevalence and the exact classification threshold. A high accuracy could result from class imbalance.
- AP is around 0.44–0.45. Compare it with the positive-class prevalence before describing it as strong or weak.

---

## 6. The “global minimum” question

The project should not be judged by whether it reaches a global minimum. Neural networks have non-convex objective functions, and ordinary stochastic gradient-based training does not generally guarantee a global minimum. Moreover, the pipeline includes a neural-network encoder followed by PCA and a separate tree-based head. The end-to-end forecasting objective is not jointly optimised as one differentiable model.

The relevant questions are:

1. Does training loss decrease and converge?
2. Does validation loss/metric improve with it?
3. Is there a widening training–validation gap?
4. Is the chosen checkpoint stable across seeds?
5. Does the complete pipeline improve the predeclared validation metric over strong baselines?
6. Does the improvement persist in fresh walk-forward or untouched data?

Do not use the lowest training loss as the selection objective. Select checkpoints based on a validation metric appropriate for each task, with early stopping and a predeclared rule.

---

## 7. Fine-tuning plan: controlled experiment, not blind epoch inflation

The report says the backbones were fine-tuned for only a few epochs. More training may help some models, but there is no evidence yet that epoch count is the principal bottleneck.

### 7.1 First inspect learning curves

For each target/backbone/configuration, record at every epoch:

- training loss;
- validation loss;
- primary validation metric;
- secondary diagnostics;
- learning rate;
- best epoch/checkpoint;
- wall-clock time and peak memory;
- random seed.

Diagnose:

- **Underfitting:** training and validation performance both continue improving or both remain poor. Consider more epochs, better optimisation or unfreezing additional layers.
- **Overfitting:** training loss improves while validation loss/metric plateaus or deteriorates. Use early stopping, regularisation, data/augmentation review or fewer trainable parameters.
- **Representation/target limitation:** training improves but validation does not, or both plateau with limited predictive performance. More epochs are unlikely to solve this alone; inspect targets, input signal, labels, data coverage and model design.

### 7.2 Controlled unfreezing/epoch sweep

For promising backbones, use a small, staged experiment:

- A: frozen pretrained encoder;
- B: fine-tune the final two transformer blocks (or analogous final backbone stages);
- C: fine-tune the final four blocks/stages;
- D: fully fine-tune the encoder;
- compare candidate budgets such as 4, 6, 8, and at most 10 epochs initially, with early stopping. These are testable budgets, not mandated epoch counts.

Use a relatively low encoder learning rate, e.g. a starting range of `1e-5` to `3e-5` for ViT-B/16, and a higher learning rate for a newly trained head where applicable, e.g. `1e-4` to `3e-4`. Treat these only as initial search ranges; use validation to select. Consider AdamW, weight decay, a short warm-up, and cosine decay or another documented schedule. Do not use test performance to pick any of these settings.

The exact learning-rate setup must be consistent with the optimizer, batch size, loss, pretraining, gradient accumulation, precision and GPU/TPU constraints in the existing code.

### 7.3 Training-seed robustness

Run at least three seeds for top validation candidates when compute allows. Report the mean and variation across seeds, not merely the best lucky run. Seed variability is not a substitute for time-series uncertainty estimates, but it reveals optimisation instability.

### 7.4 Target-specific fine-tuning

Inspect whether the same fine-tuned encoder is being used for all tasks or whether each task has a separate fine-tuned representation. Do not assume one representation is optimal for volatility regression, drawdown classification and relative-return prediction.

If encoders are trained separately for each target, version their datasets, loss functions, checkpoints and seeds. If a shared encoder/multi-task model is explored, treat it as a separate experiment and compare it against independent models. Multi-task learning can help or hurt depending on target relationships; it is not automatically superior.

### 7.5 Keep the head comparison fair

The current pipeline uses PCA-compressed embeddings and a tree-based head in many rows. Compare frozen and fine-tuned models with consistent preprocessing and validation budgets. If PCA is fitted on training data only, preserve that rule. Refit PCA inside each walk-forward fold. Do not fit PCA on a combined train + validation + test matrix.

---

## 8. Target-by-target research plan

## 8.1 V — Volatility forecasting

### Goal
Improve out-of-sample forecast quality, stability, and incremental value over standard numerical or econometric baselines.

### Immediate audits

1. Verify precisely what `logrv20` represents: log realised volatility, log realised variance, or another transformed outcome. Document units and inverse transformation.
2. Verify whether the target uses closes, intraday observations, realised variance, an annualisation factor, and/or winsorisation. Confirm dates and horizon.
3. Check if the predicted quantity is positive and on the scale required by each metric.
4. Recompute QLIKE from first principles after confirming the target scale. Standard volatility QLIKE formulations require non-negative/positive variance forecasts and realised variance. Do not apply a standard variance QLIKE formula directly to a logarithmic target without a justified transformation.
5. Audit the custom `wmR2` definition and weights. Name the weighting scheme in documentation.
6. Ensure confidence intervals and metric comparisons use exactly aligned target/prediction rows.

### Baselines to add

- Historical rolling volatility (clearly specify lookback window and estimator).
- EWMA volatility.
- GARCH-family forecasts, such as GARCH and asymmetric variants where appropriate.
- HAR-RV if reliable realised-volatility data at appropriate frequencies are available.
- Numerical-only linear/regularised regression and numerical-only gradient boosting.
- Current frozen and fine-tuned image+numerical models.

All fitted baselines must be estimated using only information available at the forecast origin and evaluated on identical dates. Do not compare models on different sample subsets without explaining it.

### Image and feature ablations

- 33 numerical features only.
- weekly image only + numbers;
- daily image only + numbers;
- hourly image only + numbers;
- weekly + daily + numbers;
- weekly + hourly + numbers;
- daily + hourly + numbers;
- all three image branches + numbers;
- images only (diagnostic, not expected to be inherently superior).

If the code already logs some of these configurations, reuse the artifacts rather than rerunning needlessly. Define a small, hypothesis-driven search instead of exhaustive unbounded tuning.

### Additional metrics

- R² and weighted R², with precise definitions;
- MAE and RMSE on a documented scale;
- correctly implemented QLIKE on a variance-consistent scale;
- rank correlation/Spearman between predicted and realised risk;
- calibration of risk buckets: predicted versus realised volatility;
- per-regime and per-sector error;
- paired difference versus each baseline with uncertainty intervals.

### Economic test

Evaluate whether volatility forecasts improve position sizing or portfolio risk targeting versus standard historical/EWMA/GARCH volatility. Track realised portfolio volatility, risk-target error, maximum drawdown, turnover, transaction costs and exposure. Do not claim improvement until the portfolio experiment is run out of sample.

## 8.2 D10 — Drawdown-risk classification

### Goal
Improve ranking and probability calibration for 10% drawdown events; determine whether forecasts support better risk decisions.

### Immediate audits

1. Confirm event definition exactly, including reference price (`next day open`), future lows versus closes, corporate-action-adjusted prices, and precise horizon inclusion.
2. Calculate positive-event prevalence on train/validation/test and by year/sector.
3. Check class weighting, sample weighting, probability outputs, threshold handling, calibration and Brier-skill reference probability.
4. Ensure the reference probability for BSS is computed from training/validation prevalence appropriate to the evaluation design and not leaked from test labels.
5. Audit ECE bin count, bin type and calibration sample. ECE is binning-sensitive; report Brier/log-loss and reliability plots as well.

### Models and experiments

- Numerical-only XGBoost/LightGBM/CatBoost if already available in the environment and licences/dependencies are acceptable; do not add new libraries without checking the project setup.
- Logistic regression as an interpretable benchmark.
- Frozen image embeddings + numbers.
- Partial and full fine-tuning + numbers.
- Carefully configured class weights or training resampling if necessary; never resample validation/test.
- Probability calibration (Platt scaling and/or isotonic regression) fitted on a dedicated validation/calibration portion or out-of-fold validation predictions only.

### Primary and secondary metrics

Predeclare whether AUC or a utility-focused metric is primary. Recommended reporting:

- ROC-AUC with dependence-aware CI;
- average precision/PR-AUC plus positive prevalence;
- Brier score, Brier skill score, log-loss;
- reliability diagram, calibration slope and intercept, ECE with binning scheme documented;
- recall, precision, specificity and alert rate at risk thresholds selected on validation only;
- precision/recall among top risk deciles or top 10%/20% risk group;
- false negatives and false positives with explicit decision costs;
- per-year/per-sector/per-regime performance.

Accuracy alone is not enough, particularly if events are imbalanced. Investigate why the detailed classification leaderboard reports accuracy around 0.71 while AUC is around 0.66 and AP around 0.45; compare the result with event prevalence and an appropriate constant-class baseline.

### Risk policy and economic test

Use validation data to choose thresholds and predefine an intervention: reduced position size, portfolio cap, additional review, or exclusion. Evaluate the complete rule against an unmodified baseline on fresh out-of-sample data. Report both avoided losses and opportunity cost, not just event-detection scores.

## 8.3 O5 — Five-day outperformance

### Goal
Determine whether a genuine cross-sectional return-prediction signal exists. Do not set an artificial goal of exceeding AUC 0.50 at any cost.

### Immediate audits

1. Verify the exact forward-return calculation, holding period, price convention, corporate actions, and missing-return treatment.
2. Verify whether “median stock” means the median of the selected 120 stocks, all point-in-time eligible stocks, a sector group, or another benchmark.
3. Verify that the cross-sectional benchmark at date t is computed only from valid outcomes for the exact intended universe and does not inadvertently vary due to survivorship or missing labels.
4. Check class balance by year, sector and regime.
5. Investigate why `rankIC` and `rankIC_t` are NaN: missing implementation, too few valid cross-sectional observations, constant scores, or another issue. Implement them correctly or document why not available.
6. Audit whether thresholding/abstention is based on calibrated probabilities and validation-selected thresholds. The current abstention experiment is not supported.

### Target options (separate pre-registered experiments)

Keep the current binary target for comparability. In a separate experiment, test continuous forward excess returns:

`excess_return_{i,t,h} = stock_return_{i,t,h} - benchmark_return_{t,h}`

The benchmark should be explicitly chosen and justified (market index, sector-relative, or cross-sectional median). Also consider a cross-sectional ranking objective if portfolio stock selection is the intended use. Do not silently replace the current target or report the alternative as a direct apples-to-apples comparison.

### Metrics

For binary classification:

- ROC-AUC with CI;
- average precision with base rate;
- log-loss, Brier score, Brier skill score;
- calibration and balanced accuracy;
- precision/recall at thresholds determined on validation;
- abstention/coverage curve only with predeclared confidence score and acceptance rule.

For continuous/ranking prediction:

- out-of-sample Pearson/Spearman cross-sectional IC by date;
- mean IC, IC volatility and an appropriate time-series IC t-statistic;
- rank-IC confidence intervals using date blocks or another dependence-aware method;
- return-prediction error as secondary diagnostics, not as the only proof of value;
- top-minus-bottom portfolio spread and rank-decile returns;
- turnover, costs, exposures, realised volatility, Sharpe, maximum drawdown and tail risk.

Do not report a simple t-statistic as reliable without addressing serial correlation, overlapping horizons, and cross-sectional dependence. A standard Newey–West/HAC approach or block bootstrap may be suitable depending on the unit of analysis and implementation; document the design rather than mechanically selecting a method.

### Model strategy

If the 5-day target remains near random across validation folds, examine target quality, label noise, horizon, benchmark and whether technical chart images have any hypothesis-supported signal. Prefer a modest, controlled set of alternatives over a large hyperparameter search. It is valid to conclude that this dataset/feature set does not support a robust five-day signal.

---

## 9. Evaluation and statistical-inference framework

### 9.1 Split design

- Reconstruct and document current split logic.
- Retain the existing 2020–2022 set as already-inspected historical test/diagnostic data for new iterations.
- Establish whether 2023–2026 holdout was opened, how often it was evaluated, and whether its outcomes influenced choices. Because reported holdout results already exist, do not assume it remains fully untouched.
- Use rolling-origin/walk-forward development folds over earlier history for model selection. Keep folds chronological.
- Purge or embargo at least the full maximum forward target horizon at boundaries, with logic verified per target. The existing 20-day gap is a sensible starting point for 20-day targets, but check label alignment and whether the implementation purges the right rows.
- Consider an independent security generalisation split: reserve a subset of stocks entirely from model training and evaluate temporal performance on those stocks. This answers a different question from testing future dates on known securities.
- If possible, create a point-in-time universe including historical constituents and delisted firms. Do not limit the universe to current survivors.

### 9.2 Dependence-aware uncertainty

Rows from the same stock are serially dependent; different stocks share market-date shocks; 20-day target windows overlap. The 8,666 test rows should not be treated as IID by default.

- Inspect the existing month-block bootstrap implementation and document exactly what gets resampled.
- Consider paired date-block bootstrap for the difference between two models evaluated on the exact same stock-date observations.
- Consider a two-way approach or clustered resampling that accounts for date and security dependence when suitable.
- Test sensitivity to block length.
- Report effective sample units, unique dates, stocks, and event counts, not only row count.
- For uncertainty in portfolio results, resample or analyse at a time unit appropriate to the portfolio return series.

Do not make claims that a particular inference method guarantees perfect coverage. Explain assumptions and sensitivity.

### 9.3 Multiple comparisons

The previous experiment tried 1,690 configurations and reports 87 test leaderboard rows. Selecting the maximum among many noisy test estimates leads to selection bias. In future work:

- predeclare a limited hypothesis-driven search;
- conduct tuning on walk-forward validation only;
- use multiple-comparison correction where inferential claims compare many models;
- make paired comparisons on identical observations;
- reserve an untouched evaluation period or use prospective paper-trading after the pipeline is frozen.

### 9.4 Report uncertainty of incremental value

For each task, explicitly report:

`delta_metric = metric(images + numerical features) - metric(numerical features only)`

Use the same test observations and a paired, dependence-aware confidence interval. For metrics where lower is better, orient the delta clearly (e.g. `baseline loss - image model loss`, so positive means improvement) and document the sign.

Statistical significance is not the same as economic value. Report both with clear caveats.

---

## 10. Image preprocessing and representation experiments

Before modifying the architecture, inspect and record the exact image renderer.

Questions to answer:

- Are all three images rendered as of the same forecast cutoff timestamp?
- Does the 1H chart contain any bars that would not be available by the actual decision time?
- Are charts normalised identically across symbols and dates?
- Does fill-to-panel scaling create artificial shape differences due to varying ranges or bar counts?
- Are y-axis labels, tick labels, gridlines, colours, volume, moving averages or any indicators included? Document the exact renderer rather than assuming.
- Are images always resized using the pretrained model's expected resolution and channel normalisation?
- Are train and live-dashboard screenshots rendered through the exact same canonical function?
- Are empty bars, holidays, early closes, missing periods and split-adjusted prices handled consistently?

Potential image experiments, each evaluated through the same folds:

1. Existing chart renderer.
2. A renderer with fixed and documented axes/scaling.
3. A normalised relative-price representation.
4. One timeframe at a time and pairwise combinations.
5. A shared multi-timeframe encoder versus separate encoders only if there is a reason to test it.

Keep these experiments hypothesis-driven. Do not add visual decoration or indicators simply because a more elaborate picture seems likely to help.

---

## 11. Numerical features and new input signals

The current model uses 33 numerical features. Audit their definitions, lookback dates, missing-value treatment, redundancy, and whether any feature leaks target/future information.

Recommended process:

1. Generate a data dictionary for all 33 features, with formula, lookback, units, timestamp alignment, missing-value policy and source.
2. Check features using unit tests on small hand-calculated examples.
3. Ensure no rolling statistic includes the forecast target period or future values.
4. Examine feature stability and drift over time; do not select features using test correlations.
5. Compare simple, interpretable groups (returns/range, volatility, gaps, volume, trend) with full features using validation-only ablations.
6. Do not add many indicators without a research hypothesis and an ablation. More features increase the risk of tuning to noise.

Potential additions should be considered only after the above audit. Candidate classes include lagged returns, rolling range, realised-volatility estimates, gap features, volume surprise/liquidity proxies and market/sector context. Confirm that each is available at the forecast time and that the same values could be produced live.

---

## 12. Monte Carlo path simulation and risk outputs

The report describes a simulation that samples volatility forecasts/errors, uses Student-t daily shocks, estimates intraday-low ranges and reweights paths so that the fraction crossing a 10% drawdown matches the drawdown model's calibrated probability. It reports approximately 89% coverage for the nominal 90% band and 51% coverage for the nominal 50% band in the test years.

This is encouraging coverage, but it does not by itself validate the full simulated path distribution or tail risks.

Recommended audits and metrics:

- Verify the forecast distribution's scale and annualisation conversion.
- Measure coverage by time period, sector, predicted-risk decile and stress regime.
- Report interval width along with coverage; trivially wide intervals can achieve coverage.
- Check coverage for 50%, 80%, 90% and 95% intervals if sample size permits.
- Evaluate drawdown probability calibration independently of the reweighting mechanism.
- Document how reweighting is performed and whether it changes other path properties.
- Validate autocorrelation, volatility clustering, skewness and tail behaviour against realised paths where feasible.
- Simulate correlated multi-stock scenarios before making portfolio-level stress claims. Independently simulated stock paths will generally understate shared market shocks.
- Recalibrate only on appropriate historical validation data and retest on later unseen periods.

Do not equate a calibrated event probability with a calibrated full path distribution.

---

## 13. Repository and engineering expectations

Claude Code should first map the actual repository and use its existing framework where possible. Do not assume file names, libraries or cloud resources before inspection.

### Expected outputs after the audit

1. `docs/triFrame_audit.md` — source-grounded review of current implementation and metric discrepancies.
2. `configs/` (or the project's existing configuration path) — immutable, versioned experiment configurations.
3. A baseline manifest with code/config/data hashes and environment versions.
4. Separate logs/predictions for each task and model.
5. A metrics table with consistent provenance fields.
6. A compact experiment matrix and a stopping/selection policy.
7. Tests for target alignment, split purging, image cutoff timestamps, PCA/scaler fitting, QLIKE implementation and metric reproducibility.
8. A results document separating validation findings, historical test diagnostics and any genuinely untouched evaluation results.

### Reproducibility

- Fix and log seeds for Python, NumPy, ML framework and data loaders where possible.
- Record library versions, device type, precision, batch size, effective batch size, gradient accumulation, optimizer, learning-rate schedule, weight decay, number of trainable parameters, epoch and checkpoint.
- Persist model predictions for each split so metrics can be reproduced without rerunning the full encoder.
- Include a configuration hash and prediction hash for each metric record.
- Do not overwrite past baseline artifacts.
- Use consistent CPU/GPU/TPU determinism settings where practical and document any nondeterministic operations.

### Unit tests worth adding

- No feature row uses a timestamp later than the forecast cutoff.
- Every target uses the correct forward horizon and reference date.
- No label interval crosses a split boundary after purge/embargo.
- Each target and prediction has aligned stock/date identifiers.
- Preprocessing, PCA, calibration and thresholds are fit only on allowed data.
- QLIKE receives values on the expected positive variance scale.
- Brier skill score uses an explicitly defined and properly estimated reference forecast.
- AUC/AP and accuracy report class prevalence and threshold conventions.
- NaN/inf metrics are surfaced as failures or explicitly documented, not silently accepted.
- The inference-time renderer gives the same dimensions/channel normalisation and feature schema as training.

---

## 14. Prioritised implementation roadmap

### Phase 0 — Baseline lock and audit (do first)

- Inspect files/code and establish the source of the discrepant metrics.
- Resolve holdout status.
- Verify targets, splits, preprocessing fit boundaries and QLIKE/BSS implementations.
- Reproduce the current reported metrics from saved predictions.
- Preserve a tagged baseline and produce a manifest.

**Exit criteria:** the exact metric table can be reproduced, all differences are explained, and leakage/data issues are documented.

### Phase 1 — Validation redesign

- Build chronological walk-forward folds for research iteration.
- Add purge/embargo that respects the longest label horizon.
- Ensure date and stock-level dependencies are considered in uncertainty estimation.
- Add a separate stock-held-out test if feasible.
- Do not use the existing test leaderboard to choose a model.

**Exit criteria:** all candidate models can be compared fairly on the same folds with reproducible metrics and confidence intervals.

### Phase 2 — Strong baselines

- Volatility: historical rolling, EWMA, GARCH-family, numerical-only regression/boosting.
- Drawdown: numerical-only logistic/boosting plus proper calibration baselines.
- O5: numerical-only classification plus continuous excess-return/ranking baselines as a separately defined experiment.

**Exit criteria:** the project knows what value exists before introducing extra visual complexity.

### Phase 3 — Focused fine-tuning

- Inspect current learning curves.
- Run frozen / partial unfreezing / full fine-tuning with an early-stopping rule.
- Use small, predeclared learning-rate and epoch ranges.
- Use multiple seeds for top candidates.
- Keep target-specific results separate.

**Exit criteria:** robust validation improvement over the frozen representation or a clear conclusion that fine-tuning adds no reliable value.

### Phase 4 — Feature/image ablations

- Perform one-timeframe and multi-timeframe ablations.
- Compare canonical rendering variants only with a defined hypothesis.
- Compare numerical-only versus image+numerical deltas using paired confidence intervals.

**Exit criteria:** evidence about which, if any, image branch adds useful information.

### Phase 5 — Economic evaluation

- Use volatility forecasts in defined portfolio-risk rules.
- Use drawdown probabilities in a defined risk-control policy.
- Use relative-return scores in a defined ranked portfolio only if the signal survives validation.
- Include transaction costs, slippage assumptions, turnover and exposures.
- Compare against simple, investable baselines and no-model control.

**Exit criteria:** demonstrate whether predictive scores translate into practical improvement. If they do not, report that honestly.

### Phase 6 — Final independent evaluation

- Freeze code, configuration, features, target definitions, selection rules and thresholds.
- Use a period or prospective paper-trading period not used in iterative selection.
- Do not reopen the final period to choose another model after seeing its result. If it is reused, label it as development/test data and establish a new future evaluation period.

---

## 15. Reporting template for each future experiment

Every experiment report should include:

### Experiment identification

- Hypothesis and why it is plausible.
- Target and exact target definition/version.
- Model/backbone/head and trainable layers.
- Input features and image-renderer version.
- Data window, universe definition and split IDs.
- Hyperparameters and selection rule.
- Seed(s), software versions and hardware.

### Predictive performance

- Primary validation metric.
- All secondary metrics relevant to the target.
- Baseline score and paired delta.
- Confidence intervals and inference methodology.
- Metrics by period, sector, market regime and stock where sample sizes permit.
- Calibration/evaluation coverage details for probabilistic models.

### Generalisation and risk

- Training versus validation curves.
- Variation across seeds.
- Performance drop across time/stock-held-out settings.
- Data quality, missingness and survivorship limitations.
- Any evidence of selection bias or multiple testing.

### Economic value

- Explicit portfolio rule, if any.
- Costs, turnover, exposure and benchmark.
- Return, Sharpe, drawdown, tail risk and stability.
- Whether the economic result was evaluated on data that did not participate in design/selection.

### Conclusion labels

Use one of these exact categories where appropriate:

- **SUPPORTED:** evidence is consistent with the hypothesis under the predeclared evaluation design.
- **NOT SUPPORTED:** no reliable evidence for the hypothesis under the evaluation design.
- **INCONCLUSIVE:** evidence is too uncertain or the experiment has unresolved limitations.
- **INVALID / NEEDS RE-RUN:** leakage, data errors, metric mismatch or holdout contamination prevents interpretation.

Avoid saying “proved” or “guaranteed.”

---

## 16. Research decisions and recommendations

### Recommended current project positioning

Describe TriFrame as a **multi-timeframe stock-risk forecasting system that tests whether deep visual representations add value beyond numerical price-derived features**.

Current findings justify these careful statements:

- The numerical and combined models show predictive information about future volatility in the reported historical test period.
- Numerical features provide moderate discrimination for the defined drawdown event.
- The current five-day relative-outperformance model has not shown a reliable edge.
- The formal incremental tests do not currently support a claim that chart images materially improve volatility or O5 predictions beyond numerical features.
- The NO TRADE flag stratifies risk in the reported test sample, but its economic value requires a portfolio intervention test.

Do not currently claim:

- that chart images outperform numerical features in a statistically established manner;
- that the system reliably predicts stock price direction;
- that more epochs necessarily improve model quality;
- that the global minimum has to be found;
- that current results prove profitability or improved investment returns;
- that the 2001–2009 training sensitivity analysis eliminates survivorship bias;
- that the 2023–2026 data remain untouched without an explicit artifact/log audit.

### Priority order

1. Reconcile metric/provenance discrepancies and establish holdout integrity.
2. Validate data and target construction.
3. Build sound walk-forward evaluation and dependence-aware uncertainty.
4. Strengthen task-specific numerical baselines.
5. Fine-tune systematically with early stopping and multiple seeds.
6. Test visual incremental value through ablations and paired comparisons.
7. Test portfolio utility only after predictive validity is established.

The objective is **not to make every metric larger at all costs**. It is to improve predictive validity, statistical reliability and economic usefulness without contaminating evaluation data.

---

## 17. Exact instruction to execute next

Claude Code: **start with read-only inspection**. Do not change model code or launch a large training run yet. Produce a report that maps the implementation to Sections 1–5, resolves or isolates the discrepancies in Section 4, and identifies the smallest set of fixes/tests needed to reproduce the existing results.

Then propose a concise experiment plan with:

- hypothesis;
- model/configuration changes;
- train/validation fold(s);
- primary metric and secondary metrics;
- selection rule;
- compute budget estimate based on the actual environment;
- leakage/overfitting controls;
- expected artifact outputs;
- explicit stopping rule.

Only after the audit should implementation begin, using versioned experimental configurations and preserving all prior results.
