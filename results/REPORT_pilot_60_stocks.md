# TriFrame CV - end-to-end run report (quick)

Generated 2026-10-19T20:19:32+00:00 from the values computed in this notebook.

## Data
- Universe: 979 US stocks; this run: 60 stocks, 27,313 chart sets (weekly 104 / daily 126 / hourly 140 bars each) with 33 numeric features.
- Splits: train ('2001-01-01', '2017-01-01'), validation ('2017-01-01', '2020-01-01'), test ('2020-01-01', '2023-01-03') (20-day purge); holdout from 2023-01-03.

## Models compared
- 10 frozen backbones (resnet18, efficientnet_b0, vit_base_patch16_224, convnext_tiny, swin_tiny_patch4_window7_224, vit_small_patch14_dinov2, mobilenetv3_large_100, resnet18_random_s0, resnet18_random_s1, resnet18_random_s2), PCA 128 per view, linear and boosted-tree heads; 1200 configurations tried on validation (all logged in ledger.csv).

## Chosen model per target (pre-registered rule) and test results (2020-2022)
| Target | Chosen inputs / head | Validation | Test |
|---|---|---|---|
| 20-day volatility (logrv20) | NUM+IMG:swin_tiny_patch4_window7_224 / tree | R2 0.5526; wmR2 0.5621; QLIKE 0.2535 | R2 0.4958; wmR2 0.5500; QLIKE 0.5563 |
| beats the median stock over 5 days (y_out5) | NUM / tree | AUC 0.5159; BSS 0.0006; ECE 0.0038 | AUC 0.5098; BSS -0.0017; ECE 0.0082 |
| drawdown of 10 % or more within 20 days (dd20_le_10) | NUM+IMG:vit_small_patch14_dinov2 / tree | AUC 0.7113; BSS 0.0946; ECE 0.0090 | AUC 0.6559; BSS 0.0617; ECE 0.0581 |

## Hypotheses (rules fixed before the test years were opened)
- H1, chart images add to the numbers (test): **not supported**. V: delta R2 +0.0039 [-0.0044, +0.0123]; O5: delta AUC -0.0071 [-0.0261, +0.0121]
- H2, backbones differ (validation): **supported** (3 of 21 pairs after Holm).
- H3, abstention improves accepted direction calls (test): **not supported** (gain at 10% coverage -0.0050 [-0.0469, +0.0365]).

## Controls (validation)
- label_shuffle_V: R2 -0.1151, wmR2 0.0046; label_shuffle_O5: AUC 0.5065; majority_class_accuracy_O5: 0.5111
- Random-weight backbones (3 seeds), image-only volatility R2: resnet18_random_s0 -0.0274, resnet18_random_s1 -0.0372, resnet18_random_s2 -0.0378

## NO TRADE rule on the test years
- Flags 56.0% of cases; realised volatility 50.2% flagged vs 27.2% other; 10% drawdowns 37.1% vs 18.0%. Direction shown as a call on the dashboard: False (validation check True, test AUC CI lower bound 0.498).

## Survivorship sensitivity (test metric with / without 2001-2009 rows in training)
- V: R2 0.4958 / 0.4881
- O5: AUC 0.5098 / 0.5114
- D10: AUC 0.6559 / 0.6580

## Holdout (opened once)
- V|all: R2 0.5497; wmR2 0.5769; QLIKE 0.2822
- V|to_2026-02-28: R2 0.5353; wmR2 0.5681; QLIKE 0.2987
- V|after_2026-02-28: R2 0.6135; wmR2 0.6290; QLIKE 0.1868
- O5|all: AUC 0.4900; BSS -0.0055; ECE 0.0235
- O5|to_2026-02-28: AUC 0.4843; BSS -0.0063; ECE 0.0295
- O5|after_2026-02-28: AUC 0.5212; BSS -0.0009; ECE 0.0176

## Where everything is
- Results, embeddings, figures, ledger, logs: https://huggingface.co/datasets/KrishnaHarshaTangirala/triframe-cv-runs (private)
- Model bundle: https://huggingface.co/KrishnaHarshaTangirala/triframe-cv-model (private)
- Dashboard: https://huggingface.co/spaces/KrishnaHarshaTangirala/triframe-cv-dashboard (app https://krishnaharshatangirala-triframe-cv-dashboard.hf.space)

Limits: 979 survivor US stocks from one vendor; live inputs from Yahoo Finance; research output, not investment advice.