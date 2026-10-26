# TriFrame CV - end-to-end run report (s500, notebook v4)

Run s500-v4-20261021T1905; code a76bc03f2e9e2cd5; config 0de8a5ceef8fe17f. Generated 2026-10-21T23:38:39+00:00 from the values computed in this notebook.

## Status of the evaluation periods
- Validation 2017-2019: used to choose and calibrate (the only selection data).
- Test 2020-2022: historical diagnostic (already read by earlier runs); used for nothing.
- 2023 onward: exposed (opened 2 times before this run); not opened in this run.

## Data
- Universe: 979 US stocks; this run: 500 stocks (containing the 60-stock and 120-stock samples of earlier runs), 236,520 chart sets (weekly 104 / daily 126 / hourly 140 bars) with 33 numeric features.
- Splits: train ('2001-01-01', '2017-01-01'), validation ('2017-01-01', '2020-01-01'), test ('2020-01-01', '2023-01-03') (20-day purge).

## Models compared
- 10 frozen backbones and 9 fine-tuned ones (ft_resnet18, ft_efficientnet_b0, ft_vit_base_patch16_224, ft_convnext_tiny, ft_swin_tiny_patch4_window7_224, ft_vit_small_patch14_dinov2, ft_mobilenetv3_large_100, ft_p25.convnext_tiny, ft_p50.swin_tiny_patch4_window7_224); PCA 128 per view; linear and boosted-tree heads; baselines HAR, rolling 22-day, EWMA, GARCH(1,1); 1881 configurations tried on validation.
- Fine-tuning: With early stopping on 2015-2016 (so the 2017-2019 validation years were never used by fine-tuning), fine-tuning raised validation volatility R² (numbers + images) for 9 of 9 runs (significantly for 7, significantly worse for 0; mean change +0.0141) and drawdown AUC for 8 of 9 (significantly for 3). Total fine-tuning time 158 GPU-minutes (15 runs, including the study and the extra seeds).

## Chosen model per target (pre-registered rule) and results
| Target | Chosen inputs / head | Validation | Test (historical diagnostic) |
|---|---|---|---|
| 20-day volatility (logrv20) | NUM+IMG:ft_convnext_tiny / tree | R2 0.5923; wmR2 0.5967; QLIKE 0.2406 | R2 0.5031; wmR2 0.5718; QLIKE 0.5996; RMSE 0.3811 |
| beats the median stock over 5 days (y_out5) | NUM / tree | AUC 0.5139; BSS 0.0006; ECE 0.0030 | AUC 0.5099; AP 0.5101; precision 0.5033; recall 0.9973; F1 0.6690; BSS -0.0008; ECE 0.0122 |
| drawdown of 10 % or more within 20 days (dd20_le_10) | NUM+IMG:ft_convnext_tiny / tree | AUC 0.7389; BSS 0.1282; ECE 0.0089 | AUC 0.6477; AP 0.4217; precision 0.3922; recall 0.6489; F1 0.4889; BSS 0.0455; ECE 0.0582 |

## Did training reach the minimum? (Stage 6D)
- Ridge head (volatility): GLOBAL minimum reached (certified) (closed form; gradient 6e-16; 5 random starts converge)
- Logistic head (drawdown): GLOBAL minimum reached (certified) (5 random starts agree to 3e-14; gradient 2e-08)
- Boosted trees (V, NUM+IMG:ft_convnext_tiny): minimum of the validation curve kept (early stopping) (validation rmse lowest at 364 trees; training keeps falling)
- Boosted trees (O5, NUM): minimum of the validation curve kept (early stopping) (validation logloss lowest at 104 trees; training keeps falling)
- Boosted trees (D10, NUM+IMG:ft_convnext_tiny): minimum of the validation curve kept (early stopping) (validation logloss lowest at 79 trees; training keeps falling)
- Fine-tuned network (ConvNeXt-Tiny (fine-tuned)): converged to a stable minimum of the held-out loss; a GLOBAL minimum cannot be certified for a network (training loss fell from 1.461 to 1.311; held-out loss lowest at epoch 2 of 4; lowest point on the line at alpha 1.0; seeds: validation R² sd 0.007)

## Evaluation metrics (Stage 7E; thresholds fixed on validation)
- D10, validation, chosen (NUM+IMG:ft_convnext_tiny / tree): AUC 0.7404, precision 0.358, recall 0.553, F1 0.434, accuracy 0.743 (majority 0.822), prevalence 0.178
- D10, validation, numbers only: AUC 0.7257, precision 0.340, recall 0.564, F1 0.424, accuracy 0.727 (majority 0.822), prevalence 0.178
- D10, test, chosen (NUM+IMG:ft_convnext_tiny / tree): AUC 0.6477, precision 0.392, recall 0.649, F1 0.489, accuracy 0.590 (majority 0.698), prevalence 0.302
- D10, test, numbers only: AUC 0.6533, precision 0.366, recall 0.741, F1 0.490, accuracy 0.535 (majority 0.698), prevalence 0.302
- O5, validation, chosen (NUM / tree): AUC 0.5163, precision 0.499, recall 1.000, F1 0.666, accuracy 0.499 (majority 0.501), prevalence 0.499
- O5, validation, numbers only: AUC 0.5139, precision 0.499, recall 0.991, F1 0.664, accuracy 0.500 (majority 0.501), prevalence 0.499
- O5, test, chosen (NUM / tree): AUC 0.5099, precision 0.503, recall 0.997, F1 0.669, accuracy 0.504 (majority 0.503), prevalence 0.503
- O5, test, numbers only: AUC 0.5101, precision 0.504, recall 0.969, F1 0.663, accuracy 0.505 (majority 0.503), prevalence 0.503
- V, validation, chosen (NUM+IMG:ft_convnext_tiny / tree): R2 0.5923, RMSE 0.3208, MAE 0.2500
- V, validation, numbers only: R2 0.5461, RMSE 0.3385, MAE 0.2632
- V, test, chosen (NUM+IMG:ft_convnext_tiny / tree): R2 0.5031, RMSE 0.3811, MAE 0.2655
- V, test, numbers only: R2 0.5004, RMSE 0.3821, MAE 0.2686

## Test leaderboard (descriptive; best head per model)
- 20-day volatility (logrv20) (R2): ViT-B/16 0.511; ResNet-18 0.511; DINOv2 ViT-S/14 0.510; ConvNeXt-Tiny (fine-tuned, last 25%) 0.510; ViT-B/16 (fine-tuned) 0.509; MobileNetV3-Large 0.509; Swin-T 0.508; ConvNeXt-Tiny 0.508
- beats the median stock over 5 days (y_out5) (AUC): ResNet-18 0.513; ConvNeXt-Tiny (fine-tuned) 0.512; Swin-T (fine-tuned) 0.512; MobileNetV3-Large 0.512; ResNet-18 (fine-tuned) 0.511; ViT-B/16 (fine-tuned) 0.511; Swin-T (fine-tuned, last 50%) 0.510; Numbers only (33 features) (chosen) 0.510
- drawdown of 10 % or more within 20 days (dd20_le_10) (AUC): ConvNeXt-Tiny (fine-tuned, last 25%) 0.659; ResNet-18 0.658; ViT-B/16 (fine-tuned) 0.657; EfficientNet-B0 (fine-tuned) 0.657; ViT-B/16 0.657; EfficientNet-B0 0.656; ConvNeXt-Tiny 0.656; DINOv2 ViT-S/14 0.656

## Hypotheses (rules fixed before the test years were read)
- H1, chart images add to the numbers (test): **not supported**. V: delta R2 +0.0027 [-0.0184, +0.0265] (quarter blocks [-0.0171, +0.0315]); O5: delta AUC -0.0023 [-0.0094, +0.0045] (quarter blocks [-0.0087, +0.0048])
- H2, backbones differ (validation): **supported** (10 of 21 pairs after Holm).
- H3, abstention improves accepted direction calls (test): **not supported** (gain at 10% coverage +0.0082 [-0.0052, +0.0194]).
- Direction as a continuous ranking (Stage 7D): NOT SUPPORTED (no reliable cross-sectional ranking signal).

## Controls (validation)
- label_shuffle_V: R2 -0.1229, wmR2 0.0054; label_shuffle_O5: AUC 0.4951; majority_class_accuracy_O5: 0.4990

## NO TRADE rule on the test years
- Flags 48.3% of cases; realised volatility 55.7% flagged vs 29.8% other; 10% drawdowns 40.1% vs 21.0% (risk stratification; no portfolio test yet).

## Monte Carlo simulation (Stage 7M)
- Student-t 5 d.f., intraday-low range 1.25; 90% band coverage validation 90.9%, test 88.4%; 50% band 50.8% / 48.6%.

## Training-history sensitivity (test metric with / without 2001-2009 rows; not a survivorship test)
- V: R2 0.5031 / 0.5007
- O5: AUC 0.5099 / 0.5112
- D10: AUC 0.6477 / 0.6476

## More data: earlier runs vs this run on the earlier runs' own stocks and rows
- pilot, 60 stocks, V R2: validation 0.5526 -> 0.5892; test 0.4958 -> 0.5102
- pilot, 60 stocks, V wmR2: validation 0.5621 -> 0.5986; test 0.5500 -> 0.5723
- pilot, 60 stocks, V MAE: validation 0.2544 -> 0.2438; test 0.2677 -> 0.2620
- pilot, 60 stocks, D10 AUC: validation 0.7113 -> 0.7296; test 0.6559 -> 0.6487
- pilot, 60 stocks, D10 BSS: validation 0.0946 -> 0.1186; test 0.0617 -> 0.0285
- pilot, 60 stocks, D10 ECE: validation 0.0090 -> 0.0191; test 0.0581 -> 0.0617
- pilot, 60 stocks, O5 AUC: validation 0.5159 -> 0.5148; test 0.5098 -> 0.5026
- v3, 120 stocks, V R2: validation 0.5831 -> 0.5944; test 0.5029 -> 0.5129
- v3, 120 stocks, V wmR2: validation 0.5877 -> 0.6002; test 0.5720 -> 0.5813
- v3, 120 stocks, V MAE: validation 0.2534 -> 0.2486; test 0.2722 -> 0.2688
- v3, 120 stocks, D10 AUC: validation 0.7126 -> 0.7368; test 0.6625 -> 0.6539
- v3, 120 stocks, D10 BSS: validation 0.1008 -> 0.1204; test 0.0886 -> 0.0468
- v3, 120 stocks, D10 ECE: validation 0.0051 -> 0.0130; test 0.0279 -> 0.0550
- v3, 120 stocks, O5 AUC: validation 0.5220 -> 0.5127; test 0.5075 -> 0.5113

## Where everything is
- Results, predictions, metric provenance (metrics_provenance.csv), figures, ledger, PDF: https://huggingface.co/datasets/KrishnaHarshaTangirala/triframe-cv-runs (private)
- Model bundle: https://huggingface.co/KrishnaHarshaTangirala/triframe-cv-model-s500 (private)
- Dashboard: streamlit_app.py in the GitHub repo (deploy on Streamlit Cloud, docs/DEPLOY.md)

Limits: survivor US stocks from one vendor; live inputs from Yahoo Finance; research output, not investment advice.