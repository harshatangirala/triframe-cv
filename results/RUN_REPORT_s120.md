# TriFrame CV - end-to-end run report (s120)

Generated 2026-10-20T12:43:14+00:00 from the values computed in this notebook.

## Data
- Universe: 979 US stocks; this run: 120 stocks (including the pilot's 60), 56,099 chart sets (weekly 104 / daily 126 / hourly 140 bars each) with 33 numeric features.
- Splits: train ('2001-01-01', '2017-01-01'), validation ('2017-01-01', '2020-01-01'), test ('2020-01-01', '2023-01-03') (20-day purge); holdout from 2023-01-03.

## Models compared
- 10 frozen backbones (resnet18, efficientnet_b0, vit_base_patch16_224, convnext_tiny, swin_tiny_patch4_window7_224, vit_small_patch14_dinov2, mobilenetv3_large_100, resnet18_random_s0, resnet18_random_s1, resnet18_random_s2) and 7 fine-tuned ones (ft_resnet18, ft_efficientnet_b0, ft_vit_base_patch16_224, ft_convnext_tiny, ft_swin_tiny_patch4_window7_224, ft_vit_small_patch14_dinov2, ft_mobilenetv3_large_100), PCA 128 per view, linear and boosted-tree heads; 1690 configurations tried on validation (all logged in ledger.csv).
- Fine-tuning: Fine-tuning raised validation volatility R² (numbers + images) for 7 of 7 backbones (significantly for 5, significantly worse for 0; mean change +0.0142) and drawdown AUC for 6 of 7 (significantly for 1). Images-only volatility R² moved from 0.192 (frozen, average) to 0.215 (fine-tuned). Total fine-tuning time 38 GPU-minutes.

## Chosen model per target (pre-registered rule) and test results (2020-2022)
| Target | Chosen inputs / head | Validation | Test |
|---|---|---|---|
| 20-day volatility (logrv20) | NUM+IMG:ft_vit_base_patch16_224 / tree | R2 0.5831; wmR2 0.5877; QLIKE 0.2443 | R2 0.5029; wmR2 0.5720; QLIKE 0.5398 |
| beats the median stock over 5 days (y_out5) | NUM / tree | AUC 0.5220; BSS 0.0013; ECE 0.0062 | AUC 0.5075; BSS -0.0011; ECE 0.0149 |
| drawdown of 10 % or more within 20 days (dd20_le_10) | NUM / tree | AUC 0.7126; BSS 0.1008; ECE 0.0051 | AUC 0.6625; BSS 0.0886; ECE 0.0279 |

## Test leaderboard (descriptive; numbers + images, best head per model)
- 20-day volatility (logrv20) (R2): ViT-B/16 0.512; ConvNeXt-Tiny 0.511; DINOv2 ViT-S/14 0.510; MobileNetV3-Large 0.510; ResNet-18 0.509; MobileNetV3-Large (fine-tuned) 0.509
- beats the median stock over 5 days (y_out5) (AUC): ConvNeXt-Tiny (fine-tuned) 0.514; ConvNeXt-Tiny 0.513; ViT-B/16 0.512; DINOv2 ViT-S/14 (fine-tuned) 0.510; ResNet-18 (fine-tuned) 0.509; Numbers only (33 features) (chosen) 0.509
- drawdown of 10 % or more within 20 days (dd20_le_10) (AUC): ViT-B/16 (fine-tuned) 0.674; ResNet-18 (fine-tuned) 0.666; ConvNeXt-Tiny (fine-tuned) 0.665; ResNet-18 0.665; EfficientNet-B0 0.664; Numbers only (33 features) (chosen) 0.664

## Hypotheses (rules fixed before the test years were opened)
- H1, chart images add to the numbers (test): **not supported**. V: delta R2 +0.0030 [-0.0156, +0.0265]; O5: delta AUC +0.0001 [-0.0136, +0.0154]
- H2, backbones differ (validation): **supported** (13 of 21 pairs after Holm).
- H3, abstention improves accepted direction calls (test): **not supported** (gain at 10% coverage +0.0045 [-0.0192, +0.0291]).

## Controls (validation)
- label_shuffle_V: R2 -0.1071, wmR2 0.0000; label_shuffle_O5: AUC 0.5040; majority_class_accuracy_O5: 0.4976
- Random-weight backbones (3 seeds), image-only volatility R2: resnet18_random_s0 0.0054, resnet18_random_s1 -0.0009, resnet18_random_s2 0.0049

## NO TRADE rule on the test years
- Flags 49.7% of cases; realised volatility 54.0% flagged vs 28.4% other; 10% drawdowns 39.5% vs 19.2%. Direction shown as a call on the dashboard: False (validation check True, test AUC CI lower bound 0.496).

## Monte Carlo simulation (Stage 7M)
- Student-t 5 degrees of freedom, intraday-low range 1, drawdown matching by default: True. 90% band coverage: validation 91.1%, test 88.7%; 50% band: validation 50.7%, test 50.6%.

## Survivorship sensitivity (test metric with / without 2001-2009 rows in training)
- V: R2 0.5029 / 0.4959
- O5: AUC 0.5075 / 0.5009
- D10: AUC 0.6625 / 0.6597

## More data: this run vs the 60-stock pilot (same 60 stocks, same test rows)
- V R2: pilot 0.4958 -> this run 0.4948
- V wmR2: pilot 0.5500 -> this run 0.5621
- V MAE: pilot 0.2677 -> this run 0.2652
- D10 AUC: pilot 0.6559 -> this run 0.6587
- D10 BSS: pilot 0.0617 -> this run 0.0768
- D10 ECE: pilot 0.0581 -> this run 0.0342
- O5 AUC: pilot 0.5098 -> this run 0.5095

## Holdout (opened once)
- V|all: R2 0.5844; wmR2 0.5995; QLIKE 0.2538
- V|to_2026-02-28: R2 0.5721; wmR2 0.5933; QLIKE 0.2668
- V|after_2026-02-28: R2 0.6372; wmR2 0.6373; QLIKE 0.1784
- O5|all: AUC 0.5079; BSS -0.0010; ECE 0.0105
- O5|to_2026-02-28: AUC 0.5090; BSS -0.0009; ECE 0.0091
- O5|after_2026-02-28: AUC 0.5050; BSS -0.0017; ECE 0.0196

## Where everything is
- Results, embeddings, figures, ledger, logs, PDF report: https://huggingface.co/datasets/KrishnaHarshaTangirala/triframe-cv-runs (private)
- Model bundle: https://huggingface.co/KrishnaHarshaTangirala/triframe-cv-model-s120 (private)
- Dashboard: streamlit_app.py in the GitHub repo (deploy on Streamlit Cloud, docs/DEPLOY.md)

Limits: survivor US stocks from one vendor; live inputs from Yahoo Finance; research output, not investment advice.