# TriFrame CV: results of the 500-stock run (notebook v4)

236,520 chart sets from 500 US stocks, 2001-2022. Seven pretrained vision networks (each frozen and fine-tuned) against a
numbers-only baseline of 33 features. Every choice was made on the 2017-2019 validation years. The 2020-2022 test years were
already read by earlier runs, so they are a **historical diagnostic**, not a fresh test. The 2023+ period was opened twice
before and is not used here. Research output, not investment advice.

## Headline results

| Target | Chosen model | Validation 2017-2019 | Test 2020-2022 (95% CI) |
|---|---|---|---|
| 20-day volatility | numbers + fine-tuned ConvNeXt-Tiny charts, boosted trees | R² 0.592 | R² 0.503 (0.25 to 0.65), RMSE 0.381 |
| Drawdown of 10% or more in 20 days | numbers + fine-tuned ConvNeXt-Tiny charts, boosted trees | AUC 0.739 | AUC 0.648 (0.59 to 0.69) |
| Beats the median stock over 5 days | numbers only, boosted trees | AUC 0.514 | AUC 0.510 (0.500 to 0.521) |

## Did the scores improve over the 60- and 120-stock runs?

Compared on the earlier run's own stocks and exactly the same rows:

| Earlier run | Target | Earlier, test | s500, test | Change |
|---|---|---|---|---|
| pilot, 60 stocks | volatility R² | 0.496 | 0.510 | +0.014 |
| pilot, 60 stocks | drawdown AUC | 0.656 | 0.649 | -0.007 |
| v3, 120 stocks | volatility R² | 0.503 | 0.513 | +0.010 |
| v3, 120 stocks | drawdown AUC | 0.663 | 0.654 | -0.009 |
| v3, 120 stocks | direction AUC | 0.508 | 0.511 | about 0 |

Validation scores rose for every target (volatility +0.036 and +0.011, drawdown +0.019 and +0.024). On the test years the
gain is small for volatility and absent for drawdown and direction. These same-row changes have no confidence interval in the
report, so shifts of about 0.01 may be noise.

## Achievements

- **Fine-tuning tested honestly.** It now early-stops on 2015-2016 inside the training years, so the validation years stay clean.
  It still raised validation volatility R² in 9 of 9 comparisons (7 significant): ConvNeXt-Tiny +0.032, ViT-B/16 +0.028,
  Swin-T +0.025. Training only the last 25% or 50% of a network gained nothing, so the whole network has to adapt to charts.
- **Global minimum.** The ridge and logistic heads are convex and their global minimum is certified (zero gradient, identical
  answers from 5 random starts, a single bowl). Boosted trees keep the minimum of their validation curve. The fine-tuned network
  sits at the bottom of its valley in the loss landscape; a global minimum cannot be certified for a neural network.
- **Metrics and curves.** Precision, recall, F1, ROC-AUC, R², RMSE; ROC and precision-recall curves, confusion matrices,
  reliability plots; training, early-stopping and validation curves for all 7 networks; boosting and learning curves.
- **Beats the classical volatility models on the test years:** rolling 22-day about 0.42, EWMA about 0.45, HAR about 0.48,
  against 0.50.
- **Same conclusions at 60, 120 and 500 stocks.** The NO TRADE flag separates risky cases (volatility 56% vs 30%, 10% drawdowns
  40% vs 21%), and the Monte Carlo 90% band held 88% of test outcomes.

## Observations

1. **Chart images add to the numbers on validation but not on test.** Validation: R² 0.592 vs 0.546. Test: +0.003 (95% CI -0.018
   to +0.026), so H1 is not supported. Direction AUC changed by -0.002 (-0.009 to +0.004).
2. **The drawdown pick looks lucky rather than better.** It had the best validation AUC but ranked last of 17 candidates on test,
   though all 17 are within about 0.01 of each other.
3. **2020 is the weak year.** Volatility R² by year: 2017 about 0.60, 2018 0.52, 2019 0.61, **2020 0.27**, 2021 0.63, 2022 0.57.
   The validation years contain no crisis.
4. **Drawdown calibration drifted.** The drawdown rate was about 18% in validation and 30% in test; calibration error rose from
   0.028 (v3) to 0.058.
5. **Two F1 scores mislead.** Direction F1 0.67 at recall 1.00 is what saying "yes" to every stock scores. Drawdown F1 0.49 compares
   with 0.46 for flagging everything, and its accuracy is below the 70% of always saying "no".
6. **More stocks has stopped helping much.** Learning curves flatten after about 250 stocks.
7. **No direction signal.** Validation log-loss is a coin flip's (0.6929 vs 0.6931); the weekly ranking experiment also failed.
8. **Controls are clean.** Best image-only R² 0.207, random-weight networks about -0.02, shuffled labels -0.12.

## What to improve next

1. Add volatility measured from the hourly bars to the numeric features (today all 33 come from daily bars).
2. Evaluate year by year across 2008-2022 so both crises are judged.
3. Average the top few models instead of picking one winner.
4. Floor the lowest drawdown probability and recalibrate on a recent window.
5. Show the "flag everything" F1 and balanced accuracy beside every F1; add confidence intervals to the same-row table.
6. Add earnings dates (needs a data source with a suitable licence).
7. Train on all 979 stocks (expect only about +0.005 R²).
