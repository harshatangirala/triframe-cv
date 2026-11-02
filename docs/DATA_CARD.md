# DATA CARD: TriFrame tier-A bars (frozen)
Frozen 2026-10-11T19:36:51 | root sha256 `f40e315ea5659f66e1dc2a11...` | 3,919 files | data through **2026-10-09**

## What is in `final/`
| Item | Content |
|---|---|
| `1day/ 1hour/ 1week/` | **979 verified stocks** (941 original + 38 additions). 6,069,829 daily, 42,129,515 hourly, 1,259,665 weekly rows |
| `meta/indicators/` | SMA20/50, MACD(12,26,9), RSI14, BB(20,2), ATR14, 20-bar volume average, band width, volume ratio |
| `meta/labels.parquet` | 5/20-session total returns (dividends included), 3-class label, cross-sectional label, volatility, downside tail, MAE/MFE, eligibility |
| `meta/candidates.parquet` | STRAT-01..04 long/short trades with stop, target, exit reason, gross and net R at three cost levels |
| `meta/split_manifest.json` | era calendar, walk-forward folds, holdout, purge and embargo rule |
| `reports/` | validator, check results, overnight moves above 40% |
| `extension_verification.csv` | per-stock verification of the Mar-Oct 2026 extension |

## Sources and construction
* 1-minute feed (Finnhub-derived, Hugging Face `mito0o852/OHLCV-1m`), cleaned and aggregated to hourly/daily/weekly; split-adjusted from Yahoo split history (5 split dates corrected to the observed price jump).
* Gaps in the feed filled from Yahoo daily bars (source = `yahoo`); Mar-Oct 2026 extension from Yahoo daily + hourly (source = `yahoo_ext`), on the same split basis through a constant factor.
* Prices are split-adjusted, **not** dividend-adjusted. Dividends are used only in labels.
* All hourly bars are regular trading hours, 09:30 aligned (7 per session, 4 on early closes).

## Verification (tier A)
| Check | Result |
|---|---|
| Daily close vs Yahoo, 941 original, full history | std 0.242% |
| Daily close vs Yahoo, 38 additions | std 0.127% |
| Hourly close vs Yahoo, last ~16 months | std 0.044% (941), 0.046% (additions) |
| Sessions vs official NYSE calendar | **0 missing, 0 extra** in 979 stocks |
| OHLC violations / duplicates / non-positive prices | 0 / 0 / 0 |
| Weekly vs aggregated daily (129,337 weeks) | 0 mismatches |
| Indicators vs seeded TA-Lib-style RSI/ATR | RSI diff 4e-7, ATR diff 2e-9 (relative) |
| Indicators on a 200-bar window vs full history | max relative diff 3e-6 |
| STRAT-01/02 triggers vs independent evaluator | 0 mismatches in 76,212 evaluations |
| Label audit (200 random samples, independent recomputation) | 0 mismatches |
| Barrier simulator hand-built cases | 7 of 7 pass |

## Known limitations (read before reporting results)
1. **Survivorship bias remains.** Tier A has no delisted names (Yahoo has no history for them). Tier B (23 delisted/acquired names, `extra/`) is unverified and for stress tests only.
2. **S&P 500 coverage:** this universe covers 57.9% of S&P 500 member-days (2001-2026), 68.7% in 2026.
3. Single-vendor lineage (Finnhub + Yahoo). No CRSP/TAQ. Volume is about 90% of consolidated volume in the Finnhub-built part; the Yahoo extension is consolidated (median step at the seam 1.29x).
4. The extension (Mar-Oct 2026) is Yahoo-only. Daily open vs first hourly open agree within 10 bps on 92% of days.
5. No earnings dates, no shares outstanding, no bid-ask spreads. Costs are fixed-bps scenarios.
6. 1-minute data stops at 2026-02-27 and is not part of the 1D/1W/1H model.
7. STRAT-03 has fewer than 2,000 development-era trades per side: flagged **low evidence**.
8. SMCI was OTC-quoted Aug 2018 to Jan 2020: those 349 sessions are Yahoo daily only (no hourly bars) and are excluded as decision dates.
9. Eligibility: >=200 bars of history, no missing session in the last 200, hourly bars complete, 20-day average dollar volume >= $1M. This keeps 5,568,837 stock-days.

## Rule-only baseline (non-skipped trades, all years; net R at 10 bps round trip)
|                       |   trades |   target_rate |   stop_rate |   avg_net_R_base |   median_bars_held |
|:----------------------|---------:|--------------:|------------:|-----------------:|-------------------:|
| ('STRAT-01', 'LONG')  |    12152 |         0.159 |       0.359 |            0.038 |                 11 |
| ('STRAT-01', 'SHORT') |     9327 |         0.122 |       0.393 |           -0.146 |                 10 |
| ('STRAT-02', 'LONG')  |    27358 |         0.09  |       0.259 |            0.079 |                 20 |
| ('STRAT-02', 'SHORT') |    33164 |         0.068 |       0.298 |           -0.107 |                 20 |
| ('STRAT-03', 'LONG')  |     1823 |         0.07  |       0.311 |            0.003 |                  3 |
| ('STRAT-03', 'SHORT') |     1862 |         0.071 |       0.356 |           -0.12  |                  3 |
| ('STRAT-04', 'LONG')  |    36459 |         0.064 |       0.087 |           -0.007 |                  3 |
| ('STRAT-04', 'SHORT') |    20594 |         0.054 |       0.079 |           -0.056 |                  3 |

These are the unfiltered rules. They are roughly break-even after costs; shorts lose on average (a rising market). The model's job is to filter them.

Skipped candidates: {'reward_to_risk_below_minimum': 2417, 'opposite_signal_same_day': 2328, 'gap_over_1p5_atr': 2298, 'entry_too_extended': 315, 'unresolved_at_data_end': 251, 'target_wrong_side': 96, 'invalid_stop_geometry': 62}
