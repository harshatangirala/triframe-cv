"""Fill the paper's tables from the result files and write TriFrame_paper.html (printed to PDF with a headless browser).
Usage: build_paper.py <run_results_s500_dir> <run_results_root>   (tables/ and fig/ sit beside this file)"""
import json, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R, ROOT = sys.argv[1:3]
S = json.load(open(os.path.join(HERE, "tables", "supplementary_results.json")))
lt = pd.read_csv(os.path.join(R, "stage7_leaderboard_test.csv"))
lv = pd.read_csv(os.path.join(R, "stage6L_leaderboard_validation.csv"))
FT = pd.read_csv(os.path.join(R, "stage5F_finetune_vs_frozen.csv"))
CH = "ConvNeXt-Tiny (fine-tuned)"
NUM = "Numbers only (33 features)"


def f(x, d=3):
    if x is None or (isinstance(x, float) and x != x):
        return "&ndash;"
    s = f"{x:.{d}f}"
    return s.replace("-", "&minus;")


def sg(x, d=3):
    return ("+" if x >= 0 else "") + f(x, d)


def ci(a, b, d=3):
    return f"[{f(a, d)}, {f(b, d)}]"


def table(num, title, head, rows, note, widths=None):
    h = "".join(f'<th{" class=l" if i == 0 else ""}>{c}</th>' for i, c in enumerate(head))
    body = ""
    for r in rows:
        if isinstance(r, str):
            body += f'<tr class="sub"><td colspan="{len(head)}">{r}</td></tr>'
        else:
            body += "<tr>" + "".join(f'<td{" class=l" if i == 0 else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return (f'<div class="tbl"><p class="ttl">Table {num}</p><p class="tname">{title}</p><p class="tnote">{note}</p>'
            f'<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>')


NM = {"IMG": "Numbers + images", "NUM": "Numbers only", "NUMH": "Numbers + hourly RV", "HAR": "HAR, daily range", "HARH": "HAR, hourly RV", "GARCH": "GARCH(1,1)", "EWMA": "EWMA", "ROLL22": "Rolling 22-day"}
T = {}

sp = pd.read_csv(os.path.join(R, "stage2_splits.csv"))
T["splits"] = table("I", "Summary Statistics by Sample Period", ["Period", "Stock-dates", "Stocks", "Dates", "Weeks", "Mean log vol.", "O5 rate", "D10 rate"],
                    [[{"train": "Train, 2001&ndash;2016", "val": "Validation, 2017&ndash;2019", "test": "Test, 2020&ndash;2022"}[r.split], f"{r.rows:,}", r.tickers, r.dates, r.weeks, f(r.vol_mean), f(r.up5_rate), f(r.dd10_rate)] for r in sp.itertuples()],
                    "Rows are chart sets for the 500-stock sample after a 20-session purge at each boundary. Mean log vol. is the mean of 20-day log realized volatility, equation (1). O5 is the rate of beating the median stock over five sessions. D10 is the rate of a 10% drawdown within 20 sessions. Weeks is the number of distinct calendar weeks.")

cards = pd.read_csv(os.path.join(R, "stage6L_model_cards.csv")); cards = cards[cards["kind"].eq("frozen") if "kind" in cards else cards.index >= 0] if "kind" in cards else cards
iv = lv[(lv.target == "V") & (lv.condition == "images only")].set_index("model")["R2"]
net_rows = [["ViT-B/16", "Transformer", "ImageNet-21k, then 1k", "85.8", 768], ["ConvNeXt-Tiny", "CNN", "ImageNet-12k, then 1k", "27.8", 768], ["Swin-T", "Transformer", "ImageNet-1k", "27.5", 768],
            ["DINOv2 ViT-S/14", "Transformer", "LVD-142M (self-supervised)", "21.6", 384], ["ResNet-18", "CNN", "ImageNet-1k", "11.2", 512], ["MobileNetV3-Large", "CNN", "ImageNet-1k", "4.2", 1280], ["EfficientNet-B0", "CNN", "ImageNet-1k", "4.0", 1280]]
T["networks"] = table("II", "Vision Networks", ["Network", "Family", "Pretraining data", "Params (M)", "Embedding", "Images-only val. R²"],
                      [r + [f(iv[r[0]])] for r in net_rows] + [["ResNet-18 &times; 3", "Control", "None (random weights)", "11.2", 512, f"{f(iv['ResNet-18 (random weights, seed 1)'])} to {f(iv['ResNet-18 (random weights, seed 0)'])}"]],
                      "Embedding is the output size per view before PCA to 128. The last column is validation R² for 20-day log volatility using frozen images alone.")

T["finetune"] = table("III", "Fine-Tuning: Validation Gains", ["Model", "Frozen R²", "Fine-tuned R²", "Gain", "95% CI", "Network&rsquo;s own R²", "D10 AUC", "Best / total epochs", "Minutes"],
                      [[r.model.replace(" (fine-tuned)", "").replace("(fine-tuned, ", "("), f(r.V_num_img_frozen), f(r.V_num_img_finetuned), sg(r.V_gain), ci(r.V_gain_lo, r.V_gain_hi), f(r.V_end_to_end), f(r.D10_num_img_finetuned), f"{r.best_epoch} / {r.epochs}", f"{r.minutes:.0f}"] for r in FT.sort_values("V_gain", ascending=False).itertuples()],
                      "Numbers plus images with the best head on 2017 to 2019. Fine-tuning learned from rows with labels ending before 2015 and stopped on 2015 to 2016. Network&rsquo;s own R² is the end-to-end network&rsquo;s forecast without a separate tree head. Numbers alone reach 0.546.")


def m(d, model, c):
    return d[(d.model == model) & (d.target == c[0]) & (d.condition == c[1])].iloc[0]


vI, vN, tI, tN = m(lv, CH, ("V", "numbers + images")), m(lv, NUM, ("V", "numbers only")), m(lt, CH, ("V", "numbers + images")), m(lt, NUM, ("V", "numbers only"))
TR = json.load(open(os.path.join(R, "stage7_test_results.json")))
T["headline"] = table("IV", "Chosen Models and Pre-Specified Tests", ["Target / metric", "Validation, chosen", "Validation, numbers", "Test, chosen", "Test, numbers"],
                      ["Volatility: numbers + fine-tuned ConvNeXt-Tiny, XGBoost",
                       ["R² (test 95% CI [0.246, 0.655])", f(vI.R2), f(vN.R2), f(tI.R2), f(tN.R2)], ["Within-month R²", f(vI.wmR2), f(vN.wmR2), f(tI.wmR2), f(tN.wmR2)], ["RMSE", f(vI.RMSE), f(vN.RMSE), f(tI.RMSE), f(tN.RMSE)], ["MAE", f(vI.MAE), f(vN.MAE), f(tI.MAE), f(tN.MAE)], ["Spearman", f(vI.Spearman), f(vN.Spearman), f(tI.Spearman), f(tN.Spearman)], ["QLIKE", f(vI.QLIKE), f(vN.QLIKE), f(tI.QLIKE), f(tN.QLIKE)],
                       "Drawdown of 10%: numbers + fine-tuned ConvNeXt-Tiny, XGBoost, calibrated", ["ROC-AUC (test 95% CI [0.589, 0.690])", "0.740", "0.726", "0.648", "0.653"], ["Precision / recall / F1", "0.36 / 0.55 / 0.43", "0.34 / 0.56 / 0.42", "0.39 / 0.65 / 0.49", "0.37 / 0.74 / 0.49"],
                       "Five-day direction: numbers only, XGBoost, calibrated", ["ROC-AUC (test 95% CI [0.500, 0.521])", "0.516", "0.514", "0.510", "0.510"],
                       "Pre-specified tests", ["H1, V: gain in test R²", "", "", "+0.003", "[&minus;0.018, +0.026]"], ["H1, O5: gain in test AUC", "", "", "&minus;0.002", "[&minus;0.009, +0.005]"], ["H2: network pairs differing (Holm)", "10 of 21", "", "", ""], ["H3: accuracy gain, top 10% confidence", "", "", "+0.008", "[&minus;0.005, +0.019]"]],
                      "Values come from the frozen run. Confidence intervals resample calendar months 1,000 times. For the volatility rows the numbers column is the XGBoost model on the 33 features. Thresholds for precision, recall and F1 are fixed on validation (0.223 for D10).")

ql = {"IMG": tI.QLIKE, "NUM": tN.QLIKE, "HAR": lt[(lt.model == "HAR volatility model") & (lt.target == "V")].iloc[0].QLIKE, "EWMA": lt[(lt.model == "EWMA volatility") & (lt.target == "V")].iloc[0].QLIKE, "ROLL22": lt[(lt.model == "Rolling 22-day volatility") & (lt.target == "V")].iloc[0].QLIKE}
rows = []
dm = S["dm"]; pr = S["paired"]
for r in S["vol_models"]:
    k = r["model"]; key = f"IMG-{k}"
    d = pr[key]["R2"] if key in pr else None; t = dm[key]["t_nw"] if key in dm else None
    rows.append([NM[k], f"{r['n']:,}", f(r["val_R2"]) if r["val_R2"] is not None else "&ndash;", f(r["R2"]), f(r["wmR2"]), f(r["RMSE"]), f(r["MAE"]), f(r["Spearman"]), f(ql.get(k)) if k in ql else "&ndash;", sg(d["delta"]) + " " + ci(d["lo"], d["hi"]) if d else "&ndash;", f(t, 2) if t is not None else "&ndash;"])
T["forecasters"] = table("V", "Volatility Forecasters on the Test Years", ["Model", "Rows", "Val. R²", "R²", "Within-month R²", "RMSE", "MAE", "Spearman", "QLIKE", "Images minus model: R²", "DM <i>t</i>"], rows,
                         "All models except the chosen image model are re-fitted in the supplementary analysis with one consistent code path. QLIKE comes from the run where available. The DM <i>t</i> is the Newey-West <i>t</i>-statistic (3 lags) on the monthly mean squared-error difference; negative values favor the image model. The chosen model&rsquo;s validation score is in Table IV.")

dc = pd.DataFrame(S["decomp"])
rows = []
for yr in ("all", 2020, 2021, 2022):
    a = dc[(dc.model == "IMG") & (dc.year.astype(str) == str(yr))].iloc[0]; b = dc[(dc.model == "NUM") & (dc.year.astype(str) == str(yr))].iloc[0]
    rows.append(["All test years" if yr == "all" else yr, f(a.mse, 4), f(b.mse, 4), f(a.mse_cross_section, 4), f(b.mse_cross_section, 4), f(a.mse - a.mse_cross_section, 4), f(b.mse - b.mse_cross_section, 4), f(a.share_market_level, 3), f(b.share_market_level, 3), f(a.mean_error, 3), f(b.mean_error, 3)])
T["decomp"] = table("VI", "Decomposing Squared Forecast Error", ["Sample", "MSE, images", "MSE, numbers", "Cross-section, images", "Cross-section, numbers", "Level, images", "Level, numbers", "Level share, images", "Level share, numbers", "Mean error, images", "Mean error, numbers"], rows,
                    "Equation (3). Level is the month-mean component of mean squared error and Cross-section the remainder. Mean error is realized minus forecast log volatility, so positive values are under-forecasts.")

rows = []
by = pd.DataFrame(S["by_year"]); by = by[(by.a == "IMG") & (by.b == "NUM")]
for y in (2020, 2021, 2022):
    a = by[(by.year == y) & (by.metric == "R2")].iloc[0]; b = by[(by.year == y) & (by.metric == "wmR2")].iloc[0]
    rows.append([y, "&ndash;", "&ndash;", sg(a.delta), ci(a.lo, a.hi), sg(b.delta), ci(b.lo, b.hi)])
for h in S["hetero"]:
    rows.append([h["group"], f"{h['n']:,}", f(h["R2_num"]), sg(h["d"]), ci(h["lo"], h["hi"]), sg(h["wm_d"]), ci(h["wm_lo"], h["wm_hi"])])
T["hetero"] = table("VII", "Gains from Images by Subsample, Test Years", ["Subsample", "Rows", "R², numbers", "Gain in R²", "95% CI", "Gain in within-month R²", "95% CI"], rows,
                    "Numbers plus images minus numbers only, with month-block bootstrap confidence intervals. Liquidity terciles sort on 20-day average dollar volume within each month.")

st = S["stress"]; p = pr["IMG-NUM"]
T["stress"] = table("VIII", "Stress Tests of the Within-Month Gain", ["Test", "Images", "Numbers", "Gain", "Interval"],
                    [["Within-month R², month blocks, 95%", f(S["vol_models"][0]["wmR2"]), f(S["vol_models"][1]["wmR2"]), sg(st["wmR2_month_block"]["delta"]), ci(*st["wmR2_month_block"]["ci95"])],
                     ["Same, 99% (Bonferroni, five metrics)", "", "", "", ci(*st["wmR2_month_block"]["ci99"])], ["Same, quarter blocks, 95%", "", "", "", ci(*st["wmR2_quarter_block"]["ci95"])], ["Same, resampling whole stocks, 95%", "", "", "", ci(*st["wmR2_stock_block"]["ci95"])],
                     ["Within-week R², 95%", f(st["within_week_R2"]["img"]), f(st["within_week_R2"]["num"]), sg(st["within_week_R2"]["img"] - st["within_week_R2"]["num"]), ci(*st["within_week_R2"]["ci95"])],
                     ["R² with month and stock fixed effects, 95%", f(st["twoway"]["img"]), f(st["twoway"]["num"]), sg(st["twoway"]["img"] - st["twoway"]["num"]), ci(*st["twoway"]["ci95"])],
                     ["Cross-sectional MSE, 95%", f(st["xs_mse"]["img"], 4), f(st["xs_mse"]["num"], 4), sg(st["xs_mse"]["delta"], 4), ci(*st["xs_mse"]["ci95"], 4)], ["Level MSE, 95%", f(st["level_mse"]["img"], 4), f(st["level_mse"]["num"], 4), sg(st["level_mse"]["delta"], 4), ci(*st["level_mse"]["ci95"], 4)],
                     ["Mean monthly cross-sectional R²", f(st["monthly_xs"]["mean_img"]), f(st["monthly_xs"]["mean_num"]), sg(st["monthly_xs"]["mean_diff"]), f"<i>t</i> = {f(st['monthly_xs']['t_nw'], 2)}; {st['monthly_xs']['share_pos'] * 100:.0f}% of months positive"],
                     ["Other paired metrics: Spearman", "", "", sg(p["Spearman"]["delta"]), ci(p["Spearman"]["lo"], p["Spearman"]["hi"])], ["Other paired metrics: MAE", "", "", sg(p["MAE"]["delta"]), ci(p["MAE"]["lo"], p["MAE"]["hi"])], ["Other paired metrics: pooled R²", "", "", sg(p["R2"]["delta"]), ci(p["R2"]["lo"], p["R2"]["hi"])]],
                    "Images is the chosen run&rsquo;s saved forecast. Numbers is the replicated XGBoost model on 33 features. Fixed-effects R² demeans outcomes and forecasts by month and by stock. In the run&rsquo;s own numbers-only fit, the within-month gain is +0.018 (0.572 against 0.554). Test years 2020 to 2022; this analysis was not pre-specified.")

e = S["encompass"]
T["encompass"] = table("IX", "Encompassing Regressions of Realized Log Volatility on Pairs of Forecasts", ["Forecast 1", "Forecast 2", "<i>b</i><sub>1</sub>", "<i>t</i>", "<i>b</i><sub>2</sub>", "<i>t</i>", "R²", "With month FE: <i>b</i><sub>1</sub>", "<i>t</i>", "<i>b</i><sub>2</sub>", "<i>t</i>", "R²"],
                       [[NM[r["f1"]], NM[r["f2"]], f(r["b1"]), f(r["t1"], 1), f(r["b2"]), f(r["t2"], 1), f(r["R2"]), f(r["fe_b1"]), f(r["fe_t1"], 1), f(r["fe_b2"]), f(r["fe_t2"], 1), f(r["fe_R2"])] for r in e],
                       "OLS on test rows, 2020 to 2022, with standard errors clustered by month. Month-fixed-effects columns demean outcomes and forecasts by month.")

dm_ = S["dd_models"]; dp = S["dd_paired"]
d_i, d_n = m(lt, CH, ("D10", "numbers + images")), m(lt, NUM, ("D10", "numbers only"))
T["drawdown"] = table("X", "Drawdown Forecasts on the Test Years", ["Model", "AUC", "Within-month AUC", "AUC 2020", "AUC 2021", "AUC 2022", "Gain over numbers: AUC", "Gain: within-month AUC"],
                      [[NM[k], f(dm_[k]["AUC"]), f(dm_[k]["wmAUC"]), f(dm_[k]["AUC_2020"]), f(dm_[k]["AUC_2021"]), f(dm_[k]["AUC_2022"]), "&ndash;" if k == "NUM" else sg(dp[f"{k}-NUM"]["AUC"]) + " " + ci(dp[f"{k}-NUM"]["AUC_lo"], dp[f"{k}-NUM"]["AUC_hi"]), "&ndash;" if k == "NUM" else sg(dp[f"{k}-NUM"]["wmAUC"]) + " " + ci(dp[f"{k}-NUM"]["wm_lo"], dp[f"{k}-NUM"]["wm_hi"])] for k in ("IMG", "NUM", "NUMH")],
                      f"Replicated numeric fits against the run&rsquo;s saved image forecasts. Run values for uncalibrated image and numeric models respectively: average precision {f(d_i.AP)} and {f(d_n.AP)} (base rate 0.302), Brier skill {f(d_i.BSS)} and {f(d_n.BSS)}, calibration error {f(d_i.ECE)} and {f(d_n.ECE)}; the calibrated image model has Brier skill 0.046 and calibration error 0.058.")

dd = S["deciles"]; ds = S["decile_spread"]
T["deciles"] = table("XI", "Stocks Sorted on Predicted Volatility, Test Years", ["Decile", "Rows", "Predicted vol.", "Realized vol.", "10% drawdown rate", "Mean 20-day return"],
                     [[("1 (low)" if r["dec"] == 0 else "10 (high)" if r["dec"] == 9 else r["dec"] + 1), f"{r['n']:,}", f"{r['pred_vol'] * 100:.1f}%", f"{r['real_vol'] * 100:.1f}%", f"{r['dd'] * 100:.1f}%", f"{r['ret20'] * 100:.2f}%".replace("-", "&minus;")] for r in dd],
                     f"Deciles form within each calendar month on the chosen model&rsquo;s forecast. Volatility is annualized. Returns include dividends, are gross of costs and come from surviving stocks only. The high-minus-low mean 20-day return is {ds['ret20_hi_lo'] * 100:.2f}% (Newey-West <i>t</i> = {f(ds['t'], 2)}).")

o = S["overlay"]
T["overlay"] = table("XII", "Equal-Weighted Monthly Series by Risk Flag, Test Years", ["Series", "Mean 20-day return", "Std. dev.", "Annualized ratio", "Worst month", "Months below &minus;10%"],
                     [[n, f"{o[k]['mean'] * 100:.2f}%".replace("-", "&minus;"), f"{o[k]['sd'] * 100:.2f}%", f(o[k]["ratio_ann"], 2), f"{o[k]['worst'] * 100:.1f}%".replace("-", "&minus;"), f"{o[k]['share_below_m10'] * 36:.0f} of 36"] for n, k in (("All stocks", "all"), ("Unflagged", "unflagged"), ("Flagged", "flagged"))],
                     f"Each month averages the forward 20-session total return of rows dated in that month, so periods overlap slightly. Not a tradable strategy: gross of costs and limited to surviving stocks. Unflagged minus flagged: {o['spread'] * 100:.2f}% (Newey-West <i>t</i> = {f(o['spread_t'], 2)}). Flag rate {o['flag_rate'] * 100:.1f}%.".replace("-1.92", "&minus;1.92"))

cv = pd.read_csv(os.path.join(R, "stage7M_coverage_by_group.csv"))
T["mc"] = table("XIII", "Monte Carlo Coverage on the Test Years", ["Group", "Rows", "90% band holds", "50% band holds", "Median width of 90% band", "Realized drawdown rate", "Simulated drawdown rate"],
                [[(f"Predicted-volatility decile {int(r.value) + 1}" if "decile" in r.group else f"Year {int(r.value)}"), f"{r.n:,}", f(r.cover_90), f(r.cover_50), f(r.median_width_90), f(r.dd_rate), f(r.dd_sim)] for r in cv.itertuples()],
                "Raw simulation with no drift, Student-<i>t</i> shocks with five degrees of freedom and a range factor of 1.25, both chosen on validation. A well-calibrated 90% band holds 0.90.")

H = {n: json.load(open(os.path.join(ROOT, d, "stage10_holdout_results.json")))["results"] for n, d in (("60 stocks", "results_quick"), ("120 stocks", "results_s120"))}
rows = []
for n, h in H.items():
    rows.append([n, h["V|all"]["n"], f(h["V|all"]["R2"]) + " " + ci(*h["V|all"]["CI"]), f(h["V|all"]["wmR2"]), f(h["V|to_2026-02-28"]["R2"]), f(h["V|after_2026-02-28"]["R2"]), f(h["O5|all"]["AUC"]) + " " + ci(*h["O5|all"]["CI"])])
T["holdout"] = table("XIV", "Earlier Runs on the 2023 Onward Holdout", ["Run", "Rows", "Volatility R² [95% CI]", "Within-month R²", "R², to Feb. 2026", "R², after Feb. 2026", "Direction AUC [95% CI]"], rows,
                     "Each configuration was opened once, on 19 and 20 October 2026, using the configuration chosen by that run. The 60-stock model uses Swin-T frozen features, and the 120-stock model a fine-tuned ViT-B/16. These are smaller samples than the main analysis, and neither is the 500-stock model. Drawdown labels were not scored. Numeric benchmarks were not scored, so no image test is available on this period.")

feat = [["Returns", "1-, 5- and 21-day return; overnight and intraday return over 21 days; latest gap"], ["Volume", "Abnormal volume; five-day volume surge"], ["Position and mean reversion", "Five-day close location; position in 20-day range; 20-day <i>z</i>-score; distance in ATR units; RSI(2), RSI(3), RSI(14, Cutler); down and up streaks; five-day variance ratio; 63-day first-order autocorrelation; 20-day trend <i>z</i>-score"], ["Volatility (logs)", "Close-to-close, Parkinson and Garman-Klass at 1, 5 and 22 bars; exponentially weighted over the window"], ["Amplitude", "ATR as a percentage of price; log range of the window; largest drawdown in the window"]]
T["features"] = table("A.I", "The 33 Numeric Features", ["Group", "Features"], feat, "All are functions of the last 126 daily bars ending on the decision date. Mean dollar volume is excluded from the model because a chart&rsquo;s volume axis has no units.")

z = pd.DataFrame(S["mz"])
T["mz"] = table("A.II", "Mincer-Zarnowitz Regressions, Test Years", ["Model", "Sample", "Intercept", "(s.e.)", "Slope", "(s.e.)", "R²"],
                [[NM[r.model], r.year, f(r.a), f"({f(r.a_se)})", f(r.b), f"({f(r.b_se)})", f(r.R2)] for r in z[z.model.isin(["IMG", "NUM", "HAR"])].itertuples()],
                "Realized log volatility regressed on the forecast, with standard errors clustered by month. An unbiased forecast has an intercept of zero and a slope of one.")

io = lt[(lt.target == "V") & (lt.condition == "images only")].merge(lv[(lv.target == "V") & (lv.condition == "images only")][["model", "R2", "wmR2"]], on="model", suffixes=("_t", "_v")).sort_values("wmR2_t", ascending=False)
T["imgonly"] = table("A.III", "Images Alone: Volatility Forecasts", ["Model", "Validation R²", "Validation within-month R²", "Test R²", "Test within-month R²"],
                     [[r.model, f(r.R2_v), f(r.wmR2_v), f(r.R2_t), f(r.wmR2_t)] for r in io.itertuples()],
                     "No numeric features enter these models. The random-weight controls reach validation R² of about &minus;0.02 and within-month R² of about 0.08; they were not scored on test years.")

rp = S["replication"]
LB = {"NUM_V_val_R2": "Numbers, V, validation R²", "NUM_V_test_R2": "Numbers, V, test R²", "NUM_D10_val_AUC": "Numbers, D10, validation AUC", "NUM_D10_test_AUC": "Numbers, D10, test AUC", "HAR_test_R2": "HAR, test R²", "EWMA_test_R2": "EWMA, test R²", "ROLL22_test_R2": "Rolling 22-day, test R²", "IMG_V_test_R2": "Image model, V, test R² (saved forecasts)", "IMG_D10_test_AUC": "Image model, D10, test AUC (saved forecasts)"}
T["replication"] = table("A.IV", "Replication of the Run", ["Quantity", "Run (GPU notebook)", "Supplementary (CPU re-fit)"], [[LB[k], f(v["run"], 4), f(v["here"], 4)] for k, v in rp.items()],
                         "The supplementary analysis re-fits with XGBoost 3.4.1 on CPU and uses OLS where the notebook uses lightly penalized ridge.")

FIG = {"charts": ["charts_triplet.png", "charts_splits.png"], "architecture": ["architecture.png"], "regimes": ["regimes.svg"], "ft_gains": ["ft_gains.svg"], "learning": ["learning.svg"], "forecasters": ["forecasters.svg"], "monthly": ["monthly.svg"], "forest": ["forest.svg"], "views": ["views.svg"],
       "val_vs_test": ["val_vs_test.svg"], "dd_eval": ["dd_eval.png"], "deciles": ["deciles.svg"], "vol_eval": ["vol_eval.png"], "mc": ["mc.png"], "ft_curves": ["ft_curves.png"], "convex": ["convex.png"], "boosting": ["boosting.png"], "landscape": ["landscape.png"], "leaderboard": ["leaderboard.png"], "direction": ["direction.png"], "imageonly": ["imageonly.png"]}


def figure(mo):
    key, title, note = mo.group(1), mo.group(2), mo.group(3)
    wide = key in ("dd_eval", "vol_eval", "mc", "convex", "boosting", "landscape", "ft_curves", "leaderboard", "charts")
    imgs = "".join(f'<img src="fig/{x}" alt="{title}">' for x in FIG[key])
    a, b = title.split(". ", 1)
    return f'<div class="fig"><p class="ttl">{a}</p><p class="tname">{b}</p>{imgs}<p class="fnote">{note}</p></div>'


body = open(os.path.join(HERE, "paper_body.html"), encoding="utf-8").read()
body = re.sub(r"\{\{FIG:(\w+)\|(.*?)\|(.*?)\}\}", figure, body, flags=re.S)
body = re.sub(r"\{\{TAB:(\w+)\}\}", lambda mo: T[mo.group(1)], body)
assert "{{" not in body
CSS = """
@page { size: Letter; margin: 1in; @bottom-center { content: counter(page); font: 10pt 'Times New Roman', serif } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact }
body { font: 11.5pt/1.5 'Times New Roman', Times, serif; color: #000; margin: 0; text-align: justify; hyphens: auto }
a { color: inherit; text-decoration: none }
h1 { font: bold 17pt/1.25 'Times New Roman', serif; text-align: center; margin: 0 0 14pt; hyphens: none }
h2 { font: bold 12pt/1.3 'Times New Roman', serif; text-align: center; margin: 20pt 0 8pt; break-after: avoid; font-variant: small-caps; letter-spacing: .3pt }
h3 { font: italic 11.5pt/1.3 'Times New Roman', serif; text-align: left; margin: 14pt 0 6pt; break-after: avoid }
p { margin: 0 0 8pt; orphans: 3; widows: 3; text-indent: 0 }
.titlepage { break-after: page; position: relative; min-height: 8.8in }
.author { text-align: center; font-size: 12.5pt; margin-bottom: 2pt } .date { text-align: center; margin-bottom: 18pt; font-size: 10.5pt }
.abstract { margin: 0 .35in 14pt; font-size: 11pt; line-height: 1.4 } .abslabel { text-align: center; font-weight: bold; letter-spacing: 1pt; margin-bottom: 5pt }
.jel { font-size: 10.5pt; margin: 0 .35in; line-height: 1.35; text-align: left }
.titlenote { position: absolute; bottom: 0; border-top: .6pt solid #000; padding-top: 5pt; font-size: 9.5pt; line-height: 1.3 } .titlenote p { margin: 0 }
.eq { text-align: center; margin: 6pt 0 10pt; position: relative; font-size: 11.5pt } .eqn { position: absolute; right: 0 }
ul.hyp { margin: 0 0 8pt; padding-left: 20pt } ul.hyp li { margin-bottom: 3pt }
.tbl, .fig { margin: 12pt 0 14pt; break-inside: avoid; text-align: center }
.ttl { font-weight: bold; margin: 0; text-align: center; line-height: 1.2 } .tname { font-style: italic; margin: 1pt 0 5pt; text-align: center; line-height: 1.2; hyphens: none }
.tnote, .fnote { font-size: 9pt; line-height: 1.25; text-align: justify; margin: 0 0 6pt } .fnote { margin: 6pt 0 0 }
table { width: 100%; border-collapse: collapse; font-size: 8.8pt; line-height: 1.2; border-top: 1pt solid #000; border-bottom: 1pt solid #000; hyphens: none }
thead th { border-bottom: .6pt solid #000; padding: 3pt 3pt; font-weight: normal; vertical-align: bottom; text-align: center }
td { padding: 2pt 3pt; text-align: center; vertical-align: top } td.l, th.l { text-align: left }
tr.sub td { text-align: left; font-style: italic; padding-top: 5pt; border-top: .3pt solid #999 }
.fig img { max-width: 100%; display: block; margin: 2pt auto }
.refs p { padding-left: 22pt; text-indent: -22pt; margin-bottom: 4pt; font-size: 10.5pt; line-height: 1.3; text-align: left; hyphens: none }
.pb { break-before: page }
"""
html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>What Do Charts Add? Multi-Horizon Price Images and the Cross-Section of Equity Risk</title><style>{CSS}</style></head><body>\n{body}\n</body></html>'
open(os.path.join(HERE, "TriFrame_paper.html"), "w", encoding="utf-8").write(html)
txt = re.sub(r"<[^>]+>", " ", re.sub(r"<table.*?</table>", " ", body, flags=re.S))
print("written; words excluding tables:", len(txt.split()))
