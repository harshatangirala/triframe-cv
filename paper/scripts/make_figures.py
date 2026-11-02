"""Publication figures for the TriFrame paper, from aggregate run outputs and supplementary_results.json.
Usage: make_figures.py <run_root> <tab_dir> <out_dir>"""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN, TAB, OUT = sys.argv[1:4]
R = os.path.join(RUN, "results_s500")
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(TAB, "supplementary_results.json")))
BLUE, ORANGE, GREY, DARK = "#1f4e8c", "#c8691c", "#9aa1aa", "#222222"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7, "legend.frameon": False, "svg.fonttype": "path", "axes.titlesize": 10.5, "axes.titleweight": "bold"})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)


# regimes
y = pd.read_csv(os.path.join(R, "stage2_event_rates_by_year.csv"))
y = y.rename(columns={y.columns[0]: "date"})
if "dd10_rate" not in y.columns:
    y = y.set_index("date").T.reset_index(names="date")
y["date"] = y["date"].astype(float).astype(int)
cols = [GREY if d < 2017 else BLUE if d < 2020 else ORANGE for d in y["date"]]
fig, ax = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
ax[0].bar(y["date"], y["dd10_rate"].astype(float) * 100, color=cols)
ax[0].set_ylabel("Percent of stock-days")
ax[0].set_title("Panel A. Stock-days followed by a 10% drawdown within 20 sessions", loc="left")
ax[1].bar(y["date"], y["mean_ann_vol"].astype(float) * 100, color=cols)
ax[1].set_ylabel("Percent per year")
ax[1].set_title("Panel B. Mean realized volatility over the next 20 sessions (annualised)", loc="left")
for c, l in ((GREY, "Train"), (BLUE, "Validation"), (ORANGE, "Test")):
    ax[0].bar(0, 0, color=c, label=l)
ax[0].set_xlim(2000.3, 2022.7)
ax[0].legend(ncol=3, loc="upper right")
ax[1].set_xticks(range(2001, 2023, 3))
save(fig, "regimes")

# fine-tuning gains
f = pd.read_csv(os.path.join(R, "stage5F_finetune_vs_frozen.csv")).sort_values("V_gain")
fig, ax = plt.subplots(figsize=(6.6, 3.3))
yy = np.arange(len(f))
partial = f["model"].str.contains("last")
ax.errorbar(f["V_gain"], yy, xerr=[f["V_gain"] - f["V_gain_lo"], f["V_gain_hi"] - f["V_gain"]], fmt="none", ecolor=DARK, elinewidth=.8, capsize=2)
ax.scatter(f["V_gain"], yy, c=[GREY if p else BLUE for p in partial], zorder=3, s=26)
ax.axvline(0, color=DARK, lw=.7)
ax.set_yticks(yy)
ax.set_yticklabels(f["model"].str.replace(" (fine-tuned)", "", regex=False).str.replace("(fine-tuned, ", "(", regex=False))
ax.set_xlabel("Gain in validation R², fine-tuned minus frozen (numbers + images)")
save(fig, "ft_gains")

# validation vs test across candidates
fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
for a, t, ttl, lab in ((ax[0], "V", "Panel A. Volatility R²", "R²"), (ax[1], "D10", "Panel B. Drawdown AUC", "AUC")):
    j = pd.read_csv(os.path.join(TAB, f"val_vs_test_{t}.csv"))
    ft = j["kind"] == "fine-tuned"
    a.scatter(j.loc[~ft, "val"], j.loc[~ft, "test"], c=GREY, s=28, label="Frozen")
    a.scatter(j.loc[ft, "val"], j.loc[ft, "test"], c=BLUE, s=28, marker="^", label="Fine-tuned")
    b = j[j["model"] == "ConvNeXt-Tiny (fine-tuned)"].iloc[0]
    a.scatter(b["val"], b["test"], s=80, facecolors="none", edgecolors=ORANGE, lw=1.3, label="Chosen on validation")
    a.set_xlabel(f"Validation {lab}")
    a.set_ylabel(f"Test {lab}")
    rho = S["val_test_rank"][t]["spearman"]
    a.set_title(f"{ttl} (rank corr. {rho:.2f})", loc="left", fontsize=9.5)
ax[0].legend(loc="lower left", fontsize=8.5)
save(fig, "val_vs_test")

# monthly: the key decomposition
m = pd.read_csv(os.path.join(TAB, "test_monthly_series.csv"))
x = pd.to_datetime(m["month"])
fig, ax = plt.subplots(3, 1, figsize=(7, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.1, 1, 1]})
ax[0].plot(x, m["real"], color=DARK, lw=1.4, label="Realized")
ax[0].plot(x, m["pred_img"], color=BLUE, lw=1.2, label="Numbers + images")
ax[0].plot(x, m["pred_num"], color=ORANGE, lw=1.1, ls="--", label="Numbers only")
ax[0].set_ylabel("Mean log volatility")
ax[0].set_title("Panel A. Market-wide level: monthly cross-sectional mean", loc="left")
ax[0].legend(ncol=3, fontsize=8.5)
ax[1].plot(x, m["xs_r2_img"], color=BLUE, lw=1.2, label="Numbers + images")
ax[1].plot(x, m["xs_r2_num"], color=ORANGE, lw=1.1, ls="--", label="Numbers only")
ax[1].set_ylabel("R²")
ax[1].set_title("Panel B. Cross-sectional fit: squared correlation within each month", loc="left")
d = (m["xs_r2_img"] - m["xs_r2_num"]) * 100
ax[2].bar(x, d, width=22, color=[BLUE if v > 0 else ORANGE for v in d])
ax[2].axhline(0, color=DARK, lw=.7)
ax[2].set_ylabel("Percentage points")
ms = S["stress"]["monthly_xs"]
ax[2].set_title(f"Panel C. Monthly gain from images (mean {ms['mean_diff'] * 100:.2f}, Newey-West t = {ms['t_nw']:.2f})", loc="left")
save(fig, "monthly")

# forest: pooled vs cross-sectional gain
rows = [("All test years", S["paired"]["IMG-NUM"]["R2"], S["paired"]["IMG-NUM"]["wmR2"])]
by = pd.DataFrame(S["by_year"])
by = by[(by.a == "IMG") & (by.b == "NUM")]
for yr in (2020, 2021, 2022):
    rows.append((str(yr), by[(by.year == yr) & (by.metric == "R2")].iloc[0].to_dict(), by[(by.year == yr) & (by.metric == "wmR2")].iloc[0].to_dict()))
for h in S["hetero"]:
    rows.append((h["group"], dict(delta=h["d"], lo=h["lo"], hi=h["hi"]), dict(delta=h["wm_d"], lo=h["wm_lo"], hi=h["wm_hi"])))
fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)
yy = np.arange(len(rows))[::-1]
for a, k, ttl, c in ((ax[0], 1, "Panel A. Pooled R²", ORANGE), (ax[1], 2, "Panel B. Within-month R²", BLUE)):
    for i, r in zip(yy, rows):
        v = r[k]
        a.plot([v["lo"], v["hi"]], [i, i], color=DARK, lw=.9)
        a.scatter(v["delta"], i, color=c, s=24, zorder=3)
    a.axvline(0, color=DARK, lw=.7, ls=":")
    a.set_title(ttl, loc="left")
    a.set_xlabel("Numbers + images minus numbers only")
ax[0].set_yticks(yy)
ax[0].set_yticklabels([r[0] for r in rows])
save(fig, "forest")

# forecasters
vm = pd.DataFrame(S["vol_models"])
nm = {"IMG": "Numbers + images", "NUM": "Numbers only", "NUMH": "Numbers + hourly RV", "HAR": "HAR (daily range)", "HARH": "HAR (hourly RV)", "GARCH": "GARCH(1,1)", "EWMA": "EWMA", "ROLL22": "Rolling 22-day"}
vm = vm.sort_values("wmR2")
fig, ax = plt.subplots(figsize=(6.6, 3.1))
yy = np.arange(len(vm))
ax.barh(yy + .19, vm["wmR2"], .36, color=BLUE, label="Within-month R²")
ax.barh(yy - .19, vm["R2"], .36, color=GREY, label="Pooled R²")
ax.set_yticks(yy)
ax.set_yticklabels([nm[k] for k in vm["model"]])
ax.set_xlabel("Test years, 2020-2022")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2)
save(fig, "forecasters")

# deciles
d = pd.DataFrame(S["deciles"])
fig, ax = plt.subplots(1, 2, figsize=(7, 2.9))
x = d["dec"] + 1
ax[0].plot(x, d["pred_vol"] * 100, "o--", color=GREY, ms=4, label="Predicted")
ax[0].plot(x, d["real_vol"] * 100, "o-", color=BLUE, ms=4, label="Realized")
ax[0].set_ylabel("Percent per year")
ax[0].set_xlabel("Decile of predicted volatility (within month)")
ax[0].set_title("Panel A. Volatility", loc="left")
ax[0].legend()
ax[1].bar(x, d["dd"] * 100, color=BLUE)
ax[1].set_ylabel("Percent")
ax[1].set_xlabel("Decile of predicted volatility (within month)")
ax[1].set_title("Panel B. Rate of 10% drawdowns", loc="left")
ax[0].set_xticks(x)
ax[1].set_xticks(x)
save(fig, "deciles")

# learning curves
lc = pd.read_csv(os.path.join(R, "stage6D_learning_curves.csv"))
print("learning columns", list(lc.columns))
fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))
sc = "val_score" if "val_score" in lc else "score" if "score" in lc else [c for c in lc.columns if c not in ("target", "feature_set", "n_stocks", "metric", "head", "stocks")][0]
v = lc[lc["target"] == "V"]
for fs, c, l in (("NUM", ORANGE, "Numbers only"), ("NUM+IMG:ft_convnext_tiny", BLUE, "Numbers + images")):
    g = v[v["feature_set"] == fs]
    ax[0].plot(g["n_stocks"], g[sc], "o-", color=c, ms=4, label=l)
g = lc[lc["target"] == "D10"]
ax[1].plot(g["n_stocks"], g[sc], "o-", color=ORANGE, ms=4)
ax[0].set_title("Panel A. Validation volatility R²", loc="left")
ax[1].set_title("Panel B. Validation drawdown AUC", loc="left")
ax[0].legend()
for a in ax:
    a.set_xlabel("Stocks used for training")
save(fig, "learning")

# views
t = pd.read_csv(os.path.join(R, "stage6_single_vs_multi_timeframe.csv"))
t = t[t["target"] == "V"]
p = t.pivot(index="backbone", columns="views", values="delta")[["W", "D", "H", "WDH"]]
names = {"convnext_tiny": "ConvNeXt-Tiny", "efficientnet_b0": "EfficientNet-B0", "mobilenetv3_large_100": "MobileNetV3-L", "resnet18": "ResNet-18", "swin_tiny_patch4_window7_224": "Swin-T", "vit_base_patch16_224": "ViT-B/16", "vit_small_patch14_dinov2": "DINOv2 ViT-S/14"}
fig, ax = plt.subplots(figsize=(6.2, 2.9))
w = .2
for i, (c, col, l) in enumerate(zip(p.columns, ("#b9c4d3", "#7f9cc4", BLUE, DARK), ("Weekly", "Daily", "Hourly", "All three"))):
    ax.bar(np.arange(len(p)) + (i - 1.5) * w, p[c], w, color=col, label=l)
ax.set_xticks(range(len(p)))
ax.set_xticklabels([names[b] for b in p.index], rotation=20, ha="right")
ax.set_ylabel("Validation R² gain over numbers")
ax.legend(ncol=4, fontsize=8.5)
save(fig, "views")
print("figures:", sorted(os.listdir(OUT)))
