# ==================================== STAGE 6L: model leaderboard - every pretrained model on every metric (validation) ====================================
# Which pretrained models were used, what they are, and how each scored on six metrics per target, frozen and fine-tuned,
# with the numbers-only baselines for reference. Validation years only: nothing here can change the frozen choices.
t0 = time.time()
recs_all = json.load(open(bpath))
LB_BOOTS = {t: T.MonthBoot(DATES[VA & OK[t]], LB_BOOT, seed=SEED) for t in TARGETS}

# ---- 1. model cards
cards = []
for b in NAMES + FT_NAMES:
    rec = recs_all.get(b, {}); base = T.base_name(b)
    fam, nm, desc = T.BACKBONE_INFO.get(base, ("?", base, ""))
    ft = rec.get("ft") or {}
    cards.append({"model": T.display_name(b), "key": b, "family": fam, "what it is": desc,
                  "pretrained on": T.pretrain_data(rec.get("timm_name", ""), rec.get("pretrained", True)),
                  "parameters (M)": rec.get("n_params", np.nan) / 1e6, "embedding size": rec.get("emb_dim"),
                  "PCA variance kept": np.mean([PCA_EXPL.get(f"{b}|{v}", np.nan) for v in T.VIEWS]),
                  "inference images/s": rec.get("infer_img_per_s"), "fine-tuning": (f"{ft.get('epochs_run')} epochs (best {ft.get('best_epoch')}), "
                                                                               f"{ft.get('minutes', 0):.0f} min" if ft else "frozen")})
CARDS = pd.DataFrame(cards)
save_df(CARDS, "stage6L_model_cards.csv")
print("The vision models compared (all from timm, ImageNet-style inputs of 224 x 224; each embeds the weekly, daily and hourly chart separately):")
print(CARDS.drop(columns=["key", "what it is"]).round(3).to_string(index=False))

# ---- 2. every metric, every model, both conditions (images only / numbers + images)
PRIMARY_LB = {"V": ("R2", T.r2), "D10": ("AUC", T.auc), "O5": ("AUC", T.auc)}


LB_S2, LB_THR = {}, {}                                      # per model: validation residual variance (QLIKE) and F1 threshold


def lb_row(t, label, kind, fs, head, cond):
    """All metrics of one model on validation. QLIKE uses the model's own validation residual variance; precision / recall / F1
    use the model's F1-maximising validation threshold (both kept for the test leaderboard). Probabilities are uncalibrated here."""
    m = VA & OK[t]; y = Y[t][m]; p = VAL_PRED[(t, fs, head)]
    if TARGETS[t] == "reg":
        LB_S2[(t, fs, head)] = float(np.var(y - p))
    else:
        LB_THR[(t, fs, head)] = T.best_f1_threshold(y, p)
    met = T.board_metrics(TARGETS[t], y, p, DATES[m], base=BASE.get(t), s2=LB_S2.get((t, fs, head)), thr=LB_THR.get((t, fs, head)))
    assert_finite(met, f"validation {t} {fs}/{head}")
    pm, fn = PRIMARY_LB[t]
    ci, _ = LB_BOOTS[t].ci(fn, y, p)
    chosen = (SEL[t]["feature_set"], SEL[t]["head"]) == (fs, head)
    for k_, v_ in met.items():
        log_metric(target_name=t, target_formula=TARGET_FORMULA[t], feature_set=fs, vision_model=C.Bundle.parse_fs(fs)[1] if ":" in fs else None,
                   frozen_or_finetuned=("fine-tuned" if ":" in fs and C.Bundle.parse_fs(fs)[1].startswith(T.FT_PREFIX) else ("frozen" if ":" in fs else "none")),
                   head_type=head, split_name="validation", split_start=SPLITS["val"][0], split_end=SPLITS["val"][1], **split_info(m),
                   probability_calibrated=False if TARGETS[t] == "clf" else None, prediction_scale="logrv20" if t == "V" else "probability",
                   target_scale="logrv20" if t == "V" else "0/1", metric_name=k_, metric_value=v_,
                   ci_method=f"month-block bootstrap, {LB_BOOT} resamples, percentile" if k_ == pm else None,
                   ci_lower=ci[0] if k_ == pm else None, ci_upper=ci[1] if k_ == pm else None)
    return dict(target=t, model=label, kind=kind, condition=cond, feature_set=fs, head=head, chosen=chosen,
                primary=pm, primary_value=met[pm], ci_lo=ci[0], ci_hi=ci[1], **met)


def best_of(t, fs):
    tab = VAL_TABLES[t]; met, sign = PRIMARY[t]
    sub = tab[tab.feature_set == fs]
    return None if sub.empty else sub.loc[(sign * sub[met]).idxmax(), "head"]


lb = []
for t in TARGETS:
    lb.append(lb_row(t, "Numbers only (33 features)", "reference", "NUM", best_of(t, "NUM"), "numbers only"))
    if t == "V":
        for fs_, lab_ in (("HAR", "HAR volatility model"), ("ROLL22", "Rolling 22-day volatility"), ("EWMA", "EWMA volatility"),
                          ("GARCH", "GARCH(1,1) volatility")):
            if ("V", fs_, "linear") in VAL_PRED:
                lb.append(lb_row(t, lab_, "reference", fs_, "linear", "numbers only"))
    for b in NAMES + FT_NAMES:
        kind = "fine-tuned" if b in FT_NAMES else ("control" if b in T.RANDOM else "frozen")
        lb.append(lb_row(t, T.display_name(b), kind, f"IMG:{b}", "linear", "images only"))
        h_ = best_of(t, f"NUM+IMG:{b}")
        if h_:
            lb.append(lb_row(t, T.display_name(b), kind, f"NUM+IMG:{b}", h_, "numbers + images"))
BOARD = pd.DataFrame(lb)
save_df(BOARD, "stage6L_leaderboard_validation.csv")
MET_COLS = {"V": ["R2", "wmR2", "QLIKE", "MAE", "RMSE", "Spearman"],
            "D10": ["AUC", "AP", "precision", "recall", "F1", "BSS", "ECE", "prevalence"],
            "O5": ["AUC", "AP", "precision", "recall", "F1", "BSS", "ECE", "prevalence"]}
for t in TARGETS:
    sub = BOARD[(BOARD.target == t) & BOARD.condition.isin(["numbers only", "numbers + images"])].copy()
    sub["model"] = sub["model"] + np.where(sub["chosen"], "  <== chosen", "")
    pm = PRIMARY_LB[t][0]
    sub = sub.sort_values("primary_value", ascending=False)
    print(f"\n{TARGET_NAME[t]} - validation {SPLITS['val'][0][:4]}-{int(SPLITS['val'][1][:4]) - 1}, numbers + images "
          f"(best head per model), sorted by {pm}:")
    print(sub[["model", "kind", "head"] + MET_COLS[t] + ["ci_lo", "ci_hi"]].round(4).to_string(index=False))
print("\nMetric guide: R2 = share of variation explained (1 is perfect, 0 = no better than the average); wmR2 = R2 within each month "
      "(stock-vs-stock skill only); QLIKE = volatility forecast loss (lower is better); MAE/RMSE = average error in log volatility; "
      "Spearman = rank correlation; AUC = chance a positive case is ranked above a negative one (0.5 = guessing); AP = average precision; "
      "precision = share of flagged cases that had the event; recall = share of events that were flagged; F1 = their harmonic mean "
      "(at each model's F1-maximising validation threshold); BSS = Brier skill vs the training base rate (0 = no skill); ECE = calibration "
      "error (lower is better); prevalence = event rate (compare AP with it). Probabilities here are uncalibrated.")

# ---- 3. figures: primary metric per model (bars with CIs), all metrics (heatmap), accuracy vs size and speed
COLORS = {"frozen": "#3b6ea5", "fine-tuned": "#e67e22", "control": "#9e9e9e", "reference": "#c0392b"}
fig, axes = plt.subplots(1, 3, figsize=(20, 7.5))
for ax, t in zip(axes, ("V", "D10", "O5")):
    sub = BOARD[(BOARD.target == t) & (BOARD.condition == "numbers + images")].sort_values("primary_value")
    ref = BOARD[(BOARD.target == t) & (BOARD.feature_set == "NUM")].iloc[0]
    yy = np.arange(len(sub))
    ax.barh(yy, sub["primary_value"], color=[COLORS[k] for k in sub["kind"]],
            xerr=np.vstack([sub["primary_value"] - sub["ci_lo"], sub["ci_hi"] - sub["primary_value"]]), capsize=2)
    ax.axvline(ref["primary_value"], color=COLORS["reference"], ls="--", label=f"numbers only ({ref['primary_value']:.3f})")
    ax.set_yticks(yy); ax.set_yticklabels([m + ("  ★" if c else "") for m, c in zip(sub["model"], sub["chosen"])], fontsize=8)
    lo_ = min(sub["ci_lo"].min(), ref["primary_value"]) - 0.01
    ax.set_xlim(lo_, max(sub["ci_hi"].max(), ref["primary_value"]) + 0.01)
    ax.set_title(f"{TARGET_NAME[t]}\nvalidation {PRIMARY_LB[t][0]} (numbers + images, 95% CI; ★ = chosen)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c, label=k) for k, c in COLORS.items() if k != "reference"], loc="upper center", ncol=3, fontsize=9)
plt.tight_layout(rect=(0, 0, 1, 0.95)); fpath = os.path.join(FIG, "stage6L_leaderboard_validation.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6L_leaderboard_validation.png")

heat_cols = [("V", c) for c in ("R2", "wmR2", "QLIKE", "MAE")] + [("D10", c) for c in ("AUC", "AP", "F1", "BSS", "ECE")] + [("O5", c) for c in ("AUC", "F1", "BSS")]
lower_better = {"QLIKE", "MAE", "RMSE", "logloss", "ECE"}
models_h = [m for m in BOARD[(BOARD.condition == "numbers + images") | (BOARD.feature_set == "NUM")].drop_duplicates("model")["model"]]
vals = np.full((len(models_h), len(heat_cols)), np.nan)
for i, mdl in enumerate(models_h):
    for j, (t, c) in enumerate(heat_cols):
        r_ = BOARD[(BOARD.target == t) & (BOARD.model == mdl) & BOARD.condition.isin(["numbers + images", "numbers only"])]
        if len(r_):
            vals[i, j] = r_.iloc[0][c]
score = np.zeros_like(vals)
for j, (t, c) in enumerate(heat_cols):
    col = vals[:, j]; rk = pd.Series(-col if c in lower_better else col).rank(pct=True).values
    score[:, j] = rk
fig, ax = plt.subplots(figsize=(14, 0.42 * len(models_h) + 2))
ax.imshow(score, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=7)
ax.set_xticks(range(len(heat_cols))); ax.set_xticklabels([f"{t}\n{c}" for t, c in heat_cols], fontsize=8)
ax.set_yticks(range(len(models_h))); ax.set_yticklabels(models_h, fontsize=8)
ax.set_title("All metrics, validation, numbers + images (colour = rank among models: green best; V = volatility, D10 = drawdown, O5 = direction)", fontsize=10)
plt.tight_layout(); fpath = os.path.join(FIG, "stage6L_metrics_heatmap.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6L_metrics_heatmap.png")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
img_v = BOARD[(BOARD.target == "V") & (BOARD.condition == "images only")].set_index("feature_set")
for b in [b for b in NAMES if b in T.PRETRAINED]:
    c0 = CARDS.set_index("key").loc[b]; r0 = img_v.loc[f"IMG:{b}", "R2"]
    for k_, (axi, xcol) in enumerate(((ax[0], "parameters (M)"), (ax[1], "inference images/s"))):
        axi.scatter(c0[xcol], r0, s=70, color=COLORS["frozen"], zorder=3)
        axi.annotate(c0["model"], (c0[xcol], r0), fontsize=8, xytext=(4, 4), textcoords="offset points")
        if T.FT_PREFIX + b in FT_NAMES:
            r1 = img_v.loc[f"IMG:{T.FT_PREFIX + b}", "R2"]
            axi.annotate("", xy=(c0[xcol], r1), xytext=(c0[xcol], r0), arrowprops=dict(arrowstyle="->", color=COLORS["fine-tuned"]))
            axi.scatter(c0[xcol], r1, s=70, marker="^", color=COLORS["fine-tuned"], zorder=3)
for axi, xl in ((ax[0], "parameters (millions, log scale)"), (ax[1], "inference speed (images per second, log scale)")):
    axi.set_xscale("log"); axi.set_xlabel(xl); axi.set_ylabel("validation volatility R² from images alone")
ax[0].set_title("Bigger is not better: accuracy vs model size (circle = frozen, triangle = fine-tuned)", fontsize=10)
ax[1].set_title("Accuracy vs speed", fontsize=10)
plt.tight_layout(); fpath = os.path.join(FIG, "stage6L_efficiency.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6L_efficiency.png")
print(f"Stage 6L done in {time.time() - t0:.0f}s")
