# ==================================== STAGE 7E: evaluation metrics, visually (v4) ====================================
# Precision, recall, F1, ROC-AUC (drawdown, direction) and R2, RMSE (volatility) for the chosen models next to the numbers-only
# model, on the validation years (where thresholds were chosen) and the test years (historical diagnostic).
from sklearn.metrics import roc_curve, precision_recall_curve
t0 = time.time()
yr_all = pd.to_datetime(pd.Series(DATES)).dt.year.values


_PREDS = {}


def preds(t, split):
    """(rows, {label: predictions}) for the chosen model (probabilities calibrated) and the numbers-only model (uncalibrated)."""
    if (t, split) not in _PREDS:
        _PREDS[(t, split)] = _preds(t, split)
    return _PREDS[(t, split)]


def _preds(t, split):
    fs, hd = SELF[t]["feature_set"], SELF[t]["head"]
    if split == "validation":
        m = VA & OK[t]; pc_ = VAL_PRED[(t, fs, hd)]; pn_ = VAL_PRED[(t, "NUM", best_of(t, "NUM"))]
    else:
        m = MTE[t]; pc_ = TEST_PRED[t]; pn_ = refit(t, "NUM", best_of(t, "NUM"), FIT_TV, MTE[t])[1]
    if TARGETS[t] == "clf":
        pc_ = cal(t, pc_) if split == "validation" else pc_
    return m, {f"chosen ({fs} / {hd})": pc_, "numbers only": pn_}


EVAL_ROWS = []
for t in ("D10", "O5"):
    fig, ax = plt.subplots(1, 5, figsize=(26, 4.8))
    thr = FZ["thresholds"][t]
    for split, ls_ in (("validation", "--"), ("test", "-")):
        m, P = preds(t, split); y = Y[t][m]
        for k, (lab, p) in enumerate(P.items()):
            fpr, tpr, _ = roc_curve(y, p); pr, rc_, _ = precision_recall_curve(y, p)
            col = ["#e67e22", "#3b6ea5"][k]
            ax[0].plot(fpr, tpr, ls_, color=col, lw=1.4, label=f"{split}, {lab.split(' (')[0]}: AUC {T.auc(y, p):.3f}")
            ax[1].plot(rc_, pr, ls_, color=col, lw=1.4, label=f"{split}, {lab.split(' (')[0]}: AP {T.avg_precision(y, p):.3f}")
            cm = T.class_metrics_at(y, p, thr if k == 0 else T.best_f1_threshold(Y[t][VA & OK[t]], VAL_PRED[(t, "NUM", best_of(t, "NUM"))]))
            EVAL_ROWS.append(dict(target=t, split=split, model=lab, AUC=T.auc(y, p), AP=T.avg_precision(y, p), **{kk: cm[kk] for kk in
                             ("threshold", "precision", "recall", "F1", "specificity", "balanced_acc", "accuracy", "majority_acc", "alert_rate", "prevalence")},
                             BSS=1 - T.brier(y, p) / T.brier(y, np.full(len(y), BASE[t])), ECE=T.ece(y, p)))
        ax[1].axhline(float(np.mean(y)), color="grey", ls=ls_, lw=0.8)
    ax[0].plot([0, 1], [0, 1], color="grey", lw=0.8); ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate (recall)")
    ax[0].set_title(f"ROC curves (dashed = validation, solid = test)", fontsize=9); ax[0].legend(fontsize=6.5)
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_title("Precision-recall (grey lines = event rate)", fontsize=9); ax[1].legend(fontsize=6.5)
    mv, Pv = preds(t, "validation"); yv_, pv_ = Y[t][mv], list(Pv.values())[0]
    grid_ = np.linspace(np.quantile(pv_, 0.01), np.quantile(pv_, 0.99), 60)
    cms = [T.class_metrics_at(yv_, pv_, g_) for g_ in grid_]
    for key_, col in (("precision", "#3b6ea5"), ("recall", "#e67e22"), ("F1", "#2e9d5b")):
        ax[2].plot(grid_, [c_[key_] for c_ in cms], color=col, label=f"{key_} (validation)")
    mt_, Pt = preds(t, "test"); ct_ = T.class_metrics_at(Y[t][mt_], list(Pt.values())[0], thr)
    for key_, col in (("precision", "#3b6ea5"), ("recall", "#e67e22"), ("F1", "#2e9d5b")):
        ax[2].plot(thr, ct_[key_], "o", color=col, ms=8, mec="black")
    ax[2].axvline(thr, color="#c0392b", ls="--", label=f"threshold fixed on validation ({thr:.3f}); dots = test")
    ax[2].set_xlabel("probability threshold"); ax[2].set_title("Precision, recall and F1 vs threshold (chosen model)", fontsize=9); ax[2].legend(fontsize=6.5)
    cmat = np.array([[ct_["TN"], ct_["FP"]], [ct_["FN"], ct_["TP"]]])
    ax[3].imshow(cmat / cmat.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i_ in range(2):
        for j_ in range(2):
            ax[3].text(j_, i_, f"{int(cmat[i_, j_]):,}\n({cmat[i_, j_] / cmat[i_].sum():.0%} of row)", ha="center", va="center", fontsize=9)
    ax[3].set_xticks([0, 1]); ax[3].set_xticklabels(["predicted no", "predicted yes"]); ax[3].set_yticks([0, 1]); ax[3].set_yticklabels(["actual no", "actual yes"])
    ax[3].set_title(f"Confusion matrix, test years, threshold {thr:.3f}\nprecision {ct_['precision']:.2f} recall {ct_['recall']:.2f} F1 {ct_['F1']:.2f}", fontsize=9)
    for split, mk in (("validation", "s"), ("test", "o")):
        m, P = preds(t, split); y, p = Y[t][m], list(P.values())[0]
        bins = np.quantile(p, np.linspace(0, 1, 11)); bins[-1] += 1e-9; k_ = np.clip(np.digitize(p, bins) - 1, 0, 9)
        ax[4].plot([p[k_ == i].mean() for i in range(10) if (k_ == i).any()], [y[k_ == i].mean() for i in range(10) if (k_ == i).any()],
                   mk + "-", label=f"{split}: ECE {T.ece(y, p):.3f}, slope {T.calibration_line(y, p)[1]:.2f}")
    lim_ = [0, max(0.6, float(np.max(list(Pt.values())[0])))]
    ax[4].plot(lim_, lim_, color="grey", lw=0.8); ax[4].set_xlabel("predicted chance (10 equal-size groups)"); ax[4].set_ylabel("observed rate")
    ax[4].set_title("Reliability: calibrated chosen model (on the line = calibrated)", fontsize=9); ax[4].legend(fontsize=7)
    fig.suptitle(f"{TARGET_NAME[t]}: evaluation", fontsize=11)
    plt.tight_layout(); fpath = os.path.join(FIG, f"stage7E_{t}_evaluation.png"); plt.savefig(fpath, dpi=130); plt.show()
    push(fpath, f"figures_{RUN_TAG}/stage7E_{t}_evaluation.png")

# ---- volatility: predicted vs actual, residuals, calibration by decile, R2 by year
fig, ax = plt.subplots(1, 4, figsize=(24, 4.8))
for split, col in (("validation", "#3b6ea5"), ("test", "#e67e22")):
    m, P = preds("V", split); y = Y["V"][m]
    for k, (lab, p) in enumerate(P.items()):
        EVAL_ROWS.append(dict(target="V", split=split, model=lab, R2=T.r2(y, p), RMSE=T.rmse(y, p), MAE=float(np.mean(np.abs(y - p))),
                              Spearman=T.spearman(y, p), RMSE_annual_vol=T.rmse(C.ann_vol(y), C.ann_vol(p))))
    p = list(P.values())[0]
    if split == "test":
        hb_ = ax[0].hexbin(C.ann_vol(p), C.ann_vol(y), gridsize=45, bins="log", cmap="viridis", extent=(0, 1.5, 0, 1.5))
        ax[0].plot([0, 1.5], [0, 1.5], color="#c0392b", lw=1); plt.colorbar(hb_, ax=ax[0], label="log10(count)")
        ax[0].set_title(f"Test years: predicted vs realised volatility (a year)\nR² {T.r2(y, p):.3f} on log volatility, RMSE {T.rmse(y, p):.3f}", fontsize=9)
        ax[0].set_xlabel("predicted"); ax[0].set_ylabel("realised")
    ax[1].hist(y - p, bins=60, alpha=0.55, density=True, color=col, label=f"{split}: mean {np.mean(y - p):+.3f}, sd {np.std(y - p):.3f}")
    q_ = pd.qcut(p, 10, labels=False, duplicates="drop")
    ax[2].plot([C.ann_vol(p[q_ == i]).mean() for i in range(int(q_.max()) + 1)], [C.ann_vol(y[q_ == i]).mean() for i in range(int(q_.max()) + 1)],
               "o-", color=col, label=split)
ax[1].set_title("Errors (realised - predicted log volatility)", fontsize=9); ax[1].legend(fontsize=7)
ax[2].plot([0, 1.2], [0, 1.2], color="grey", lw=0.8); ax[2].set_xlabel("mean predicted volatility (a year), per tenth")
ax[2].set_ylabel("mean realised"); ax[2].set_title("Calibration by predicted-risk decile (on the line = right level)", fontsize=9); ax[2].legend(fontsize=7)
r2y = []
for split, msk in (("validation", VA & OK["V"]), ("test", MTE["V"])):
    pp_ = VAL_PRED[("V", SELF["V"]["feature_set"], SELF["V"]["head"])] if split == "validation" else TEST_PRED["V"]
    yy_, yrs_ = Y["V"][msk], yr_all[msk]
    for yr_ in np.unique(yrs_):
        r2y.append((int(yr_), T.r2(yy_[yrs_ == yr_], pp_[yrs_ == yr_]), split))
ax[3].bar([str(a) for a, _, _ in r2y], [b for _, b, _ in r2y], color=["#3b6ea5" if s == "validation" else "#e67e22" for _, _, s in r2y])
ax[3].set_title("Volatility R² by year (blue = validation, orange = test)", fontsize=9)
plt.tight_layout(); fpath = os.path.join(FIG, "stage7E_V_evaluation.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage7E_V_evaluation.png")

EVAL = pd.DataFrame(EVAL_ROWS); save_df(EVAL, "stage7E_evaluation_metrics.csv")
print("Evaluation metrics (chosen model vs numbers only; thresholds fixed on validation; test = historical diagnostic):")
for t in ("V", "D10", "O5"):
    cols_ = ["R2", "RMSE", "MAE", "Spearman"] if t == "V" else ["AUC", "AP", "precision", "recall", "F1", "specificity", "accuracy", "majority_acc", "BSS", "ECE"]
    print(f"\n{TARGET_NAME[t]}:"); print(EVAL[EVAL.target == t][["split", "model"] + cols_].round(4).to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(21, 4.6))
for ax, (t, cols_) in zip(axes, (("V", ["R2", "RMSE", "Spearman"]), ("D10", ["AUC", "AP", "precision", "recall", "F1"]), ("O5", ["AUC", "precision", "recall", "F1"]))):
    g = EVAL[EVAL.target == t]; xx = np.arange(len(cols_)); w_ = 0.2
    for k, ((split, model), r_) in enumerate(g.groupby(["split", "model"], sort=False)):
        ax.bar(xx + (k - 1.5) * w_, r_[cols_].iloc[0].values, w_, label=f"{split}: {model.split(' (')[0]}")
    ax.set_xticks(xx); ax.set_xticklabels(cols_); ax.set_title(TARGET_NAME[t], fontsize=9); ax.legend(fontsize=7)
plt.tight_layout(); fpath = os.path.join(FIG, "stage7E_metric_bars.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage7E_metric_bars.png")
print(f"Stage 7E done in {time.time() - t0:.0f}s")
