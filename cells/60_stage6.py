# ==================================== STAGE 6: compare, select (pre-registered rule), calibrate, freeze ====================================
import matplotlib; import matplotlib.pyplot as plt
t0 = time.time()
METRIC_FN = {"V": T.r2, "O5": T.logloss, "D10": T.brier}
PREREG = {
    "selection": "Per target the best validation score wins (V: R2, O5: log loss, D10: Brier). A model that uses chart images is "
                 "chosen only if its validation gain over the best numbers-only model has a month-block 95% CI that excludes 0.",
    "H1": "On the test years, the best numbers+images model (chosen on validation among pretrained backbones) beats the numbers-only "
          "model: the 95% CI of delta-R2 (volatility) or delta-AUC (direction) lies above 0.",
    "H2": "On validation, image-only volatility R2 differs between at least two pretrained backbones (pairwise month-block "
          "bootstrap, Holm-adjusted p < 0.05).",
    "H3": "On the test years, accuracy of direction calls at 10% coverage (confidence threshold fixed on validation) exceeds "
          "accuracy at 100% coverage: the 95% CI of the difference lies above 0.",
    "no_trade": "NO TRADE unless direction is usable, confidence passes its validation threshold, and neither predicted volatility "
                "nor drawdown probability is in its validation top 20%. Direction is usable only if (a) on validation the Wilson "
                "lower bound of accuracy exceeds 0.5 at some coverage AND (b) on the test years the direction model's AUC 95% CI "
                "lies above 0.5 (validation alone is optimistic because it also chose the model).",
}
BOOT_VA = {t: T.MonthBoot(DATES[VA & OK[t]], N_BOOT, seed=SEED) for t in TARGETS}
SEL, VAL_SUMMARY = {}, {}
for t in TARGETS:
    tab = VAL_TABLES[t].copy()
    cand = tab[(tab.feature_set == "NUM") | tab.feature_set.str.startswith(("NUM+IMG:", "IMG:"))]
    cand = cand[~cand.feature_set.str.contains("random")]
    met, sign = PRIMARY[t]
    cand = cand.assign(score=sign * cand[met]).sort_values("score", ascending=False)
    best, num = cand.iloc[0], cand[cand.feature_set == "NUM"].iloc[0]
    y = Y[t][VA & OK[t]]
    key = lambda r: (t, r["feature_set"], r["head"])
    chosen, gate = num, None
    if best["feature_set"] != "NUM":
        ci, _ = BOOT_VA[t].diff(METRIC_FN[t], y, VAL_PRED[key(best)], VAL_PRED[key(num)])
        gate = dict(candidate=f"{best['feature_set']} / {best['head']}", delta=float(best[met] - num[met]), ci=ci)
        if (ci[0] > 0) if sign > 0 else (ci[1] < 0):
            chosen = best
    SEL[t] = dict(feature_set=chosen["feature_set"], head=chosen["head"], gate=gate,
                  fit={k: v for k, v in FITS[key(chosen)].items() if k not in ("model", "coef", "intercept")})
    # H1 comparison pair, fixed now: best numbers-only head and best numbers + all-three-views model (pretrained backbones)
    multi = cand[cand.feature_set.str.fullmatch(r"NUM\+IMG:[^:]+")]
    SEL[t]["h1_numbers"] = dict(feature_set="NUM", head=num["head"])
    SEL[t]["h1_images"] = dict(feature_set=multi.iloc[0]["feature_set"], head=multi.iloc[0]["head"]) if len(multi) else None
    VAL_SUMMARY[t] = {k: float(chosen[k]) for k in chosen.index if k in ("R2", "wmR2", "QLIKE", "MAE", "AUC", "logloss", "brier", "BSS", "ECE", "rankIC", "rankIC_t")}
    print(f"{t}: chosen {chosen['feature_set']} / {chosen['head']} ({met} {chosen[met]:.4f})"
          + (f"; best image model {gate['candidate']} delta {gate['delta']:+.4f}, CI [{gate['ci'][0]:+.4f}, {gate['ci'][1]:+.4f}]" if gate else ""))

# ---- H2 (validation): do pretrained backbones differ in image-only volatility R2?
yV, bV = Y["V"][VA & OK["V"]], BOOT_VA["V"]
img_r2 = {}
for b in NAMES:
    p = VAL_PRED[("V", f"IMG:{b}", "linear")]
    img_r2[b] = dict(R2=T.r2(yV, p), CI=bV.ci(T.r2, yV, p)[0])
pv = {}
pre = [b for b in NAMES if b in T.PRETRAINED]
for i, a in enumerate(pre):
    for b in pre[i + 1:]:
        _, draws = bV.diff(T.r2, yV, VAL_PRED[("V", f"IMG:{a}", "linear")], VAL_PRED[("V", f"IMG:{b}", "linear")])
        pv[f"{a} vs {b}"] = float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))
padj = T.holm(pv) if pv else {}
H2 = dict(supported=bool(any(p < 0.05 for p in padj.values())), n_pairs=len(pv),
          significant_pairs=[k for k, p in padj.items() if p < 0.05], image_only_R2=img_r2)
print(f"\nH2 (backbones differ, validation): {'SUPPORTED' if H2['supported'] else 'NOT SUPPORTED'} "
      f"- {len(H2['significant_pairs'])} of {len(pv)} pairs differ after Holm correction.")
print(pd.DataFrame({b: {"image-only R2": v["R2"], "lo": v["CI"][0], "hi": v["CI"][1]} for b, v in img_r2.items()}).T.round(3).to_string())

# ---- per-view (single timeframe) vs all three views, validation delta vs numbers only
rows = []
for b in pre:
    for t, met in (("V", "R2"), ("O5", "AUC")):
        tab = VAL_TABLES[t].set_index(["feature_set", "head"])
        base_ = tab.loc[("NUM", "linear"), met]
        for v in ("W", "D", "H", None):
            fs = f"NUM+IMG:{b}" + (f":{v}" if v else "")
            if (fs, "linear") in tab.index:
                rows.append(dict(backbone=b, target=t, views=v or "WDH", delta=float(tab.loc[(fs, "linear"), met] - base_)))
views_df = pd.DataFrame(rows)
save_df(views_df, "stage6_single_vs_multi_timeframe.csv")
print("\nValidation gain over numbers only (linear heads): single timeframe vs all three")
print(views_df.pivot_table(index=["target", "backbone"], columns="views", values="delta").round(4).to_string())

# ---- controls promised in the proposal: label shuffle (within each training month) and the naive majority-class baseline
rng_s = np.random.default_rng(SEED)
CONTROLS = {}
for t, fs in (("V", SEL["V"]["h1_images"]["feature_set"] if SEL["V"]["h1_images"] else "NUM"), ("O5", "NUM")):
    X = design(fs); mt, mv = TR & OK[t], VA & OK[t]
    ys = Y[t][mt].copy()
    if t == "V":                                          # volatility: shuffle across ALL training rows (a within-month shuffle
        ys = ys[rng_s.permutation(len(ys))]               # keeps month averages, which still teach a usable cross-sectional slope)
    else:                                                 # direction: within-month shuffle keeps the 50/50 base rate per period
        mon_tr, _ = T.month_codes(DATES[mt])
        for k in np.unique(mon_tr):
            ii = np.flatnonzero(mon_tr == k); ys[ii] = ys[rng_s.permutation(ii)]
    if t == "V":
        _, _, p = T.ridge_path(X[mt], ys, X[mv], [1000.0])[1000.0]
        yv = Y[t][mv]; c_, _ = T.month_codes(DATES[mv])
        CONTROLS["label_shuffle_V"] = dict(feature_set=fs, R2=T.r2(yv, p), wmR2=T.wm_r2(yv, p, c_))
    else:
        _, _, p = T.logit_path(X[mt], ys, X[mv], [0.01], device=DEVICE)[0.01]
        CONTROLS["label_shuffle_O5"] = dict(feature_set=fs, AUC=T.auc(Y[t][mv], p))
maj = int(BASE["O5"] >= 0.5)
CONTROLS["majority_class_accuracy_O5"] = float((Y["O5"][VA & OK["O5"]] == maj).mean())
print("\nControls (validation): " + "; ".join(f"{k}: " + ", ".join(f"{a} {b:.4f}" for a, b in v.items() if isinstance(b, float))
                                          if isinstance(v, dict) else f"{k}: {v:.4f}" for k, v in CONTROLS.items()))
print("   (with shuffled labels the pipeline must find nothing: volatility R2 at or below 0, within-month R2 near 0, direction AUC near 0.5)")

# ---- calibration, volatility residual band, NO TRADE rule (all from validation predictions)
def _vp(t):
    return VAL_PRED[(t, SEL[t]["feature_set"], SEL[t]["head"])]
CAL = {t: T.isotonic_arrays(_vp(t), Y[t][VA & OK[t]]) for t in ("O5", "D10")}
cal = lambda t, p: np.interp(p, CAL[t]["x"], CAL[t]["y"])
res = Y["V"][VA & OK["V"]] - _vp("V")
VRES = dict(res_q10=float(np.quantile(res, 0.10)), res_q50=float(np.quantile(res, 0.50)), res_q90=float(np.quantile(res, 0.90)), s2=float(np.var(res)))
pO5_va = cal("O5", _vp("O5")); yO5_va = Y["O5"][VA & OK["O5"]]
abst_va, THR = T.abstention_table(pO5_va, yO5_va)
usable = abst_va[(abst_va.coverage_target < 1) & (abst_va.wilson_lower > 0.5)]
cov_dir = float(usable.coverage_target.max()) if len(usable) else None
RULE = dict(vol_ann_q80=float(np.quantile(C.ann_vol(_vp("V")), 0.8)), p_dd10_q80=float(np.quantile(cal("D10", _vp("D10")), 0.8)),
            direction_usable=bool(len(usable)), coverage_for_direction=cov_dir,
            conf_threshold=float(THR[cov_dir]) if cov_dir else 0.5, thresholds_by_coverage={str(k): float(v) for k, v in THR.items()})
save_df(abst_va, "stage6_abstention_validation.csv")
print("\nAbstention on validation (calibrated direction probabilities):")
print(abst_va.round(4).to_string(index=False))
print(f"Direction usable: {RULE['direction_usable']} | NO TRADE risk thresholds: volatility >= {RULE['vol_ann_q80']:.1%} a year "
      f"or P(drawdown >= 10%) >= {RULE['p_dd10_q80']:.1%}")

# ---- figures
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
names_ = list(img_r2)
ax[0].bar(range(len(names_)), [img_r2[b]["R2"] for b in names_], yerr=np.array([[img_r2[b]["R2"] - img_r2[b]["CI"][0] for b in names_],
          [img_r2[b]["CI"][1] - img_r2[b]["R2"] for b in names_]]), color=["#3b6ea5" if b in T.PRETRAINED else "#999999" for b in names_], capsize=3)
ax[0].axhline(VAL_SUMMARY["V"].get("R2", np.nan) if SEL["V"]["feature_set"] == "NUM" else T.r2(yV, VAL_PRED[("V", "NUM", "linear")]), color="#c0392b", ls="--", label="numbers only")
ax[0].set_xticks(range(len(names_))); ax[0].set_xticklabels(names_, rotation=40, ha="right", fontsize=8)
ax[0].set_title("Volatility R2 from chart images alone (validation; grey = random weights)"); ax[0].legend()
vd = views_df[views_df.target == "V"].pivot_table(index="backbone", columns="views", values="delta")
im = ax[1].imshow(vd.values, cmap="RdBu", vmin=-0.02, vmax=0.02, aspect="auto")
ax[1].set_xticks(range(vd.shape[1])); ax[1].set_xticklabels(vd.columns); ax[1].set_yticks(range(vd.shape[0])); ax[1].set_yticklabels(vd.index, fontsize=8)
ax[1].set_title("Validation R2 gain over numbers: single view vs all three"); plt.colorbar(im, ax=ax[1])
plt.tight_layout(); fpath = os.path.join(FIG, "stage6_validation.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6_validation.png")

# ---- freeze: everything the test and holdout stages are allowed to use
FROZEN = dict(run_tag=RUN_TAG, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), splits=SPLITS,
              holdout_start=HOLDOUT_START, n_rows={s: int(m.sum()) for s, m in (("train", TR), ("val", VA), ("test", TE))},
              split_ids_sha256=_sha(os.path.join(RES, f"split_ids_{RUN_TAG}.parquet")),
              selection=SEL, calibrators=CAL, volatility_residuals=VRES, rule=RULE, prereg=PREREG, H2=H2, controls=CONTROLS,
              n_configurations_tried=len(LEDGER), backbones=NAMES, pca_dim=PCA_DIM, seed=SEED)
FROZEN_PATH = os.path.join(WORK, "frozen", f"frozen_config_{RUN_TAG}.json")
json.dump(FROZEN, open(FROZEN_PATH, "w"), indent=1, default=float)
shutil.copy(FROZEN_PATH, os.path.join(DRIVE_DIR, os.path.basename(FROZEN_PATH)))
push(FROZEN_PATH, f"frozen/{os.path.basename(FROZEN_PATH)}")
print(f"\nFrozen: {FROZEN_PATH} (also on Drive and Hugging Face). Nothing below may change these choices.")
print(f"Stage 6 done in {time.time() - t0:.0f}s")
