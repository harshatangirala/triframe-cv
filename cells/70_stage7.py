# ==================================== STAGE 7: one honest test of the frozen choices (2020-2022) ====================================
t0 = time.time()
marker = os.path.join(WORK, "frozen", f"test_opened_{RUN_TAG}.json")
if os.path.exists(marker) and not os.environ.get("TFCV_TEST_REUSE_REASON"):
    raise RuntimeError(f"The {RUN_TAG} test set was already evaluated ({marker}). Re-running it would turn it into a tuning set. "
                       "If you must re-run (e.g. a crash), set os.environ['TFCV_TEST_REUSE_REASON'] = '<why>' and it is logged.")
FZ = json.load(open(FROZEN_PATH))                         # only frozen choices are used below
SELF = FZ["selection"]
cal = lambda t, p: np.interp(p, FZ["calibrators"][t]["x"], FZ["calibrators"][t]["y"])
FIT_TV = TR | VA


def refit(t, fs, head, fit_mask, pred_mask):
    """Refit a frozen configuration on fit_mask rows (same hyper-parameters) and predict pred_mask rows."""
    X = design(fs); mf = fit_mask & OK[t]; f = FITS[(t, fs, head)]
    if head == "linear":
        if TARGETS[t] == "reg":
            w, b, p = T.ridge_path(X[mf], Y[t][mf], X[pred_mask], [f["param"]])[f["param"]]
        else:
            w, b, p = T.logit_path(X[mf], Y[t][mf], X[pred_mask], [f["param"]], device=DEVICE)[f["param"]]
        return dict(head="linear", coef=w, intercept=float(b)), p
    m, _, _ = T.tree_fit(X[mf], Y[t][mf], None, None, TARGETS[t], f["cfg"], gpu=(DEVICE == "cuda"), seed=SEED, n_estimators=f["n_estimators"])
    return dict(head="tree", model=m), T.tree_predict(m, X[pred_mask], TARGETS[t])


MTE = {t: TE & OK[t] for t in TARGETS}
BOOT_TE = {t: T.MonthBoot(DATES[MTE[t]], N_BOOT, seed=SEED) for t in TARGETS}
FINAL, TEST_PRED, TEST_SUMMARY = {}, {}, {}
for t in TARGETS:
    s = SELF[t]
    FINAL[t], p = refit(t, s["feature_set"], s["head"], FIT_TV, MTE[t])
    if TARGETS[t] == "clf":
        p = cal(t, p)
    TEST_PRED[t] = p
    y = Y[t][MTE[t]]; bt = BOOT_TE[t]
    met = eval_target(t, y, p, DATES[MTE[t]], s2=FZ["volatility_residuals"]["s2"] if t == "V" else None,
                      xret=XRET[MTE[t]] if t == "O5" else None, base=BASE.get(t))
    if t == "V":
        met["R2_CI"] = bt.ci(T.r2, y, p)[0]; met["wmR2_CI"] = bt.ci(T.wm_r2, y, p, uses_months=True)[0]
    else:
        met["AUC_CI"] = bt.ci(T.auc, y, p)[0]
        met["BSS_CI"] = bt.ci(lambda yy, pp: 1 - T.brier(yy, pp) / T.brier(yy, np.full(len(yy), BASE[t])), y, p)[0]
    TEST_SUMMARY[t] = met
    print(f"{t} ({s['feature_set']} / {s['head']}): " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in met.items()))

# ---- H1: numbers + all three chart views vs numbers only, on the test years (pre-registered pair)
H1 = {}
for t, fn, nm in (("V", T.r2, "R2"), ("O5", T.auc, "AUC")):
    pn_, pi_ = SELF[t]["h1_numbers"], SELF[t]["h1_images"]
    if pi_ is None:
        continue
    _, p_num = refit(t, pn_["feature_set"], pn_["head"], FIT_TV, MTE[t])
    _, p_img = refit(t, pi_["feature_set"], pi_["head"], FIT_TV, MTE[t])
    y = Y[t][MTE[t]]
    ci, _ = BOOT_TE[t].diff(fn, y, p_img, p_num)
    H1[t] = dict(images_model=f"{pi_['feature_set']} / {pi_['head']}", numbers_model=f"NUM / {pn_['head']}",
                 metric=nm, numbers=fn(y, p_num), images=fn(y, p_img), delta=fn(y, p_img) - fn(y, p_num), CI=ci)
H1_supported = bool(any(v["CI"][0] > 0 for v in H1.values()))
print(f"\nH1 (three timeframes of chart images add to the numbers, test years): {'SUPPORTED' if H1_supported else 'NOT SUPPORTED'}")
for t, v in H1.items():
    print(f"   {t}: {v['images_model']} vs {v['numbers_model']}: delta {v['metric']} {v['delta']:+.4f}, 95% CI [{v['CI'][0]:+.4f}, {v['CI'][1]:+.4f}]")

# ---- H3: does abstention improve accepted direction calls? (thresholds fixed on validation)
thr10 = FZ["rule"]["thresholds_by_coverage"]["0.1"]
yo, po = Y["O5"][MTE["O5"]].astype(int), TEST_PRED["O5"]
abst_te, _ = T.abstention_table(po, yo, thresholds={float(k): v for k, v in FZ["rule"]["thresholds_by_coverage"].items()})


def _gain(yy, pp):
    keep = np.abs(pp - 0.5) >= thr10
    return (((pp[keep] > 0.5) == yy[keep]).mean() if keep.any() else np.nan) - ((pp > 0.5) == yy).mean()


h3_ci, _ = BOOT_TE["O5"].ci(_gain, yo, po)
H3 = dict(gain_10pct_vs_all=float(_gain(yo, po)), CI=h3_ci, supported=bool(h3_ci[0] > 0), table=abst_te.to_dict("records"))
print(f"H3 (abstention raises accuracy of accepted calls, test years): {'SUPPORTED' if H3['supported'] else 'NOT SUPPORTED'} "
      f"- accuracy gain at 10% coverage {H3['gain_10pct_vs_all']:+.4f}, 95% CI [{h3_ci[0]:+.4f}, {h3_ci[1]:+.4f}]")
print(abst_te.round(4).to_string(index=False))

# ---- NO TRADE rule on the test years: what does the risk flag separate?
common = MTE["V"] & MTE["O5"] & MTE["D10"]
def _on(t):                                            # test predictions restricted to rows valid for all targets
    return TEST_PRED[t][common[MTE[t]]]
vol_hat, pdd = C.ann_vol(_on("V")), _on("D10")
risk = (vol_hat >= FZ["rule"]["vol_ann_q80"]) | (pdd >= FZ["rule"]["p_dd10_q80"])
real_vol, real_dd = C.ann_vol(Y["V"][common]), Y["D10"][common]
NT = dict(share_flagged=float(risk.mean()),
          realised_vol_flagged=float(real_vol[risk].mean()), realised_vol_other=float(real_vol[~risk].mean()),
          dd10_rate_flagged=float(real_dd[risk].mean()), dd10_rate_other=float(real_dd[~risk].mean()))
if FZ["rule"]["direction_usable"]:
    callm = (~risk) & (np.abs(_on("O5") - 0.5) >= FZ["rule"]["conf_threshold"])
    NT["direction_calls_share"] = float(callm.mean())
    NT["direction_calls_accuracy"] = float(((_on("O5")[callm] > 0.5) == Y["O5"][common][callm]).mean()) if callm.any() else None
print(f"\nNO TRADE risk flag on the test years: flags {NT['share_flagged']:.1%} of cases; realised volatility "
      f"{NT['realised_vol_flagged']:.1%} vs {NT['realised_vol_other']:.1%} a year; 10% drawdowns {NT['dd10_rate_flagged']:.1%} vs {NT['dd10_rate_other']:.1%}")

# ---- survivorship sensitivity: refit without the 2001-2009 cache (the cache most exposed to survivorship)
SURV = {}
no_early = FIT_TV & (df["src"].values != "cache_2001_2009")
for t in TARGETS:
    s = SELF[t]
    _, p2 = refit(t, s["feature_set"], s["head"], no_early, MTE[t])
    p2 = cal(t, p2) if TARGETS[t] == "clf" else p2
    y = Y[t][MTE[t]]
    SURV[t] = dict(with_2001_2009=T.r2(y, TEST_PRED[t]) if t == "V" else T.auc(y, TEST_PRED[t]),
                   without_2001_2009=T.r2(y, p2) if t == "V" else T.auc(y, p2), metric="R2" if t == "V" else "AUC")
print("Survivorship sensitivity (test metric with / without the 2001-2009 rows in training):",
      {t: f"{v['metric']} {v['with_2001_2009']:.4f} / {v['without_2001_2009']:.4f}" for t, v in SURV.items()})

# ---- deployment rule = frozen rule + the pre-registered test-year check on direction (part b of PREREG['no_trade'])
RULE_DEPLOY = dict(FZ["rule"])
RULE_DEPLOY["direction_usable"] = bool(FZ["rule"]["direction_usable"] and TEST_SUMMARY["O5"]["AUC_CI"][0] > 0.5)
print(f"Direction shown as a call on the dashboard: {RULE_DEPLOY['direction_usable']} "
      f"(validation check {FZ['rule']['direction_usable']}, test AUC CI {TEST_SUMMARY['O5']['AUC_CI'][0]:.3f}-{TEST_SUMMARY['O5']['AUC_CI'][1]:.3f})")
TEST_RESULTS = dict(summary=TEST_SUMMARY, H1=H1, H1_supported=H1_supported, H3=H3, no_trade=NT, survivorship=SURV,
                    rule_deploy=RULE_DEPLOY, n_test_rows={t: int(MTE[t].sum()) for t in TARGETS})
tpath = os.path.join(RES, "stage7_test_results.json")
json.dump(TEST_RESULTS, open(tpath, "w"), indent=1, default=float)
push(tpath, f"results_{RUN_TAG}/stage7_test_results.json")
pred_df = df.loc[MTE["V"], ["sample_id", "ticker", "date"]].assign(pred_logrv20=TEST_PRED["V"])
ppath = os.path.join(RES, "stage7_test_predictions.parquet"); pred_df.to_parquet(ppath, index=False)
push(ppath, f"results_{RUN_TAG}/stage7_test_predictions.parquet")
json.dump(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), reuse_reason=os.environ.get("TFCV_TEST_REUSE_REASON", "")),
          open(marker, "w"))
push(marker, f"frozen/{os.path.basename(marker)}")
print(f"\nStage 7 done in {time.time() - t0:.0f}s")
