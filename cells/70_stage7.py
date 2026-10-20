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

# ---- every candidate model on the test years. DESCRIPTIVE ONLY: the frozen choices above are not changed by this table.
t1 = time.time()
TB = []
for t in TARGETS:
    y = Y[t][MTE[t]]; pm, fn = PRIMARY_LB[t]

    def add_tb(label, kind, fs, head, cond):
        _, p = refit(t, fs, head, FIT_TV, MTE[t])                     # same frozen hyper-parameters, refitted on train + validation
        met = T.board_metrics(TARGETS[t], y, p, DATES[MTE[t]], base=BASE.get(t))
        ci, _ = BOOT_TE[t].ci(fn, y, p)
        TB.append(dict(target=t, model=label, kind=kind, condition=cond, feature_set=fs, head=head,
                       chosen=(SELF[t]["feature_set"], SELF[t]["head"]) == (fs, head), primary=pm, primary_value=met[pm],
                       ci_lo=ci[0], ci_hi=ci[1], **met))
    add_tb("Numbers only (33 features)", "reference", "NUM", best_of(t, "NUM"), "numbers only")
    for b in [b for b in NAMES if b in T.PRETRAINED] + FT_NAMES:
        kind = "fine-tuned" if b in FT_NAMES else "frozen"
        add_tb(T.display_name(b), kind, f"NUM+IMG:{b}", best_of(t, f"NUM+IMG:{b}"), "numbers + images")
        add_tb(T.display_name(b), kind, f"IMG:{b}", "linear", "images only")
BOARD_TEST = pd.DataFrame(TB)
save_df(BOARD_TEST, "stage7_leaderboard_test.csv")
for t in TARGETS:
    sub = BOARD_TEST[(BOARD_TEST.target == t) & (BOARD_TEST.condition != "images only")].sort_values("primary_value", ascending=False)
    sub = sub.assign(model=sub["model"] + np.where(sub["chosen"], "  <== chosen", ""))
    print(f"\n{TARGET_NAME[t]} - TEST years, numbers + images (descriptive; probabilities uncalibrated here):")
    print(sub[["model", "kind", "head"] + MET_COLS[t] + ["ci_lo", "ci_hi"]].round(4).to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, t in zip(axes, ("V", "D10", "O5")):
    sub = BOARD_TEST[(BOARD_TEST.target == t) & (BOARD_TEST.condition == "numbers + images")].sort_values("primary_value")
    ref = BOARD_TEST[(BOARD_TEST.target == t) & (BOARD_TEST.feature_set == "NUM")].iloc[0]
    yy = np.arange(len(sub))
    ax.barh(yy, sub["primary_value"], color=[COLORS[k] for k in sub["kind"]],
            xerr=np.vstack([sub["primary_value"] - sub["ci_lo"], sub["ci_hi"] - sub["primary_value"]]), capsize=2)
    ax.axvline(ref["primary_value"], color=COLORS["reference"], ls="--", label=f"numbers only ({ref['primary_value']:.3f})")
    ax.set_yticks(yy); ax.set_yticklabels([m + ("  ★" if c else "") for m, c in zip(sub["model"], sub["chosen"])], fontsize=8)
    ax.set_xlim(min(sub["ci_lo"].min(), ref["primary_value"]) - 0.01, max(sub["ci_hi"].max(), ref["primary_value"]) + 0.01)
    ax.set_title(f"{TARGET_NAME[t]}\nTEST {PRIMARY_LB[t][0]} (numbers + images, 95% CI; ★ = model chosen on validation)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); fpath = os.path.join(FIG, "stage7_leaderboard_test.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage7_leaderboard_test.png")
print(f"Test leaderboard: {len(BOARD_TEST)} model rows in {time.time() - t1:.0f}s")

# ---- 120 vs 60 stocks: this run's frozen models vs the pilot's, on the SAME 60 stocks and the SAME test rows
VS_PILOT = None
if PILOT_SET:
    pc = None
    try:
        src = os.environ.get("TFCV_PILOT_BUNDLE")
        pc = json.load(open(os.path.join(src, "config.json") if src else
                            hf_hub_download(PILOT_MODEL_REPO, "config.json", token=HF_TOKEN, local_dir=os.path.join(WORK, "_pilot"))))
    except Exception as e:
        print(f"Pilot bundle {PILOT_MODEL_REPO} not readable ({type(e).__name__}); skipping the 60 vs 120 comparison.")
    if pc:
        mp = df["ticker"].isin(PILOT_SET).values
        rv = []
        for t, mets in (("V", ("R2", "wmR2", "MAE")), ("D10", ("AUC", "BSS", "ECE")), ("O5", ("AUC",))):
            sm = mp[MTE[t]]
            now = eval_target(t, Y[t][MTE[t]][sm], TEST_PRED[t][sm], DATES[MTE[t]][sm],
                              s2=FZ["volatility_residuals"]["s2"] if t == "V" else None, base=BASE.get(t))
            for m_ in mets:
                rv.append(dict(target=t, metric=m_, pilot_model_60=pc["metrics"]["test"][t].get(m_), this_model_on_pilot_60=now.get(m_),
                               this_model_all_stocks=TEST_SUMMARY[t].get(m_), rows_pilot=pc["metrics"]["test"][t].get("n"), rows_now=int(sm.sum()),
                               pilot_choice=pc["targets"][t]["feature_set"] + " / " + pc["targets"][t]["head"],
                               this_choice=SELF[t]["feature_set"] + " / " + SELF[t]["head"]))
        VS_PILOT = pd.DataFrame(rv)
        save_df(VS_PILOT, "stage7_vs_pilot.csv")
        same = (VS_PILOT["rows_pilot"] == VS_PILOT["rows_now"]).all()
        print(f"\nMore data: 60-stock pilot model vs this {df['ticker'].nunique()}-stock model on the pilot's 60 stocks, test years "
              f"({'identical' if same else 'DIFFERENT - check'} test rows: {VS_PILOT['rows_now'].iloc[0]:,}). Lower is better for MAE and ECE.")
        print(VS_PILOT[["target", "metric", "pilot_model_60", "this_model_on_pilot_60", "this_model_all_stocks", "pilot_choice", "this_choice"]].round(4).to_string(index=False))
        fig, ax = plt.subplots(figsize=(10, 4))
        lab = VS_PILOT["target"] + " " + VS_PILOT["metric"]; xx = np.arange(len(VS_PILOT))
        ax.bar(xx - 0.2, VS_PILOT["pilot_model_60"], 0.4, label="pilot model (trained on 60 stocks)", color="#9e9e9e")
        ax.bar(xx + 0.2, VS_PILOT["this_model_on_pilot_60"], 0.4, label=f"this model (trained on {df['ticker'].nunique()} stocks)", color="#3b6ea5")
        ax.set_xticks(xx); ax.set_xticklabels(lab, fontsize=8); ax.legend(fontsize=8)
        ax.set_title("Does more training data help? Same 60 stocks, same test rows (2020-2022)", fontsize=10)
        plt.tight_layout(); fpath = os.path.join(FIG, "stage7_vs_pilot.png"); plt.savefig(fpath, dpi=130); plt.show()
        push(fpath, f"figures_{RUN_TAG}/stage7_vs_pilot.png")
TEST_DD10_RATE = float(np.mean(Y["D10"][MTE["D10"]]))

# ---- deployment rule = frozen rule + the pre-registered test-year check on direction (part b of PREREG['no_trade'])
RULE_DEPLOY = dict(FZ["rule"])
RULE_DEPLOY["direction_usable"] = bool(FZ["rule"]["direction_usable"] and TEST_SUMMARY["O5"]["AUC_CI"][0] > 0.5)
print(f"Direction shown as a call on the dashboard: {RULE_DEPLOY['direction_usable']} "
      f"(validation check {FZ['rule']['direction_usable']}, test AUC CI {TEST_SUMMARY['O5']['AUC_CI'][0]:.3f}-{TEST_SUMMARY['O5']['AUC_CI'][1]:.3f})")
TEST_RESULTS = dict(summary=TEST_SUMMARY, H1=H1, H1_supported=H1_supported, H3=H3, no_trade=NT, survivorship=SURV,
                    rule_deploy=RULE_DEPLOY, n_test_rows={t: int(MTE[t].sum()) for t in TARGETS}, dd10_rate_test=TEST_DD10_RATE,
                    leaderboard=BOARD_TEST.to_dict("records"), vs_pilot=None if VS_PILOT is None else VS_PILOT.to_dict("records"))
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
