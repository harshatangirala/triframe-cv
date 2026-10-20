# ==================================== STAGE 11: report (markdown + PDF, computed values only), provenance and final upload ====================================
def _row(t, d):
    keys = ("R2", "wmR2", "QLIKE", "RMSE") if t == "V" else ("AUC", "AP", "precision", "recall", "F1", "BSS", "ECE")
    return "; ".join(f"{k} {d[k]:.4f}" for k in keys if k in d and isinstance(d[k], float) and np.isfinite(d[k]))


yv_ = lambda k: f"{SPLITS[k][0][:4]}-{int(SPLITS[k][1][:4]) - 1}"
lines = [f"# TriFrame CV - end-to-end run report ({RUN_TAG}, notebook {NOTEBOOK_VERSION})", "",
         f"Run {RUN_ID}; code {CODE_HASH}; config {CONFIG_HASH}. Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} "
         "from the values computed in this notebook.", "",
         "## Status of the evaluation periods",
         f"- Validation {yv_('val')}: used to choose and calibrate (the only selection data).",
         f"- Test {yv_('test')}: historical diagnostic (already read by earlier runs); used for nothing.",
         f"- 2023 onward: exposed (opened {len(HOLDOUT_OPENINGS)} times before this run); "
         + ("opened again here and labelled as such." if "holdout" in BUNDLE_CFG["metrics"] else "not opened in this run."), "",
         "## Data", f"- Universe: {len(all_tickers)} US stocks; this run: {df['ticker'].nunique()} stocks"
         + (" (containing the " + " and ".join(f"{n}-stock" for n in SUBSETS if n < QUICK_TICKERS) + " samples of earlier runs)" if len(SUBSETS) > 1 else "")
         + f", {len(df):,} chart sets (weekly 104 / daily 126 / hourly 140 bars) with 33 numeric features.",
         f"- Splits: train {SPLITS['train']}, validation {SPLITS['val']}, test {SPLITS['test']} (20-day purge).",
         "", "## Models compared",
         f"- {len(NAMES)} frozen backbones and {len(FT_NAMES)} fine-tuned ones ({', '.join(FT_NAMES) or 'none'}); PCA {PCA_DIM} per view; "
         f"linear and boosted-tree heads; baselines HAR, rolling 22-day, EWMA, GARCH(1,1); {FZ['n_configurations_tried']} configurations tried on validation.",
         f"- Fine-tuning: {FT_VERDICT}", "",
         "## Chosen model per target (pre-registered rule) and results",
         "| Target | Chosen inputs / head | Validation | Test (historical diagnostic) |", "|---|---|---|---|"]
for t in TARGETS:
    lines.append(f"| {TARGET_NAME[t]} | {SELF[t]['feature_set']} / {SELF[t]['head']} | {_row(t, VAL_SUMMARY[t])} | {_row(t, TEST_SUMMARY[t])} |")
lines += ["", "## Did training reach the minimum? (Stage 6D)"] + [f"- {r.component}: {r.verdict} ({r.evidence})" for r in OPT.itertuples()]
lines += ["", "## Evaluation metrics (Stage 7E; thresholds fixed on validation)"]
for r in EVAL.itertuples():
    if r.target == "V":
        lines.append(f"- V, {r.split}, {r.model}: R2 {r.R2:.4f}, RMSE {r.RMSE:.4f}, MAE {r.MAE:.4f}")
    else:
        lines.append(f"- {r.target}, {r.split}, {r.model}: AUC {r.AUC:.4f}, precision {r.precision:.3f}, recall {r.recall:.3f}, F1 {r.F1:.3f}, "
                     f"accuracy {r.accuracy:.3f} (majority {r.majority_acc:.3f}), prevalence {r.prevalence:.3f}")
lines += ["", "## Test leaderboard (descriptive; best head per model)"]
for t in TARGETS:
    sub = BOARD_TEST[(BOARD_TEST.target == t) & (BOARD_TEST.condition != "images only")].sort_values("primary_value", ascending=False)
    lines.append(f"- {TARGET_NAME[t]} ({sub['primary'].iloc[0]}): " + "; ".join(
        f"{r.model}{' (chosen)' if r.chosen else ''} {r.primary_value:.3f}" for r in sub.head(8).itertuples()))
lines += ["", "## Hypotheses (rules fixed before the test years were read)",
          f"- H1, chart images add to the numbers (test): **{'supported' if H1_supported else 'not supported'}**. "
          + "; ".join(f"{t}: delta {v['metric']} {v['delta']:+.4f} [{v['CI'][0]:+.4f}, {v['CI'][1]:+.4f}] (quarter blocks "
                      f"[{v['CI_quarter_blocks'][0]:+.4f}, {v['CI_quarter_blocks'][1]:+.4f}])" for t, v in H1.items()),
          f"- H2, backbones differ (validation): **{'supported' if FZ['H2']['supported'] else 'not supported'}** "
          f"({len(FZ['H2']['significant_pairs'])} of {FZ['H2']['n_pairs']} pairs after Holm).",
          f"- H3, abstention improves accepted direction calls (test): **{'supported' if H3['supported'] else 'not supported'}** "
          f"(gain at 10% coverage {H3['gain_10pct_vs_all']:+.4f} [{H3['CI'][0]:+.4f}, {H3['CI'][1]:+.4f}]).",
          f"- Direction as a continuous ranking (Stage 7D): {DIRECTION_VERDICT}.",
          "", "## Controls (validation)",
          "- " + "; ".join(f"{k}: " + ", ".join(f"{a} {b:.4f}" for a, b in v.items() if isinstance(b, float)) if isinstance(v, dict)
                          else f"{k}: {v:.4f}" for k, v in FZ["controls"].items()),
          "", "## NO TRADE rule on the test years",
          f"- Flags {NT['share_flagged']:.1%} of cases; realised volatility {NT['realised_vol_flagged']:.1%} flagged vs {NT['realised_vol_other']:.1%} other; "
          f"10% drawdowns {NT['dd10_rate_flagged']:.1%} vs {NT['dd10_rate_other']:.1%} (risk stratification; no portfolio test yet).",
          "", "## Monte Carlo simulation (Stage 7M)",
          f"- Student-t {MC['t_df']:g} d.f., intraday-low range {MC['low_k']:g}; 90% band coverage validation {MC['coverage']['val']['90%']:.1%}, "
          f"test {MC['coverage']['test']['90%']:.1%}; 50% band {MC['coverage']['val']['50%']:.1%} / {MC['coverage']['test']['50%']:.1%}.",
          "", "## Training-history sensitivity (test metric with / without 2001-2009 rows; not a survivorship test)"]
lines += [f"- {t}: {v['metric']} {v['with_2001_2009']:.4f} / {v['without_2001_2009']:.4f}" for t, v in SURV.items()]
if VS_PILOT is not None:
    lines += ["", "## More data: earlier runs vs this run on the earlier runs' own stocks and rows"]
    lines += [f"- {r.earlier_run}, {r.target} {r.metric}: validation {r.earlier_val:.4f} -> {r.this_val:.4f}; test {r.earlier_test:.4f} -> {r.this_test:.4f}"
              for r in VS_PILOT.itertuples()]
if "holdout" in BUNDLE_CFG["metrics"]:
    lines += ["", "## 2023+ period (previously viewed; NOT an untouched holdout)"] + [f"- {k}: " + _row(k.split('|')[0], v) for k, v in BUNDLE_CFG["metrics"]["holdout"].items()]
lines += ["", "## Where everything is",
          f"- Results, predictions, metric provenance (metrics_provenance.csv), figures, ledger, PDF: https://huggingface.co/datasets/{RUNS_REPO} (private)",
          f"- Model bundle: https://huggingface.co/{MODEL_REPO} (private)",
          f"- Dashboard: {DASHBOARD_URL or 'streamlit_app.py in the GitHub repo (deploy on Streamlit Cloud, docs/DEPLOY.md)'}",
          "", "Limits: survivor US stocks from one vendor; live inputs from Yahoo Finance; research output, not investment advice."]
REPORT_PATH = os.path.join(RES, f"REPORT_{RUN_TAG}.md")
open(REPORT_PATH, "w", encoding="utf-8").write("\n".join(lines))
shutil.copy(REPORT_PATH, os.path.join(DRIVE_DIR, os.path.basename(REPORT_PATH)))
save_df(pd.DataFrame(LEDGER), "ledger.csv")
save_df(pd.DataFrame(METRICS_LOG), "metrics_provenance.csv")
push(REPORT_PATH, f"results_{RUN_TAG}/{os.path.basename(REPORT_PATH)}")
print("\n".join(lines))

# ---- the plain-language PDF report, from the values above
if REPORT_PDF:
    yrs = lambda k: (int(SPLITS[k][0][:4]), int(SPLITS[k][1][:4]) - (1 if SPLITS[k][1][5:7] == "01" else 0) + 1)
    F_ = lambda n: os.path.join(FIG, n)
    figs = dict(pipeline=F_("architecture.png"), leaderboard=F_("stage7_leaderboard_test.png"),
                finetune=F_("stage5F_learning_curves.png") if FT_NAMES else None, finetune_bars=F_("stage5F_finetuning.png") if FT_NAMES else None,
                mc=F_("stage7M_monte_carlo.png"), convex=F_("stage6D_convex_heads.png"), landscape=F_("stage6D_loss_landscape.png"),
                boosting=F_("stage6D_boosting_curves.png"), learning=F_("stage6D_learning_curves.png"), eval_D10=F_("stage7E_D10_evaluation.png"),
                eval_V=F_("stage7E_V_evaluation.png"), metric_bars=F_("stage7E_metric_bars.png"), vs_prev=F_("stage7_vs_previous_runs.png"),
                timeline=RP.fig_timeline(F_("timeline.png"), [("train", *yrs("train"), "#3b6ea5"), ("validation", *yrs("val"), "#e67e22"),
                                                             ("test (diagnostic)", *yrs("test"), "#2e9d5b"), ("2023+ (exposed)", yrs("test")[1], 2027, "#7f8c8d")]),
                example=RP.fig_example(F_("example_charts.png")))
    hold = BUNDLE_CFG["metrics"].get("holdout", {})
    ts = TEST_SUMMARY; o5_ci = ts["O5"]["AUC_CI"]
    chosen = []
    for t, lab, test, test_ci, hold_txt, plain in (
            ("V", "Volatility, next 20 days", f"R² {ts['V']['R2']:.2f}", f"R² {ts['V']['R2']:.3f} ({ts['V']['R2_CI'][0]:.2f} to {ts['V']['R2_CI'][1]:.2f})",
             f"R² {hold['V|all']['R2']:.2f}" if "V|all" in hold else "not opened",
             f"explains about {ts['V']['R2']:.0%} of the variation in volatility"),
            ("D10", "10%+ drawdown within 20 days", f"AUC {ts['D10']['AUC']:.2f}", f"AUC {ts['D10']['AUC']:.3f} ({ts['D10']['AUC_CI'][0]:.2f} to {ts['D10']['AUC_CI'][1]:.2f})",
             f"AUC {hold['D10|all']['AUC']:.2f}" if "D10|all" in hold else ("not scored" if hold else "not opened"),
             f"ranks a stock that drops 10% above one that does not {ts['D10']['AUC']:.0%} of the time"),
            ("O5", "Beats the median stock, 5 days", f"AUC {ts['O5']['AUC']:.3f}", f"AUC {ts['O5']['AUC']:.3f} ({o5_ci[0]:.3f} to {o5_ci[1]:.3f})",
             f"AUC {hold['O5|all']['AUC']:.3f}" if "O5|all" in hold else "not opened",
             "no better than a coin flip" if o5_ci[0] <= 0.5 else "a small but real edge")):
        miu = MIU.iloc[["V", "D10", "O5"].index(t)]
        chosen.append(dict(target=lab, model=f"{miu['Vision network']} + {miu['Head']}" if SELF[t]["feature_set"] != "NUM" else f"numbers only + {miu['Head']}",
                           inputs=miu["Inputs"], network=miu["Vision network"], head=miu["Head"],
                           val=(f"R² {VAL_SUMMARY[t]['R2']:.3f}" if t == "V" else f"AUC {VAL_SUMMARY[t]['AUC']:.3f}"),
                           test=test, test_ci=test_ci, holdout=hold_txt, plain=plain))
    cards_pdf = []
    for r in CARDS.to_dict("records"):
        if r["key"].startswith(T.FT_PREFIX):
            continue
        ftr = records.get(T.ft_name(r["key"]), {}).get("ft") if FT_NAMES else None
        cards_pdf.append(dict(model=r["model"], family=r["family"], pretrained_on=r["pretrained on"], params_m=r["parameters (M)"],
                              finetuning=(f"{ftr['epochs_run']} epochs (best {ftr['best_epoch']}), {ftr['minutes']:.0f} min" if ftr else "-")))
    img_best = max((v["R2"] for b, v in FZ["H2"]["image_only_R2"].items() if b in T.PRETRAINED), default=float("nan"))
    rnd = [v["R2"] for b, v in FZ["H2"]["image_only_R2"].items() if b in T.RANDOM]
    ev = lambda t, split, model_prefix: EVAL[(EVAL.target == t) & (EVAL.split == split) & EVAL.model.str.startswith(model_prefix)].iloc[0]
    eval_rows = []
    for t in ("V", "D10", "O5"):
        for split in ("validation", "test"):
            for mp in ("chosen", "numbers only"):
                r_ = ev(t, split, mp)
                eval_rows.append([TARGET_NAME[t].split(" (")[0], split, "chosen model" if mp == "chosen" else "numbers only"] +
                                 ([f"R² {r_.R2:.3f}", f"RMSE {r_.RMSE:.3f}", "-", "-"] if t == "V" else
                                  [f"AUC {r_.AUC:.3f}", f"precision {r_.precision:.2f}", f"recall {r_.recall:.2f}", f"F1 {r_.F1:.2f}"]))
    prog = []
    if VS_PILOT is not None:
        for r in VS_PILOT[VS_PILOT.metric.isin(["R2", "AUC"])].itertuples():
            prog.append([r.earlier_run, TARGET_NAME[r.target].split(" (")[0], r.metric, f"{r.earlier_val:.3f}", f"{r.this_val:.3f}",
                         f"{r.earlier_test:.3f}", f"{r.this_test:.3f}"])
    R = dict(
        title_run=f"{df['ticker'].nunique()}-stock run ({RUN_TAG}, notebook {NOTEBOOK_VERSION})", version="v4", n_stocks=int(df["ticker"].nunique()),
        universe=len(all_tickers), n_rows=len(df), pilot_included=bool(PILOT_SET),
        splits={"train": (f"{yrs('train')[0]}-{yrs('train')[1] - 1}", int(TR.sum()), int(df.loc[TR, 'ticker'].nunique()), "fit the models"),
                "validation": (f"{yrs('val')[0]}-{yrs('val')[1] - 1}", int(VA.sum()), int(df.loc[VA, 'ticker'].nunique()), "tune, select, calibrate, set thresholds and the simulation"),
                "test": (f"{yrs('test')[0]}-{yrs('test')[1] - 1}", int(TE.sum()), int(df.loc[TE, 'ticker'].nunique()), "historical diagnostic: already read by earlier runs"),
                "2023+": (f"{HOLDOUT_START[:4]}-2026", 0, 0, f"exposed: opened {len(HOLDOUT_OPENINGS)} times before" + (" and again here" if hold else ""))},
        cards=cards_pdf, n_configs=int(len(LEDGER)), chosen=chosen,
        metrics=dict(test_V_R2=ts["V"]["R2"], direction_usable=bool(RULE_DEPLOY["direction_usable"])),
        hypotheses=dict(
            H1=(bool(H1_supported), "On the test years, adding all three chart views to the numbers changed " + " and ".join(
                f"{'volatility R²' if t == 'V' else 'direction AUC'} by {v['delta']:+.3f} (95% CI {v['CI'][0]:+.3f} to {v['CI'][1]:+.3f})" for t, v in H1.items())
                + (". Both intervals include zero." if not H1_supported else ".")),
            H2=(bool(FZ["H2"]["supported"]), f"{len(FZ['H2']['significant_pairs'])} of {FZ['H2']['n_pairs']} pairs of frozen pretrained networks "
                f"differ in how well their pictures alone predict volatility (validation, Holm-corrected); the best reaches R² {img_best:.3f}."),
            H3=(bool(H3["supported"]), f"Keeping only the 10% most confident direction calls changed their accuracy by {H3['gain_10pct_vs_all']:+.3f} "
                f"(95% CI {H3['CI'][0]:+.3f} to {H3['CI'][1]:+.3f}) on the test years.")),
        controls=(f"With shuffled labels the pipeline finds nothing (volatility R² {FZ['controls']['label_shuffle_V']['R2']:.3f}, direction AUC "
                  f"{FZ['controls']['label_shuffle_O5']['AUC']:.3f}), and randomly initialised ResNet-18s reach volatility R² {min(rnd):.3f} to "
                  f"{max(rnd):.3f} from pictures alone, against {img_best:.3f} for the best pretrained network." if rnd else None),
        no_trade=dict(NT, vol_q80=FZ["rule"]["vol_ann_q80"], pdd_q80=FZ["rule"]["p_dd10_q80"]),
        finetune=(dict(intro=f"<b>In this run every pretrained network was fine-tuned</b> (early stopping on {FT_INNER[0][:4]}-{int(FT_INNER[1][:4]) - 1}, "
                             f"inside the training years), plus a controlled study and extra seeds.", verdict=FT_VERDICT,
                       rows=[[r.model, f"{r.V_num_img_frozen:.3f}", f"{r.V_num_img_finetuned:.3f}", f"{r.V_gain:+.3f} ({r.V_gain_lo:+.3f} to {r.V_gain_hi:+.3f})",
                              f"{r.minutes:.0f} min"] for r in FT_CMP.itertuples()]) if FT_NAMES else None),
        optimisation=dict(rows=[[r.component, r.convex, r.verdict] for r in OPT.itertuples()]),
        evaluation=dict(rows=eval_rows, thresholds=FZ["thresholds"]),
        progress=dict(rows=prog) if prog else None,
        direction=DIRECTION_VERDICT,
        mc=dict(t_df=MC["t_df"], tilt_default=MC["tilt_default"], test_cov90=MC["coverage"]["test"]["90%"], val_cov90=MC["coverage"]["val"]["90%"],
                test_cov50=MC["coverage"]["test"]["50%"]),
        audit=dict(openings=len(HOLDOUT_OPENINGS), fixes=["early stopping of fine-tuning moved off the selection years",
                                                          "QLIKE in leaderboards uses validation residual variance (was test)",
                                                          "rank IC computed per week with a Newey-West t-statistic (was empty)",
                                                          "precision / recall / F1 at thresholds fixed on validation, shown with the event rate",
                                                          "every metric row carries its provenance (metrics_provenance.csv)"]),
        dashboard=dict(url=DASHBOARD_URL, models=MIU.to_dict("records"), multi=True),
        limits=[f"{df['ticker'].nunique()} of {len(all_tickers)} stocks; the full-universe run uses the same notebook with QUICK = False.",
                "Survivor-biased universe (only stocks that still exist) from one data vendor, so risk may be understated for stocks that later fail.",
                "The test years were already read by earlier runs and the 2023+ period has been opened before: a truly fresh test needs new data.",
                "The chart renderer scales each picture to fill its panel, so pictures cannot show the volatility level; numbers carry that.",
                "Research output, not investment advice. NO TRADE is the default decision; no portfolio backtest has been run."],
        figures=figs, leaderboard_caption="Every candidate on the test years (historical diagnostic; 95% CIs; ★ = the model chosen on validation).")
    PDF_PATH = os.path.join(RES, f"TriFrame_CV_report_{RUN_TAG}.pdf")
    RP.build_pdf(R, PDF_PATH)
    shutil.copy(PDF_PATH, os.path.join(DRIVE_DIR, os.path.basename(PDF_PATH)))
    push(PDF_PATH, f"results_{RUN_TAG}/{os.path.basename(PDF_PATH)}")
    print(f"\nPDF report: {PDF_PATH}\n  also on Google Drive: {DRIVE_DIR}/{os.path.basename(PDF_PATH)} (My Drive > triframe_cv)")
    if IN_COLAB:
        from google.colab import files
        files.download(PDF_PATH)                                    # lands in your computer's Downloads folder

push(FIG, f"figures_{RUN_TAG}")
retry = list(FAILED_UPLOADS); FAILED_UPLOADS.clear()
for local, pir, rid, rtype in retry:
    push(local, pir, repo_id=rid, repo_type=rtype)
print(f"\nUploads still failing: {len(FAILED_UPLOADS)}" + (" - re-run this cell" if FAILED_UPLOADS else " (none)"))
