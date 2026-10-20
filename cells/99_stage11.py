# ==================================== STAGE 11: report (markdown + 5-page PDF, computed values only) and final upload ====================================
def _row(t, d):
    keys = ("R2", "wmR2", "QLIKE") if t == "V" else ("AUC", "BSS", "ECE") + (("rankIC",) if t == "O5" else ())
    return "; ".join(f"{k} {d[k]:.4f}" for k in keys if k in d and isinstance(d[k], float) and np.isfinite(d[k]))


lines = [f"# TriFrame CV - end-to-end run report ({RUN_TAG})", "",
         f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} from the values computed in this notebook.", "",
         "## Data", f"- Universe: {len(all_tickers)} US stocks; this run: {df['ticker'].nunique()} stocks"
         + (f" (including the pilot's {len(PILOT_SET)})" if PILOT_SET else "") + f", {len(df):,} chart sets "
         f"(weekly 104 / daily 126 / hourly 140 bars each) with 33 numeric features.",
         f"- Splits: train {SPLITS['train']}, validation {SPLITS['val']}, test {SPLITS['test']} (20-day purge); holdout from {HOLDOUT_START}.",
         "", "## Models compared",
         f"- {len(NAMES)} frozen backbones ({', '.join(NAMES)})" + (f" and {len(FT_NAMES)} fine-tuned ones ({', '.join(FT_NAMES)})" if FT_NAMES else "")
         + f", PCA {PCA_DIM} per view, linear and boosted-tree heads; {FZ['n_configurations_tried']} configurations tried on validation (all logged in ledger.csv).",
         f"- Fine-tuning: {FT_VERDICT}",
         "", "## Chosen model per target (pre-registered rule) and test results (2020-2022)",
         "| Target | Chosen inputs / head | Validation | Test |", "|---|---|---|---|"]
for t in TARGETS:
    lines.append(f"| {TARGET_NAME[t]} | {SELF[t]['feature_set']} / {SELF[t]['head']} | {_row(t, VAL_SUMMARY[t])} | {_row(t, TEST_SUMMARY[t])} |")
lines += ["", "## Test leaderboard (descriptive; numbers + images, best head per model)"]
for t in TARGETS:
    sub = BOARD_TEST[(BOARD_TEST.target == t) & (BOARD_TEST.condition != "images only")].sort_values("primary_value", ascending=False)
    lines.append(f"- {TARGET_NAME[t]} ({sub['primary'].iloc[0]}): " + "; ".join(
        f"{r.model}{' (chosen)' if r.chosen else ''} {r.primary_value:.3f}" for r in sub.head(6).itertuples()))
lines += ["", "## Hypotheses (rules fixed before the test years were opened)",
          f"- H1, chart images add to the numbers (test): **{'supported' if H1_supported else 'not supported'}**. "
          + "; ".join(f"{t}: delta {v['metric']} {v['delta']:+.4f} [{v['CI'][0]:+.4f}, {v['CI'][1]:+.4f}]" for t, v in H1.items()),
          f"- H2, backbones differ (validation): **{'supported' if FZ['H2']['supported'] else 'not supported'}** "
          f"({len(FZ['H2']['significant_pairs'])} of {FZ['H2']['n_pairs']} pairs after Holm).",
          f"- H3, abstention improves accepted direction calls (test): **{'supported' if H3['supported'] else 'not supported'}** "
          f"(gain at 10% coverage {H3['gain_10pct_vs_all']:+.4f} [{H3['CI'][0]:+.4f}, {H3['CI'][1]:+.4f}]).",
          "", "## Controls (validation)",
          "- " + "; ".join(f"{k}: " + ", ".join(f"{a} {b:.4f}" for a, b in v.items() if isinstance(b, float)) if isinstance(v, dict)
                          else f"{k}: {v:.4f}" for k, v in FZ["controls"].items()),
          "- Random-weight backbones (3 seeds), image-only volatility R2: "
          + ", ".join(f"{b} {FZ['H2']['image_only_R2'][b]['R2']:.4f}" for b in NAMES if b in T.RANDOM),
          "", "## NO TRADE rule on the test years",
          f"- Flags {NT['share_flagged']:.1%} of cases; realised volatility {NT['realised_vol_flagged']:.1%} flagged vs {NT['realised_vol_other']:.1%} other; "
          f"10% drawdowns {NT['dd10_rate_flagged']:.1%} vs {NT['dd10_rate_other']:.1%}. Direction shown as a call on the dashboard: "
          f"{RULE_DEPLOY['direction_usable']} (validation check {FZ['rule']['direction_usable']}, test AUC CI lower bound "
          f"{TEST_SUMMARY['O5']['AUC_CI'][0]:.3f}).",
          "", "## Monte Carlo simulation (Stage 7M)",
          f"- Student-t {MC['t_df']:g} degrees of freedom, intraday-low range {MC['low_k']:g}, drawdown matching by default: {MC['tilt_default']}. "
          f"90% band coverage: validation {MC['coverage']['val']['90%']:.1%}, test {MC['coverage']['test']['90%']:.1%}; "
          f"50% band: validation {MC['coverage']['val']['50%']:.1%}, test {MC['coverage']['test']['50%']:.1%}.",
          "", "## Survivorship sensitivity (test metric with / without 2001-2009 rows in training)"]
lines += [f"- {t}: {v['metric']} {v['with_2001_2009']:.4f} / {v['without_2001_2009']:.4f}" for t, v in SURV.items()]
if VS_PILOT is not None:
    lines += ["", "## More data: this run vs the 60-stock pilot (same 60 stocks, same test rows)"]
    lines += [f"- {r.target} {r.metric}: pilot {r.pilot_model_60:.4f} -> this run {r.this_model_on_pilot_60:.4f}" for r in VS_PILOT.itertuples()]
if "holdout" in BUNDLE_CFG["metrics"]:
    lines += ["", "## Holdout (opened once)"] + [f"- {k}: " + _row(k.split('|')[0], v) for k, v in BUNDLE_CFG["metrics"]["holdout"].items()]
lines += ["", "## Where everything is",
          f"- Results, embeddings, figures, ledger, logs, PDF report: https://huggingface.co/datasets/{RUNS_REPO} (private)",
          f"- Model bundle: https://huggingface.co/{MODEL_REPO} (private)",
          f"- Dashboard: {DASHBOARD_URL or 'streamlit_app.py in the GitHub repo (deploy on Streamlit Cloud, docs/DEPLOY.md)'}",
          "", "Limits: survivor US stocks from one vendor; live inputs from Yahoo Finance; research output, not investment advice."]
REPORT_PATH = os.path.join(RES, f"REPORT_{RUN_TAG}.md")
open(REPORT_PATH, "w", encoding="utf-8").write("\n".join(lines))
shutil.copy(REPORT_PATH, os.path.join(DRIVE_DIR, os.path.basename(REPORT_PATH)))
save_df(pd.DataFrame(LEDGER), "ledger.csv")
push(REPORT_PATH, f"results_{RUN_TAG}/{os.path.basename(REPORT_PATH)}")
print("\n".join(lines))

# ---- the plain-language PDF report (about 5-6 pages), from the values above
if REPORT_PDF:
    yrs = lambda k: (int(SPLITS[k][0][:4]), int(SPLITS[k][1][:4]) - (1 if SPLITS[k][1][5:7] == "01" else 0) + 1)
    figs = dict(pipeline=os.path.join(FIG, "architecture.png"), leaderboard=os.path.join(FIG, "stage7_leaderboard_test.png"),
                finetune=os.path.join(FIG, "stage5F_finetuning.png") if FT_NAMES else None, mc=os.path.join(FIG, "stage7M_monte_carlo.png"),
                timeline=RP.fig_timeline(os.path.join(FIG, "timeline.png"), [("train", *yrs("train"), "#3b6ea5"), ("validation", *yrs("val"), "#e67e22"),
                                                                           ("test", *yrs("test"), "#2e9d5b"), ("holdout (sealed)", yrs("test")[1], 2027, "#7f8c8d")]),
                example=RP.fig_example(os.path.join(FIG, "example_charts.png")))
    hold = BUNDLE_CFG["metrics"].get("holdout", {})
    ts = TEST_SUMMARY
    o5_ci = ts["O5"]["AUC_CI"]
    chosen = []
    for t, lab, test, test_ci, hold_txt, plain in (
            ("V", "Volatility, next 20 days", f"R² {ts['V']['R2']:.2f}", f"R² {ts['V']['R2']:.3f} ({ts['V']['R2_CI'][0]:.2f} to {ts['V']['R2_CI'][1]:.2f})",
             f"R² {hold['V|all']['R2']:.2f}" if "V|all" in hold else "not opened",
             f"explains about {ts['V']['R2']:.0%} of the variation in volatility"),
            ("D10", "10%+ drawdown within 20 days", f"AUC {ts['D10']['AUC']:.2f}", f"AUC {ts['D10']['AUC']:.3f} ({ts['D10']['AUC_CI'][0]:.2f} to {ts['D10']['AUC_CI'][1]:.2f})",
             f"AUC {hold['D10|all']['AUC']:.2f}" if "D10|all" in hold else "not opened",
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
        ftr = records.get(T.FT_PREFIX + r["key"], {}).get("ft") if FT_NAMES else None
        cards_pdf.append(dict(model=r["model"], family=r["family"], pretrained_on=r["pretrained on"], params_m=r["parameters (M)"],
                              finetuning=(f"{ftr['epochs_run']} epochs (best {ftr['best_epoch']}), {ftr['minutes']:.0f} min" if ftr else "-")))
    img_best = max((v["R2"] for b, v in FZ["H2"]["image_only_R2"].items() if b in T.PRETRAINED), default=float("nan"))
    rnd = [v["R2"] for b, v in FZ["H2"]["image_only_R2"].items() if b in T.RANDOM]
    R = dict(
        title_run=f"{df['ticker'].nunique()}-stock run ({RUN_TAG}, notebook v3)", version="v3", n_stocks=int(df["ticker"].nunique()),
        universe=len(all_tickers), n_rows=len(df), pilot_included=bool(PILOT_SET),
        splits={"train": (f"{yrs('train')[0]}-{yrs('train')[1] - 1}", int(TR.sum()), int(df.loc[TR, 'ticker'].nunique()), "fit the models"),
                "validation": (f"{yrs('val')[0]}-{yrs('val')[1] - 1}", int(VA.sum()), int(df.loc[VA, 'ticker'].nunique()), "tune, select, calibrate, set the NO TRADE rule and the simulation"),
                "test": (f"{yrs('test')[0]}-{yrs('test')[1] - 1}", int(TE.sum()), int(df.loc[TE, 'ticker'].nunique()), "one honest check of the frozen choices"),
                "holdout": (f"{HOLDOUT_START[:4]}-2026", 0, 0, "sealed; opened once by the owner (Stage 10)" + (" - opened" if hold else " - not opened in this run"))},
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
        finetune=(dict(intro=f"<b>In this run all {len(FT_NAMES)} pretrained networks were fine-tuned</b> (up to {FT_EPOCHS} epochs on the training "
                             "years, early stopping on validation) and competed with their frozen versions.", verdict=FT_VERDICT,
                       rows=[[r.model, f"{r.V_num_img_frozen:.3f}", f"{r.V_num_img_finetuned:.3f}", f"{r.V_gain:+.3f} ({r.V_gain_lo:+.3f} to {r.V_gain_hi:+.3f})",
                              f"{r.minutes:.0f} min"] for r in FT_CMP.itertuples()]) if FT_NAMES else None),
        mc=dict(t_df=MC["t_df"], tilt_default=MC["tilt_default"], test_cov90=MC["coverage"]["test"]["90%"], val_cov90=MC["coverage"]["val"]["90%"],
                test_cov50=MC["coverage"]["test"]["50%"]),
        vs_pilot=(dict(text=f"The pilot trained on 60 stocks; this run trained on {df['ticker'].nunique()}, including those 60. Scoring both "
                            "frozen models on the pilot's 60 stocks and the same test rows isolates the effect of more training data.",
                       rows=[[TARGET_NAME[r.target].split(' (')[0], r.metric, f"{r.pilot_model_60:.3f}", f"{r.this_model_on_pilot_60:.3f}",
                              f"{r.this_model_all_stocks:.3f}" if r.this_model_all_stocks == r.this_model_all_stocks else "-"] for r in VS_PILOT.itertuples()])
                  if VS_PILOT is not None else None),
        dashboard=dict(url=DASHBOARD_URL, models=MIU.to_dict("records")),
        limits=[f"{df['ticker'].nunique()} of {len(all_tickers)} stocks; the full-universe run uses the same notebook with QUICK = False.",
                "Survivor-biased universe (only stocks that still exist) from one data vendor, so risk may be understated for stocks that later fail.",
                "The chart renderer scales each picture to fill its panel, so pictures cannot show the volatility level; numbers carry that.",
                "Live dashboard inputs come from Yahoo Finance, which can differ slightly from the training vendor.",
                "Research output, not investment advice. NO TRADE is the default decision."],
        figures=figs, leaderboard_caption="Every candidate on the test years (numbers + images, 95% CIs, ★ = the model chosen on "
                                          "validation). Descriptive only: the choice was frozen before these numbers were computed.")
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
