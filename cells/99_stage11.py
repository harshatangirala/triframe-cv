# ==================================== STAGE 11: report (computed values only) and final upload ====================================
def _row(t, d):
    keys = ("R2", "wmR2", "QLIKE") if t == "V" else ("AUC", "BSS", "ECE") + (("rankIC",) if t == "O5" else ())
    return "; ".join(f"{k} {d[k]:.4f}" for k in keys if k in d and isinstance(d[k], float) and np.isfinite(d[k]))


lines = [f"# TriFrame CV - end-to-end run report ({RUN_TAG})", "",
         f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} from the values computed in this notebook.", "",
         "## Data", f"- Universe: {len(all_tickers)} US stocks; this run: {df['ticker'].nunique()} stocks, {len(df):,} chart sets "
         f"(weekly 104 / daily 126 / hourly 140 bars each) with 33 numeric features.",
         f"- Splits: train {SPLITS['train']}, validation {SPLITS['val']}, test {SPLITS['test']} (20-day purge); holdout from {HOLDOUT_START}.",
         "", "## Models compared",
         f"- {len(NAMES)} frozen backbones ({', '.join(NAMES)}), PCA {PCA_DIM} per view, linear and boosted-tree heads; "
         f"{FZ['n_configurations_tried']} configurations tried on validation (all logged in ledger.csv).",
         "", "## Chosen model per target (pre-registered rule) and test results (2020-2022)",
         "| Target | Chosen inputs / head | Validation | Test |", "|---|---|---|---|"]
for t in TARGETS:
    lines.append(f"| {TARGET_NAME[t]} | {SELF[t]['feature_set']} / {SELF[t]['head']} | {_row(t, VAL_SUMMARY[t])} | {_row(t, TEST_SUMMARY[t])} |")
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
          "", "## Survivorship sensitivity (test metric with / without 2001-2009 rows in training)"]
lines += [f"- {t}: {v['metric']} {v['with_2001_2009']:.4f} / {v['without_2001_2009']:.4f}" for t, v in SURV.items()]
if "holdout" in BUNDLE_CFG["metrics"]:
    lines += ["", "## Holdout (opened once)"] + [f"- {k}: " + _row(k.split('|')[0], v) for k, v in BUNDLE_CFG["metrics"]["holdout"].items()]
lines += ["", "## Where everything is",
          f"- Results, embeddings, figures, ledger, logs: https://huggingface.co/datasets/{RUNS_REPO} (private)",
          f"- Model bundle: https://huggingface.co/{MODEL_REPO} (private)",
          f"- Dashboard: {DASHBOARD_URL or 'streamlit_app.py in the GitHub repo (deploy on Streamlit Cloud, docs/DEPLOY.md)'}",
          "", "Limits: 979 survivor US stocks from one vendor; live inputs from Yahoo Finance; research output, not investment advice."]
REPORT_PATH = os.path.join(RES, f"REPORT_{RUN_TAG}.md")
open(REPORT_PATH, "w").write("\n".join(lines))
shutil.copy(REPORT_PATH, os.path.join(DRIVE_DIR, os.path.basename(REPORT_PATH)))
save_df(pd.DataFrame(LEDGER), "ledger.csv")
push(REPORT_PATH, f"results_{RUN_TAG}/{os.path.basename(REPORT_PATH)}")
push(FIG, f"figures_{RUN_TAG}")
retry = list(FAILED_UPLOADS); FAILED_UPLOADS.clear()
for local, pir, rid, rtype in retry:
    push(local, pir, repo_id=rid, repo_type=rtype)
print("\n".join(lines))
print(f"\nUploads still failing: {len(FAILED_UPLOADS)}" + (" - re-run this cell" if FAILED_UPLOADS else " (none)"))
