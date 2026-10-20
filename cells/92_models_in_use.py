# ==================================== WHICH MODELS THE DASHBOARD USES (v3) ====================================
# The dashboard loads MODEL_REPO and uses exactly the frozen models below (the same code produces this table in the app).
import dashboard_logic as DL
import tfcv_report as RP
importlib.reload(DL); importlib.reload(RP)
MIU = DL.models_in_use(BUNDLE_CFG)
print(f"The dashboard loads {MODEL_REPO} (private) and uses these models:\n")
print(MIU.to_string(index=False))
for t in TARGETS:
    s = BUNDLE_CFG["targets"][t]
    print(f"\n{TARGET_NAME[t]}: {s['feature_set']} / {s['head']}"
          + (f" - chosen over numbers only because the validation gain {SELF[t]['gate']['delta']:+.4f} had a CI that excludes 0 "
             f"[{SELF[t]['gate']['ci'][0]:+.4f}, {SELF[t]['gate']['ci'][1]:+.4f}]" if s["feature_set"] != "NUM" and SELF[t].get("gate") else
             " - numbers only: no image model cleared the pre-registered gate" if s["feature_set"] == "NUM" else ""))
print(f"\nVision networks the dashboard has to run: {BD.needed_backbones() or 'none'}"
      + (" (fine-tuned weights are shipped inside the private bundle)" if any(b.startswith(T.FT_PREFIX) for b in BD.needed_backbones()) else ""))
fpath = RP.fig_pipeline(os.path.join(FIG, "architecture.png"), ft=bool(FT_NAMES))
push(fpath, f"figures_{RUN_TAG}/architecture.png")

# ---- the dashboard's wording on one real test-year example (as of that date; the numbers are the frozen bundle's)
ex_row = df[TE & np.isfinite(Y["V"])].sample(1, random_state=SEED + 5).iloc[0]
ex_emb = {b: {v: T.assemble(emb_dir(b), f"{b}|{v}", np.array([ex_row.sample_id]))[0] for v in T.VIEWS} for b in BD.needed_backbones()}
ex_out = BD.predict_row(ex_row[FEATS].to_numpy(float), ex_emb)
ex_res = dict(ok=True, ticker=ex_row.ticker, asof=pd.Timestamp(ex_row.date).date(), how="test-year example", out=ex_out, p0=100.0)
ex_sim = DL.run_mc(ex_res, BUNDLE_CFG, n_paths=5000)
print("\n" + "=" * 100 + "\nExample of the dashboard's explanation (a test-year row; price shown as an index of 100):\n")
print(DL.report_markdown(ex_res, BUNDLE_CFG, ex_sim))
print(f"\nWhat actually happened next: realised volatility {C.ann_vol(ex_row.logrv20):.0%} a year; "
      f"10% drawdown {'yes' if ex_row.dd20_le_10 == 1 else 'no'}; 20-day return {ex_row.ret20:+.1%}.")
