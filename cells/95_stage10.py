# ==================================== STAGE 10 (owner only): the 2023-2026 period ====================================
# STATUS: EXPOSED. This period was opened for the pilot and for v3 (Stage 0 shows the log), so it is no longer an untouched test.
# Opening it again for this run is allowed by the guard (a new frozen config) but its results must be labelled "previously viewed
# period", not "sealed holdout", and nothing may be changed because of them. A genuinely new test needs data after 2026-10-09
# (e.g. a prospective paper-trading period with the frozen model).
# To run it anyway: compute the frozen config's SHA-256 yourself (!sha256sum {FROZEN_PATH}), set RUN_HOLDOUT = True, paste it when asked.
RUN_HOLDOUT = False
if RUN_HOLDOUT:
    from getpass import getpass
    os.environ["TRIFRAME_HOLDOUT_CONFIG"] = FROZEN_PATH
    os.environ["TRIFRAME_HOLDOUT_UNLOCK"] = getpass("SHA-256 of the frozen config (owner): ").strip()
    t0 = time.time()
    st_h = ImageStore(who=f"tfcv_holdout_{RUN_TAG}", root=DST + "/", allow_holdout=True)      # logged, single use per config
    push(os.path.join(GOV, "holdout_access_log.csv"), "governance/holdout_access_log.csv")
    idx_h = st_h.index()
    hd = st_h.join_features(idx_h).merge(st_h.derived_labels()[["sample_id", "xret5_med", "dd20_le_10"]], on="sample_id", how="left")
    hd = hd[(pd.to_datetime(hd["date"]) >= HOLDOUT_START) & hd["ticker"].isin(df["ticker"].unique())]
    hd = hd.sort_values(["ticker", "date"]).reset_index(drop=True)
    need_b = BD.needed_backbones()
    emb_h = {}
    if need_b:
        HOLD_DIR = os.path.join(WORK, f"emb_holdout_{RUN_TAG}")
        rows_h = st_h._idx.set_index("sample_id").loc[hd["sample_id"].values].reset_index()
        T.extract_embeddings(st_h, rows_h, need_b, HOLD_DIR, DEVICE, chunk_tickers=CHUNK_TICKERS, batch=BATCH, factory=globals().get("FACTORY"))
        emb_h = {b: {v: T.assemble(HOLD_DIR, f"{b}|{v}", hd["sample_id"].values) for v in T.VIEWS} for b in need_b}
    ph = BD.predict_matrix(hd[FEATS].to_numpy(float), emb_h)                 # the deployed bundle, nothing refitted
    HOLD = {}
    for t, col, key in (("V", "logrv20", "logrv20"), ("O5", "y_out5", "p_out5"), ("D10", "dd20_le_10", "p_dd10")):
        y = hd[col].to_numpy(float); ok = np.isfinite(y)
        for part, m in (("all", ok), ("to_2026-02-28", ok & (pd.to_datetime(hd["date"]) <= "2026-02-28").values),
                        ("after_2026-02-28", ok & (pd.to_datetime(hd["date"]) > "2026-02-28").values)):
            if m.sum() < 50:
                continue
            res_ = eval_target(t, y[m], ph[key][m], hd["date"].values[m], s2=FZ["volatility_residuals"]["s2"] if t == "V" else None,
                               xret=hd["xret5_med"].to_numpy(float)[m] if t == "O5" else None, base=BASE.get(t))
            bt = T.MonthBoot(hd["date"].values[m], N_BOOT, seed=SEED)
            res_["CI"] = bt.ci(T.r2 if t == "V" else T.auc, y[m], ph[key][m])[0]
            HOLD[f"{t}|{part}"] = res_
            print(f"{t} {part}: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in res_.items()))
    hpath = os.path.join(RES, "stage10_holdout_results.json")
    json.dump(dict(results=HOLD, rows=len(hd), utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), open(hpath, "w"), indent=1, default=float)
    push(hpath, f"results_{RUN_TAG}/stage10_holdout_results.json")
    BUNDLE_CFG["metrics"]["holdout"] = HOLD
    json.dump(BUNDLE_CFG, open(os.path.join(B_DIR, "config.json"), "w"), indent=1, default=float)
    push(os.path.join(B_DIR, "config.json"), "config.json", repo_id=MODEL_REPO, repo_type="model", msg="add holdout results")
    print(f"2023+ period (previously viewed, NOT an untouched holdout) evaluated in {time.time() - t0:.0f}s and logged in {GOV}.")
else:
    print("2023+ period not opened (RUN_HOLDOUT = False, the v4 default: the period is exposed, so opening it adds little evidence).")
