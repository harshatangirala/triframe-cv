# ==================================== STAGE 8: the deployable model bundle + two parity checks + model card ====================================
t0 = time.time()
B_DIR = os.path.join(WORK, "bundle")
shutil.rmtree(B_DIR, ignore_errors=True); os.makedirs(B_DIR)
arrays = {"num_fill": num_fill, "num_mean": num_mean, "num_std": num_std}
tcfg, need = {}, {}
for t in TARGETS:
    s, mdl = SELF[t], FINAL[t]
    spec = dict(task=TARGETS[t], feature_set=s["feature_set"], params=s["fit"])
    if mdl["head"] == "linear":
        spec["head"] = "linear"; arrays[f"coef|{t}"] = mdl["coef"]; arrays[f"intercept|{t}"] = np.array(mdl["intercept"])
    else:
        fname = f"{t}.json" if T.tree_backend() == "xgb" else f"{t}.txt"
        spec["head"] = T.tree_save(mdl["model"], os.path.join(B_DIR, fname)); spec["file"] = fname
    if t == "V":
        spec.update(FZ["volatility_residuals"])
    else:
        spec["calibrator"] = FZ["calibrators"][t]
    tcfg[t] = spec
    use_num, b, views = C.Bundle.parse_fs(s["feature_set"])
    if b is not None:
        need.setdefault(b, set()).update(views)
        for v in views:
            mz = np.load(os.path.join(PCA_DIR, f"{b}__{v}_meta.npz"))
            arrays[f"pca_mean|{b}|{v}"], arrays[f"pca_comp|{b}|{v}"] = mz["mean"], mz["comp"]
        _, mu_, sd_ = img_matrix(b, views)
        arrays[f"img_mean|{b}|{''.join(views)}"], arrays[f"img_std|{b}|{''.join(views)}"] = mu_, sd_
recs = json.load(open(os.path.join(RES, "backbones.json")))
for b in need:                                              # a fine-tuned backbone ships its weights inside the (private) bundle
    if b.startswith(T.FT_PREFIX):
        shutil.copy(T.FT_WEIGHTS[b], os.path.join(B_DIR, recs[b]["weights_file"]))
LB_COMPACT = [dict(target=r["target"], model=r["model"], kind=r["kind"], condition=r["condition"], metric=r["primary"],
                   value=r["primary_value"], ci=[r["ci_lo"], r["ci_hi"]], chosen=bool(r["chosen"])) for r in TEST_RESULTS["leaderboard"]]
versions = {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}
for mod in ("xgboost", "lightgbm", "timm", "sklearn"):
    try:
        versions[mod] = __import__(mod).__version__
    except Exception:
        pass
BUNDLE_CFG = dict(version="tfcv-1", run_tag=RUN_TAG, created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  features=FEATS, img_clip=10.0, views=list(T.VIEWS), targets=tcfg,
                  backbones={b: recs[b] for b in need}, rule=RULE_DEPLOY, base_rates=BASE, splits=SPLITS,
                  metrics=dict(val=VAL_SUMMARY, test=TEST_SUMMARY), hypotheses=dict(H1_supported=H1_supported, H2=FZ["H2"]["supported"], H3=H3["supported"]),
                  prereg=FZ["prereg"], frozen_config_sha256=_sha(FROZEN_PATH), versions=versions,
                  data=dict(universe=len(all_tickers), tickers_used=int(df["ticker"].nunique()), rows=len(df)),
                  mc={k: MC[k] for k in ("t_df", "low_k", "res_quantiles", "tilt_default", "horizon", "drift", "coverage", "dd_brier")},
                  risk_quantiles=FZ["risk_quantiles"], reference=dict(FZ["reference"], dd10_rate_test=TEST_DD10_RATE),
                  leaderboard_test=LB_COMPACT, finetune=dict(FZ["finetune"], verdict=FT_VERDICT, backbones=FT_NAMES))
json.dump(BUNDLE_CFG, open(os.path.join(B_DIR, "config.json"), "w"), indent=1, default=float)
np.savez(os.path.join(B_DIR, "arrays.npz"), **{k: np.asarray(v) for k, v in arrays.items()})

# ---- parity 1: the saved bundle reproduces the notebook's own test predictions (same rows, raw inputs)
BD = C.Bundle(B_DIR)
common = MTE["V"] & MTE["O5"] & MTE["D10"]
sel_rows = np.flatnonzero(common)[:3000]
emb = {b: {v: T.assemble(emb_dir(b), f"{b}|{v}", df["sample_id"].values[sel_rows]) for v in T.VIEWS} for b in need}
pb = BD.predict_matrix(Xraw[sel_rows], emb)
pos = {t: np.searchsorted(np.flatnonzero(MTE[t]), sel_rows) for t in TARGETS}
dif = {"V": np.abs(pb["logrv20"] - TEST_PRED["V"][pos["V"]]), "O5": np.abs(pb["p_out5"] - TEST_PRED["O5"][pos["O5"]]),
       "D10": np.abs(pb["p_dd10"] - TEST_PRED["D10"][pos["D10"]])}
print("Bundle vs notebook test predictions:", {k: f"median {np.median(v):.1e}, max {v.max():.1e}, rows > 1e-3: {(v > 1e-3).mean():.2%}" for k, v in dif.items()})
# float32 rounding can flip a tree split on a handful of rows; anything systematic fails here
assert all(np.median(v) < 1e-4 and (v > 1e-3).mean() < 0.005 for v in dif.values()), "the saved bundle does not reproduce the notebook predictions"

# ---- parity 2: dashboard input pipeline (raw bars -> features + charts) equals the training inputs
if os.path.isdir(f"{DST}/final/1day"):
    import pandas_market_calendars as mcal
    sch = mcal.get_calendar("XNYS").schedule("2000-12-01", "2026-12-31")
    early = set(pd.DatetimeIndex(sch.index[sch.market_close.dt.tz_convert("America/New_York").dt.hour < 16]).normalize())
    pick = df[TE].sample(min(40, int(TE.sum())), random_state=SEED)
    fbad = tbad = 0
    for _, r in pick.iterrows():
        d0 = pd.read_parquet(f"{DST}/final/1day/{r.ticker}.parquet")[["date", "open", "high", "low", "close", "volume"]]
        h0 = C.rth_only(pd.read_parquet(f"{DST}/final/1hour/{r.ticker}.parquet"), early)
        k = int(np.searchsorted(d0["date"].values.astype("datetime64[ns]"), np.datetime64(r.date, "ns")))
        f_ = C.features_at(d0, k); ref = r[FEATS].to_numpy(float)
        fbad += int(not np.all(np.isclose(f_, ref, rtol=1e-7, atol=1e-9) | (np.isnan(f_) & np.isnan(ref))))
        tr_ = C.render_at(d0, h0, k); ref_img = st.triplet(r.sample_id)
        tbad += int(not all(np.array_equal(tr_[v], ref_img[v]) for v in T.VIEWS))
    print(f"Dashboard input pipeline vs training inputs on {len(pick)} test rows: feature mismatches {fbad}, chart mismatches {tbad}")
    assert fbad == 0 and tbad == 0, "the dashboard would not see what the model was trained on"

# ---- parity 3: the dashboard's own embedding code (CPU, fp32, weights loaded the way the dashboard loads them) reproduces the
#      notebook's embeddings (GPU, fp16) for the backbones the bundle needs
if need:
    pick3 = df[TE].sample(min(4, int(TE.sum())), random_state=SEED)
    worst = 1.0
    for _, r in pick3.iterrows():
        trip3 = st.triplet(r.sample_id)
        for b in need:
            e_dash = C.embed_views({v: trip3[v] for v in T.VIEWS}, BUNDLE_CFG["backbones"][b], B_DIR) if globals().get("FACTORY") is None else None
            if e_dash is None:
                continue
            for v in T.VIEWS:
                e_nb = T.assemble(emb_dir(b), f"{b}|{v}", np.array([r.sample_id]))[0]
                cos = float(np.dot(e_dash[v], e_nb) / (np.linalg.norm(e_dash[v]) * np.linalg.norm(e_nb) + 1e-12)); worst = min(worst, cos)
    print(f"Dashboard embedding code vs notebook embeddings ({len(pick3)} test rows x {len(need)} backbones x 3 views): lowest cosine similarity {worst:.5f}")
    assert worst > 0.995, "the dashboard would compute different image features from the notebook"

# ---- model card (only computed values)
def _fmt(d):
    return ", ".join(f"{k} {v:.3f}" for k, v in d.items() if isinstance(v, float))
card = [f"# TriFrame CV model bundle ({RUN_TAG} run)", "",
        "Frozen models behind the *Computer Vision with Deep Learning Project Dashboard*. Private: data and weights are not licensed for publication.", "",
        "| Target | Inputs | Head | Test metrics (2020-2022) |", "|---|---|---|---|"]
for t in TARGETS:
    card.append(f"| {TARGET_NAME[t]} | {tcfg[t]['feature_set']} | {tcfg[t]['head']} | {_fmt(TEST_SUMMARY[t])} |")
card += ["", f"H1 (images add to numbers, test): {'supported' if H1_supported else 'not supported'}; "
         f"H2 (backbones differ, validation): {'supported' if FZ['H2']['supported'] else 'not supported'}; "
         f"H3 (abstention helps, test): {'supported' if H3['supported'] else 'not supported'}.", "",
         f"Splits: {SPLITS}. Holdout from {HOLDOUT_START} (see Stage 10 results if run). Configurations tried: {FZ['n_configurations_tried']}.",
         "Limits: 979 survivor US stocks, one data vendor; not investment advice."]
open(os.path.join(B_DIR, "README.md"), "w", encoding="utf-8").write("\n".join(card))
push(B_DIR, "", repo_id=MODEL_REPO, repo_type="model", msg=f"bundle {RUN_TAG}")
print(f"Bundle written to {B_DIR}" + (f" and pushed to {MODEL_REPO} (private)" if UPLOAD else " (upload off)")
      + f". Backbones needed at inference: {sorted(need) or 'none'}")
print(f"Stage 8 done in {time.time() - t0:.0f}s")
