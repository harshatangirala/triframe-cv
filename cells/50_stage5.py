# ==================================== STAGE 5: PCA, then image and combined heads with hyper-parameter tuning ====================================
t0 = time.time()
PCA_DIR = os.path.join(WORK, f"pca_{RUN_TAG}")
PCA_EXPL = {}
order_ids = df["sample_id"].values
for b in NAMES:
    for v in T.VIEWS:
        f, fm = os.path.join(PCA_DIR, f"{b}__{v}.npy"), os.path.join(PCA_DIR, f"{b}__{v}_meta.npz")
        if not (os.path.exists(f) and os.path.exists(fm)):
            Xe = T.assemble(EMB_DIR, f"{b}|{v}", order_ids)
            Z, mu, comp, expl = T.pca_fit_transform(Xe, TR, min(PCA_DIM, Xe.shape[1]), seed=SEED, device=DEVICE)
            np.save(f, Z.astype(np.float32)); np.savez(fm, mean=mu, comp=comp, expl=expl)
            del Xe, Z; gc.collect()
        PCA_EXPL[f"{b}|{v}"] = float(np.load(fm)["expl"])
        push(fm, f"pca_{RUN_TAG}/{os.path.basename(fm)}")
print(f"PCA ({PCA_DIM} components per view, fitted on training rows) - variance kept, mean over views:")
print(pd.Series({b: np.mean([PCA_EXPL[f'{b}|{v}'] for v in T.VIEWS]) for b in NAMES}).round(3).to_string())

for b in NAMES:
    tb = time.time()
    for t in TARGETS:
        fit_one(t, f"IMG:{b}", design(f"IMG:{b}"), heads=("linear",), stage="5_images_only")
        heads = ("linear", "tree") if (b in T.PRETRAINED or b == "resnet18_random_s0") else ("linear",)
        fit_one(t, f"NUM+IMG:{b}", design(f"NUM+IMG:{b}"), heads=heads, stage="5_numbers_plus_images")
    if b in T.PRETRAINED:
        for v in T.VIEWS:                                   # single-timeframe conditions (H1 breakdown)
            for t in ("V", "O5"):
                fit_one(t, f"NUM+IMG:{b}:{v}", design(f"NUM+IMG:{b}:{v}"), heads=("linear",), stage="5_single_view")
    print(f"   {b}: fitted in {time.time() - tb:.0f}s")

VAL_TABLES = {}
for t in TARGETS:
    VAL_TABLES[t] = val_table(t)
    save_df(VAL_TABLES[t], f"stage5_validation_{t}.csv")
ledger_df = pd.DataFrame(LEDGER)
save_df(ledger_df, "ledger.csv")
print(f"\nStage 5 done in {(time.time() - t0) / 60:.1f} min. Configurations tried (all logged): {len(ledger_df):,}")
