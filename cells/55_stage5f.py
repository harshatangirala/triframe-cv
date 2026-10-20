# ==================================== STAGE 5F: fine-tune the pretrained backbones end to end (v3) ====================================
# Until v2 the backbones were FROZEN: ImageNet features were extracted once and only the small heads were trained. Here each
# pretrained backbone is trained further on the charts themselves (training years only; early stopping on validation; no
# augmentation). Its fine-tuned embeddings then go through the same PCA and heads and compete under the same selection rule.
import matplotlib; import matplotlib.pyplot as plt
t0 = time.time()
FT_NAMES, FT_REC, FT_HIST, FT_E2E = [], {}, [], {}
if FINETUNE:
    bases = list(T.PRETRAINED) if FT_BACKBONES == "all" else [b for b in FT_BACKBONES if b in T.PRETRAINED]
    FT_NAMES = [T.FT_PREFIX + b for b in bases]
    rng_ft = np.random.default_rng(SEED)
    tr_rows, va_rows = np.flatnonzero(TR), np.flatnonzero(VA)
    if len(tr_rows) > FT_MAX_TRAIN_ROWS:
        tr_rows = np.sort(rng_ft.choice(tr_rows, FT_MAX_TRAIN_ROWS, replace=False))
    if len(va_rows) > FT_MAX_VAL_ROWS:
        va_rows = np.sort(rng_ft.choice(va_rows, FT_MAX_VAL_ROWS, replace=False))
    sel_ft = np.concatenate([tr_rows, va_rows])
    rows_ft = st._idx.set_index("sample_id").loc[df["sample_id"].values[sel_ft]].reset_index()
    tp = time.time()
    PACKED = T.packed_rows(st, rows_ft)
    g0 = rows_ft[rows_ft["ticker"] == rows_ft["ticker"].iloc[0]].iloc[:16]
    raw = T.views_from_packed(PACKED[:len(g0)], "cpu", resize=False); ref = T.FastTriplets(st).read(g0)
    assert all(np.array_equal(raw[v][:, 0].numpy().astype(np.uint8), ref[v]) for v in T.VIEWS), "packed chart decoder differs from the store"
    print(f"Fine-tuning data: {len(tr_rows):,} training and {len(va_rows):,} validation chart sets packed in memory "
          f"({PACKED.nbytes / 1e9:.2f} GB, {time.time() - tp:.0f}s); decoder checked against the store.")
    xn_ft, ys_ft = XN[sel_ft], (Y["V"][sel_ft], Y["O5"][sel_ft], Y["D10"][sel_ft])
    i_tr, i_va = np.arange(len(tr_rows)), np.arange(len(tr_rows), len(sel_ft))
    FT_BATCH = 64 if GPU_GB > 30 else 32
    FT_DRIVE = os.path.join(DRIVE_DIR, f"ft_{RUN_TAG}")
    os.makedirs(FT_DRIVE, exist_ok=True)

    def ft_fetch(fname):
        local = os.path.join(WORK, "ft", fname)
        if os.path.exists(local):
            return local
        if os.path.exists(os.path.join(FT_DRIVE, fname)):
            shutil.copy(os.path.join(FT_DRIVE, fname), local); return local
        if UPLOAD:
            try:
                got = hf_hub_download(RUNS_REPO, f"ft_{RUN_TAG}/{fname}", repo_type="dataset", local_dir=os.path.join(WORK, "_dl"), token=HF_TOKEN)
                shutil.copy(got, local); return local
            except Exception:
                return None
        return None

    for b in bases:
        nm = T.FT_PREFIX + b
        wfile, mfile = f"{nm}.safetensors", f"{nm}.json"
        w_, m_ = ft_fetch(wfile), ft_fetch(mfile)
        if w_ and m_:                                           # finished in an earlier session: reuse (resumable)
            meta = json.load(open(m_))
            T.FT_WEIGHTS[nm] = w_
            print(f"{nm}: fine-tuned weights found (from an earlier session), skipping training.")
        else:
            print(f"Fine-tuning {b} (up to {FT_EPOCHS} epochs, batch {FT_BATCH} chart sets = {3 * FT_BATCH} images)...")
            m, mu, sd, rec, hist, e2e = T.finetune(b, PACKED, xn_ft, ys_ft, i_tr, i_va, DEVICE, epochs=FT_EPOCHS, batch=FT_BATCH,
                                                   lr_backbone=FT_LR_BACKBONE, lr_head=FT_LR_HEAD, seed=SEED,
                                                   factory=globals().get("FACTORY"))
            w_ = os.path.join(WORK, "ft", wfile); T.save_backbone(m, w_); T.FT_WEIGHTS[nm] = w_
            meta = dict(record=rec, history=hist.to_dict("records"), e2e_val={k: np.asarray(v).tolist() for k, v in e2e.items()},
                        val_sample_ids=df["sample_id"].values[va_rows].tolist(), weights_sha256=_sha(w_))
            m_ = os.path.join(WORK, "ft", mfile); json.dump(meta, open(m_, "w"), default=float)
            for f_ in (w_, m_):
                shutil.copy(f_, os.path.join(FT_DRIVE, os.path.basename(f_)))
                push(f_, f"ft_{RUN_TAG}/{os.path.basename(f_)}")       # private runs repo only (weights are never published)
            del m; gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        FT_REC[nm] = meta["record"]; FT_HIST += meta["history"]
        FT_E2E[nm] = {k: np.asarray(v) for k, v in meta["e2e_val"].items()}
    del PACKED; gc.collect()

    # ---- embeddings of every chart from the fine-tuned backbones (own folder per set of weights, so stale chunks are never reused)
    wsig = hashlib.sha256("".join(_sha(T.FT_WEIGHTS[n]) for n in FT_NAMES).encode()).hexdigest()[:10]
    EMB_FT_DIR = os.path.join(WORK, f"emb_ft_{RUN_TAG}_{wsig}"); os.makedirs(EMB_FT_DIR, exist_ok=True)
    have_ft, keep_ft = chunk_store(EMB_FT_DIR, f"emb_ft_{RUN_TAG}_{wsig}")
    rec_ft = T.extract_embeddings(st, rows_e, FT_NAMES, EMB_FT_DIR, DEVICE, chunk_tickers=CHUNK_TICKERS, batch=BATCH,
                                  factory=globals().get("FACTORY"), on_chunk=keep_ft, have_chunk=have_ft)
    for nm in FT_NAMES:
        rec_ft[nm]["ft"] = FT_REC[nm].get("ft", {})
    records.update(rec_ft)
    json.dump(records, open(bpath, "w"), indent=1)
    push(bpath, f"results_{RUN_TAG}/backbones.json")

    # ---- the same PCA and heads as the frozen backbones
    for nm in FT_NAMES:                                      # new weights => the PCA must be refitted
        mark, cur = os.path.join(PCA_DIR, f"{nm}.weights"), _sha(T.FT_WEIGHTS[nm])
        if not os.path.exists(mark) or open(mark).read() != cur:
            for v in T.VIEWS:
                for suf in (".npy", "_meta.npz"):
                    if os.path.exists(os.path.join(PCA_DIR, f"{nm}__{v}{suf}")):
                        os.remove(os.path.join(PCA_DIR, f"{nm}__{v}{suf}"))
            open(mark, "w").write(cur)
    pca_backbones(FT_NAMES)
    _IMG_CACHE.clear()
    fit_backbones(FT_NAMES, stage="5F")
    refresh_val_tables()

    # ---- did fine-tuning help? frozen vs fine-tuned on validation (month-block bootstrap CIs on the same resamples)
    BOOT_FT = {t: T.MonthBoot(DATES[VA & OK[t]], min(N_BOOT, 500), seed=SEED) for t in TARGETS}

    def best_head(t, fs):
        tab = VAL_TABLES[t]; met, sign = PRIMARY[t]
        sub = tab[tab.feature_set == fs]
        return sub.loc[(sign * sub[met]).idxmax(), "head"]

    rows_ft_cmp = []
    for b in bases:
        nm = T.FT_PREFIX + b
        r = dict(backbone=b, model=T.display_name(b))
        for t, fn, met in (("V", T.r2, "R2"), ("D10", T.auc, "AUC"), ("O5", T.auc, "AUC")):
            y = Y[t][VA & OK[t]]
            for cond, fs in (("img", "IMG:{}"), ("num_img", "NUM+IMG:{}")):
                hf, hz = best_head(t, fs.format(b)), best_head(t, fs.format(nm))
                pf, pz = VAL_PRED[(t, fs.format(b), hf)], VAL_PRED[(t, fs.format(nm), hz)]
                r[f"{t}_{cond}_frozen"], r[f"{t}_{cond}_finetuned"] = fn(y, pf), fn(y, pz)
                if cond == "num_img":
                    ci, _ = BOOT_FT[t].diff(fn, y, pz, pf)
                    r[f"{t}_gain"], r[f"{t}_gain_lo"], r[f"{t}_gain_hi"] = fn(y, pz) - fn(y, pf), ci[0], ci[1]
            yv_ = Y[t][va_rows]; okv_ = np.isfinite(yv_)            # the network's own predictions (validation rows it was scored on)
            r[f"{t}_end_to_end"] = fn(yv_[okv_], FT_E2E[nm][t][okv_]) if okv_.sum() > 10 else np.nan
        fr = FT_REC[nm].get("ft", {})
        r.update(epochs=fr.get("epochs_run"), best_epoch=fr.get("best_epoch"), minutes=fr.get("minutes"), train_img_per_s=fr.get("train_img_per_s"))
        rows_ft_cmp.append(r)
    FT_CMP = pd.DataFrame(rows_ft_cmp)
    save_df(FT_CMP, "stage5F_finetune_vs_frozen.csv")
    save_df(pd.DataFrame(FT_HIST), "stage5F_training_history.csv")
    n_up = int((FT_CMP["V_gain"] > 0).sum()); n_sig = int((FT_CMP["V_gain_lo"] > 0).sum()); n_sig_dn = int((FT_CMP["V_gain_hi"] < 0).sum())
    d_up = int((FT_CMP["D10_gain"] > 0).sum()); d_sig = int((FT_CMP["D10_gain_lo"] > 0).sum())
    FT_VERDICT = (f"Fine-tuning raised validation volatility R² (numbers + images) for {n_up} of {len(FT_CMP)} backbones "
                  f"(significantly for {n_sig}, significantly worse for {n_sig_dn}; mean change {FT_CMP['V_gain'].mean():+.4f}) and drawdown "
                  f"AUC for {d_up} of {len(FT_CMP)} (significantly for {d_sig}). Images-only volatility R² moved from "
                  f"{FT_CMP['V_img_frozen'].mean():.3f} (frozen, average) to {FT_CMP['V_img_finetuned'].mean():.3f} (fine-tuned). "
                  f"Total fine-tuning time {FT_CMP['minutes'].sum():.0f} GPU-minutes.")
    print("\nFrozen vs fine-tuned on validation (numbers + images, best head; gain = fine-tuned minus frozen, 95% month-block CI):")
    show = FT_CMP[["model", "V_num_img_frozen", "V_num_img_finetuned", "V_gain", "V_gain_lo", "V_gain_hi", "D10_num_img_frozen",
                   "D10_num_img_finetuned", "V_img_frozen", "V_img_finetuned", "V_end_to_end", "epochs", "best_epoch", "minutes"]]
    print(show.round(4).to_string(index=False))
    print("\n" + FT_VERDICT)

    # ---- figures: training curves and frozen vs fine-tuned
    H_ = pd.DataFrame(FT_HIST)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    for i, (b, g) in enumerate(H_.groupby("backbone", sort=False)):
        ax[0].plot(g["epoch"], g["val_loss"], marker="o", label=T.display_name(b))
        ax[1].plot(g["epoch"], g["val_R2_V"], marker="o", label=T.display_name(b))
    ax[0].set_title("Validation loss per epoch (lower is better)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=7)
    ax[1].set_title("Validation volatility R² of the network itself"); ax[1].set_xlabel("epoch")
    x_ = np.arange(len(FT_CMP)); w_b = 0.38
    ax[2].bar(x_ - w_b / 2, FT_CMP["V_num_img_frozen"], w_b, label="frozen", color="#3b6ea5")
    ax[2].bar(x_ + w_b / 2, FT_CMP["V_num_img_finetuned"], w_b, label="fine-tuned", color="#e67e22",
              yerr=np.vstack([FT_CMP["V_gain"] - FT_CMP["V_gain_lo"], FT_CMP["V_gain_hi"] - FT_CMP["V_gain"]]), capsize=3)
    ax[2].axhline(T.r2(Y["V"][VA & OK["V"]], VAL_PRED[("V", "NUM", best_head("V", "NUM"))]), color="#c0392b", ls="--", label="numbers only")
    lo_ = min(FT_CMP[["V_num_img_frozen", "V_num_img_finetuned"]].min()) - 0.03
    ax[2].set_ylim(lo_, None); ax[2].set_xticks(x_); ax[2].set_xticklabels(FT_CMP["model"], rotation=35, ha="right", fontsize=8)
    ax[2].set_title("Volatility R² (numbers + images, validation)"); ax[2].legend(fontsize=8)
    plt.tight_layout(); fpath = os.path.join(FIG, "stage5F_finetuning.png"); plt.savefig(fpath, dpi=130); plt.show()
    push(fpath, f"figures_{RUN_TAG}/stage5F_finetuning.png")
else:
    FT_CMP, FT_VERDICT = None, "Fine-tuning was switched off (FINETUNE = False): all image models use frozen backbones."
    print(FT_VERDICT)
print(f"Stage 5F done in {(time.time() - t0) / 60:.1f} min.")
