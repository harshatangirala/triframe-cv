# ==================================== STAGE 5F: fine-tune the pretrained backbones (v4 protocol) ====================================
# Each pretrained backbone is trained end to end on the training years. v4 changes the protocol (review brief, section 7):
#  * early stopping is decided on an INNER validation slice (FT_INNER, inside the training years); the 2017-2019 validation years
#    are only logged, so they stay clean for model selection in Stage 6 (in v3 they did both jobs, which flattered fine-tuning);
#  * up to FT_EPOCHS epochs with patience FT_PATIENCE, a fresh random draw of FT_ROWS_PER_EPOCH chart sets every epoch;
#  * a controlled study on the FT_STUDY_TOP best backbones: train only the last 25% / 50% of the network, or full fine-tuning on
#    the two risk targets only; per backbone the variant with the lowest inner-validation risk loss (volatility + drawdown) is
#    carried forward;
#  * the best setup is retrained with FT_SEEDS to show how much results move between runs.
import matplotlib; import matplotlib.pyplot as plt
t0 = time.time()
FT_NAMES, FT_REC, FT_HIST, FT_STEPS, FT_E2E, FT_HEADS, FT_RUNS = [], {}, [], [], {}, {}, {}
FT_STUDY_TABLE = FT_SEED_TABLE = FT_CMP = None
LS_DATA = None
if FINETUNE:
    bases = list(T.PRETRAINED) if FT_BACKBONES == "all" else [b for b in FT_BACKBONES if b in T.PRETRAINED]
    rng_ft = np.random.default_rng(SEED)
    i0 = np.datetime64(pd.Timestamp(FT_INNER[0]))
    dts, e20 = pd.to_datetime(df["date"]).values, pd.to_datetime(df["end20"]).values
    learn_rows = np.flatnonzero(TR & (e20 < i0))                       # learns from training rows whose label ends before the slice
    stop_rows = np.flatnonzero(TR & (dts >= i0))                       # inner validation slice: decides early stopping
    val_rows = np.flatnonzero(VA)                                      # 2017-2019: scored and logged only
    if len(stop_rows) > FT_MAX_STOP_ROWS:
        stop_rows = np.sort(rng_ft.choice(stop_rows, FT_MAX_STOP_ROWS, replace=False))
    if len(val_rows) > FT_MAX_VAL_ROWS:
        val_rows = np.sort(rng_ft.choice(val_rows, FT_MAX_VAL_ROWS, replace=False))
    sel_ft = np.concatenate([learn_rows, stop_rows, val_rows])
    rows_ft = st._idx.set_index("sample_id").loc[df["sample_id"].values[sel_ft]].reset_index()
    tp = time.time()
    PACKED = T.packed_rows(st, rows_ft)
    g0 = rows_ft[rows_ft["ticker"] == rows_ft["ticker"].iloc[0]].iloc[:16]
    raw = T.views_from_packed(PACKED[:len(g0)], "cpu", resize=False); ref = T.FastTriplets(st).read(g0)
    assert all(np.array_equal(raw[v][:, 0].numpy().astype(np.uint8), ref[v]) for v in T.VIEWS), "packed chart decoder differs from the store"
    xn_ft, ys_ft = XN[sel_ft], (Y["V"][sel_ft], Y["O5"][sel_ft], Y["D10"][sel_ft])
    i_learn = np.arange(len(learn_rows)); i_stop = np.arange(len(learn_rows), len(learn_rows) + len(stop_rows))
    i_val = np.arange(len(learn_rows) + len(stop_rows), len(sel_ft))
    print(f"Fine-tuning data: learns from {len(learn_rows):,} chart sets (labels ending before {FT_INNER[0]}), early-stops on "
          f"{len(stop_rows):,} ({FT_INNER[0][:4]}-{int(FT_INNER[1][:4]) - 1}), logs {len(val_rows):,} validation chart sets; "
          f"{PACKED.nbytes / 1e9:.2f} GB in memory, packed in {time.time() - tp:.0f}s; decoder checked against the store.")
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

    def run_ft(nm, base, kw, seed=SEED):
        """Fine-tune `base` with settings kw as run `nm`; resumable (weights, heads and history kept on Drive and in the private runs repo)."""
        tag = nm if seed == SEED else f"{nm}.seed{seed}"
        files = [f"{tag}.safetensors", f"{tag}.heads.safetensors", f"{tag}.json"]
        got = [ft_fetch(f) for f in files]
        if all(got):
            print(f"{tag}: found from an earlier session, not retrained.")
            meta = json.load(open(got[2]))
        else:
            print(f"Fine-tuning {T.display_name(nm)}" + (f", seed {seed}" if seed != SEED else "") + f" (up to {FT_EPOCHS} epochs)...")
            m, mu, sd, rec, hist, steps, e2e, heads = T.finetune(base, PACKED, xn_ft, ys_ft, i_learn, i_stop, DEVICE, val_idx=i_val,
                                                                 epochs=FT_EPOCHS, patience=FT_PATIENCE, rows_per_epoch=FT_ROWS_PER_EPOCH,
                                                                 batch=FT_BATCH, lr_backbone=FT_LR_BACKBONE, lr_head=FT_LR_HEAD, seed=seed,
                                                                 factory=globals().get("FACTORY"), **kw)
            from safetensors.torch import save_file
            got = [os.path.join(WORK, "ft", f) for f in files]
            T.save_backbone(m, got[0])
            save_file({f"{k}.{n}": v.contiguous() for k, d in heads.items() for n, v in d.items()}, got[1])
            meta = dict(name=nm, base=base, seed=seed, settings={k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()},
                        record=rec, history=hist.to_dict("records"), steps=steps.to_dict("records"),
                        e2e_val={k: np.asarray(v).tolist() for k, v in (e2e or {}).items()}, weights_sha256=_sha(got[0]))
            json.dump(meta, open(got[2], "w"), default=float)
            for f_ in got:
                shutil.copy(f_, os.path.join(FT_DRIVE, os.path.basename(f_)))
                push(f_, f"ft_{RUN_TAG}/{os.path.basename(f_)}")       # private runs repo only (weights are never published)
            del m; gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        meta["weights"], meta["heads_file"] = got[0], got[1]
        meta["history"] = [dict(h, run=tag) for h in meta["history"]]
        FT_RUNS[tag] = meta
        return meta

    # ---- 1) every pretrained backbone, full fine-tuning on all three targets
    for b in bases:
        run_ft(T.ft_name(b), b, {})
    risk_of = lambda tag: FT_RUNS[tag]["record"]["ft"]["best_stop_risk_loss"]
    ranked = sorted(bases, key=lambda b: risk_of(T.ft_name(b)))
    print("\nInner-validation risk loss (volatility + drawdown, lower is better):", ", ".join(f"{b} {risk_of(T.ft_name(b)):.4f}" for b in ranked))
    # ---- 2) controlled study on the best backbones
    carried = {b: T.ft_name(b) for b in bases}
    if FT_STUDY:
        st_rows = []
        for b in ranked[:FT_STUDY_TOP]:
            opts = [T.ft_name(b)]
            for v, kw in FT_STUDY_VARIANTS.items():
                opts.append(run_ft(T.ft_name(b, v), b, kw)["name"])
            for nm in opts:
                r_ = FT_RUNS[nm]["record"]["ft"]
                st_rows.append(dict(backbone=b, run=T.display_name(nm), name=nm, inner_risk_loss=r_["best_stop_risk_loss"],
                                    best_epoch=r_["best_epoch"], epochs=r_["epochs_run"], trainable_M=r_["n_trainable_backbone"] / 1e6,
                                    minutes=r_["minutes"], val_R2_V_network=[h["val_R2_V"] for h in FT_RUNS[nm]["history"] if h["epoch"] == r_["best_epoch"]][0]))
            carried[b] = min(opts, key=risk_of)
        FT_STUDY_TABLE = pd.DataFrame(st_rows)
        save_df(FT_STUDY_TABLE, "stage5F_study.csv")
        print("\nControlled fine-tuning study (chosen per backbone by the INNER validation risk loss, never by 2017-2019):")
        print(FT_STUDY_TABLE.drop(columns=["name"]).round(4).to_string(index=False))
    FT_NAMES = [T.ft_name(b) for b in bases] + [nm for nm in carried.values() if T.ft_variant(nm)]
    for nm in FT_NAMES:
        T.FT_WEIGHTS[nm] = FT_RUNS[nm]["weights"]
    # ---- 3) seeds: how much does the best setup move when only the random seed changes?
    BEST_FT = min(FT_RUNS, key=risk_of)
    seed_rows = []
    for sd_ in [SEED] + list(FT_SEEDS):
        meta = FT_RUNS[BEST_FT] if sd_ == SEED else run_ft(BEST_FT, FT_RUNS[BEST_FT]["base"], {k: tuple(v) if isinstance(v, list) else v
                                                                                                for k, v in FT_RUNS[BEST_FT]["settings"].items()}, seed=sd_)
        r_ = meta["record"]["ft"]; hb = [h for h in meta["history"] if h["epoch"] == r_["best_epoch"]][0]
        seed_rows.append(dict(seed=sd_, best_epoch=r_["best_epoch"], inner_risk_loss=r_["best_stop_risk_loss"], val_R2_V=hb["val_R2_V"],
                              val_AUC_D10=hb["val_AUC_D10"], train_loss=hb["train_loss"]))
    FT_SEED_TABLE = pd.DataFrame(seed_rows)
    save_df(FT_SEED_TABLE, "stage5F_seeds.csv")
    print(f"\nSeeds for the best setup ({T.display_name(BEST_FT)}): the network's own validation volatility R2 "
          f"{FT_SEED_TABLE.val_R2_V.mean():.3f} +- {FT_SEED_TABLE.val_R2_V.std(ddof=0):.3f} over {len(FT_SEED_TABLE)} seeds; "
          f"best epoch {sorted(FT_SEED_TABLE.best_epoch.tolist())}.")
    for nm, meta in FT_RUNS.items():
        FT_HIST += meta["history"]; FT_STEPS += [dict(s, run=nm) for s in meta["steps"]]
    # rows kept for the loss-landscape plots in Stage 6D (small fixed subsets), then the big array is freed
    ls_learn = np.sort(rng_ft.choice(i_learn, min(LANDSCAPE_ROWS, len(i_learn)), replace=False))
    ls_val = np.sort(rng_ft.choice(i_val, min(LANDSCAPE_ROWS, len(i_val)), replace=False))
    keep_i = np.concatenate([ls_learn, ls_val])
    LS_DATA = dict(packed=PACKED[keep_i], xnum=xn_ft[keep_i], ys=tuple(np.asarray(y_)[keep_i] for y_ in ys_ft),
                   sets={"training rows": np.arange(len(ls_learn)), "validation rows": np.arange(len(ls_learn), len(keep_i))})
    del PACKED; gc.collect()

    # ---- 4) embeddings of every chart from the carried fine-tuned backbones (own folder per set of weights: stale chunks never reused)
    wsig = hashlib.sha256("".join(_sha(T.FT_WEIGHTS[n]) for n in FT_NAMES).encode()).hexdigest()[:10]
    EMB_FT_DIR = os.path.join(WORK, f"emb_ft_{RUN_TAG}_{wsig}"); os.makedirs(EMB_FT_DIR, exist_ok=True)
    have_ft, keep_ft = chunk_store(EMB_FT_DIR, f"emb_ft_{RUN_TAG}_{wsig}")
    rec_ft = T.extract_embeddings(st, rows_e, FT_NAMES, EMB_FT_DIR, DEVICE, chunk_tickers=CHUNK_TICKERS, batch=BATCH,
                                  factory=globals().get("FACTORY"), on_chunk=keep_ft, have_chunk=have_ft)
    for nm in FT_NAMES:
        rec_ft[nm]["ft"] = FT_RUNS[nm]["record"].get("ft", {})
    records.update(rec_ft)
    json.dump(records, open(bpath, "w"), indent=1)
    push(bpath, f"results_{RUN_TAG}/backbones.json")
    for nm in FT_NAMES:                                       # new weights => the PCA must be refitted
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

    # ---- 5) frozen vs fine-tuned on validation (paired month-block CIs)
    BOOT_FT = {t: T.MonthBoot(DATES[VA & OK[t]], min(N_BOOT, 500), seed=SEED) for t in TARGETS}

    def best_head(t, fs):
        tab = VAL_TABLES[t]; met, sign = PRIMARY[t]
        sub = tab[tab.feature_set == fs]
        return sub.loc[(sign * sub[met]).idxmax(), "head"]

    rows_cmp = []
    for nm in FT_NAMES:
        b = T.base_name(nm); r = dict(model=T.display_name(nm), name=nm, backbone=b)
        for t, fn in (("V", T.r2), ("D10", T.auc), ("O5", T.auc)):
            y = Y[t][VA & OK[t]]
            for cond, fs in (("img", "IMG:{}"), ("num_img", "NUM+IMG:{}")):
                pf, pz = VAL_PRED[(t, fs.format(b), best_head(t, fs.format(b)))], VAL_PRED[(t, fs.format(nm), best_head(t, fs.format(nm)))]
                r[f"{t}_{cond}_frozen"], r[f"{t}_{cond}_finetuned"] = fn(y, pf), fn(y, pz)
                if cond == "num_img":
                    ci, _ = BOOT_FT[t].diff(fn, y, pz, pf)
                    r[f"{t}_gain"], r[f"{t}_gain_lo"], r[f"{t}_gain_hi"] = fn(y, pz) - fn(y, pf), ci[0], ci[1]
            e2e = np.asarray(FT_RUNS[nm]["e2e_val"].get(t, []))
            yv_ = Y[t][sel_ft[i_val]]; ok_ = np.isfinite(yv_)
            r[f"{t}_end_to_end"] = fn(yv_[ok_], e2e[ok_]) if len(e2e) and ok_.sum() > 10 else np.nan
        fr = FT_RUNS[nm]["record"]["ft"]
        r.update(epochs=fr["epochs_run"], best_epoch=fr["best_epoch"], minutes=fr["minutes"], inner_risk_loss=fr["best_stop_risk_loss"])
        rows_cmp.append(r)
    FT_CMP = pd.DataFrame(rows_cmp)
    save_df(FT_CMP, "stage5F_finetune_vs_frozen.csv")
    save_df(pd.DataFrame(FT_HIST), "stage5F_training_history.csv"); save_df(pd.DataFrame(FT_STEPS), "stage5F_training_steps.csv")
    n_up, n_sig, n_dn = int((FT_CMP.V_gain > 0).sum()), int((FT_CMP.V_gain_lo > 0).sum()), int((FT_CMP.V_gain_hi < 0).sum())
    FT_VERDICT = (f"With early stopping on {FT_INNER[0][:4]}-{int(FT_INNER[1][:4]) - 1} (so the 2017-2019 validation years were never used "
                  f"by fine-tuning), fine-tuning raised validation volatility R² (numbers + images) for {n_up} of {len(FT_CMP)} runs "
                  f"(significantly for {n_sig}, significantly worse for {n_dn}; mean change {FT_CMP.V_gain.mean():+.4f}) and drawdown AUC "
                  f"for {int((FT_CMP.D10_gain > 0).sum())} of {len(FT_CMP)} (significantly for {int((FT_CMP.D10_gain_lo > 0).sum())}). "
                  f"Total fine-tuning time {sum(m_['record']['ft']['minutes'] for m_ in FT_RUNS.values()):.0f} GPU-minutes "
                  f"({len(FT_RUNS)} runs, including the study and the extra seeds).")
    print("\nFrozen vs fine-tuned on validation (numbers + images, best head; gain = fine-tuned minus frozen, 95% month-block CI):")
    print(FT_CMP[["model", "V_num_img_frozen", "V_num_img_finetuned", "V_gain", "V_gain_lo", "V_gain_hi", "D10_num_img_finetuned",
                  "V_end_to_end", "epochs", "best_epoch", "minutes"]].round(4).to_string(index=False))
    print("\n" + FT_VERDICT)

    # ---- 6) figures: learning curves per run (training vs early-stopping slice vs validation), step losses, study and seeds
    H_ = pd.DataFrame(FT_HIST)
    base_runs = [T.ft_name(b) for b in bases]
    ncol = 4; nrow = int(np.ceil(len(base_runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), squeeze=False)
    for ax, nm in zip(axes.flat, base_runs):
        g = pd.DataFrame(FT_RUNS[nm]["history"])
        ax.plot(g.epoch, g.train_loss, "o-", label="training (fixed rows, eval mode)")
        ax.plot(g.epoch, g.stop_loss, "s-", label=f"early-stopping slice {FT_INNER[0][:4]}-{int(FT_INNER[1][:4]) - 1}")
        ax.plot(g.epoch, g.val_loss, "^--", label="validation 2017-19 (logged only)")
        be_ = FT_RUNS[nm]["record"]["ft"]["best_epoch"]
        ax.axvline(be_, color="#c0392b", ls=":", lw=1.5)
        ax.set_title(f"{T.display_name(nm)}: best epoch {be_}", fontsize=9); ax.set_xlabel("epoch", fontsize=8); ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(base_runs):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7); fig.suptitle("Fine-tuning learning curves: loss per epoch (lower is better); red line = epoch kept", fontsize=11)
    plt.tight_layout(); fpath = os.path.join(FIG, "stage5F_learning_curves.png"); plt.savefig(fpath, dpi=130); plt.show()
    push(fpath, f"figures_{RUN_TAG}/stage5F_learning_curves.png")
    S_ = pd.DataFrame(FT_STEPS)
    if len(S_):
        fig, ax = plt.subplots(1, 3, figsize=(19, 4.3))
        for nm in base_runs:
            s_ = S_[S_.run == nm]
            if len(s_):
                ax[0].plot(s_.step, s_.loss.rolling(5, min_periods=1).mean(), label=T.display_name(nm), lw=1)
                ax[1].plot(s_.step, s_.grad_norm.rolling(5, min_periods=1).mean(), lw=1)
                ax[2].plot(s_.step, s_.lr_backbone, lw=1)
        ax[0].set_title("Training loss per step (smoothed)"); ax[1].set_title("Gradient norm per step (before clipping at 1.0)")
        ax[2].set_title("Backbone learning rate (warm-up, then cosine decay)"); ax[0].legend(fontsize=7)
        for a in ax:
            a.set_xlabel("optimisation step")
        plt.tight_layout(); fpath = os.path.join(FIG, "stage5F_training_steps.png"); plt.savefig(fpath, dpi=130); plt.show()
        push(fpath, f"figures_{RUN_TAG}/stage5F_training_steps.png")
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
    x_ = np.arange(len(FT_CMP)); w_b = 0.38
    ax[0].bar(x_ - w_b / 2, FT_CMP.V_num_img_frozen, w_b, label="frozen", color="#3b6ea5")
    ax[0].bar(x_ + w_b / 2, FT_CMP.V_num_img_finetuned, w_b, label="fine-tuned", color="#e67e22",
              yerr=np.vstack([FT_CMP.V_gain - FT_CMP.V_gain_lo, FT_CMP.V_gain_hi - FT_CMP.V_gain]), capsize=3)
    ax[0].axhline(T.r2(Y["V"][VA & OK["V"]], VAL_PRED[("V", "NUM", best_head("V", "NUM"))]), color="#c0392b", ls="--", label="numbers only")
    ax[0].set_ylim(min(FT_CMP[["V_num_img_frozen", "V_num_img_finetuned"]].min()) - 0.03, None)
    ax[0].set_xticks(x_); ax[0].set_xticklabels(FT_CMP.model, rotation=35, ha="right", fontsize=7)
    ax[0].set_title("Volatility R² (numbers + images, validation)"); ax[0].legend(fontsize=8)
    if FT_STUDY_TABLE is not None:
        for i, (b, g) in enumerate(FT_STUDY_TABLE.groupby("backbone", sort=False)):
            ax[1].plot(range(len(g)), g.inner_risk_loss, "o-", label=T.display_name(b))
            ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels(["full", "last 25%", "last 50%", "risk only"][:len(g)], fontsize=8)
        ax[1].set_title("Study: inner-validation risk loss per variant (lower is better)"); ax[1].legend(fontsize=8)
    ax[2].errorbar([0], [FT_SEED_TABLE.val_R2_V.mean()], yerr=[FT_SEED_TABLE.val_R2_V.std(ddof=0)], fmt="o", capsize=8, color="#e67e22")
    ax[2].scatter(np.zeros(len(FT_SEED_TABLE)) + 0.05, FT_SEED_TABLE.val_R2_V, color="#3b6ea5", label="each seed")
    ax[2].set_xticks([0]); ax[2].set_xticklabels([T.display_name(BEST_FT)], fontsize=8)
    ax[2].set_title("Seeds: the best setup's own validation R² (mean +- sd)"); ax[2].legend(fontsize=8)
    plt.tight_layout(); fpath = os.path.join(FIG, "stage5F_finetuning.png"); plt.savefig(fpath, dpi=130); plt.show()
    push(fpath, f"figures_{RUN_TAG}/stage5F_finetuning.png")
else:
    FT_VERDICT = "Fine-tuning was switched off (FINETUNE = False): all image models use frozen backbones."
    print(FT_VERDICT)
print(f"Stage 5F done in {(time.time() - t0) / 60:.1f} min.")
