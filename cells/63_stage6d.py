# ==================================== STAGE 6D: did training reach the minimum? (optimisation diagnostics, v4) ====================================
# What can honestly be shown for each kind of model:
#  * ridge and logistic heads are CONVEX: they have exactly one minimum, so reaching it can be certified (zero gradient, the same
#    answer from every random starting point, a bowl-shaped loss around it);
#  * boosted trees are built greedily: the training loss falls with every tree and the validation loss has a lowest point, which
#    early stopping keeps;
#  * fine-tuned networks are NOT convex: no method can certify a global minimum for them. What can be shown is that training
#    converged, the kept checkpoint is the minimum of the held-out loss, the trained network sits at the bottom of its basin
#    (loss-landscape plots), and different random seeds land at similar losses.
# Nothing in this stage changes a frozen choice: it describes the models chosen in Stage 6.
t0 = time.time()
OPT = []                                                     # the summary table at the end

# ---- A. convex heads: global-minimum certificates
mt_v = TR & OK["V"]
rc = T.ridge_global_check(XN[mt_v], Y["V"][mt_v], FITS[("V", "NUM", "linear")]["param"], seed=SEED)
mt_d = TR & OK["D10"]
lc = T.logistic_global_check(XN[mt_d], Y["D10"][mt_d], FITS[("D10", "NUM", "linear")]["param"], seed=SEED)
fig, ax = plt.subplots(1, 4, figsize=(21, 4.3))
for i, tr in enumerate(rc["traces"]):
    ax[0].semilogy(np.maximum(tr, 1e-16), "o-", ms=3, label=f"random start {i + 1}")
ax[0].set_title("Ridge (volatility, 33 numbers): gap to the\nclosed-form global minimum, per iteration", fontsize=9)
ax[0].set_xlabel("conjugate-gradient iteration"); ax[0].legend(fontsize=7)
for sl in rc["slices"]:
    ax[1].plot(rc["ts"], sl, lw=1.5)
ax[1].axvline(0, color="#c0392b", ls="--"); ax[1].set_title("Ridge loss along 3 random lines through the\nsolution: lowest at 0 (the solution)", fontsize=9)
ax[1].set_xlabel("distance from the solution (x its size)")
gg = np.asarray(rc["grid"]); cs_ = ax[2].contourf(gg, gg, np.log10(np.asarray(rc["surface"]) - rc["J_star"] + 1e-9).T, 20, cmap="viridis")
ax[2].plot(0, 0, "r*", ms=14); plt.colorbar(cs_, ax=ax[2], label="log10(loss - minimum)")
ax[2].set_title("Ridge loss on a random plane: a single bowl,\nthe star (solution) at the bottom", fontsize=9)
for i, tr in enumerate(lc["traces"]):
    ax[3].semilogy(np.maximum(np.asarray(tr) - min(lc["final_losses"]), 1e-12), lw=1.2, label=f"random start {i + 1}")
ax[3].set_title("Logistic (drawdown, 33 numbers): L-BFGS from 5\nrandom starts, gap to the best final loss", fontsize=9)
ax[3].set_xlabel("function evaluation"); ax[3].legend(fontsize=7)
plt.tight_layout(); fpath = os.path.join(FIG, "stage6D_convex_heads.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6D_convex_heads.png")
print(f"Ridge (volatility): gradient norm at the closed-form solution {rc['grad_norm']:.1e}; all {len(rc['traces'])} random starts "
      f"reach it (final gaps {max(t[-1] for t in rc['traces']):.0e}). Condition number {rc['cond']:.0f}.")
print(f"Logistic (drawdown): {len(lc['final_losses'])} random starts end at the same loss (spread {lc['loss_spread']:.1e}) and the same "
      f"weights (largest difference {lc['max_weight_spread']:.1e}); gradient norm {lc['grad_norm']:.1e}.")
OPT += [dict(component="Ridge head (volatility)", convex="yes", evidence=f"closed form; gradient {rc['grad_norm']:.0e}; 5 random starts converge",
             verdict="GLOBAL minimum reached (certified)"),
        dict(component="Logistic head (drawdown)", convex="yes", evidence=f"5 random starts agree to {lc['loss_spread']:.0e}; gradient {lc['grad_norm']:.0e}",
             verdict="GLOBAL minimum reached (certified)")]

# ---- B. boosted trees: training vs validation loss against the number of trees, for the chosen tree heads
tree_keys = [(t, SEL[t]["feature_set"]) for t in TARGETS if SEL[t]["head"] == "tree"] + [(t, "NUM") for t in TARGETS if SEL[t]["head"] != "tree"]
fig, ax = plt.subplots(1, len(tree_keys), figsize=(6.2 * len(tree_keys), 4.2), squeeze=False)
for a, (t, fs) in zip(ax[0], tree_keys):
    cv_ = FITS[(t, fs, "tree")].get("curve")
    if not cv_:
        a.axis("off"); continue
    a.plot(cv_["train"], label="training"); a.plot(cv_["val"], label="validation")
    a.axvline(cv_["best"] - 1, color="#c0392b", ls="--", label=f"kept: {cv_['best']} trees (validation minimum)")
    a.set_title(f"{TARGET_NAME[t]}\n{fs} / boosted trees ({cv_['metric']})", fontsize=9); a.set_xlabel("number of trees"); a.legend(fontsize=7)
    OPT.append(dict(component=f"Boosted trees ({t}, {fs})", convex="no (greedy)", evidence=f"validation {cv_['metric']} lowest at {cv_['best']} trees; training keeps falling",
                    verdict="minimum of the validation curve kept (early stopping)"))
plt.tight_layout(); fpath = os.path.join(FIG, "stage6D_boosting_curves.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6D_boosting_curves.png")

# ---- C. validation curves: training and validation score against the regularisation strength (from the ledger)
L_ = pd.DataFrame(LEDGER)
curves = [("V", "NUM", "ridge strength alpha", "R2"), ("D10", "NUM", "logistic C (inverse strength)", "-log loss")]
if SEL["V"]["feature_set"] != "NUM":
    curves.insert(1, ("V", SEL["V"]["feature_set"], "ridge strength alpha", "R2"))
fig, ax = plt.subplots(1, len(curves), figsize=(6.2 * len(curves), 4.2), squeeze=False)
for a, (t, fs, xl, yl) in zip(ax[0], curves):
    g = L_[(L_.target == t) & (L_.feature_set == fs) & (L_["head"] == "linear")].copy()
    g["param"] = g["param"].astype(float); g = g.sort_values("param")
    a.semilogx(g.param, g.train_score, "o-", label="training"); a.semilogx(g.param, g.val_score, "s-", label="validation")
    kb = FITS[(t, fs, "linear")]["param"]; a.axvline(kb, color="#c0392b", ls="--", label=f"chosen {kb:g}")
    a.set_title(f"{TARGET_NAME[t]}\n{fs}: score vs regularisation", fontsize=9); a.set_xlabel(xl); a.set_ylabel(yl); a.legend(fontsize=7)
plt.tight_layout(); fpath = os.path.join(FIG, "stage6D_validation_curves.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6D_validation_curves.png")

# ---- D. learning curves: validation score against the number of training STOCKS (does more data help?)
tick_tr = np.array(sorted(df.loc[TR, "ticker"].unique())); rng_lc = np.random.default_rng(SEED)
order_t = rng_lc.permutation(tick_tr); FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
LC_ROWS = []
for t, fs in [("V", "NUM"), ("V", SEL["V"]["feature_set"]), ("D10", "NUM")]:
    if (t, fs) in [(r_["target"], r_["feature_set"]) for r_ in LC_ROWS]:
        continue
    X = design(fs); f = FITS[(t, fs, "tree")] if (t, fs, "tree") in FITS else None
    if f is None:
        continue
    mv = VA & OK[t]
    for fr in FRACS:
        sub_t = set(order_t[:max(2, int(round(fr * len(order_t))))])
        mtr = TR & OK[t] & df["ticker"].isin(sub_t).values
        m_, _, _ = T.tree_fit(X[mtr], Y[t][mtr], None, None, TARGETS[t], f["cfg"], gpu=(DEVICE == "cuda"), seed=SEED, n_estimators=f["n_estimators"])
        p_ = T.tree_predict(m_, X[mv], TARGETS[t])
        LC_ROWS.append(dict(target=t, feature_set=fs, share_of_stocks=fr, n_stocks=len(sub_t), n_rows=int(mtr.sum()),
                            score=T.r2(Y[t][mv], p_) if t == "V" else T.auc(Y[t][mv], p_)))
LC = pd.DataFrame(LC_ROWS); save_df(LC, "stage6D_learning_curves.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for (t, fs), g in LC.groupby(["target", "feature_set"], sort=False):
    a = ax[0] if t == "V" else ax[1]
    a.plot(g.n_stocks, g.score, "o-", label=fs)
ax[0].set_title("Learning curve: validation volatility R² vs training stocks"); ax[1].set_title("Learning curve: validation drawdown AUC vs training stocks")
for a in ax:
    a.set_xlabel("number of stocks in training"); a.legend(fontsize=8)
plt.tight_layout(); fpath = os.path.join(FIG, "stage6D_learning_curves.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage6D_learning_curves.png")
print("Learning curves (validation score as the number of training stocks grows):")
print(LC.pivot_table(index=["target", "feature_set"], columns="n_stocks", values="score").round(4).to_string())

# ---- E. the fine-tuned network: convergence, the minimum of the held-out loss, the loss landscape and the seeds
LS = None
if FINETUNE and LANDSCAPE and LS_DATA is not None:
    b_v = C.Bundle.parse_fs(SEL["V"]["feature_set"])[1]
    ls_run = b_v if (b_v and b_v.startswith(T.FT_PREFIX) and b_v in FT_RUNS) else BEST_FT
    meta = FT_RUNS[ls_run]; fr_ = meta["record"]["ft"]
    from safetensors.torch import load_file
    hz = load_file(meta["heads_file"])
    heads_ = {k: {n[len(k) + 1:]: v for n, v in hz.items() if n.startswith(k + ".")} for k in ("init", "best")}
    tw_ = tuple(meta["settings"].get("task_weights", (1.0, 1.0, 1.0)))
    tl = time.time()
    LS = T.ft_landscape(meta["base"], meta["weights"], heads_, fr_["dim"], LS_DATA["packed"], LS_DATA["xnum"], LS_DATA["ys"], LS_DATA["sets"],
                        DEVICE, fr_["v_mean"], fr_["v_std"], task_weights=tw_, grid=np.linspace(-1, 1, LANDSCAPE_GRID), seed=SEED,
                        factory=globals().get("FACTORY"))
    json.dump(LS, open(os.path.join(RES, "stage6D_landscape.json"), "w"))
    push(os.path.join(RES, "stage6D_landscape.json"), f"results_{RUN_TAG}/stage6D_landscape.json")
    fig = plt.figure(figsize=(21, 4.6))
    a1 = fig.add_subplot(1, 4, 1)
    for nm, ln in LS["line"].items():
        a1.plot(LS["alphas"], ln, "o-", ms=3, label=nm)
    a1.axvline(0, color="grey", ls=":"); a1.axvline(1, color="#c0392b", ls="--")
    a1.set_title("Straight line from the starting point (0)\nto the trained network (1)", fontsize=9); a1.set_xlabel("alpha"); a1.legend(fontsize=7)
    gl = np.asarray(LS["grid"])
    for k_, (nm, sf) in enumerate(LS["surface"].items()):
        a = fig.add_subplot(1, 4, 2 + k_)
        cs_ = a.contourf(gl, gl, np.asarray(sf).T, 25, cmap="viridis"); a.plot(0, 0, "r*", ms=14); plt.colorbar(cs_, ax=a)
        a.set_title(f"Loss on a random plane through the trained\nnetwork ({nm}); star = trained network", fontsize=9)
    a3 = fig.add_subplot(1, 4, 4, projection="3d")
    GX, GY = np.meshgrid(gl, gl); sf0 = np.asarray(list(LS["surface"].values())[0]).T
    a3.plot_surface(GX, GY, sf0, cmap="viridis", alpha=0.9); a3.set_title("The same training-loss surface in 3-D", fontsize=9)
    plt.tight_layout(); fpath = os.path.join(FIG, "stage6D_loss_landscape.png"); plt.savefig(fpath, dpi=130); plt.show()
    push(fpath, f"figures_{RUN_TAG}/stage6D_loss_landscape.png")
    ci_ = len(gl) // 2; line_tr = np.asarray(list(LS["line"].values())[0]); a_min = LS["alphas"][int(np.argmin(line_tr))]
    print(f"Loss landscape of {T.display_name(ls_run)} ({time.time() - tl:.0f}s). Along the straight line from the starting point, the "
          f"training loss is lowest at alpha = {a_min:.1f} (1.0 = the trained network).")
    LS_SUMMARY = {}
    for nm, sf in LS["surface"].items():
        sf = np.asarray(sf); at_c, low = float(sf[ci_, ci_]), float(sf.min()); ij = np.unravel_index(np.argmin(sf), sf.shape)
        dist = float(np.hypot(gl[ij[0]], gl[ij[1]]))
        LS_SUMMARY[nm] = dict(at_trained=at_c, lowest_on_plane=low, gap=at_c - low, distance_of_lowest=dist)
        print(f"   {nm}: loss at the trained network {at_c:.4f}; lowest point found on the random plane {low:.4f}, "
              f"{'at the trained network' if at_c <= low + 1e-9 else f'{dist:.2f} step(s) away (gap {at_c - low:.4f})'}.")
    print("   Reading: training was stopped where the held-out loss was lowest, so the TRAINING loss can still go lower nearby (going "
          "there would fit the training years more closely and generalise worse). That is the intended kind of minimum: the minimum "
          "of the held-out loss along training, not the lowest possible training loss.")
    hb = pd.DataFrame(meta["history"])
    OPT.append(dict(component=f"Fine-tuned network ({T.display_name(ls_run)})", convex="no",
                    evidence=(f"training loss fell from {hb.train_loss.iloc[0]:.3f} to {hb.train_loss.min():.3f}; held-out loss lowest at epoch "
                              f"{fr_['best_epoch']} of {fr_['epochs_run']}; lowest point on the line at alpha {a_min:.1f}; seeds: validation R² "
                              f"sd {FT_SEED_TABLE.val_R2_V.std(ddof=0):.3f}"),
                    verdict="converged to a stable minimum of the held-out loss; a GLOBAL minimum cannot be certified for a network"))
OPT = pd.DataFrame(OPT); save_df(OPT, "stage6D_optimisation_summary.csv")
print("\nOptimisation summary (did each part of the model reach its minimum?):")
print(OPT.to_string(index=False))
print(f"Stage 6D done in {(time.time() - t0) / 60:.1f} min.")
