# ==================================== STAGE 7M: Monte Carlo price simulation - calibrate on validation, check on the test years ====================================
# The dashboard simulates the next 20 sessions from the frozen volatility forecast. Here the simulation is tuned on validation
# data (tail shape of daily moves by return coverage, then the intraday-low range by drawdown Brier) and checked once on test.
t0 = time.time()
vspec = dict(FZ["volatility_residuals"])
sampler = C.residual_sampler(vspec)
y20 = np.log1p(df["ret20"].to_numpy(float))                 # realised 20-day log return (entry at the next open, as the drawdown label)


def mc_rows(mask_rows, pv, pd_):
    """Rows valid for volatility, drawdown and the 20-day return; returns (forecast, log return, drawdown hit, drawdown prob)."""
    ok = np.isfinite(y20[mask_rows]) & np.isfinite(Y["D10"][mask_rows])
    return pv[ok], y20[mask_rows][ok], Y["D10"][mask_rows][ok], pd_[ok]


# validation arrays: frozen volatility model's validation predictions and the calibrated drawdown probabilities on the same rows
mv = VA & OK["V"] & OK["D10"]
pv_va = pd.Series(VAL_PRED[("V", SELF["V"]["feature_set"], SELF["V"]["head"])], index=np.flatnonzero(VA & OK["V"])).loc[np.flatnonzero(mv)].values
pd_va = pd.Series(cal("D10", VAL_PRED[("D10", SELF["D10"]["feature_set"], SELF["D10"]["head"])]), index=np.flatnonzero(VA & OK["D10"])).loc[np.flatnonzero(mv)].values
xv, lv, hv, qv = mc_rows(mv, pv_va, pd_va)
mt = TE & OK["V"] & OK["D10"]
pv_te = pd.Series(TEST_PRED["V"], index=np.flatnonzero(MTE["V"])).loc[np.flatnonzero(mt)].values
pd_te = pd.Series(TEST_PRED["D10"], index=np.flatnonzero(MTE["D10"])).loc[np.flatnonzero(mt)].values
xt, lt, ht, qt = mc_rows(mt, pv_te, pd_te)
print(f"Rows: validation {len(xv):,}, test {len(xt):,} (each with a volatility forecast, a realised 20-day return and a drawdown label).")

# 1) tail shape: coverage of the realised 20-day returns on validation
grid = []
for df_ in MC_TDF_GRID:
    U, D = T.mc_universal(MC_DRAWS, df_, sampler, seed=SEED, low_k=1.0)
    s_ = T.mc_scores(U, D, xv, lv, hv)
    grid.append(dict(t_df=df_, coverage_error=s_["coverage_error"], **{f"cov_{k}": v for k, v in s_["coverage"].items()}))
GRID_T = pd.DataFrame(grid)
TDF = float(GRID_T.sort_values("coverage_error").iloc[0]["t_df"])
# 2) intraday-low range: Brier of the simulated drawdown chance on validation
grid = []
for k_ in MC_LOWK_GRID:
    U, D = T.mc_universal(MC_DRAWS, TDF, sampler, seed=SEED, low_k=k_)
    s_ = T.mc_scores(U, D, xv, lv, hv)
    grid.append(dict(low_k=k_, dd_brier=s_["dd_brier"], dd_mean_pred=s_["dd_mean_pred"], dd_rate=s_["dd_rate"]))
GRID_K = pd.DataFrame(grid)
LOWK = float(GRID_K.sort_values("dd_brier").iloc[0]["low_k"])
print("Tail shape (validation coverage error, lower is better):"); print(GRID_T.round(4).to_string(index=False))
print("Intraday-low range (validation drawdown Brier, lower is better):"); print(GRID_K.round(4).to_string(index=False))

# 3) the chosen simulation, with and without matching the drawdown model, on validation (choice) and test (check)
U, D = T.mc_universal(MC_DRAWS, TDF, sampler, seed=SEED, low_k=LOWK)
SC = {"val_raw": T.mc_scores(U, D, xv, lv, hv), "val_matched": T.mc_scores(U, D, xv, lv, hv, p_model=qv),
      "test_raw": T.mc_scores(U, D, xt, lt, ht), "test_matched": T.mc_scores(U, D, xt, lt, ht, p_model=qt)}
TILT_DEFAULT = bool(SC["val_matched"]["coverage_error"] <= SC["val_raw"]["coverage_error"] + 0.01)
use = "matched" if TILT_DEFAULT else "raw"
MC = dict(t_df=TDF, low_k=LOWK, res_quantiles=vspec["res_quantiles"], tilt_default=TILT_DEFAULT, horizon=20, drift=0.0,
          coverage={"val": SC[f"val_{use}"]["coverage"], "test": SC[f"test_{use}"]["coverage"]},
          dd_brier={"val": SC[f"val_{use}"]["dd_brier"], "test": SC[f"test_{use}"]["dd_brier"]},
          scores={k: {kk: vv for kk, vv in v.items()} for k, v in SC.items()},
          grid_tdf=GRID_T.to_dict("records"), grid_lowk=GRID_K.to_dict("records"))
tab_mc = pd.DataFrame([{"set": k, **{f"{lvl} band holds": v["coverage"][lvl] for lvl in v["coverage"]},
                        "coverage error": v["coverage_error"], "mean PIT": v["mean_pit"], "drawdown Brier": v["dd_brier"],
                        "drawdown rate": v["dd_rate"], "simulated drawdown chance": v["dd_mean_pred"]} for k, v in SC.items()])
print(f"\nChosen: Student-t with {TDF:g} degrees of freedom, intraday-low range {LOWK:g}; match the drawdown model by default: {TILT_DEFAULT}.")
print("How often the real 20-day outcome fell inside the simulation's central bands (a 90% band should hold about 90%):")
print(tab_mc.round(4).to_string(index=False))
print("(mean PIT above 0.5 = real returns tended to come in above the simulation's middle, i.e. the stocks drifted up; the "
      "simulation deliberately has no drift because the direction model has no edge)")
save_df(tab_mc, "stage7M_monte_carlo_check.csv")

# ---- coverage where it matters: by predicted-risk decile and by year, with the width of the 90% band (wide bands cover trivially)
q05_u, q95_u = np.quantile(U, [0.05, 0.95])
yrs_t = pd.to_datetime(pd.Series(DATES[mt])).dt.year.values[np.isfinite(y20[mt]) & np.isfinite(Y["D10"][mt])]
MC_GROUPS = []
for grp_name, codes in (("predicted-volatility decile", pd.qcut(xt, 10, labels=False, duplicates="drop")), ("year", yrs_t)):
    for g_ in np.unique(codes):
        k_ = codes == g_
        s_ = T.mc_scores(U, D, xt[k_], lt[k_], ht[k_])
        width = float(np.median(np.expm1(np.exp(xt[k_]) * q95_u) - np.expm1(np.exp(xt[k_]) * q05_u)))
        MC_GROUPS.append(dict(group=grp_name, value=int(g_), n=int(k_.sum()), cover_90=s_["coverage"]["90%"], cover_50=s_["coverage"]["50%"],
                              median_width_90=width, dd_rate=s_["dd_rate"], dd_sim=s_["dd_mean_pred"]))
MC_GROUPS = pd.DataFrame(MC_GROUPS); save_df(MC_GROUPS, "stage7M_coverage_by_group.csv")
print("\nTest-year coverage of the 90% band by predicted-volatility decile and by year (raw simulation), with the band's median width:")
print(MC_GROUPS.round(3).to_string(index=False))
mpath_mc = os.path.join(RES, "stage7M_monte_carlo.json"); json.dump(MC, open(mpath_mc, "w"), indent=1, default=float)
push(mpath_mc, f"results_{RUN_TAG}/stage7M_monte_carlo.json")

# 4) figures: coverage (nominal vs real), PIT histograms, and an example simulation
fig, ax = plt.subplots(1, 3, figsize=(19, 4.6))
lv_ = [0.5, 0.8, 0.9, 0.95]; xx = np.arange(len(lv_))
for i, (k, c) in enumerate((("val_" + use, "#3b6ea5"), ("test_" + use, "#e67e22"))):
    ax[0].bar(xx + (i - 0.5) * 0.36, [SC[k]["coverage"][f"{int(l * 100)}%"] for l in lv_], 0.36, color=c, label=k.replace("_", " "))
ax[0].plot(xx, lv_, "k_", markersize=30, mew=2, label="should be")
ax[0].set_xticks(xx); ax[0].set_xticklabels([f"{int(l * 100)}% band" for l in lv_]); ax[0].set_ylim(0.4, 1.0)
ax[0].set_title("Share of real 20-day outcomes inside the simulated bands"); ax[0].legend(fontsize=8)
for i, (k, c) in enumerate((("val_" + use, "#3b6ea5"), ("test_" + use, "#e67e22"))):
    ax[1].bar(np.arange(10) + (i - 0.5) * 0.4, SC[k]["pit_hist"], 0.4, color=c, label=k.replace("_", " "))
ax[1].axhline(0.1, color="k", ls="--", lw=1); ax[1].set_xticks(range(10)); ax[1].set_xticklabels([f"{i / 10:.1f}" for i in range(10)], fontsize=8)
ax[1].set_title("Where real outcomes fell in the simulated distribution\n(flat at 10% = well calibrated)", fontsize=10); ax[1].legend(fontsize=8)
ex_i = int(np.argsort(xt)[len(xt) // 2])
sim_ex = C.simulate_paths(100.0, xt[ex_i], vspec, mc=MC, n_paths=4000, p_dd_target=qt[ex_i] if TILT_DEFAULT else None, seed=SEED)
for pth in sim_ex["sample_paths"][:40]:
    ax[2].plot(sim_ex["days"], pth, color="#3b6ea5", alpha=0.15, lw=0.8)
ax[2].fill_between(sim_ex["days"], sim_ex["fan"][0], sim_ex["fan"][4], color="#3b6ea5", alpha=0.15, label="90% of paths")
ax[2].fill_between(sim_ex["days"], sim_ex["fan"][1], sim_ex["fan"][3], color="#3b6ea5", alpha=0.3, label="50% of paths")
ax[2].plot(sim_ex["days"], sim_ex["fan"][2], color="#1f4e79", lw=2, label="median")
ax[2].axhline(90, color="#c0392b", ls="--", lw=1, label="10% drawdown level")
ax[2].set_title(f"Example: a typical test-year stock (volatility {C.ann_vol(xt[ex_i]):.0%} a year), start = 100", fontsize=10)
ax[2].set_xlabel("trading days ahead"); ax[2].legend(fontsize=8)
plt.tight_layout(); fpath = os.path.join(FIG, "stage7M_monte_carlo.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage7M_monte_carlo.png")
print(f"Stage 7M done in {time.time() - t0:.0f}s")
