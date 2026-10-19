# ==================================== STAGE 3: numeric baselines (the bar every image model must clear) ====================================
t0 = time.time()
FEATS = C.DEPLOY                                            # 33 features from the last 126 daily bars (project rule)
Xraw = df[FEATS].to_numpy(np.float64)
num_fill = np.nanmedian(Xraw[TR], 0); num_fill = np.where(np.isfinite(num_fill), num_fill, 0.0)
Xf = np.where(np.isfinite(Xraw), Xraw, num_fill)
num_mean, num_std = T.standardize(Xf[TR])
XN = ((Xf - num_mean) / num_std).astype(np.float32)
Y = {"V": df["logrv20"].to_numpy(float), "O5": df["y_out5"].to_numpy(float), "D10": df["dd20_le_10"].to_numpy(float)}
OK = {t: np.isfinite(Y[t]) for t in Y}
BASE = {t: float(np.mean(Y[t][TR & OK[t]])) for t in ("O5", "D10")}
DATES, XRET = df["date"].values, df["xret5_med"].to_numpy(float)

# M0: training mean / base rate; HAR-range OLS for volatility; then linear and tree heads on the 33 numbers
VAL_PRED[("V", "M0", "constant")] = np.full(int((VA & OK["V"]).sum()), float(np.mean(Y["V"][TR & OK["V"]])))
for t in ("O5", "D10"):
    VAL_PRED[(t, "M0", "constant")] = np.full(int((VA & OK[t]).sum()), BASE[t])
HAR = [f"x_{k}_{w}" for k in ("cc", "park", "gk") for w in (1, 5, 22)] + ["x_ewma_win"]
har_ix = [FEATS.index(c) for c in HAR]
m_tr, m_va = TR & OK["V"], VA & OK["V"]
w_, b_, p_ = T.ridge_path(XN[m_tr][:, har_ix], Y["V"][m_tr], XN[m_va][:, har_ix], [1e-6])[1e-6]
VAL_PRED[("V", "HAR", "ols")] = p_
for t in TARGETS:
    fit_one(t, "NUM", XN, heads=("linear", "tree"), stage="3_numeric")
    tab = val_table(t)
    print(f"\n{TARGET_NAME[t]} - validation {SPLITS['val'][0][:4]}-{int(SPLITS['val'][1][:4]) - 1}:")
    print(tab.drop(columns=["target"]).round(4).to_string(index=False))
    save_df(tab, f"stage3_numeric_{t}.csv")
print(f"\nStage 3 done in {time.time() - t0:.0f}s")
