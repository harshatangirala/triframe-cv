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
BASE = {t: float(np.mean(Y[t][TR & OK[t]])) for t in ("O5", "D10")}          # Brier-skill reference = TRAINING base rate
DATES, XRET = df["date"].values, df["xret5_med"].to_numpy(float)

# ---- volatility baselines (all mapped to logrv20 by a linear fit on training rows only, all scored on the same validation rows)
VAL_PRED[("V", "M0", "constant")] = np.full(int((VA & OK["V"]).sum()), float(np.mean(Y["V"][TR & OK["V"]])))
for t in ("O5", "D10"):
    VAL_PRED[(t, "M0", "constant")] = np.full(int((VA & OK[t]).sum()), BASE[t])
BASEX["HAR"] = XN[:, [FEATS.index(f"x_{k}_{w}") for k in ("cc", "park", "gk") for w in (1, 5, 22)] + [FEATS.index("x_ewma_win")]]
BASEX["ROLL22"] = XN[:, [FEATS.index("x_cc_22")]]            # 22-day close-to-close volatility
BASEX["EWMA"] = XN[:, [FEATS.index("x_ewma_win")]]           # exponentially weighted volatility
GARCH_COVER = None
if GARCH_BASELINE and os.path.isdir(f"{DST}/final/1day"):
    tg = time.time(); g = np.full(len(df), np.nan)
    for tk, rows in df.groupby("ticker").groups.items():
        try:
            s_ = T.garch_logrv20(pd.read_parquet(f"{DST}/final/1day/{tk}.parquet", columns=["date", "close"]), SPLITS["train"][1])
        except Exception:
            s_ = None
        if s_ is not None:
            g[rows] = s_.reindex(pd.to_datetime(df.loc[rows, "date"]).values).values
    GARCH_COVER = float(np.isfinite(g).mean())
    ok_ = np.isfinite(g) & TR & np.isfinite(Xraw[:, FEATS.index("x_cc_22")])   # stocks without enough pre-2017 history: filled from the 22-day volatility
    b_ = np.polyfit(Xraw[ok_, FEATS.index("x_cc_22")], g[ok_], 1)
    g = np.where(np.isfinite(g), g, np.polyval(b_, Xf[:, FEATS.index("x_cc_22")]))
    BASEX["GARCH"] = ((g - g[TR].mean()) / g[TR].std()).astype(np.float32)[:, None]
    print(f"GARCH(1,1) forecasts for {GARCH_COVER:.1%} of rows (parameters estimated on pre-{SPLITS['train'][1][:4]} returns only) "
          f"in {time.time() - tg:.0f}s.")
for fs in BASEX:
    fit_one("V", fs, BASEX[fs], heads=("linear",), stage="3_baseline")
for t in TARGETS:
    fit_one(t, "NUM", XN, heads=("linear", "tree"), stage="3_numeric")
    tab = val_table(t)
    print(f"\n{TARGET_NAME[t]} - validation {SPLITS['val'][0][:4]}-{int(SPLITS['val'][1][:4]) - 1}:")
    print(tab.drop(columns=["target"]).round(4).to_string(index=False))
    save_df(tab, f"stage3_numeric_{t}.csv")
print("\nBaselines: M0 = training average; HAR = range-based volatility over 1/5/22 days; ROLL22 = 22-day volatility; EWMA = "
      "exponentially weighted volatility; GARCH = GARCH(1,1); NUM = all 33 numbers (linear and boosted trees).")
print(f"Stage 3 done in {time.time() - t0:.0f}s")
