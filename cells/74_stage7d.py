# ==================================== STAGE 7D: the direction signal re-examined (v4, separate experiment) ====================================
# The binary target (beat the median stock over 5 days) throws away how much a stock beat it by. As a SEPARATE, pre-declared
# experiment (it does not replace the dashboard's model), this fits the 33 numbers to the continuous 5-day excess return and scores
# rankings the way a stock-picker would use them: the weekly rank IC (Spearman correlation across stocks within each week), with a
# Newey-West t-statistic for overlapping 5-day windows. Rule declared here: a ranking signal counts only if the validation IC is
# positive AND the test-years t-statistic exceeds 2 (the test years are a historical diagnostic, so even then it is "promising").
t0 = time.time()
yx = df["xret5_med"].to_numpy(float); okx = np.isfinite(yx)
mtr, mva, mte = TR & okx, VA & okx, TE & okx
best_x = None
for cfg in T.TREE_GRID:
    m_, n_, pv_ = T.tree_fit(XN[mtr], yx[mtr], XN[mva], yx[mva], "reg", cfg, gpu=(DEVICE == "cuda"), seed=SEED)
    ic_v = T.rank_ic(DATES[mva], pv_, yx[mva])
    if best_x is None or ic_v[0] > best_x[0]:
        best_x = (ic_v[0], cfg, n_, pv_)
m_x, _, _ = T.tree_fit(XN[TR & okx | VA & okx], yx[TR & okx | VA & okx], None, None, "reg", best_x[1], gpu=(DEVICE == "cuda"), seed=SEED, n_estimators=best_x[2])
pt_x = T.tree_predict(m_x, XN[mte], "reg")
DIR = []
for lab, split, msk, score in (("continuous excess-return model", "validation", mva, best_x[3]),
                               ("continuous excess-return model", "test", mte, pt_x),
                               ("chosen binary model (O5)", "validation", VA & OK["O5"] & okx, cal("O5", VAL_PRED[("O5", SELF["O5"]["feature_set"], SELF["O5"]["head"])])[okx[VA & OK["O5"]]]),
                               ("chosen binary model (O5)", "test", MTE["O5"] & okx, TEST_PRED["O5"][okx[MTE["O5"]]])):
    ic, t_nw, n_w = T.rank_ic(DATES[msk], score, yx[msk])
    yb_ = (yx[msk] > 0).astype(float)
    DIR.append(dict(model=lab, split=split, weekly_rank_IC=ic, newey_west_t=t_nw, weeks=n_w, AUC_vs_beating_median=T.auc(yb_, score),
                    top_minus_bottom_decile_5d=float(np.mean(yx[msk][score >= np.quantile(score, 0.9)]) - np.mean(yx[msk][score <= np.quantile(score, 0.1)]))))
DIR = pd.DataFrame(DIR); save_df(DIR, "stage7D_direction.csv")
cont_ok = bool(DIR.query("model == 'continuous excess-return model' and split == 'validation'").weekly_rank_IC.iloc[0] > 0 and
               DIR.query("model == 'continuous excess-return model' and split == 'test'").newey_west_t.iloc[0] > 2)
DIRECTION_VERDICT = ("PROMISING (validation IC > 0 and test-years t > 2; the test years are a historical diagnostic, so this needs a "
                     "fresh period before any claim)" if cont_ok else "NOT SUPPORTED (no reliable cross-sectional ranking signal)")
print(DIR.round(4).to_string(index=False))
print(f"\nContinuous-target ranking signal: {DIRECTION_VERDICT}. The top-minus-bottom decile column is the average 5-day excess return of "
      "the 10% highest-scored stocks minus the 10% lowest, before any trading costs (not a backtest).")
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for split, msk, score in (("validation", mva, best_x[3]), ("test", mte, pt_x)):
    d_ = pd.DataFrame({"w": pd.to_datetime(pd.Series(DATES[msk])).dt.to_period("W-FRI").values, "p": score, "x": yx[msk]})
    ic_s = d_.groupby("w").apply(lambda z: z.p.rank().corr(z.x.rank()) if len(z) >= 20 else np.nan, include_groups=False).dropna()
    ax[0].plot(ic_s.index.to_timestamp(), ic_s.rolling(13, min_periods=4).mean(), label=f"{split} (13-week average)")
    dec = pd.qcut(score, 10, labels=False, duplicates="drop")
    ax[1].plot(range(int(dec.max()) + 1), [100 * np.mean(yx[msk][dec == i]) for i in range(int(dec.max()) + 1)], "o-", label=split)
ax[0].axhline(0, color="grey", lw=0.8); ax[0].set_title("Weekly rank IC of the continuous model", fontsize=10); ax[0].legend(fontsize=8)
ax[1].axhline(0, color="grey", lw=0.8); ax[1].set_xlabel("score decile (0 = lowest)"); ax[1].set_ylabel("mean 5-day excess return (%)")
ax[1].set_title("Excess return by score decile", fontsize=10); ax[1].legend(fontsize=8)
plt.tight_layout(); fpath = os.path.join(FIG, "stage7D_direction.png"); plt.savefig(fpath, dpi=130); plt.show()
push(fpath, f"figures_{RUN_TAG}/stage7D_direction.png")
print(f"Stage 7D done in {time.time() - t0:.0f}s")
