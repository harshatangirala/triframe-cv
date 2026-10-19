# ==================================== STAGE 1: load the data (979 stocks, three timeframes) ====================================
t0 = time.time()
st = ImageStore(who=f"tfcv_e2e_{RUN_TAG}", root=DST + "/")            # sealed: rows dated 2023-01-03 onward are invisible here
idx = st.index()
df = st.join_features(idx).merge(st.derived_labels()[["sample_id", "xret5_med", "dd20_pct", "dd20_le_10"]],
                                  on="sample_id", how="left", validate="one_to_one")
all_tickers = sorted(df["ticker"].unique())
if QUICK:
    keep = sorted(np.random.default_rng(SEED).choice(all_tickers, QUICK_TICKERS, replace=False))
    df = df[df["ticker"].isin(keep)]
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

raw_span = "n/a"
if os.path.isdir(f"{DST}/final/1day"):
    _d = pd.read_parquet(f"{DST}/final/1day/{all_tickers[0]}.parquet", columns=["date"])
    raw_span = f"{_d['date'].min().date()} to {_d['date'].max().date()} ({(_d['date'].max() - _d['date'].min()).days / 365.25:.1f} years)"
yr = pd.to_datetime(df["date"]).dt.year
s1 = pd.DataFrame({"rows": df.groupby(yr).size(), "tickers": df.groupby(yr)["ticker"].nunique(),
                   "full_1W_window": df.groupby(yr)["n_valid_W"].apply(lambda s: float((s == 104).mean())),
                   "full_1D_window": df.groupby(yr)["n_valid_D"].apply(lambda s: float((s == 126).mean())),
                   "full_1H_window": df.groupby(yr)["n_valid_H"].apply(lambda s: float((s == 140).mean()))}).reset_index(names="year")
print(f"Universe: {len(all_tickers)} US stocks in the data ({df['ticker'].nunique()} used in this run).")
print(f"Raw daily/hourly bars (first ticker): {raw_span}.")
print(f"Chart sets loaded: {len(df):,} rows, decision dates {pd.to_datetime(df['date']).min().date()} to {pd.to_datetime(df['date']).max().date()}.")
print("Each row = one stock on one decision date: a 104-bar weekly chart, a 126-bar daily chart, a 140-bar hourly chart,")
print("33 numeric features from the last 126 daily bars, and the labels. Rows from 2023-01-03 on stay sealed until Stage 10.")
print(f"Label base rates: beats median over 5 days {df['y_out5'].mean():.3f} | drawdown >= 10 % in 20 days {df['dd20_le_10'].mean():.3f}")
print(s1.round(3).to_string(index=False))
save_df(s1, "stage1_rows_by_year.csv")
print(f"Stage 1 done in {time.time() - t0:.0f}s")
