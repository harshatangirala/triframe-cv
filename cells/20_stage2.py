# ==================================== STAGE 2: chronological split with a 20-day purge ====================================
d_, e_ = pd.to_datetime(df["date"]), pd.to_datetime(df["end20"])
split = np.full(len(df), "purged", dtype=object)
for name, (a, b) in SPLITS.items():
    m = ((d_ >= a) & (d_ < b) & (e_ < b)).values        # a row belongs to a split only if its 20-day label also ends inside it
    split[m] = name
df["split"] = split
n_purged = int((df["split"] == "purged").sum())
df = df[df["split"] != "purged"].reset_index(drop=True)
TR, VA, TE = (df["split"].values == s for s in ("train", "val", "test"))
for a_, b_ in (("train", "val"), ("val", "test")):          # unit test: no label window crosses a boundary (the longest label is 20 days)
    assert pd.to_datetime(df.loc[df.split == a_, "end20"]).max() < pd.to_datetime(df.loc[df.split == b_, "date"]).min(), f"{a_}/{b_} overlap"
s2_tab = df.groupby("split").agg(rows=("sample_id", "size"), tickers=("ticker", "nunique"), dates=("date", "nunique"), first=("date", "min"),
                                 last=("date", "max"), vol_mean=("logrv20", "mean"), up5_rate=("y_out5", "mean"),
                                 dd10_rate=("dd20_le_10", "mean"), dd10_events=("dd20_le_10", "sum"))
s2_tab["weeks"] = [pd.to_datetime(df.loc[df.split == s, "date"]).dt.to_period("W-FRI").nunique() for s in s2_tab.index]
s2_tab = s2_tab.loc[["train", "val", "test"]].reset_index()
print(s2_tab.to_string(index=False))
print(f"Purged {n_purged:,} rows whose 20-day label window crossed a split boundary. Holdout (from {HOLDOUT_START}) is not loaded.")
print("Rows are stock-days with overlapping 20-day label windows, so they are NOT independent: the effective sample is closer to "
      "the number of distinct weeks than to the number of rows (hence block-bootstrap confidence intervals throughout).")
ev_year = df.groupby(pd.to_datetime(df["date"]).dt.year).agg(rows=("sample_id", "size"), dd10_rate=("dd20_le_10", "mean"),
                                                             up5_rate=("y_out5", "mean"), mean_ann_vol=("logrv20", lambda s: float(np.mean(C.ann_vol(s)))))
print("\nEvent rates and volatility by year (the drawdown rate swings from year to year: compare AP and accuracy with it):")
print(ev_year.round(3).T.to_string())
save_df(s2_tab, "stage2_splits.csv"); save_df(ev_year.reset_index(names="year"), "stage2_event_rates_by_year.csv")
spath = os.path.join(RES, f"split_ids_{RUN_TAG}.parquet")
df[["sample_id", "ticker", "date", "split"]].to_parquet(spath, index=False)
push(spath, f"results_{RUN_TAG}/split_ids.parquet")
