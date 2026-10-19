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
for a_, b_ in (("train", "val"), ("val", "test")):
    assert pd.to_datetime(df.loc[df.split == a_, "end20"]).max() < pd.to_datetime(df.loc[df.split == b_, "date"]).min(), f"{a_}/{b_} overlap"
s2_tab = df.groupby("split").agg(rows=("sample_id", "size"), tickers=("ticker", "nunique"), first=("date", "min"), last=("date", "max"),
                                 vol_mean=("logrv20", "mean"), up5_rate=("y_out5", "mean"), dd10_rate=("dd20_le_10", "mean"))
s2_tab = s2_tab.loc[["train", "val", "test"]].reset_index()
print(s2_tab.to_string(index=False))
print(f"Purged {n_purged:,} rows whose 20-day label window crossed a split boundary. Holdout (from {HOLDOUT_START}) is not loaded.")
save_df(s2_tab, "stage2_splits.csv")
spath = os.path.join(RES, f"split_ids_{RUN_TAG}.parquet")
df[["sample_id", "ticker", "date", "split"]].to_parquet(spath, index=False)
push(spath, f"results_{RUN_TAG}/split_ids.parquet")
