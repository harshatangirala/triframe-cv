import sys, numpy as np, pandas as pd, time
R = "C:/Users/harsh/Downloads/us_stocks_ohlcv/"
sys.path.insert(0, R + "scripts"); sys.path.insert(0, sys.argv[1])
from triframe_loader import ImageStore
import tfcv_core as C
import pandas_market_calendars as mcal
sch = mcal.get_calendar("XNYS").schedule("2000-12-01", "2026-12-31")
early = set(pd.DatetimeIndex(sch.index[sch.market_close.dt.tz_convert("America/New_York").dt.hour < 16]).normalize())
st = ImageStore(who="claude_audit_parity", root=R)
idx = st.join_features(st.index())
rng = np.random.default_rng(3)
pick = pd.concat([idx.sample(60, random_state=3), idx[idx.patched].sample(20, random_state=3)])
t0 = time.time(); fbad = tbad = 0; maxrel = 0.0; cache = {}
for _, r in pick.iterrows():
    t = r.ticker
    if t not in cache:
        d = pd.read_parquet(f"{R}final/1day/{t}.parquet"); h = pd.read_parquet(f"{R}final/1hour/{t}.parquet")
        cache[t] = (d[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True), C.rth_only(h, early))
    d, h = cache[t]
    k = int(np.searchsorted(d.date.values.astype("datetime64[ns]"), np.datetime64(r.date, "ns")))
    assert d.date.iloc[k] == r.date
    f = C.features_at(d, k); ref = r[C.DEPLOY].values.astype(float)
    ok = np.isclose(f, ref, rtol=1e-7, atol=1e-9) | (np.isnan(f) & np.isnan(ref))
    if not ok.all(): fbad += 1; maxrel = max(maxrel, float(np.nanmax(np.abs(f - ref) / (np.abs(ref) + 1e-12))))
    trip = C.render_at(d, h, k); refimg = st.triplet(r.sample_id)
    if not all(np.array_equal(trip[v], refimg[v]) for v in C.VIEWS): tbad += 1
print(f"{len(pick)} rows ({int(pick.patched.sum())} patched) in {time.time()-t0:.0f}s | feature mismatches: {fbad} (max rel {maxrel:.2e}) | chart mismatches: {tbad}")
