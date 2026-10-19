import sys, os, time, glob, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
from screen_reader import read_header
D = "C:/Users/harsh/Downloads/us_stocks_ohlcv/screenshots/"
man = pd.read_csv(D + "screenshot_manifest.csv")
t0 = time.time(); rows = []
for _, r in man.iterrows():
    out = read_header(D + "images/" + r.image)
    ok_t = out["ticker"] == r.ticker.replace(".", "-"); ok_f = out["timeframe"] == r.tf
    ok_c = None
    if out["ohlc"]:
        ok_c = all(abs(out["ohlc"][k] - r[k]) / r[k] < 0.002 for k in ("open", "high", "low", "close"))
    rows.append(dict(image=r.image, ticker=r.ticker, tf=r.tf, got_ticker=out["ticker"], got_tf=out["timeframe"], ok_ticker=ok_t, ok_tf=ok_f, ohlc_ok=ok_c, text=out["header_text"][:110]))
R = pd.DataFrame(rows)
print(f"{len(R)} screenshots in {time.time()-t0:.0f}s | ticker {R.ok_ticker.mean():.0%} | timeframe {R.ok_tf.mean():.0%} | legend OHLC read {R.ohlc_ok.notna().mean():.0%}, correct when read {R.ohlc_ok.dropna().mean():.0%}")
bad = R[~(R.ok_ticker & R.ok_tf)]
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(bad[["image","got_ticker","got_tf","text"]].to_string())
