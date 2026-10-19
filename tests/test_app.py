import os, sys, time, numpy as np
S, W = sys.argv[1], sys.argv[2]
sys.path[:0] = [W, "C:/Users/harsh/Downloads/us_stocks_ohlcv/scripts", S + "/pylib"]
os.environ["BUNDLE_DIR"] = W + "/bundle"
import tfcv_core as C
def stub_embed(trip, spec):                       # the local bundle was trained on stand-in nets: shape-compatible stand-in
    rng = np.random.default_rng(0); return {v: rng.normal(size=32) for v in C.VIEWS}
C.embed_views = stub_embed
from PIL import Image
t0 = time.time(); import app; print(f"app imported, UI built in {time.time()-t0:.1f}s; bundle targets:", {t: s['feature_set'] for t, s in app.CFG['targets'].items()})
D = "C:/Users/harsh/Downloads/us_stocks_ohlcv/screenshots/images/"
for tk in ("AMD", "BRKR"):
    t0 = time.time()
    md, img, tab = app.analyse(*[Image.open(f"{D}{tk}_{tf}.png") for tf in ("1W", "1D", "1H")], "")
    print(f"\n===== {tk} ({time.time()-t0:.0f}s) =====\n{md}\nchart image: {None if img is None else img.size}\n{tab.to_string(index=False) if tab is not None else ''}")
t0 = time.time(); md, img, tab = app.analyse(None, None, None, "nvda")
print(f"\n===== typed ticker only ({time.time()-t0:.0f}s) =====\n{md.splitlines()[0]}\n{md.splitlines()[1]}")
