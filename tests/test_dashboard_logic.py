"""End-to-end test of the dashboard logic on real TradingView screenshots with the real bundle, plus peak-memory reporting.
Usage: BUNDLE_DIR=<bundle folder> python tests/test_dashboard_logic.py <screenshots/images folder>"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import psutil
from PIL import Image
import dashboard_logic as DL

proc = psutil.Process(os.getpid())
rss = lambda: proc.memory_info().rss / 1e6
D = sys.argv[1].rstrip("/") + "/"
t0 = time.time(); B = DL.load_bundle(); print(f"bundle loaded in {time.time()-t0:.1f}s | RSS {rss():.0f} MB | needs {B.needed_backbones()}")
peak = rss(); results = []
for tk in ("AMD", "BRKR", "NVDA"):
    imgs = [Image.open(f"{D}{tk}_{tf}.png") for tf in ("1W", "1D", "1H")]
    t0 = time.time(); md, charts, tab = DL.analyse(B, *imgs, ""); dt = time.time() - t0; peak = max(peak, rss())
    head = md.splitlines()[0]; dec = md.splitlines()[1]
    print(f"{tk}: {dt:.0f}s | RSS {rss():.0f} MB | {head} | {dec}")
    assert head.startswith(f"### {tk} ") and charts is not None, md[:300]
    results.append(md)
# typed ticker only, and a mid-session style fallback (typed ticker with a screenshot of another stock)
t0 = time.time(); md, charts, tab = DL.analyse(B, None, None, None, "msft"); peak = max(peak, rss())
print(f"typed MSFT: {time.time()-t0:.0f}s | {md.splitlines()[0]}"); assert md.startswith("### MSFT")
imgs = [Image.open(f"{D}AMD_{tf}.png") for tf in ("1W", "1D", "1H")]
md, charts, tab = DL.analyse(B, *imgs, "nvda"); print("typed NVDA with AMD screenshots ->", md.splitlines()[0])
md, charts, tab = DL.analyse(B, None, None, None, "ZZZZQ"); print("bad ticker ->", md[:90].replace("\n", " "))
md, _, _ = DL.analyse(B, None, None, None, ""); print("empty ->", md)
print(f"\nPEAK RSS {peak:.0f} MB (Streamlit Cloud: 690 MB guaranteed, 2.7 GB maximum)")
print("\n--- sample output (AMD) ---\n" + results[0])
