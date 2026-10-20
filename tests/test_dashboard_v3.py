"""Dashboard v3 logic test on real TradingView screenshots with a real bundle: several companies, risk tiers, explanations,
Monte Carlo and every chart. Usage: BUNDLE_DIR=<bundle folder> python tests/test_dashboard_v3.py <screenshots/images folder>"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from PIL import Image
import dashboard_logic as DL
import dashboard_charts as CH

D = sys.argv[1].rstrip("/") + "/"
B = DL.load_bundle(); cfg = B.cfg
print(DL.models_in_use(cfg).to_string(index=False))
results, sims = [], []
for tk in ("AMD", "BRKR", "AOS"):
    t0 = time.time()
    res = DL.analyse_company(B, {s: Image.open(f"{D}{tk}_{s}.png") for s in DL.SLOTS}, "")
    assert res["ok"], res["message"]
    assert res["ticker"] == tk, (tk, res["ticker"])
    sim = DL.run_mc(res, cfg, n_paths=5000, match_dd=True, seed=1)
    raw = DL.run_mc(res, cfg, n_paths=5000, match_dd=False, seed=1)
    assert abs(sim["p_dd"] - res["out"]["p_dd10"]) < 1e-6, "reweighted paths must match the drawdown model"
    assert sim["fan"].shape == (5, 21) and np.all(np.diff(sim["fan"], axis=0) >= -1e-9), "fan percentiles out of order"
    assert abs(sim["fan"][2, 0] - res["p0"]) < 1e-9
    ex = DL.explain(res, cfg, sim)
    rp = ex["risk"]
    print(f"\n{tk}: {time.time() - t0:.0f}s | as of {res['asof']} ({res['how']}) | close {res['p0']:.2f} | vol {res['out']['vol_ann']:.1%} | p_dd {res['out']['p_dd10']:.1%} "
          f"(raw sim {raw['p_dd']:.1%}) | tier {rp['tier']} {rp['score']:.2f} | sim 90% {sim['ret_q05']:+.1%}..{sim['ret_q95']:+.1%} "
          f"| ESS {sim['ess']:.0f}")
    for f in (CH.vol_gauge(res["out"], cfg["rule"]), CH.dd_donut(res["out"], cfg["base_rates"]["D10"], cfg["rule"]),
              CH.direction_bar(res["out"]), CH.risk_components(rp), CH.mc_fan(sim, tk), CH.mc_hist(sim)):
        f.to_dict()
    results.append(res); sims.append(sim)
tab, txt = DL.compare(results, cfg, sims)
print("\n" + tab.to_string(index=False)); print(txt)
for f in (CH.risk_map(results, cfg["rule"]), CH.compare_bars(results, cfg["rule"], cfg["base_rates"]["D10"]),
          CH.compare_fan(results, sims), CH.compare_box(results, sims), CH.tier_donut(tab)):
    f.to_dict()
print("\n" + DL.report_markdown(results[0], cfg, sims[0]))
md, img, chk = DL.analyse(B, None, None, None, "msft")
assert md.startswith("### MSFT"), md[:200]
bad = DL.analyse_company(B, {}, "ZZZZQ"); print("\nbad ticker ->", bad["message"][:80])
print("\nDASHBOARD V3 LOGIC TEST PASSED")
