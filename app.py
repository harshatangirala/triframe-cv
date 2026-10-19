"""Computer Vision with Deep Learning Project Dashboard (TriFrame) - Hugging Face Space.

Flow: three TradingView screenshots (1W, 1D, 1H) -> read ticker, timeframe and last-bar legend from each image ->
confirm the ticker by matching the legend against Yahoo Finance bars -> rebuild the exact model inputs (33 numeric
features + the three training-style charts) as of the screenshot's date -> frozen model bundle -> outputs."""
import os, traceback
import numpy as np
import pandas as pd
import gradio as gr
from huggingface_hub import snapshot_download
import tfcv_core as C
from screen_reader import read_header

MODEL_REPO = os.environ.get("MODEL_REPO", "KrishnaHarshaTangirala/triframe-cv-model")
BUNDLE_DIR = os.environ.get("BUNDLE_DIR") or snapshot_download(MODEL_REPO, token=os.environ.get("HF_TOKEN"))
B = C.Bundle(BUNDLE_DIR)
CFG = B.cfg
M = CFG.get("metrics", {})
TITLE = "Computer Vision with Deep Learning Project Dashboard"


def _fmt_pct(x, d=0):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _model_line(t):
    s = CFG["targets"][t]
    fs = "numbers only (33 features)" if s["feature_set"] == "NUM" else f"numbers + {s['feature_set'].split(':', 1)[1]} chart features"
    return f"{fs}, {'linear' if s['head'] == 'linear' else 'gradient-boosted trees'}"


def _metric(path, default=None):
    cur = M
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


DESCRIPTION = f"""
**TriFrame** asks whether chart images at three time scales (weekly, daily and hourly) add information about a US stock's
next-20-day volatility, drawdown risk and 5-day relative direction beyond plain numbers computed from the same price history.
Seven frozen ImageNet-pretrained vision backbones were compared against a numerical baseline on 979 US stocks
(2001 to 2022 for development, with 2023 onward kept as a sealed test).

**How to use:** upload three TradingView screenshots of the same stock: weekly (1W), daily (1D) and hourly (1H).
The app reads the ticker and the last bar from each picture, checks them against market data, rebuilds the exact chart
images and numbers the model was trained on, and reports the outputs below. You can type the ticker if it is not detected.

**Models in use:** volatility: {_model_line('V')}; drawdown: {_model_line('D10')}; direction: {_model_line('O5')}.
"""

LIMITS = """
*Limits: 979 survivor stocks from one data vendor (stocks that later failed are missing, so risk may be understated);
live inputs come from Yahoo Finance. Research output, not investment advice. NO TRADE is the default decision.*
"""


def _confirm(cands, reads, cache):
    """First candidate whose market data match at least one screenshot legend. Returns (ticker, as_of_index, how, daily, hourly)."""
    for tk in cands:
        try:
            if tk not in cache:
                d, h = C.fetch_yahoo(tk)
                cache[tk] = (C.drop_incomplete_session(d), h)
            d, h = cache[tk]
        except Exception:
            continue
        hits = {}
        if reads.get("1D") and reads["1D"]["ohlc"]:
            k = C.match_daily(d, reads["1D"]["ohlc"]); hits["1D"] = k
        if reads.get("1W") and reads["1W"]["ohlc"]:
            k = C.match_weekly(d, reads["1W"]["ohlc"]); hits["1W"] = k
        if reads.get("1H") and reads["1H"]["ohlc"]:
            k = C.match_hourly(d, h, reads["1H"]["ohlc"]); hits["1H"] = k
        found = {s: k for s, k in hits.items() if k is not None}
        if found:
            for s in ("1D", "1W", "1H"):
                if s in found:
                    return tk, found[s], f"matched the {s} screenshot's last bar", d, h
        if not hits and len(d) > 130:                       # no legend readable: accept the first ticker with data, unverified
            return tk, len(d) - 1, "latest completed session (screenshots could not be verified)", d, h
    return None, None, None, None, None


def analyse(img_w, img_d, img_h, ticker_text):
    checks = []
    try:
        imgs = {"1W": img_w, "1D": img_d, "1H": img_h}
        if all(v is None for v in imgs.values()) and not (ticker_text or "").strip():
            return "Upload the three screenshots (or type a ticker) and press Analyse.", None, None
        reads = {}
        for slot, im in imgs.items():
            if im is None:
                checks.append([slot, "-", "-", "no image"]); continue
            r = read_header(im)
            reads[slot] = r
            note = "ok" if r["timeframe"] in (None, slot) else f"looks like a {r['timeframe']} chart"
            lg = "-" if not r["ohlc"] else f"O {r['ohlc']['open']:g} H {r['ohlc']['high']:g} L {r['ohlc']['low']:g} C {r['ohlc']['close']:g}"
            checks.append([slot, ", ".join(r["candidates"][:4]) or "-", r["timeframe"] or "-", f"{lg} ({note})"])

        typed = (ticker_text or "").strip().upper().replace(".", "-")
        cands = [typed] if typed else []
        for s in ("1D", "1W", "1H"):
            for c in (reads.get(s) or {}).get("candidates", []):
                if c not in cands:
                    cands.append(c)
        cache = {}
        tk, k, how, d, h = _confirm(cands, reads, cache)
        if tk is None:
            deep = []
            for s, im in imgs.items():
                if im is not None:
                    for c in read_header(im, deep=True)["candidates"]:
                        if c not in cands and c not in deep:
                            deep.append(c)
            tk, k, how, d, h = _confirm(deep, reads, cache)
        if tk is None:
            return ("Could not confirm the ticker from the screenshots. Type the ticker in the box and try again.",
                    None, pd.DataFrame(checks, columns=["slot", "ticker candidates", "timeframe", "legend read"]))

        x = C.features_at(d, k)
        trip = C.render_at(d, h, k)
        emb = {b: C.embed_views(trip, CFG["backbones"][b]) for b in B.needed_backbones()}
        out = B.predict_row(x, emb)
        asof = pd.Timestamp(d["date"].iloc[k]).date()
        test_auc = _metric(["test", "O5", "AUC"])
        edge = "no reliable edge" if not CFG["rule"]["direction_usable"] else "weak edge"
        base_dd = CFG.get("base_rates", {}).get("D10")
        md = [f"### {tk} — analysis as of {asof} ({how})",
              f"**Decision: {out['decision']}**",
              "".join(f"\n- {r}" for r in out["reasons"]),
              "",
              "| Output | Value | Notes |", "|---|---|---|",
              f"| Expected volatility, next 20 trading days | **{_fmt_pct(out['vol_ann'])} a year** "
              f"(likely {_fmt_pct(out['vol_ann_lo'])}–{_fmt_pct(out['vol_ann_hi'])}) | test R² {(_metric(['test', 'V', 'R2']) or float('nan')):.2f} on log volatility |",
              f"| Chance of a drawdown of 10 % or more within 20 trading days | **{_fmt_pct(out['p_dd10'])}** | "
              f"training base rate {_fmt_pct(base_dd)}; calibrated on validation data |",
              f"| Chance of beating the median stock over the next 5 trading days | **{_fmt_pct(out['p_out5'])}** | "
              f"test AUC {(test_auc or float('nan')):.3f}: {edge} |",
              "", LIMITS]
        return "\n".join(md), C.triplet_png(trip), pd.DataFrame(checks, columns=["slot", "ticker candidates", "timeframe", "legend read"])
    except Exception as e:
        traceback.print_exc()
        return f"Something went wrong: {e}", None, pd.DataFrame(checks, columns=["slot", "ticker candidates", "timeframe", "legend read"]) if checks else None


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        w = gr.Image(type="pil", label="1 Week chart screenshot (1W)")
        dd = gr.Image(type="pil", label="1 Day chart screenshot (1D)")
        hh = gr.Image(type="pil", label="1 Hour chart screenshot (1H)")
    with gr.Row():
        tk = gr.Textbox(label="Ticker (optional; detected from the screenshots if left empty)", scale=3)
        btn = gr.Button("Analyse", variant="primary", scale=1)
    out_md = gr.Markdown()
    out_img = gr.Image(label="What the model sees: the three charts rebuilt from market data", type="pil")
    out_tab = gr.Dataframe(label="Screenshot checks")
    btn.click(analyse, inputs=[w, dd, hh, tk], outputs=[out_md, out_img, out_tab])

if __name__ == "__main__":
    demo.launch()
