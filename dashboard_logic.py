"""Dashboard logic shared by the Streamlit app (streamlit_app.py) and the Gradio app (app.py).

Flow: three TradingView screenshots (1W, 1D, 1H) -> read ticker candidates, timeframe and last-bar legend from each image ->
confirm the ticker by matching the legend against Yahoo Finance bars -> rebuild the exact model inputs (33 numeric features +
the three training-style charts) as of the screenshot's date -> frozen model bundle -> outputs and NO TRADE decision."""
import os
import time
import traceback

import numpy as np
import pandas as pd
from PIL import Image

import tfcv_core as C
from screen_reader import read_header

TITLE = "Computer Vision with Deep Learning Project Dashboard"
MODEL_REPO_DEFAULT = "KrishnaHarshaTangirala/triframe-cv-model"
CHECK_COLUMNS = ["slot", "ticker candidates", "timeframe", "legend read"]
MAX_PIXELS = 40_000_000                 # refuse absurdly large uploads before OCR touches them
MARKET_TTL = 600                        # seconds a ticker's market data is reused (spares Yahoo, speeds up repeat clicks)
_MARKET = {}

LIMITS = """
*Limits: the models were trained on survivor stocks from one data vendor (stocks that later failed are missing, so risk may be
understated); live inputs come from Yahoo Finance. Research output, not investment advice. NO TRADE is the default decision.*
"""


# ------------------------------------------------------------------------------------------------ model bundle
def load_bundle(token=None, model_repo=None):
    """Load the frozen bundle from BUNDLE_DIR (local folder) or from the private Hugging Face model repo."""
    path = os.environ.get("BUNDLE_DIR")
    if not path:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_repo or os.environ.get("MODEL_REPO", MODEL_REPO_DEFAULT), token=token or os.environ.get("HF_TOKEN"))
    return C.Bundle(path)


def _fmt_pct(x, d=0):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _metric(cfg, path, default=None):
    cur = cfg.get("metrics", {})
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _model_line(cfg, t):
    s = cfg["targets"][t]
    fs = "numbers only (33 features)" if s["feature_set"] == "NUM" else f"numbers + {s['feature_set'].split(':', 1)[1]} chart features"
    return f"{fs}, {'linear' if s['head'] == 'linear' else 'gradient-boosted trees'}"


def describe(B):
    cfg = B.cfg
    d = cfg.get("data", {})
    scope = (f"These models come from a **{cfg.get('run_tag', '')} run on {d.get('tickers_used', '?')} of {d.get('universe', '?')} stocks**."
             if d.get("tickers_used") and d.get("universe") and d["tickers_used"] != d["universe"] else
             f"These models were trained on {d.get('universe', '?')} stocks.")
    return f"""
**TriFrame** asks whether chart images at three time scales (weekly, daily and hourly) add information about a US stock's
next-20-day volatility, drawdown risk and 5-day relative direction beyond plain numbers computed from the same price history.
Seven frozen ImageNet-pretrained vision backbones were compared with a numerical baseline, using chronological training,
validation and test periods and a sealed final holdout. {scope}

**How to use:** upload three TradingView screenshots of the same stock: weekly (1W), daily (1D) and hourly (1H). The app reads the
ticker and the last bar from each picture, checks them against market data, rebuilds the exact chart images and numbers the model
was trained on, and reports the outputs below. You can also type the ticker.

**Models in use:** volatility: {_model_line(cfg, 'V')}; drawdown: {_model_line(cfg, 'D10')}; direction: {_model_line(cfg, 'O5')}.
"""


# ------------------------------------------------------------------------------------------------ market data
def market_data(ticker):
    """Daily + regular-hours hourly bars for a ticker (cached for MARKET_TTL seconds, retried on transient failures)."""
    now = time.time()
    hit = _MARKET.get(ticker)
    if hit and now - hit[0] < MARKET_TTL:
        return hit[1], hit[2]
    last = None
    for attempt, wait in enumerate((0, 1.5, 4)):
        if wait:
            time.sleep(wait)
        try:
            d, h = C.fetch_yahoo(ticker)
            d = C.drop_incomplete_session(d)
            if len(d) < C.FV.W + 5:
                raise ValueError(f"only {len(d)} daily bars for {ticker}")
            _MARKET[ticker] = (now, d, h)
            return d, h
        except ValueError as e:                  # unknown ticker or too little history: one quick retry, then give up
            last = e
            if attempt >= 1:
                break
        except Exception as e:                   # network / rate limit: keep retrying
            last = e
    raise last


def confirm(cands, typed, reads):
    """First candidate whose market data match at least one screenshot legend. A ticker typed by the user is accepted
    on the latest completed session even if no legend matches (for example a screenshot taken mid-session).
    Returns (ticker, daily index, how, daily, hourly) or (None,)*5."""
    for tk in cands:
        try:
            d, h = market_data(tk)
        except Exception:
            continue
        found = {}
        if (reads.get("1D") or {}).get("ohlc"):
            found["1D"] = C.match_daily(d, reads["1D"]["ohlc"])
        if (reads.get("1W") or {}).get("ohlc"):
            found["1W"] = C.match_weekly(d, reads["1W"]["ohlc"])
        if (reads.get("1H") or {}).get("ohlc"):
            found["1H"] = C.match_hourly(d, h, reads["1H"]["ohlc"])
        for slot in ("1D", "1W", "1H"):
            if found.get(slot) is not None:
                return tk, found[slot], f"matched the {slot} screenshot's last bar", d, h
        if tk == typed:
            why = "the screenshots' last bars did not match market data" if found else "no screenshot legend was readable"
            return tk, len(d) - 1, f"latest completed session; {why}", d, h
    return None, None, None, None, None


# ------------------------------------------------------------------------------------------------ the analysis
def _check_image(im):
    if im is None:
        return None
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    if im.size[0] * im.size[1] > MAX_PIXELS:
        raise ValueError("That image is too large; please upload a normal screenshot.")
    return im.convert("RGB")


def analyse(B, img_w, img_d, img_h, ticker_text=""):
    """Returns (markdown, PIL image of the three rebuilt charts or None, DataFrame of screenshot checks or None)."""
    cfg = B.cfg
    checks = []
    frame = lambda: pd.DataFrame(checks, columns=CHECK_COLUMNS) if checks else None
    try:
        imgs = {"1W": _check_image(img_w), "1D": _check_image(img_d), "1H": _check_image(img_h)}
        typed = (ticker_text or "").strip().upper().replace(".", "-")
        if all(v is None for v in imgs.values()) and not typed:
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

        cands = [typed] if typed else []
        for s in ("1D", "1W", "1H"):
            for c in (reads.get(s) or {}).get("candidates", []):
                if c not in cands:
                    cands.append(c)
        tk, k, how, d, h = confirm(cands, typed, reads)
        if tk is None:                                           # slower pass: also read the price-axis tag
            deep = []
            for im in imgs.values():
                if im is not None:
                    deep += [c for c in read_header(im, deep=True)["candidates"] if c not in cands and c not in deep]
            tk, k, how, d, h = confirm(deep, typed, reads)
        if tk is None:
            return ("Could not confirm the ticker from the screenshots. Type the ticker in the box and try again. "
                    "(If a valid ticker keeps failing, Yahoo Finance may be rate-limiting this server; try again in a few minutes.)",
                    None, frame())

        x = C.features_at(d, k)
        trip = C.render_at(d, h, k)
        emb = {b: C.embed_views(trip, cfg["backbones"][b]) for b in B.needed_backbones()}
        out = B.predict_row(x, emb)
        asof = pd.Timestamp(d["date"].iloc[k]).date()
        test_auc = _metric(cfg, ["test", "O5", "AUC"])
        hold_auc = _metric(cfg, ["holdout", "O5|all", "AUC"])
        edge = "no reliable edge" if not cfg["rule"]["direction_usable"] else "weak edge"
        base_dd = cfg.get("base_rates", {}).get("D10")
        r2_t, r2_h = _metric(cfg, ["test", "V", "R2"]), _metric(cfg, ["holdout", "V|all", "R2"])
        r2_note = f"test R² {r2_t:.2f}" + (f", sealed holdout R² {r2_h:.2f}" if r2_h is not None else "") + " on log volatility"
        auc_note = f"test AUC {test_auc:.3f}" + (f", holdout {hold_auc:.3f}" if hold_auc is not None else "") + f": {edge}"
        md = [f"### {tk} — analysis as of {asof} ({how})",
              f"**Decision: {out['decision']}**",
              "".join(f"\n- {r}" for r in out["reasons"]),
              "",
              "| Output | Value | Notes |", "|---|---|---|",
              f"| Expected volatility, next 20 trading days | **{_fmt_pct(out['vol_ann'])} a year** "
              f"(likely {_fmt_pct(out['vol_ann_lo'])}–{_fmt_pct(out['vol_ann_hi'])}) | {r2_note} |",
              f"| Chance of a drawdown of 10 % or more within 20 trading days | **{_fmt_pct(out['p_dd10'])}** | "
              f"training base rate {_fmt_pct(base_dd)}; calibrated on validation data |",
              f"| Chance of beating the median stock over the next 5 trading days | **{_fmt_pct(out['p_out5'])}** | {auc_note} |",
              "", LIMITS]
        return "\n".join(md), C.triplet_png(trip), frame()
    except ValueError as e:
        return f"{e}", None, frame()
    except Exception as e:
        traceback.print_exc()
        return f"Something went wrong: {type(e).__name__}: {e}", None, frame()
