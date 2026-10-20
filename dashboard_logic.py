"""Dashboard logic shared by the Streamlit app (streamlit_app.py) and the Gradio app (app.py).

Flow per company: three TradingView screenshots (1W, 1D, 1H) -> read ticker candidates, timeframe and last-bar legend from each
image -> confirm the ticker by matching the legend against Yahoo Finance bars -> rebuild the exact model inputs (33 numeric
features + the three training-style charts) as of the screenshot's date -> frozen model bundle -> outputs, NO TRADE decision,
risk tier, plain-language explanation and a Monte Carlo simulation of the next 20 sessions. Up to four companies are compared."""
import math
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
SLOTS = ("1W", "1D", "1H")

LIMITS = """
*Limits: the models were trained on survivor stocks from one data vendor (stocks that later failed are missing, so risk may be
understated); live inputs come from Yahoo Finance. Research output, not investment advice. NO TRADE is the default decision.*
"""

# risk tiers on the risk score = the larger of (volatility / its NO TRADE threshold) and (drawdown chance / its threshold);
# 1.0 is the NO TRADE risk line (the top 20 % of the model's readings in the validation years)
TIERS = (("Very high", 1.5, "#b71c1c"), ("High", 1.0, "#e53935"), ("Moderate", 0.75, "#f39c12"), ("Low", 0.0, "#2e9d5b"))

BACKBONE_NAMES = {
    "resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0", "vit_base_patch16_224": "ViT-B/16",
    "convnext_tiny": "ConvNeXt-Tiny", "swin_tiny_patch4_window7_224": "Swin-T", "vit_small_patch14_dinov2": "DINOv2 ViT-S/14",
    "mobilenetv3_large_100": "MobileNetV3-Large",
}


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
    return cur if not (isinstance(cur, float) and not np.isfinite(cur)) else default


def _years(cfg, split):
    a, b = cfg.get("splits", {}).get(split, ("", ""))
    if not a:
        return {"train": "2001-2016", "val": "2017-2019", "test": "2020-2022"}[split]
    last = int(b[:4]) - 1 if b[5:7] == "01" else int(b[:4])
    return f"{a[:4]}-{last}"


def backbone_label(b):
    """'ft_convnext_tiny' -> 'ConvNeXt-Tiny (fine-tuned)'; 'swin_tiny_patch4_window7_224' -> 'Swin-T (frozen)'."""
    ft = b.startswith("ft_")
    base = b[3:] if ft else b
    return f"{BACKBONE_NAMES.get(base, base)} ({'fine-tuned' if ft else 'frozen'})"


def pretrain_words(timm_name):
    tag = timm_name.split(".", 1)[1] if "." in timm_name else ""
    for k, w in (("lvd142m", "LVD-142M (self-supervised)"), ("in21k_ft_in1k", "ImageNet-21k + 1k"),
                 ("in12k_ft_in1k", "ImageNet-12k + 1k"), ("in1k", "ImageNet-1k")):
        if k in tag:
            return w
    return "ImageNet"


def models_in_use(cfg):
    """One row per output: what is predicted, the inputs, the network behind the chart features, the head, and its test score."""
    q = {"V": "Volatility over the next 20 trading days", "D10": "Chance of a 10%+ drawdown within 20 trading days",
         "O5": "Chance of beating the median stock over 5 trading days"}
    score = {"V": lambda: f"R² {_metric(cfg, ['test', 'V', 'R2'], float('nan')):.2f}",
             "D10": lambda: f"AUC {_metric(cfg, ['test', 'D10', 'AUC'], float('nan')):.2f}",
             "O5": lambda: f"AUC {_metric(cfg, ['test', 'O5', 'AUC'], float('nan')):.3f}"}
    rows = []
    for t in ("V", "D10", "O5"):
        s = cfg["targets"][t]
        use_num, b, views = C.Bundle.parse_fs(s["feature_set"])
        if b is None:
            inputs, net = "33 numbers from the last 126 daily bars", "none (numbers only)"
        else:
            vw = "weekly + daily + hourly charts" if len(views) == 3 else f"{'/'.join(views)} chart"
            inputs = ("33 numbers + " if use_num else "") + vw
            net = backbone_label(b) + f", pretrained on {pretrain_words(cfg.get('backbones', {}).get(b, {}).get('timm_name', ''))}"
        head = {"linear": "linear (ridge / logistic)", "xgb": "gradient-boosted trees (XGBoost)",
                "lgbm": "gradient-boosted trees (LightGBM)"}.get(s["head"], s["head"])
        rows.append({"Output": q[t], "Inputs": inputs, "Vision network": net, "Head": head,
                     "Test score": score[t]()})
    return pd.DataFrame(rows)


def describe(B):
    cfg = B.cfg
    d = cfg.get("data", {})
    scope = (f"The models in use come from a **run on {d.get('tickers_used', '?')} of {d.get('universe', '?')} stocks** "
             f"({d.get('rows', 0):,} chart sets)."
             if d.get("tickers_used") and d.get("universe") and d["tickers_used"] != d["universe"] else
             f"The models were trained on {d.get('universe', '?')} stocks.")
    return f"""
**TriFrame** asks whether chart images at three time scales (weekly, daily and hourly) add information about a US stock's
next-20-day volatility, drawdown risk and 5-day relative direction beyond plain numbers computed from the same price history.
Seven ImageNet-pretrained vision networks were compared with a numerical baseline, using chronological training, validation and
test periods and a sealed final holdout. {scope}

**How to use:** choose how many companies to analyse (1 to 4). For each one, upload three TradingView screenshots of the same
stock: weekly (1W), daily (1D) and hourly (1H), or just type the ticker. The app reads the ticker and the last bar of each
picture, checks them against market data, rebuilds the exact chart images and numbers the models were trained on, and reports
risk, a plain-language explanation and a Monte Carlo simulation of the next 20 trading days for every company.
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


# ------------------------------------------------------------------------------------------------ one company
def _check_image(im):
    if im is None:
        return None
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    if im.size[0] * im.size[1] > MAX_PIXELS:
        raise ValueError("That image is too large; please upload a normal screenshot.")
    return im.convert("RGB")


def analyse_company(B, images, ticker_text=""):
    """images: {"1W": file or PIL image or None, "1D": ..., "1H": ...}. Returns a result dict:
    ok, message, ticker, asof, how, out (model outputs + decision + reasons), p0 (as-of close), charts (PIL), checks (DataFrame)."""
    cfg = B.cfg
    checks = []
    frame = lambda: pd.DataFrame(checks, columns=CHECK_COLUMNS) if checks else None
    res = dict(ok=False, message="", ticker=None, asof=None, how=None, out=None, p0=None, charts=None, checks=None)
    try:
        imgs = {s: _check_image((images or {}).get(s)) for s in SLOTS}
        typed = (ticker_text or "").strip().upper().replace(".", "-")
        if all(v is None for v in imgs.values()) and not typed:
            res["message"] = "Upload the three screenshots (or type a ticker) and press Analyse."
            return res
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
            res.update(message="Could not confirm the ticker from the screenshots. Type the ticker in the box and try again. "
                               "(If a valid ticker keeps failing, Yahoo Finance may be rate-limiting this server; try again in a few minutes.)",
                       checks=frame())
            return res

        x = C.features_at(d, k)
        trip = C.render_at(d, h, k)
        emb = {b: C.embed_views(trip, cfg["backbones"][b], B.path) for b in B.needed_backbones()}
        out = B.predict_row(x, emb)
        res.update(ok=True, ticker=tk, asof=pd.Timestamp(d["date"].iloc[k]).date(), how=how, out=out,
                   p0=float(d["close"].iloc[k]), charts=C.triplet_png(trip), checks=frame())
        return res
    except ValueError as e:
        res.update(message=f"{e}", checks=frame())
        return res
    except Exception as e:
        traceback.print_exc()
        res.update(message=f"Something went wrong: {type(e).__name__}: {e}", checks=frame())
        return res


# ------------------------------------------------------------------------------------------------ risk tier, Monte Carlo
def _percentile(cfg, key, value):
    """Where a reading falls among the model's own validation-period readings, in percent (None if the bundle has no table)."""
    rq = cfg.get("risk_quantiles") or {}
    if key not in rq or not rq.get("levels"):
        return None
    return float(100 * np.interp(value, rq[key], rq["levels"]))


def _pct_words(pct, levels_top=95, levels_bottom=5):
    if pct is None:
        return ""
    if pct >= levels_top:
        return f"in the top {100 - levels_top}% of the model's readings"
    if pct <= levels_bottom:
        return f"in the bottom {levels_bottom}% of the model's readings"
    return f"higher than about {pct:.0f}% of the model's readings"


def risk_profile(out, cfg):
    """Risk tier from the two risk readings, each divided by its NO TRADE threshold (1.0 = the top 20 % of validation readings)."""
    rule = cfg["rule"]
    vr, dr = out["vol_ann"] / rule["vol_ann_q80"], out["p_dd10"] / rule["p_dd10_q80"]
    score = max(vr, dr)
    tier, color = next((n, c) for n, lo, c in TIERS if score >= lo)
    return dict(score=float(score), tier=tier, color=color, vol_ratio=float(vr), dd_ratio=float(dr),
                vol_pct=_percentile(cfg, "vol_ann", out["vol_ann"]), dd_pct=_percentile(cfg, "p_dd10", out["p_dd10"]),
                flagged=bool(score >= 1.0))


def run_mc(res, cfg, n_paths=5000, match_dd=True, seed=7):
    """Monte Carlo simulation of the next 20 sessions for one analysed company (see tfcv_core.simulate_paths)."""
    out = res["out"]
    return C.simulate_paths(res["p0"], out["logrv20"], cfg["targets"]["V"], mc=cfg.get("mc"), n_paths=int(n_paths),
                            p_dd_target=out["p_dd10"] if match_dd else None, seed=int(seed))


def _one_in(p):
    if p >= 0.95:
        return "close to certain by the model's reckoning"
    if p >= 0.5:
        return "more likely than not"
    if p < 0.04:
        return "less than 1 in 25"
    return f"about 1 in {max(2, round(1 / p))}"


# ------------------------------------------------------------------------------------------------ plain-language explanation
def explain(res, cfg, sim=None):
    """The detailed, plain-language reading of one company's outputs. Returns sections (title + labelled paragraphs), the
    summary table rows and the primary takeaway. Every number comes from the model, the bundle or the simulation."""
    out, tk = res["out"], res["ticker"]
    rule, rp = cfg["rule"], risk_profile(res["out"], cfg)
    v, lo, hi = out["vol_ann"], out["vol_ann_lo"], out["vol_ann_hi"]
    day, month = v / math.sqrt(252), v * math.sqrt(20 / 252)
    ref = cfg.get("reference", {})
    yv, yt, ytr = _years(cfg, "val"), _years(cfg, "test"), _years(cfg, "train")

    # 1. volatility
    vol_level = {"Very high": "very high", "High": "high", "Moderate": "moderate", "Low": "on the calm side"}[
        next(n for n, lo_, _ in TIERS if rp["vol_ratio"] >= lo_)]
    comp = "For comparison, broad US stock indices such as the S&P 500 have historically run at roughly 15-20% a year"
    if ref.get("vol_ann_median_train"):
        comp += f", and the typical (median) stock in this project's training data ran at about {ref['vol_ann_median_train']:.0%}"
    pct = f" It is {_pct_words(rp['vol_pct'])} in the validation years ({yv})." if rp["vol_pct"] is not None else ""
    r2t, r2h = _metric(cfg, ["test", "V", "R2"]), _metric(cfg, ["holdout", "V|all", "R2"])
    acc = (f"On the {yt} test years the volatility model explained {r2t:.0%} of the variation in (log) volatility across stocks "
           f"and dates (R² {r2t:.2f})" if r2t is not None else "The volatility model was scored on held-out years")
    acc += f"; on the sealed 2023-2026 holdout it explained {r2h:.0%} (R² {r2h:.2f})." if r2h is not None else "."
    acc += (" That is a solid score for this task: volatility is persistent (calm stocks tend to stay calm and wild ones wild), "
            "which makes it one of the more forecastable quantities in finance. It forecasts the size of the moves, not their direction.")
    s1 = dict(title=f"1. Expected volatility: {v:.0%} a year (likely {lo:.0%}-{hi:.0%})", items=[
        ("What it means", f"The model expects {tk}'s price to move at an annualised rate of about {v:.0%} over the next 20 trading "
                          f"days (about one month). In everyday terms that is a typical daily move of about ±{day:.1%} and a typical "
                          f"one-month move of about ±{month:.0%}. That is {vol_level} for a single US stock. {comp}.{pct}"),
        ("The range", f"{lo:.0%}-{hi:.0%} is an 80% prediction range. In the validation years ({yv}) the actual volatility landed "
                      f"inside a range this wide around the forecast about 8 times out of 10: the low end is a calm month, the high "
                      f"end a turbulent one. About 1 time in 10 it was calmer than {lo:.0%}, and about 1 time in 10 wilder than {hi:.0%}."),
        ("How accurate the model is", acc)])

    # 2. drawdown
    p, base = out["p_dd10"], cfg.get("base_rates", {}).get("D10")
    ratio = p / base if base else None
    rel = ("well above" if ratio >= 1.3 else "above" if ratio >= 1.1 else "close to" if ratio >= 0.9 else "below") if ratio else ""
    auc_d, ece_d = _metric(cfg, ["test", "D10", "AUC"]), _metric(cfg, ["test", "D10", "ECE"])
    ctx = (f"Across all stocks and dates in the training years ({ytr}) a drop like this happened {base:.0%} of the time, so this reading "
           f"is {ratio:.1f}x the typical rate ({rel} normal). " if base else "")
    ctx += (f"The NO TRADE rule treats {rule['p_dd10_q80']:.0%} or more as high risk (the top 20% of the model's readings in the "
            f"validation years).")
    cal = (f"The model's raw scores were calibrated on the validation years ({yv}) with isotonic regression, so a stated chance "
           f"matched how often the drop really happened in that period.")
    if auc_d is not None:
        cal += (f" On the {yt} test years it ranked a stock that went on to drop 10% above one that did not {auc_d:.0%} of the time "
                f"(AUC {auc_d:.2f}; 50% would be guessing), a useful but far from perfect separation.")
    if ece_d is not None:
        cal += f" Its stated chances were off by {100 * ece_d:.1f} percentage points on average in those years."
    s2 = dict(title=f"2. Chance of a 10%+ drawdown within 20 trading days: {p:.0%}", items=[
        ("What it means", f"The model gives a {p:.0%} chance ({_one_in(p)}) that at some point in the next 20 trading days {tk} "
                          f"trades at least 10% below the next session's opening price, even if it recovers afterwards."),
        ("The context", ctx), ("The calibration", cal)])

    # 3. direction
    q = out["p_out5"]
    auc_o, ci_o, hold_o = _metric(cfg, ["test", "O5", "AUC"]), _metric(cfg, ["test", "O5", "AUC_CI"]), _metric(cfg, ["holdout", "O5|all", "AUC"])
    gap = abs(q - 0.5)
    lean = ("exactly at" if gap < 0.005 else ("slightly " if gap < 0.03 else "" if gap < 0.1 else "well ") + ("above" if q > 0.5 else "below"))
    if not rule["direction_usable"]:
        take = "Treat this as a coin flip."
        if auc_o is not None:
            take += f" On the {yt} test years the direction model's AUC was {auc_o:.3f}"
            take += f" (95% range {ci_o[0]:.3f}-{ci_o[1]:.3f}, which includes 0.5)" if ci_o else ""
            take += f", and {hold_o:.3f} on the holdout" if hold_o is not None else ""
            take += ": no reliable edge."
        take += " That is why the dashboard never turns this number into a buy or sell call."
    else:
        take = (f"The direction model passed its validation and test checks, but the edge is small (test AUC {auc_o:.3f}); "
                "it is shown as a low-confidence signal only when risk is not elevated.")
    s3 = dict(title=f"3. Chance of beating the median stock over 5 trading days: {q:.0%}", items=[
        ("What it means", f"A {q:.0%} probability that {tk}'s return over the next 5 trading days beats the return of the median "
                          f"(middle) US stock over the same days. {q:.0%} is {lean} 50%."),
        ("The takeaway", take)])

    sections = [s1, s2, s3]
    if sim is not None:
        cov = (cfg.get("mc") or {}).get("coverage", {}).get("test", {})
        check = (f" On the {yt} test years the simulation's 90% band contained {cov['90%']:.0%} of the real 20-day outcomes."
                 if cov.get("90%") is not None else "")
        sections.append(dict(title=f"4. Monte Carlo simulation of the next 20 trading days ({sim['n_paths']:,} paths)", items=[
            ("How it works", "Each simulated path draws a volatility from the model's forecast and its measured error, then draws "
                             f"fat-tailed daily moves (Student-t, {sim['t_df']:g} degrees of freedom) with exactly that volatility. "
                             "There is no upward or downward drift, because the direction model has no reliable edge."
                             + (" The paths are reweighted so the share that touches a 10% drawdown equals the drawdown model's "
                                f"calibrated {p:.0%} (raw simulation: {sim['p_dd_sim']:.0%})." if sim["tilted"] else "") + check),
            ("What it shows", f"Starting from ${sim['p0']:,.2f}, the middle 90% of simulated 20-day outcomes range from "
                              f"{sim['ret_q05']:+.0%} to {sim['ret_q95']:+.0%} (${sim['price_q05']:,.2f} to ${sim['price_q95']:,.2f}). "
                              f"In the worst 5% of paths the average loss is {sim['es95']:.0%}. {sim['p_down10']:.0%} of paths end "
                              f"10% or more lower, and {sim['p_up']:.0%} end higher than they started."),
            ("How to read it", "This is a picture of the range of plausible outcomes given the predicted risk, not a forecast of "
                               "where the price will go.")]))

    tier_note = {"Very high": "Very high: far above the high-risk line", "High": "High: above the NO TRADE risk line",
                 "Moderate": "Moderate: close to, but below, the high-risk line", "Low": "Low: well below the high-risk line"}[rp["tier"]]
    rows = [
        ("Short-term direction (5 days)", f"{q:.0%} chance to beat the median stock",
         "Neutral: no reliable 5-day edge" if not rule["direction_usable"] else "Weak signal only"),
        ("Volatility (20 days)", f"{v:.0%} a year (±{month:.0%} a month)", f"{vol_level.capitalize()} swings expected"),
        ("Drawdown risk (20 days)", f"{p:.0%} chance of a 10%+ drop", f"{ratio:.1f}x the typical rate" if ratio else "-"),
        ("Overall risk tier", f"{rp['tier']} (risk score {rp['score']:.2f})", tier_note),
        ("Decision (pre-registered rule)", out["decision"], "; ".join(x.rstrip(".") for x in out["reasons"][:2]) + "."),
    ]
    if sim is not None:
        rows.insert(3, ("Simulated 20-day range", f"{sim['ret_q05']:+.0%} to {sim['ret_q95']:+.0%}",
                        f"worst-5% average {-sim['es95']:+.0%}"))

    if rp["tier"] in ("High", "Very high"):
        tk_ = (f"{tk} is in the {rp['tier'].lower()}-risk zone right now: expected swings of about ±{month:.0%} over the next month and a "
               f"{p:.0%} chance of a 10% drop" + (f" ({ratio:.1f}x the typical rate)" if ratio else "") + ", while the 5-day "
               "direction signal is no better than a coin flip. The risk side of this reading is the reliable part; the direction "
               "side is not.")
    elif rp["tier"] == "Moderate":
        tk_ = (f"{tk}'s risk is moderate: swings of about ±{month:.0%} a month and a {p:.0%} chance of a 10% drop put it close to, "
               "but below, the high-risk line. The 5-day direction signal offers no reliable edge.")
    else:
        tk_ = (f"{tk} is on the calmer side right now: swings of about ±{month:.0%} a month and a {p:.0%} chance of a 10% drop. "
               "Lower risk is not a buy signal, though: the 5-day direction signal still offers no reliable edge.")
    tk_ += f" Under the pre-registered rule the decision is {out['decision']}."
    return dict(sections=sections, summary=pd.DataFrame(rows, columns=["Metric", "Current signal", "What it implies"]),
                takeaway=tk_, risk=rp)


def report_markdown(res, cfg, sim=None):
    """One company's full report as markdown (used by the Gradio app and as a text fallback)."""
    if not res.get("ok"):
        return res.get("message") or "No result."
    ex = explain(res, cfg, sim)
    out = res["out"]
    md = [f"### {res['ticker']} — analysis as of {res['asof']} ({res['how']})", f"**Decision: {out['decision']}** · "
          f"**Risk tier: {ex['risk']['tier']}** (score {ex['risk']['score']:.2f})", ""]
    for s in ex["sections"]:
        md.append(f"#### {s['title']}")
        md += [f"- **{a}:** {b}" for a, b in s["items"]]
        md.append("")
    md += ["#### Summary", "| Metric | Current signal | What it implies |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in ex["summary"].itertuples(index=False)]
    md += ["", f"**Primary takeaway:** {ex['takeaway']}", "", LIMITS]
    return "\n".join(md)


def analyse(B, img_w, img_d, img_h, ticker_text=""):
    """Single-company analysis, returns (markdown, PIL image of the three rebuilt charts or None, DataFrame of checks or None)."""
    res = analyse_company(B, {"1W": img_w, "1D": img_d, "1H": img_h}, ticker_text)
    if not res["ok"]:
        return res["message"], None, res["checks"]
    sim = run_mc(res, B.cfg)
    return report_markdown(res, B.cfg, sim), res["charts"], res["checks"]


# ------------------------------------------------------------------------------------------------ several companies
def compare(results, cfg, sims=None):
    """Ranking of the analysed companies from highest to lowest risk, and a one-paragraph verdict."""
    rows = []
    for i, r in enumerate(results):
        if not r.get("ok"):
            continue
        rp, out = risk_profile(r["out"], cfg), r["out"]
        row = {"Ticker": r["ticker"], "Risk tier": rp["tier"], "Risk score": round(rp["score"], 2),
               "Volatility (a year)": out["vol_ann"], "Chance of 10%+ drawdown": out["p_dd10"],
               "Chance to beat median (5 days)": out["p_out5"], "Decision": out["decision"], "As of": str(r["asof"])}
        if sims is not None and sims[i] is not None:
            row["Simulated 20-day range"] = f"{sims[i]['ret_q05']:+.0%} to {sims[i]['ret_q95']:+.0%}"
        rows.append(row)
    if not rows:
        return None, ""
    tab = pd.DataFrame(rows).sort_values("Risk score", ascending=False).reset_index(drop=True)
    tab.insert(0, "Rank (1 = riskiest)", range(1, len(tab) + 1))
    if len(tab) == 1:
        r = tab.iloc[0]
        return tab, f"**{r['Ticker']}** is in the **{r['Risk tier'].lower()}** risk tier (risk score {r['Risk score']:.2f})."
    hi_, lo_ = tab.iloc[0], tab.iloc[-1]
    txt = (f"**Highest risk: {hi_['Ticker']}** ({hi_['Risk tier']}, volatility {hi_['Volatility (a year)']:.0%} a year, "
           f"{hi_['Chance of 10%+ drawdown']:.0%} chance of a 10% drawdown). **Lowest risk: {lo_['Ticker']}** ({lo_['Risk tier']}, "
           f"volatility {lo_['Volatility (a year)']:.0%} a year, {lo_['Chance of 10%+ drawdown']:.0%} chance of a 10% drawdown).")
    flagged = tab[tab["Risk score"] >= 1.0]["Ticker"].tolist()
    txt += (f" Above the NO TRADE risk line: {', '.join(flagged)}." if flagged else " None is above the NO TRADE risk line.")
    txt += (" The direction signal has no reliable edge for any of them, so the ranking is about risk, not about which "
            "stock will do better." if not cfg["rule"]["direction_usable"] else "")
    return tab, txt
