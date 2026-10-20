"""TriFrame CV - the plain-language PDF project report (about 5-6 pages) and its figures.
Used by the notebook's Stage 11, from the values computed in the run (the same function made the 60-stock pilot's report).
Every number in the report comes from the dict R passed to build_pdf; nothing is typed in by hand."""
import os
import datetime as dt
from xml.sax.saxutils import escape

import numpy as np

BLUE, ORANGE, GREY, RED, GREEN = "#3b6ea5", "#e67e22", "#9e9e9e", "#c0392b", "#2e9d5b"


# ------------------------------------------------------------------------------------------------ figures (matplotlib, no pyplot state)
def _fig(w, h):
    from matplotlib.figure import Figure
    return Figure(figsize=(w, h), dpi=150)


def fig_pipeline(path, ft=True):
    """The architecture as a flow diagram."""
    from matplotlib.patches import FancyBboxPatch
    fig = _fig(11, 4.4); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=color, ec="#33415c", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.3))
    # top row, left to right: pictures -> renderer -> vision network -> embeddings -> head
    top, h = 3.05, 1.2
    box(0.1, top, 1.65, h, "3 chart pictures\nweekly 104 bars\ndaily 126 bars\nhourly 140 bars", "#dde8f5")
    box(2.05, top, 1.75, h, "Renderer\n128-px black and\nwhite images (same\nfor training and\nthe dashboard)", "#dde8f5")
    box(4.1, top, 2.05, h, "Pretrained vision network\n(one of 7, shared by\nthe 3 views)\n" + ("frozen OR fine-tuned" if ft else "frozen"), "#fde9d4")
    box(6.45, top, 1.75, h, "Embeddings\n(384-768 numbers\nper picture)\n-> PCA to 128 each", "#fde9d4")
    box(8.5, top, 2.4, h, "Head, tuned on validation:\nridge / logistic regression\nor boosted trees (XGBoost),\none per target", "#e3f1e6")
    for a, b in ((1.75, 2.05), (3.8, 4.1), (6.15, 6.45), (8.2, 8.5)):
        arrow(a, top + h / 2, b, top + h / 2)
    # the 33 numbers join at the head
    box(6.45, 1.75, 1.75, 0.95, "33 numbers from the\nlast 126 daily bars", "#dde8f5")
    arrow(8.2, 2.22, 9.0, top)
    # bottom row, right to left: calibration -> outputs -> NO TRADE rule -> Monte Carlo
    bot, hb = 0.15, 1.15
    box(8.5, bot, 2.4, hb, "Calibration (isotonic,\nvalidation years):\n'30%' means 30%", "#e3f1e6")
    box(5.3, bot, 2.9, hb, "Outputs: 20-day volatility,\nchance of a 10% drawdown,\nchance to beat the median", "#e3f1e6")
    box(2.65, bot, 2.35, hb, "NO TRADE rule\n(fixed before the\ntest years)", "#f6d5d5")
    box(0.1, bot, 2.25, hb, "Monte Carlo:\n20-day price paths\n+ risk tier", "#f6d5d5")
    arrow(9.7, top, 9.7, bot + hb)
    for a, b in ((8.5, 8.2), (5.3, 5.0), (2.65, 2.35)):
        arrow(a, bot + hb / 2, b, bot + hb / 2)
    fig.savefig(path, bbox_inches="tight"); return path


def fig_timeline(path, splits_years):
    """Train / validation / test / holdout on a time axis. splits_years: [(name, first_year, last_year + 1, colour)]."""
    fig = _fig(11, 1.5); ax = fig.add_axes([0.03, 0.32, 0.94, 0.55])
    for name, a, b, c in splits_years:
        ax.barh(0, b - a, left=a, color=c, edgecolor="white", height=0.8)
        ax.text((a + b) / 2, 0, f"{name}\n{a}-{b - 1}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_yticks([]); ax.set_xlim(splits_years[0][1], splits_years[-1][2]); ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_xticks(range(splits_years[0][1], splits_years[-1][2] + 1, 2)); ax.tick_params(labelsize=8)
    fig.savefig(path, bbox_inches="tight"); return path


def fig_example(path, seed=3):
    """What the model sees: the renderer applied to SYNTHETIC prices (a random walk), so no licensed data appears in the report."""
    import pandas as pd
    import tfcv_core as C
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2021-01-04", periods=700)
    r = rng.standard_t(4, len(days)) * 0.012 + 0.0003
    close = 100 * np.exp(np.cumsum(r)); op = close * np.exp(rng.normal(0, 0.004, len(days)))
    hi = np.maximum(op, close) * np.exp(np.abs(rng.normal(0, 0.006, len(days)))); lo = np.minimum(op, close) * np.exp(-np.abs(rng.normal(0, 0.006, len(days))))
    daily = pd.DataFrame({"date": days, "open": op, "high": hi, "low": lo, "close": close, "volume": rng.integers(1e5, 1e6, len(days)).astype(float)})
    rows = []
    for d, o, c in zip(days[-40:], op[-40:], close[-40:]):
        path_ = np.linspace(o, c, 8) * np.exp(np.cumsum(rng.normal(0, 0.003, 8)))
        for j in range(7):
            ts = pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(hours=9 + j, minutes=30)
            a, b = path_[j], path_[j + 1]
            rows.append(dict(timestamp=ts.tz_convert("UTC"), open=a, high=max(a, b) * 1.002, low=min(a, b) * 0.998, close=b, volume=1e4))
    hourly = pd.DataFrame(rows)
    trip = C.render_at(daily, hourly, len(daily) - 1)
    fig = _fig(11, 1.9)
    for i, (v, lab) in enumerate((("W", "weekly: 104 bars"), ("D", "daily: 126 bars"), ("H", "hourly: 140 bars"))):
        ax = fig.add_axes([0.005 + i * 0.333, 0.02, 0.32, 0.8]); ax.imshow(trip[v], cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab, fontsize=9)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); return path


def fig_bars(path, names, values, ci=None, ref=None, ref_label="numbers only", title="", colors=None, xlabel=""):
    """Horizontal bars with optional CIs and a reference line."""
    fig = _fig(9, 0.32 * len(names) + 1.2); ax = fig.add_axes([0.32, 0.18, 0.64, 0.7])
    yy = np.arange(len(names)); v = np.asarray(values, float)
    err = None if ci is None else np.vstack([v - np.asarray(ci)[:, 0], np.asarray(ci)[:, 1] - v])
    ax.barh(yy, v, color=colors or BLUE, xerr=err, capsize=2)
    if ref is not None:
        ax.axvline(ref, color=RED, ls="--", lw=1.2, label=f"{ref_label} ({ref:.3f})"); ax.legend(fontsize=7, loc="lower right")
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8); ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.savefig(path, bbox_inches="tight"); return path


# ------------------------------------------------------------------------------------------------ the PDF
def _fonts():
    """A Unicode font (DejaVu ships with matplotlib) so ², ±, ≥ and – print correctly."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    try:
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(d, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(d, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(d, "DejaVuSans-Oblique.ttf")))
        from reportlab.lib.fonts import addMapping
        addMapping("DejaVu", 0, 0, "DejaVu"); addMapping("DejaVu", 1, 0, "DejaVu-Bold"); addMapping("DejaVu", 0, 1, "DejaVu-Oblique")
        addMapping("DejaVu", 1, 1, "DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _pct(x, d=0):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}%"


def build_pdf(R, out_path):
    """R: dict with the run's values (see make_pilot_report.py and the notebook's Stage 11 for how it is filled).
    Required keys: title_run, version, n_stocks, universe, n_rows, splits, cards, n_configs, chosen, metrics, hypotheses,
    no_trade, dashboard, figures. Optional: finetune, mc, vs_pilot, controls, holdout_note."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from PIL import Image as PILImage

    FONT, BOLD = _fonts()
    st = dict(
        title=ParagraphStyle("t", fontName=BOLD, fontSize=19, leading=23, textColor=colors.HexColor("#1d2533"), spaceAfter=4),
        sub=ParagraphStyle("s", fontName=FONT, fontSize=10, leading=13, textColor=colors.HexColor("#555555"), spaceAfter=10),
        h1=ParagraphStyle("h1", fontName=BOLD, fontSize=13.5, leading=17, textColor=colors.HexColor("#1f4e79"), spaceBefore=10, spaceAfter=5,
                          keepWithNext=1),
        body=ParagraphStyle("b", fontName=FONT, fontSize=9.6, leading=13.4, alignment=TA_LEFT, spaceAfter=5),
        bullet=ParagraphStyle("bl", fontName=FONT, fontSize=9.6, leading=13.4, leftIndent=12, bulletIndent=2, spaceAfter=2.5),
        cell=ParagraphStyle("c", fontName=FONT, fontSize=8.2, leading=10.4),
        cellb=ParagraphStyle("cb", fontName=BOLD, fontSize=8.2, leading=10.4, textColor=colors.white),
        box=ParagraphStyle("x", fontName=FONT, fontSize=9.8, leading=13.6, backColor=colors.HexColor("#eef4fb"), borderPadding=7,
                           borderColor=colors.HexColor("#b9cde6"), borderWidth=0.6, spaceBefore=4, spaceAfter=10),
        small=ParagraphStyle("sm", fontName=FONT, fontSize=7.8, leading=10, textColor=colors.HexColor("#555555")))
    E = lambda s: escape(str(s))
    P = lambda s, k="body": Paragraph(s, st[k])
    story = []

    def bullets(items):
        for it in items:
            story.append(Paragraph(it, st["bullet"], bulletText="•"))

    def table(rows, widths, header=True):
        data = [[Paragraph(str(c), st["cellb" if (header and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79") if header else colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d3e0")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fb")]),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        story.append(t); story.append(Spacer(1, 6))

    def image(key, width_in, caption=None, head=None):
        p = R.get("figures", {}).get(key)
        if not p or not os.path.exists(p):
            if head:
                story.append(P(head, "h1"))
            return
        w, h = PILImage.open(p).size
        items = ([P(head, "h1")] if head else []) + [Image(p, width=width_in * inch, height=width_in * inch * h / w)]
        if caption:
            items.append(Paragraph(caption, st["small"]))
        story.append(KeepTogether(items)); story.append(Spacer(1, 6))

    M, NT, H = R["metrics"], R["no_trade"], R["hypotheses"]
    sp = R["splits"]
    v3 = R.get("version") == "v3"

    # ---------------------------------------------------------------- page 1: summary
    story.append(P("TriFrame: can pictures of stock charts help predict risk?", "title"))
    story.append(P(f"Computer Vision with Deep Learning project · end-to-end report · {E(R['title_run'])} · generated "
                   f"{dt.date.today().isoformat()}", "sub"))
    images_add = "did not add a significant improvement" if not H["H1"][0] else "added a small but significant improvement"
    direction = ("there is no reliable 5-day direction signal" if not M.get("direction_usable") else "the 5-day direction signal is weak")
    story.append(P(
        f"<b>In one paragraph.</b> We asked whether pictures of a stock's price chart at three time scales (weekly, daily and hourly) "
        f"help a deep-learning model predict three things about the coming weeks: how much the price will swing (<b>volatility</b>), "
        f"whether it will fall 10% or more at some point (<b>drawdown</b>), and whether it will beat the typical stock over the next "
        f"5 days (<b>direction</b>). We tested {len([c for c in R['cards'] if c.get('family') != 'Control'])} pretrained vision models"
        f"{' (each both frozen and fine-tuned)' if R.get('finetune') else ''} on {R['n_stocks']} US stocks ({R['n_rows']:,} chart sets, "
        f"2001-2022), always against a simple baseline that uses only 33 numbers computed from the same prices. <b>The answer:</b> risk "
        f"is predictable (volatility R² {_f(M['test_V_R2'], 2)} on the unseen 2020-2022 test years), but the chart pictures {images_add} "
        f"over the numbers there, and {direction}, so the system's default decision is <b>NO TRADE</b>.", "box"))
    rows = [["Target", "Model chosen", "Test score (2020-2022)", "Sealed holdout (2023-2026)", "In plain words"]]
    for c in R["chosen"]:
        rows.append([c["target"], c["model"], c["test"], c.get("holdout", "not opened"), c["plain"]])
    table(rows, [1.45, 1.75, 1.15, 1.05, 2.1])
    if v3:
        story.append(P("<b>What is new in this version (v3)</b>"))
        bullets([f"Trained on <b>{R['n_stocks']} stocks</b> instead of 60 (the new sample contains the pilot's 60, so the two runs compare like for like).",
                 "Every pretrained model is <b>fine-tuned</b> on the charts as well as used frozen, and both compete under the same rule.",
                 "A <b>leaderboard</b> scores every model on six metrics per target, with confidence intervals and charts.",
                 "A <b>Monte Carlo simulation</b> of the next 20 trading days, calibrated on validation data and checked on the test years.",
                 "The dashboard analyses <b>up to four companies at once</b>, ranks them by risk and explains every number in plain words."])

    # ---------------------------------------------------------------- steps 1-2: data and splits
    story.append(P("Steps 1-2: the data and how it was split", "h1"))
    story.append(P(
        "Each example, a <b>chart set</b>, is one stock on one day: three black-and-white chart pictures drawn from its price history "
        "(104 weekly bars, 126 daily bars and 140 hourly bars) plus 33 ordinary numbers (returns, price ranges, volatility measures) "
        "computed from the last 126 daily bars. The labels are what happened next: the realised volatility over the next 20 trading "
        "days, whether the price traded 10% below the next day's open within 20 days, and whether the stock beat the median stock over "
        f"5 days. This run uses <b>{R['n_stocks']} of {R['universe']} US stocks</b>"
        + (" (chosen at random, and including the 60 stocks of the earlier pilot)" if R.get("pilot_included") else " (chosen at random)")
        + f", {R['n_rows']:,} chart sets in total."))
    story.append(P(
        "Time is never shuffled. Models learn from the <b>training</b> years, are tuned and selected on the <b>validation</b> years, and "
        "are scored once on the <b>test</b> years. 2023 onward is a sealed <b>holdout</b> that can be opened only once, by the owner, "
        "after everything is frozen. A 20-day gap at each boundary stops any label from leaking into the next period."))
    image("timeline", 6.9)
    table([["Period", "Years", "Chart sets", "Stocks", "Used for"]] + [[k, v[0], f"{v[1]:,}" if v[1] else "-", v[2] or "-", v[3]] for k, v in sp.items()],
          [1.0, 1.0, 1.0, 0.8, 3.7])
    image("example", 6.9, "What a model sees: the three pictures of one chart set (drawn here from synthetic prices). Each picture is "
                          "scaled to fill its panel, so it shows the shape of the moves, not their size in dollars.")

    # ---------------------------------------------------------------- step 3: pretrained models
    story.append(P("Step 3: the pretrained vision models", "h1"))
    story.append(P(
        "A <b>pretrained</b> model is a neural network that has already learned to recognise edges, shapes and textures from millions of "
        "ordinary photos. We reuse that knowledge instead of training a network from scratch. We compared seven well-known designs: "
        "convolutional networks (CNNs), which scan the picture with small filters, and transformers, which split it into patches that "
        "'pay attention' to each other. Three ResNet-18s with <b>random, untrained weights</b> serve as a control: if they did as well, "
        "pretraining would not be doing anything."))
    cols = ["Model", "Type", "Pretrained on", "Parameters (millions)"] + (["Fine-tuning"] if R.get("finetune") else [])
    rows = [cols]
    for c in R["cards"]:
        rows.append([c["model"], c["family"], c["pretrained_on"], _f(c.get("params_m"), 1)] + ([c.get("finetuning", "")] if R.get("finetune") else []))
    table(rows, [1.7, 0.9, 2.3, 1.1] + ([1.4] if R.get("finetune") else []))
    story.append(P(
        "<b>Frozen or fine-tuned?</b> 'Frozen' means the network's own weights are left exactly as they came from pretraining; we only "
        "read its output (an <b>embedding</b>, a list of a few hundred numbers summarising the picture) and train small models on top. "
        "'Fine-tuned' means the network's weights are also adjusted on our charts. "
        + (R["finetune"]["intro"] if R.get("finetune") else
           "<b>This run used frozen networks only; no fine-tuning was done.</b> Version 3 of the notebook adds fine-tuning (Stage 5F).")))

    # ---------------------------------------------------------------- step 4: architecture
    image("pipeline", 7.0, head="Step 4: the architecture")
    bullets(["The renderer draws the three charts as 128-pixel-high black-and-white pictures; the dashboard uses the very same code.",
             "One vision network turns each picture into an embedding; <b>PCA</b> compresses each embedding to 128 numbers "
             "(fitted on training years only).",
             "The three compressed embeddings and the 33 numbers go into a <b>head</b>: ridge or logistic regression, or "
             "gradient-boosted decision trees (XGBoost). One head per target.",
             "Probabilities are <b>calibrated</b> on the validation years so that '30%' really means 30%.",
             "A <b>NO TRADE rule</b>, fixed before the test years were opened, turns the outputs into a decision, and a "
             "<b>Monte Carlo simulation</b> turns the volatility forecast into a fan of possible price paths."])

    # ---------------------------------------------------------------- step 5: training and tuning
    story.append(P("Step 5: training and hyper-parameter tuning", "h1"))
    story.append(P(
        f"We tried <b>{R['n_configs']:,} model configurations</b> and logged every one: ridge strengths, logistic regularisation, tree "
        "depth, learning rate and the number of trees (by early stopping), for each target, each vision network and each combination of "
        "inputs (images only, numbers + all three timeframes, numbers + one timeframe). Every choice was made on the validation years "
        "only, and all preprocessing (gap filling, scaling, PCA) was fitted on the training years only. <b>The selection rule was written "
        "down before the test years were opened:</b> per target the best validation score wins, and a model that uses chart pictures "
        "wins only if its improvement over the numbers-only model is clearly above zero (its 95% confidence interval excludes 0)."))

    # ---------------------------------------------------------------- step 6: results
    story.append(P("Step 6: results - which models won, and how good they are", "h1"))
    rows = [["Target", "Inputs", "Vision network", "Head", "Validation", "Test (95% CI)"]]
    for c in R["chosen"]:
        rows.append([c["target"], c["inputs"], c["network"], c["head"], c["val"], c["test_ci"]])
    table(rows, [1.2, 1.3, 1.45, 1.2, 0.9, 1.25])
    image("leaderboard", 7.0, R.get("leaderboard_caption"))
    hyp = [f"<b>Do chart pictures add to the numbers? {'Yes' if H['H1'][0] else 'No'}.</b> {H['H1'][1]}",
           f"<b>Do the vision models differ from each other? {'Yes, a little' if H['H2'][0] else 'No'}.</b> {H['H2'][1]}",
           f"<b>Does waiting for confident direction calls help? {'Yes' if H['H3'][0] else 'No'}.</b> {H['H3'][1]}"]
    if R.get("controls"):
        hyp.append(f"<b>Sanity checks.</b> {R['controls']}")
    bullets(hyp)

    # ---------------------------------------------------------------- step 7: fine-tuning
    story.append(P("Step 7: did we fine-tune, and did it help?", "h1"))
    if R.get("finetune"):
        F = R["finetune"]
        story.append(P(F["verdict"]))
        if F.get("rows"):
            table([["Model", "Frozen R²", "Fine-tuned R²", "Change (95% CI)", "Training time"]] + F["rows"], [2.0, 1.0, 1.1, 1.7, 1.2])
        image("finetune", 7.0)
    else:
        story.append(P(
            "<b>No.</b> In this run every vision network was <b>frozen</b>: the pretrained weights were used as they are, and only the small "
            "heads on top were trained. Fine-tuning a network means continuing its training on our own charts so its features adapt to "
            "price patterns. Version 3 of the notebook does exactly that for all seven pretrained networks (training years only, early "
            "stopping on validation, no image augmentation because flipping or cropping a chart changes its meaning) and lets the "
            "fine-tuned networks compete with the frozen ones under the same selection rule, so the question 'is fine-tuning worth it?' "
            "gets a measured answer, with confidence intervals and the extra training time."))

    # ---------------------------------------------------------------- step 8: decision + Monte Carlo
    story.append(P("Step 8: from predictions to a decision, and the Monte Carlo simulation", "h1"))
    story.append(P(
        f"<b>NO TRADE rule.</b> A stock is flagged as high risk when its predicted volatility is at least {_pct(NT['vol_q80'])} a year "
        f"or its drawdown chance at least {_pct(NT['pdd_q80'])} (the top 20% of the model's readings in the validation years). "
        f"On the test years the flag picked out {_pct(NT['share_flagged'])} of cases, and those cases really were riskier: realised "
        f"volatility {_pct(NT['realised_vol_flagged'])} vs {_pct(NT['realised_vol_other'])} a year, and 10% drawdowns "
        f"{_pct(NT['dd10_rate_flagged'])} vs {_pct(NT['dd10_rate_other'])} of the time. A direction call would only ever be shown if "
        f"the direction model had passed its validation <i>and</i> test checks; "
        f"{'it did not, so every decision is NO TRADE' if not M.get('direction_usable') else 'it passed, but the edge is small'}."))
    if R.get("mc"):
        mc = R["mc"]
        story.append(P(
            f"<b>Monte Carlo.</b> For each stock the dashboard simulates thousands of 20-day price paths. Each path draws a volatility "
            f"from the model's forecast and its measured error on validation data, then fat-tailed daily moves (Student-t with "
            f"{mc['t_df']:g} degrees of freedom, chosen on validation) with exactly that volatility; daily lows reach below the closes by "
            f"a range also fitted on validation. There is no drift, because the direction model has no edge. "
            + (f"By default the paths are reweighted, as little as possible and without changing the average return, so the share that "
               f"touches a 10% drawdown equals the calibrated drawdown model's chance. " if mc.get("tilt_default") else "")
            + f"<b>Check on the test years:</b> the simulation's 90% band contained {_pct(mc['test_cov90'])} of the real 20-day outcomes "
              f"(validation: {_pct(mc['val_cov90'])}) and its 50% band {_pct(mc['test_cov50'])}."))
        image("mc", 7.0)
    else:
        story.append(P("<b>Monte Carlo.</b> Added in version 3: a simulation of 20-day price paths driven by the volatility forecast "
                       "and its measured error, calibrated on validation data and checked on the test years."))

    # ---------------------------------------------------------------- step 9: dashboard
    story.append(P("Step 9: the dashboard - which models it uses", "h1"))
    D = R["dashboard"]
    multi = v3 or D.get("multi")
    story.append(P(
        "The <i>Computer Vision with Deep Learning Project Dashboard</i> runs the frozen models from the private model repository. "
        + ("You choose 1 to 4 companies and upload three TradingView screenshots (1W, 1D, 1H) for each. " if multi else
           "You upload three TradingView screenshots of one stock (1W, 1D, 1H). ")
        + "The app reads the ticker and the last bar of every screenshot, confirms them against market data, rebuilds exactly the "
          "inputs the models were trained on, and shows the outputs with charts and a plain-language explanation"
        + (", a risk tier, a Monte Carlo fan of price paths and, for several companies, a ranking from riskiest to safest." if multi else ".")
        + (f" Live at {E(D['url'])}." if D.get("url") else "")))
    rows = [["Output", "Inputs", "Vision network", "Head", "Test score"]] + [[r["Output"], r["Inputs"], r["Vision network"], r["Head"], r["Test score"]] for r in D["models"]]
    table(rows, [1.9, 1.5, 1.9, 1.3, 0.75])

    if R.get("vs_pilot"):
        n0 = len(story)
        story.append(P(f"Does more data help? {R['n_stocks']} stocks vs the 60-stock pilot", "h1"))
        story.append(P(R["vs_pilot"]["text"]))
        table([["Target", "Metric", "Pilot model (60 stocks)", "This model, same 60 stocks", "This model, all stocks"]] + R["vs_pilot"]["rows"],
              [1.6, 0.8, 1.5, 1.7, 1.5])
        story[n0:] = [KeepTogether(story[n0:])]

    # ---------------------------------------------------------------- limits + glossary
    story.append(P("Limits", "h1"))
    bullets([E(x) for x in R.get("limits", [])])
    story.append(P("Glossary", "h1"))
    table([["Term", "Meaning"],
           ["R²", "Share of the ups and downs in the target that the model explains. 1 = perfect, 0 = no better than always guessing the average."],
           ["AUC", "Chance that the model ranks a case where the event happened above one where it did not. 0.5 = coin flip, 1 = perfect."],
           ["Brier skill score", "How much better the probabilities are than always predicting the average rate (0 = no better)."],
           ["Calibration", "Whether stated chances come true as often as stated (30% events happen about 30% of the time)."],
           ["Embedding / PCA", "The numbers a vision network outputs for a picture / a way to compress them to the most informative 128."],
           ["Month-block bootstrap", "Confidence intervals made by resampling whole calendar months, because stocks move together within a month."],
           ["Holdout", "The most recent data (2023 onward), sealed until all choices are frozen and then opened once."]],
          [1.5, 5.6])

    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont(FONT, 7.5); canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(0.6 * inch, 0.45 * inch, f"TriFrame CV · {R['title_run']} · research output, not investment advice")
        canvas.drawRightString(7.9 * inch, 0.45 * inch, f"page {doc.page}"); canvas.restoreState()
    doc = SimpleDocTemplate(out_path, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch,
                            bottomMargin=0.7 * inch, title="TriFrame CV project report", author="Krishna Harsha Tangirala")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path
