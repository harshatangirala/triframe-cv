"""Plotly figures for the Streamlit dashboard (one company, and up to four side by side)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]          # companies 1-4
RED, DARK_RED, AMBER, GREEN, GREY = "#e53935", "#b71c1c", "#f39c12", "#2e9d5b", "#8a8f98"


def _rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _layout(fig, height=300, title=None, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=(76 if "<br>" in title else 56) if title else 20, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13), title=dict(text=title, x=0.02, font=dict(size=15)) if title else None,
                      **kw)
    return fig


# ------------------------------------------------------------------------------------------------ one company
def vol_gauge(out, rule):
    """Expected volatility on a gauge whose colours are the risk tiers (green < 0.75x, amber < 1x, red < 1.5x the high-risk line)."""
    v, lo, hi = 100 * out["vol_ann"], 100 * out["vol_ann_lo"], 100 * out["vol_ann_hi"]
    q80 = 100 * rule["vol_ann_q80"]
    top = 10 * np.ceil(max(1.15 * hi, 1.8 * q80, 40) / 10)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={"suffix": "%", "valueformat": ".0f", "font": {"size": 40}},
        gauge={"axis": {"range": [0, top], "ticksuffix": "%"}, "bar": {"color": "#1f4e79", "thickness": 0.28},
               "steps": [{"range": [0, 0.75 * q80], "color": _rgba(GREEN, 0.35)}, {"range": [0.75 * q80, q80], "color": _rgba(AMBER, 0.4)},
                         {"range": [q80, 1.5 * q80], "color": _rgba(RED, 0.35)}, {"range": [1.5 * q80, top], "color": _rgba(DARK_RED, 0.45)}],
               "threshold": {"line": {"color": RED, "width": 4}, "thickness": 0.85, "value": q80}}))
    return _layout(fig, 300, f"Expected volatility (a year)<br><sup>likely {lo:.0f}%–{hi:.0f}% · red line: high-risk threshold {q80:.0f}%</sup>")


def dd_donut(out, base=None, rule=None):
    """Chance of a 10%+ drawdown as a pie (donut)."""
    p = float(out["p_dd10"])
    fig = go.Figure(go.Pie(values=[p, 1 - p], labels=["10%+ drawdown", "No 10% drawdown"], hole=0.64, sort=False,
                           direction="clockwise", marker=dict(colors=[RED, _rgba(GREEN, 0.55)]), textinfo="none",
                           hovertemplate="%{label}: %{percent}<extra></extra>"))
    fig.add_annotation(text=f"<b>{p:.0%}</b>", x=0.5, y=0.5, showarrow=False, font=dict(size=30))
    sub = []
    if base:
        sub.append(f"typical rate {base:.0%}")
    if rule:
        sub.append(f"high risk from {rule['p_dd10_q80']:.0%}")
    return _layout(fig, 300, "Chance of a 10%+ drawdown (20 days)" + (f"<br><sup>{' · '.join(sub)}</sup>" if sub else ""),
                   showlegend=True, legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))


def direction_bar(out, usable=False):
    """Chance of beating vs lagging the median stock over 5 days, with the 50% coin-flip line."""
    q = float(out["p_out5"])
    fig = go.Figure()
    fig.add_bar(y=["5 days"], x=[100 * q], orientation="h", name="Beats the median stock", marker_color="#4C78A8",
                text=[f"{q:.0%}"], textposition="inside", hovertemplate="beats: %{x:.0f}%<extra></extra>")
    fig.add_bar(y=["5 days"], x=[100 * (1 - q)], orientation="h", name="Lags the median stock", marker_color=_rgba(GREY, 0.55),
                text=[f"{1 - q:.0%}"], textposition="inside", hovertemplate="lags: %{x:.0f}%<extra></extra>")
    fig.add_vline(x=50, line_dash="dash", line_color=RED, annotation_text="coin flip (50%)", annotation_position="top")
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 100], ticksuffix="%"), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    return _layout(fig, 300, "Beat the median stock over 5 days?<br><sup>" + ("weak signal only" if usable else "no reliable edge: treat as a coin flip") + "</sup>")


def risk_components(rp):
    """The two parts of the risk score against the NO TRADE line (1.0)."""
    vals = [rp["vol_ratio"], rp["dd_ratio"]]
    cols = [DARK_RED if x >= 1.5 else RED if x >= 1 else AMBER if x >= 0.75 else GREEN for x in vals]
    fig = go.Figure(go.Bar(x=["Volatility", "Drawdown chance"], y=vals, marker_color=cols, text=[f"{x:.2f}x" for x in vals],
                           textposition="outside", hovertemplate="%{x}: %{y:.2f}x the high-risk threshold<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dash", line_color=RED)
    fig.update_yaxes(range=[0, max(1.6, 1.25 * max(vals))], title="x the high-risk threshold")
    return _layout(fig, 300, f"Why the tier is {rp['tier']}<br><sup>each reading ÷ its high-risk threshold · dashed line = NO TRADE risk line</sup>")


def mc_fan(sim, ticker, color=PALETTE[0]):
    """Simulated price paths: 5-95% and 25-75% bands, median, sample paths, the start price and the 10% drawdown level."""
    d, f, p0 = sim["days"], sim["fan"], sim["p0"]
    fig = go.Figure()
    for path in sim["sample_paths"]:
        fig.add_scatter(x=d, y=path, mode="lines", line=dict(width=0.7, color=_rgba(color, 0.25)), hoverinfo="skip", showlegend=False)
    for lo_i, hi_i, a, nm in ((0, 4, 0.16, "90% of paths"), (1, 3, 0.30, "50% of paths")):
        fig.add_scatter(x=d, y=f[hi_i], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=d, y=f[lo_i], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=_rgba(color, a), name=nm,
                        hovertemplate="day %{x}: %{y:$,.2f}<extra>" + nm + "</extra>")
    fig.add_scatter(x=d, y=f[2], mode="lines", line=dict(width=2.5, color=color), name="median path",
                    hovertemplate="day %{x}: %{y:$,.2f}<extra>median</extra>")
    fig.add_hline(y=p0, line_dash="dot", line_color=GREY, annotation_text=f"start ${p0:,.2f}", annotation_position="top left")
    fig.add_hline(y=0.9 * p0, line_dash="dash", line_color=RED, annotation_text="10% drawdown level", annotation_position="bottom left")
    fig.update_xaxes(title="trading days ahead", dtick=5)
    fig.update_yaxes(title="price ($)", tickprefix="$")
    return _layout(fig, 380, f"{ticker}: {sim['n_paths']:,} simulated paths for the next 20 trading days",
                   legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))


def mc_hist(sim, color=PALETTE[0]):
    """Distribution of simulated 20-day returns (weighted when the paths are matched to the drawdown model)."""
    r = 100 * sim["ret"]
    edges = np.linspace(np.percentile(r, 0.5), np.percentile(r, 99.5), 41)
    h, e = np.histogram(np.clip(r, edges[0], edges[-1]), bins=edges, weights=sim["w"])
    mid = 0.5 * (e[1:] + e[:-1])
    cols = [_rgba(RED, 0.75) if m <= 100 * sim["ret_q05"] else _rgba(color, 0.75) for m in mid]
    fig = go.Figure(go.Bar(x=mid, y=100 * h, marker_color=cols, width=(e[1] - e[0]) * 0.95,
                           hovertemplate="%{x:.1f}%: %{y:.1f}% of paths<extra></extra>"))
    fig.add_vline(x=0, line_color=GREY, line_dash="dot")
    fig.add_vline(x=100 * sim["ret_q05"], line_color=RED, line_dash="dash", annotation_text=f"worst 5%: {sim['ret_q05']:+.0%}",
                  annotation_position="top left")
    fig.add_vline(x=100 * sim["ret_q95"], line_color=GREEN, line_dash="dash", annotation_text=f"best 5%: {sim['ret_q95']:+.0%}",
                  annotation_position="top right")
    fig.update_xaxes(title="20-day return", ticksuffix="%")
    fig.update_yaxes(title="share of paths", ticksuffix="%")
    return _layout(fig, 380, "Where the simulated paths end after 20 days")


# ------------------------------------------------------------------------------------------------ several companies
def risk_map(rows, rule):
    """Each company on volatility (x) and drawdown chance (y); the shaded area is above the NO TRADE risk line."""
    xs = [100 * r["out"]["vol_ann"] for r in rows]; ys = [100 * r["out"]["p_dd10"] for r in rows]
    qx, qy = 100 * rule["vol_ann_q80"], 100 * rule["p_dd10_q80"]
    xmax, ymax = max(1.4 * qx, 1.15 * max(xs)), max(1.4 * qy, 1.15 * max(ys))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=qx, x1=xmax, y0=0, y1=ymax, fillcolor=_rgba(RED, 0.08), line_width=0)
    fig.add_shape(type="rect", x0=0, x1=qx, y0=qy, y1=ymax, fillcolor=_rgba(RED, 0.08), line_width=0)
    fig.add_vline(x=qx, line_dash="dash", line_color=RED); fig.add_hline(y=qy, line_dash="dash", line_color=RED)
    for i, r in enumerate(rows):
        fig.add_scatter(x=[xs[i]], y=[ys[i]], mode="markers+text", text=[r["ticker"]], textposition="top center", name=r["ticker"],
                        marker=dict(size=22, color=PALETTE[i % 4], line=dict(width=1, color="white")),
                        hovertemplate=f"{r['ticker']}: volatility %{{x:.0f}}%, drawdown chance %{{y:.0f}}%<extra></extra>")
    fig.add_annotation(text="shaded = above the NO TRADE risk line", x=xmax, y=0, xanchor="right", yanchor="bottom", showarrow=False,
                       font=dict(color=RED, size=12))
    fig.update_xaxes(title="expected volatility (a year)", ticksuffix="%", range=[0, xmax])
    fig.update_yaxes(title="chance of a 10%+ drawdown", ticksuffix="%", range=[0, ymax])
    return _layout(fig, 400, "Risk map: further up and right = riskier", showlegend=False)


def compare_bars(rows, rule, base=None):
    """Grouped bars: volatility, drawdown chance and chance to beat the median, one colour per company."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Expected volatility (a year)", "Chance of a 10%+ drawdown", "Chance to beat the median (5 days)"))
    for i, r in enumerate(rows):
        o = r["out"]
        for j, val in enumerate((o["vol_ann"], o["p_dd10"], o["p_out5"])):
            fig.add_bar(x=[r["ticker"]], y=[100 * val], marker_color=PALETTE[i % 4], text=[f"{val:.0%}"], textposition="outside",
                        name=r["ticker"], showlegend=(j == 0), row=1, col=j + 1, hovertemplate="%{x}: %{y:.0f}%<extra></extra>")
    fig.add_hline(y=100 * rule["vol_ann_q80"], line_dash="dash", line_color=RED, row=1, col=1)
    fig.add_hline(y=100 * rule["p_dd10_q80"], line_dash="dash", line_color=RED, row=1, col=2)
    if base:
        fig.add_hline(y=100 * base, line_dash="dot", line_color=GREY, row=1, col=2)
    fig.add_hline(y=50, line_dash="dash", line_color=GREY, row=1, col=3)
    for c, top in ((1, max(1.25 * max(100 * r["out"]["vol_ann"] for r in rows), 120 * rule["vol_ann_q80"])),
                   (2, max(1.25 * max(100 * r["out"]["p_dd10"] for r in rows), 120 * rule["p_dd10_q80"])), (3, 100)):
        fig.update_yaxes(range=[0, top], ticksuffix="%", row=1, col=c)
    fig.update_annotations(font_size=13)
    return _layout(fig, 360, None, legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))


def compare_fan(rows, sims):
    """Simulated paths of every company on one scale (start = 100): median and 5-95% band."""
    fig = go.Figure()
    for i, (r, s) in enumerate(zip(rows, sims)):
        c = PALETTE[i % 4]; f = 100 * s["fan"] / s["p0"]
        fig.add_scatter(x=s["days"], y=f[4], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=s["days"], y=f[0], mode="lines", line=dict(width=0), fill="tonexty", fillcolor=_rgba(c, 0.15),
                        hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=s["days"], y=f[2], mode="lines", line=dict(width=2.5, color=c), name=f"{r['ticker']} (median, 90% band)",
                        hovertemplate=f"{r['ticker']} day %{{x}}: %{{y:.1f}}<extra></extra>")
        for k in (0, 4):
            fig.add_scatter(x=s["days"], y=f[k], mode="lines", line=dict(width=1, color=c, dash="dot"), hoverinfo="skip", showlegend=False)
    fig.add_hline(y=100, line_dash="dot", line_color=GREY); fig.add_hline(y=90, line_dash="dash", line_color=RED,
                                                                          annotation_text="10% below the start", annotation_position="bottom left")
    fig.update_xaxes(title="trading days ahead", dtick=5); fig.update_yaxes(title="price, start = 100")
    return _layout(fig, 400, "Monte Carlo side by side: the wider the band, the riskier the stock",
                   legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))


def compare_box(rows, sims, seed=0):
    """Distribution of simulated 20-day returns per company (violin)."""
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    for i, (r, s) in enumerate(zip(rows, sims)):
        draw = rng.choice(s["ret"], size=min(4000, len(s["ret"])), p=s["w"] / s["w"].sum())
        fig.add_violin(y=100 * draw, name=r["ticker"], line_color=PALETTE[i % 4], box_visible=True, meanline_visible=False,
                       points=False, hoverinfo="skip")
    lo_ = min(np.percentile(100 * s["ret"], 0.5) for s in sims); hi_ = max(np.percentile(100 * s["ret"], 99.5) for s in sims)
    fig.add_hline(y=0, line_dash="dot", line_color=GREY); fig.add_hline(y=-10, line_dash="dash", line_color=RED)
    fig.update_yaxes(title="20-day return", ticksuffix="%", range=[lo_, hi_])
    return _layout(fig, 400, "Spread of simulated 20-day returns", showlegend=False)


def leaderboard_bars(rows, target, title):
    """Every candidate model's test score for one target (numbers + images, 95% CI); ★ = the model chosen on validation."""
    kinds = {"frozen": "#3b6ea5", "fine-tuned": "#e67e22", "reference": RED, "control": GREY}
    sub = sorted((r for r in rows if r["target"] == target and r["condition"] in ("numbers + images", "numbers only")), key=lambda r: r["value"])
    if not sub:
        return None
    fig = go.Figure(go.Bar(
        x=[r["value"] for r in sub], y=[r["model"] + (" ★" if r["chosen"] else "") for r in sub], orientation="h",
        marker_color=[kinds.get(r["kind"], GREY) for r in sub],
        error_x=dict(type="data", symmetric=False, array=[r["ci"][1] - r["value"] for r in sub], arrayminus=[r["value"] - r["ci"][0] for r in sub]),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    lo, hi = min(r["ci"][0] for r in sub), max(r["ci"][1] for r in sub)
    fig.update_xaxes(range=[lo - 0.01, hi + 0.01], title=sub[0]["metric"])
    return _layout(fig, 60 + 26 * len(sub), title + "<br><sup>blue = frozen, orange = fine-tuned, red = numbers only · ★ = chosen on validation</sup>")


def tier_donut(tab):
    """How many of the analysed companies fall in each risk tier."""
    order = [("Very high", DARK_RED), ("High", RED), ("Moderate", AMBER), ("Low", GREEN)]
    counts = [(n, c, int((tab["Risk tier"] == n).sum())) for n, c in order]
    counts = [x for x in counts if x[2] > 0]
    fig = go.Figure(go.Pie(values=[x[2] for x in counts], labels=[x[0] for x in counts], hole=0.6, sort=False,
                           marker=dict(colors=[x[1] for x in counts]), textinfo="label+value",
                           hovertemplate="%{label}: %{value} of " + str(len(tab)) + "<extra></extra>"))
    return _layout(fig, 300, "Risk tiers of the companies", showlegend=False)
