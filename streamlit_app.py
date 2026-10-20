"""Computer Vision with Deep Learning Project Dashboard (TriFrame): Streamlit app.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file streamlit_app.py. Secrets (App settings > Secrets):
                   HF_TOKEN   = "hf_..."   (a READ-only Hugging Face token for the private model repo)
                   MODEL_REPO = "KrishnaHarshaTangirala/triframe-cv-model"   (optional; this is the default)
"""
import os

import streamlit as st

import dashboard_charts as CH
import dashboard_logic as DL

st.set_page_config(page_title=DL.TITLE, page_icon="📈", layout="wide")
MAX_COMPANIES = 4
SLOT_LABELS = (("1W", "1 Week chart (1W)"), ("1D", "1 Day chart (1D)"), ("1H", "1 Hour chart (1H)"))


def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:                       # no secrets file (local run) or key missing
        return os.environ.get(name, default)


@st.cache_resource(show_spinner="Loading the model bundle...")
def get_bundle():
    return DL.load_bundle(token=secret("HF_TOKEN"), model_repo=secret("MODEL_REPO"))


def md(text):
    """Streamlit markdown reads $...$ as maths; prices need a literal dollar sign."""
    return text.replace("$", "\\$")


def plot(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


st.title(DL.TITLE)
try:
    B = get_bundle()
except Exception as e:                      # private repo without a token, wrong repo name, no network
    st.error("The model bundle could not be loaded. If you own this app, add a read-only Hugging Face token as the secret "
             f"HF_TOKEN in the app settings. ({type(e).__name__})")
    st.stop()
cfg = B.cfg

st.markdown(DL.describe(B))
st.markdown("**Models in use** (chosen by the pre-registered rule on validation data, then scored once on the "
            f"{DL._years(cfg, 'test')} test years):")
st.dataframe(DL.models_in_use(cfg), hide_index=True, width="stretch")
if cfg.get("leaderboard_test"):
    with st.expander("How every candidate model scored on the test years (why these models)"):
        for tab_, (t, name) in zip(st.tabs(["Volatility", "Drawdown", "Direction"]),
                                   (("V", "Volatility R²"), ("D10", "Drawdown AUC"), ("O5", "Direction AUC"))):
            with tab_:
                fig_ = CH.leaderboard_bars(cfg["leaderboard_test"], t, f"{name} on the test years")
                if fig_ is not None:
                    st.plotly_chart(fig_, width="stretch", key=f"lb_{t}", config={"displaylogo": False})
        if (cfg.get("finetune") or {}).get("verdict"):
            st.caption(cfg["finetune"]["verdict"])


@st.cache_data(show_spinner=False, max_entries=128)
def simulate(ticker, asof, p0, logrv20, p_dd10, n_paths, match_dd, seed):
    return DL.run_mc(dict(p0=p0, out=dict(logrv20=logrv20, p_dd10=p_dd10)), cfg, n_paths, match_dd, seed)


# ------------------------------------------------------------------------------------------------ inputs
st.subheader("1. Upload the screenshots")
n = st.radio("How many companies do you want to analyse at once?", list(range(1, MAX_COMPANIES + 1)), horizontal=True,
             key="n_companies")
inputs = []
for i, tab in enumerate(st.tabs([f"Company {i + 1}" for i in range(n)])):
    with tab:
        ups = {}
        for col, (slot, label) in zip(st.columns(3), SLOT_LABELS):
            with col:
                up = st.file_uploader(label, type=["png", "jpg", "jpeg"], key=f"up_{i}_{slot}")
                if up is not None:
                    st.image(up, width="stretch")
                ups[slot] = up
        tk = st.text_input("Ticker (optional; detected from the screenshots if left empty)", key=f"ticker_{i}")
        inputs.append((ups, tk))

with st.expander("Monte Carlo settings"):
    c1, c2, c3, c4 = st.columns(4)
    n_paths = c1.select_slider("Simulated paths", options=[1000, 2000, 5000, 10000, 20000], value=5000, key="mc_paths")
    match_dd = c2.toggle("Match the drawdown model", value=bool((cfg.get("mc") or {}).get("tilt_default", True)), key="mc_match",
                         help="Reweights the simulated paths so the share that touches a 10% drawdown equals the calibrated "
                              "drawdown model's probability.")
    n_show = c3.slider("Paths drawn on the chart", 0, 100, 40, step=10, key="mc_show")
    seed = int(c4.number_input("Random seed", 0, 10_000, 7, key="mc_seed"))

if st.button("Analyse", type="primary"):
    todo = [(i, ups, tk) for i, (ups, tk) in enumerate(inputs) if any(v is not None for v in ups.values()) or tk.strip()]
    if not todo:
        st.warning("Upload the three screenshots (or type a ticker) for at least one company, then press Analyse.")
        st.session_state.pop("results", None)
    else:
        bar = st.progress(0.0, text="Starting...")
        results = []
        for j, (i, ups, tk) in enumerate(todo):
            bar.progress(j / len(todo), text=f"Company {i + 1} of {len(todo)}: reading the screenshots, rebuilding the model inputs "
                                             "and predicting (the first run also downloads the vision models, about a minute)...")
            res = DL.analyse_company(B, ups, tk)
            res["slot"] = i + 1
            results.append(res)
        bar.empty()
        st.session_state["results"] = results


# ------------------------------------------------------------------------------------------------ results
def company_view(r, sim, idx):
    out, color = r["out"], CH.PALETTE[idx % 4]
    ex = DL.explain(r, cfg, sim)
    rp = ex["risk"]
    st.markdown(f"### {r['ticker']} — analysis as of {r['asof']}")
    st.caption(md(f"Company {r['slot']} · ticker confirmed: {r['how']} · last close ${r['p0']:,.2f}"))
    st.markdown(f"<div style='padding:12px 16px;border-radius:8px;background:{rp['color']}22;border-left:6px solid {rp['color']};"
                f"font-size:17px'><b>Decision: {out['decision']}</b> &nbsp;·&nbsp; <b>Risk tier: {rp['tier']}</b> "
                f"(risk score {rp['score']:.2f}; 1.00 = the NO TRADE risk line)</div>", unsafe_allow_html=True)
    for reason in out["reasons"]:
        st.markdown(f"- {reason}")
    k = st.columns(4)
    k[0].metric("Expected volatility", f"{out['vol_ann']:.0%} a year", help=f"likely {out['vol_ann_lo']:.0%}–{out['vol_ann_hi']:.0%}")
    k[1].metric("Chance of a 10%+ drawdown", f"{out['p_dd10']:.0%}", help="within the next 20 trading days")
    k[2].metric("Chance to beat the median", f"{out['p_out5']:.0%}", help="over the next 5 trading days")
    k[3].metric("Risk tier", rp["tier"], help="the larger of volatility and drawdown chance, each divided by its high-risk threshold")
    c = st.columns(2)
    with c[0]:
        plot(CH.vol_gauge(out, cfg["rule"]), f"g{idx}")
        plot(CH.direction_bar(out, cfg["rule"]["direction_usable"]), f"o{idx}")
    with c[1]:
        plot(CH.dd_donut(out, cfg.get("base_rates", {}).get("D10"), cfg["rule"]), f"d{idx}")
        plot(CH.risk_components(rp), f"r{idx}")
    for s in ex["sections"][:3]:
        with st.container(border=True):
            st.markdown(f"#### {md(s['title'])}")
            for a, b in s["items"]:
                st.markdown(md(f"**{a}:** {b}"))
    if sim is not None:
        with st.container(border=True):
            s = ex["sections"][3]
            st.markdown(f"#### {md(s['title'])}")
            c1, c2 = st.columns([3, 2])
            with c1:
                plot(CH.mc_fan(dict(sim, sample_paths=sim["sample_paths"][:n_show]), r["ticker"], color), f"f{idx}")
            with c2:
                plot(CH.mc_hist(sim, color), f"h{idx}")
            m = st.columns(4)
            m[0].metric("Middle 90% of 20-day outcomes", f"{sim['ret_q05']:+.0%} to {sim['ret_q95']:+.0%}")
            m[1].metric("Average loss in the worst 5%", f"{-sim['es95']:+.1%}")
            m[2].metric("Paths touching a 10% drawdown", f"{sim['p_dd']:.0%}", help=f"raw simulation {sim['p_dd_sim']:.0%}")
            m[3].metric("Paths ending higher", f"{sim['p_up']:.0%}")
            for a, b in s["items"]:
                st.markdown(md(f"**{a}:** {b}"))
    st.markdown("#### Summary")
    st.dataframe(ex["summary"], hide_index=True, width="stretch")
    st.info(md(f"**Primary takeaway:** {ex['takeaway']}"))
    with st.expander("What the model sees, and the screenshot checks"):
        if r["charts"] is not None:
            st.image(r["charts"], caption="The three charts rebuilt from market data, exactly as the model was trained on them", width="stretch")
        if r["checks"] is not None:
            st.dataframe(r["checks"], hide_index=True, width="stretch")


def comparison_view(ok, sims):
    tab, txt = DL.compare(ok, cfg, sims)
    st.markdown(md(txt))
    show = tab.copy()
    for col in ("Volatility (a year)", "Chance of 10%+ drawdown", "Chance to beat median (5 days)"):
        show[col] = show[col].map(lambda x: f"{x:.0%}")
    colors = {n: c for n, _, c in DL.TIERS}
    show["Risk score"] = show["Risk score"].map(lambda x: f"{x:.2f}")
    st.dataframe(show.style.map(lambda v: f"background-color:{colors.get(v, '')}33;font-weight:600" if v in colors else "",
                                subset=["Risk tier"]), hide_index=True, width="stretch")
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(CH.risk_map(ok, cfg["rule"]), "cmp_map")
    with c2:
        plot(CH.tier_donut(tab), "cmp_tiers")
    plot(CH.compare_bars(ok, cfg["rule"], cfg.get("base_rates", {}).get("D10")), "cmp_bars")
    c1, c2 = st.columns([3, 2])
    with c1:
        plot(CH.compare_fan(ok, sims), "cmp_fan")
    with c2:
        plot(CH.compare_box(ok, sims), "cmp_box")
    st.caption("Risk score = the larger of (expected volatility ÷ its high-risk threshold) and (drawdown chance ÷ its threshold). "
               "1.00 is the NO TRADE risk line: the top 20% of the model's readings in the validation years. Tiers: Low < 0.75 ≤ "
               "Moderate < 1.00 ≤ High < 1.50 ≤ Very high.")


results = st.session_state.get("results")
if results:
    st.subheader("2. Results")
    ok = [r for r in results if r["ok"]]
    for r in results:
        if not r["ok"]:
            st.error(f"Company {r['slot']}: {r['message']}")
            if r.get("checks") is not None:
                st.dataframe(r["checks"], hide_index=True, width="stretch")
    sims = [simulate(r["ticker"], str(r["asof"]), r["p0"], r["out"]["logrv20"], r["out"]["p_dd10"], n_paths, match_dd, seed)
            for r in ok]
    if ok:
        names = (["📊 Comparison"] if len(ok) > 1 else []) + [f"{i + 1}. {r['ticker']}" for i, r in enumerate(ok)]
        tabs = st.tabs(names)
        if len(ok) > 1:
            with tabs[0]:
                comparison_view(ok, sims)
        for i, (r, s) in enumerate(zip(ok, sims)):
            with tabs[i + (1 if len(ok) > 1 else 0)]:
                company_view(r, s, i)
    st.markdown(DL.LIMITS)
