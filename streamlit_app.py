"""Computer Vision with Deep Learning Project Dashboard (TriFrame): Streamlit app.

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud, main file streamlit_app.py. Secrets (App settings > Secrets):
                   HF_TOKEN   = "hf_..."   (a READ-only Hugging Face token for the private model repo)
                   MODEL_REPO = "KrishnaHarshaTangirala/triframe-cv-model"   (optional; this is the default)
"""
import os

import streamlit as st

import dashboard_logic as DL

st.set_page_config(page_title=DL.TITLE, page_icon="📈", layout="wide")


def secret(name, default=None):
    try:
        return st.secrets[name]
    except Exception:                       # no secrets file (local run) or key missing
        return os.environ.get(name, default)


@st.cache_resource(show_spinner="Loading the model bundle...")
def get_bundle():
    return DL.load_bundle(token=secret("HF_TOKEN"), model_repo=secret("MODEL_REPO"))


st.title(DL.TITLE)

try:
    B = get_bundle()
except Exception as e:                      # private repo without a token, wrong repo name, no network
    st.error("The model bundle could not be loaded. If you own this app, add a read-only Hugging Face token as the secret "
             f"HF_TOKEN in the app settings. ({type(e).__name__})")
    st.stop()

st.markdown(DL.describe(B))

cols = st.columns(3)
uploads = {}
for col, (slot, label) in zip(cols, (("1W", "1 Week chart screenshot (1W)"), ("1D", "1 Day chart screenshot (1D)"),
                                     ("1H", "1 Hour chart screenshot (1H)"))):
    with col:
        up = st.file_uploader(label, type=["png", "jpg", "jpeg"], key=f"up_{slot}")
        if up is not None:
            st.image(up, width="stretch")
        uploads[slot] = up

ticker = st.text_input("Ticker (optional; detected from the screenshots if left empty)", key="ticker")
go = st.button("Analyse", type="primary")

if go:
    with st.spinner("Reading the screenshots, rebuilding the model inputs and predicting (the first run also downloads the vision models, about a minute)..."):
        md, charts, checks = DL.analyse(B, uploads["1W"], uploads["1D"], uploads["1H"], ticker)
    st.markdown(md)
    if charts is not None:
        st.image(charts, caption="What the model sees: the three charts rebuilt from market data", width="stretch")
    if checks is not None:
        st.subheader("Screenshot checks")
        st.dataframe(checks, hide_index=True, width="stretch")
