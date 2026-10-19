"""Gradio version of the dashboard (kept for Hugging Face Spaces; the Streamlit app streamlit_app.py is the primary front end).
Both use dashboard_logic.py, so they behave identically."""
import os

import gradio as gr

import dashboard_logic as DL

B = DL.load_bundle()                       # BUNDLE_DIR, or MODEL_REPO + HF_TOKEN from the environment / Space secrets


def analyse(img_w, img_d, img_h, ticker_text):
    return DL.analyse(B, img_w, img_d, img_h, ticker_text)


with gr.Blocks(title=DL.TITLE) as demo:
    gr.Markdown(f"# {DL.TITLE}")
    gr.Markdown(DL.describe(B))
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
