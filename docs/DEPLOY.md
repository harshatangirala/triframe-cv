# Deploying the dashboard on Streamlit Community Cloud

The dashboard is `streamlit_app.py` in this repository. It loads the frozen model bundle from a **private** Hugging Face model
repo at start-up, so the trained weights never touch GitHub. Streamlit Community Cloud hosts it for free.

## What you need
- This repository on GitHub (public), already containing `streamlit_app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`.
- A Hugging Face **read-only** token that can read the model repo (steps below).

## 1. Make a read-only Hugging Face token (about 2 minutes)
1. https://huggingface.co/settings/tokens → **Create new token** → type **Fine-grained**.
2. Name it `streamlit-readonly`. Under *Repositories permissions*, add `KrishnaHarshaTangirala/triframe-cv-model` (and
   `KrishnaHarshaTangirala/triframe-cv-model-s120` / `-full` once those runs exist) with **Read access to contents** only.
3. Create it and copy the `hf_...` value once. Do not put it in any file or message.

## 2. Create the app (about 5 minutes)
1. https://share.streamlit.io → **Continue with GitHub** (account `harshatangirala`) → allow access.
2. **Create app** → *Deploy a public app from GitHub*.
3. Repository `harshatangirala/triframe-cv`, branch `main`, main file path `streamlit_app.py`.
4. App URL: pick a name such as `triframe-cv` (gives `https://triframe-cv.streamlit.app`).
5. **Advanced settings**:
   - Python version: **3.12** (the requirements pin the CPU-only PyTorch wheel for 3.12 and 3.13).
   - Secrets (paste exactly, with your own token):
     ```toml
     HF_TOKEN = "hf_your_read_only_token"
     MODEL_REPO = "KrishnaHarshaTangirala/triframe-cv-model"
     ```
6. **Deploy**. The first build takes several minutes (it installs PyTorch's CPU build, about 190 MB). The first analysis after
   a cold start also downloads two vision models (about a minute).

## 3. Check it
Choose how many companies (1 to 4), upload three TradingView screenshots (1W, 1D, 1H) for each, or type the tickers, and press
Analyse. For each company you should get a decision, a risk tier, expected volatility, the drawdown and direction probabilities with
charts and plain-language explanations, and a Monte Carlo fan of 20-day price paths; with two or more companies, a comparison tab
ranks them from riskiest to safest. Changing the Monte Carlo settings re-simulates without re-reading the screenshots.

## Things to know
- **Sleep:** a free app that nobody visits for about 12 hours goes to sleep. The next visitor sees a "wake up" button and waits a
  minute or two. Opening the app once before showing it to someone avoids that.
- **Memory:** a local test peaked at about 0.8 GB with the pilot's two vision networks (several companies reuse the same loaded
  networks; a fine-tuned ViT-B/16, if a run ever chooses it, adds about 0.35 GB). Streamlit Community Cloud documents 690 MB guaranteed and up to 2.7 GB per
  app. If the app ever shows "over its resource limits", reboot it from the app menu; if that keeps happening, host it on a
  paid tier or another provider.
- **Yahoo Finance:** live prices come from Yahoo Finance, which can rate-limit shared cloud servers. The app retries and caches
  for 10 minutes. If a valid ticker keeps failing, try again later.
- **Switching to the 120-stock (v3) model:** after the v3 run, give the read-only token access to
  `KrishnaHarshaTangirala/triframe-cv-model-s120`, change the secret `MODEL_REPO` to that name, then **Reboot app**. Nothing else
  changes: the app then also shows the leaderboard of all candidate models, percentile-based risk wording and the Monte Carlo
  settings calibrated in Stage 7M. The full 979-stock run works the same way with `triframe-cv-model-full`.
- **Embedding on krishnaharsha.com:** a public Streamlit app embeds with `<iframe src="https://<your-app>.streamlit.app/?embed=true">`.
- **Local run:** `pip install -r requirements.txt`, then `BUNDLE_DIR=<folder with the downloaded bundle> streamlit run streamlit_app.py`
  (or set `HF_TOKEN` and `MODEL_REPO` instead of `BUNDLE_DIR`).
