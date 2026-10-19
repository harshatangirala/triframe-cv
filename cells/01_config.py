# ==================================== CONFIG: edit here, then Runtime > Run all ====================================
import numpy as np

QUICK = True               # True = smoke test on QUICK_TICKERS stocks (about 45-90 min on an A100). False = full 979-stock run.
QUICK_TICKERS = 60
SEED = 11
HF_USER = "KrishnaHarshaTangirala"
DATA_REPO = f"{HF_USER}/triframe-us-equities-chart-ohlcv"   # existing private dataset: charts, features, labels, project code
RUNS_REPO = f"{HF_USER}/triframe-cv-runs"                   # new private dataset: embeddings, results, figures, ledger, logs
MODEL_REPO = f"{HF_USER}/triframe-cv-model"                 # new private model repo: the frozen bundle the dashboard loads
SPACE_REPO = f"{HF_USER}/triframe-cv-dashboard"             # Gradio Space that krishnaharsha.com embeds
SPACE_PRIVATE = False      # a website iframe needs a public Space; the model files stay in the private MODEL_REPO
UPLOAD = True              # push every stage's outputs to Hugging Face
UPLOAD_EMBEDDINGS = True   # full run = about 20 GB of embeddings; if your Hugging Face storage is too small, set False
                           # (they are then kept on Google Drive instead, which is what lets a run resume after a disconnect)
GRADIO_VERSION = "6.20.0"  # version the dashboard was tested with

SPLITS = {"train": ("2001-01-01", "2017-01-01"),            # fit the models
          "val":   ("2017-01-01", "2020-01-01"),            # tune, select, calibrate, set the NO TRADE rule
          "test":  ("2020-01-01", "2023-01-03")}            # one honest check of the frozen choice (includes COVID-2020)
HOLDOUT_START = "2023-01-03"                                # sealed by holdout_guard; opened once, by the owner, in Stage 10

BACKBONE_NAMES = None      # None = the 7 pretrained backbones + 3 random-weight controls (tfcv_train.BACKBONES)
PCA_DIM = 128              # per backbone and view, fitted on training rows only
ALPHAS = [float(a) for a in np.logspace(-2, 6, 17)]         # ridge grid (volatility)
CS = [float(c) for c in np.logspace(-4, 1, 6)]              # logistic grid (direction, drawdown)
N_BOOT = 1000              # month-block bootstrap resamples for every confidence interval
CHUNK_TICKERS = 40         # embedding work unit: one resumable file per chunk of tickers
