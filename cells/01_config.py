# ==================================== CONFIG: edit here, then Runtime > Run all ====================================
import numpy as np

QUICK = True               # True = a fixed sample of QUICK_TICKERS stocks. False = all 979 stocks (several hours).
QUICK_TICKERS = 120        # v3: 120 stocks (the pilot used 60). About 2-3 hours on an A100, including fine-tuning.
PILOT_TICKERS = 60         # the 120-stock sample CONTAINS the pilot's 60 stocks (+ 60 new ones), so the two runs compare like for like
SEED = 11
HF_USER = "KrishnaHarshaTangirala"
DATA_REPO = f"{HF_USER}/triframe-us-equities-chart-ohlcv"   # existing private dataset: charts, features, labels, project code
RUNS_REPO = f"{HF_USER}/triframe-cv-runs"                   # private dataset: embeddings, results, figures, ledger, logs, report
MODEL_REPO = f"{HF_USER}/triframe-cv-model" + (f"-s{QUICK_TICKERS}" if QUICK else "-full")   # private model repo: the frozen bundle
                                                           # (each run gets its own repo, so the live dashboard keeps its model until you switch it)
PILOT_MODEL_REPO = f"{HF_USER}/triframe-cv-model"          # the 60-stock pilot's bundle, read only for the 60 vs 120 comparison
DEPLOY_HF_SPACE = False    # Gradio Spaces need a paid Hugging Face plan; the dashboard is deployed on Streamlit Cloud instead (see docs/DEPLOY.md)
SPACE_REPO = f"{HF_USER}/triframe-cv-dashboard"             # only used if DEPLOY_HF_SPACE = True
SPACE_PRIVATE = False      # a website iframe needs a public Space; the model files stay in the private MODEL_REPO
DASHBOARD_URL = ""         # the live dashboard address, e.g. "https://triframe-cv.streamlit.app" (fills the website snippet)
UPLOAD = True              # push every stage's outputs to Hugging Face
UPLOAD_EMBEDDINGS = True   # full run = about 20 GB of embeddings; if your Hugging Face storage is too small, set False
                           # (they are then kept on Google Drive instead, which is what lets a run resume after a disconnect)
GRADIO_VERSION = "6.20.0"  # version the optional Gradio app was tested with

SPLITS = {"train": ("2001-01-01", "2017-01-01"),            # fit the models
          "val":   ("2017-01-01", "2020-01-01"),            # tune, select, calibrate, set the NO TRADE rule and the simulation
          "test":  ("2020-01-01", "2023-01-03")}            # one honest check of the frozen choice (includes COVID-2020)
HOLDOUT_START = "2023-01-03"                                # sealed by holdout_guard; opened once, by the owner, in Stage 10

BACKBONE_NAMES = None      # None = the 7 pretrained backbones + 3 random-weight controls (tfcv_train.BACKBONES)
PCA_DIM = 128              # per backbone and view, fitted on training rows only
ALPHAS = [float(a) for a in np.logspace(-2, 6, 17)]         # ridge grid (volatility)
CS = [float(c) for c in np.logspace(-4, 1, 6)]              # logistic grid (direction, drawdown)
N_BOOT = 1000              # month-block bootstrap resamples for every confidence interval
CHUNK_TICKERS = 40         # embedding work unit: one resumable file per chunk of tickers

# ---- Stage 5F: fine-tuning (v3). Each pretrained backbone is trained end to end on the training years (early stopping on
#      validation); its fine-tuned embeddings then compete with the frozen ones under the same selection rule.
FINETUNE = True
FT_BACKBONES = "all"       # "all" = the 7 pretrained backbones, or a list such as ["resnet18", "convnext_tiny"]
FT_EPOCHS = 4              # maximum epochs; stops early when the validation loss stops improving
FT_MAX_TRAIN_ROWS = 60000  # cap on training chart sets (a fixed random subset if there are more; the 120-stock run has ~37k)
FT_MAX_VAL_ROWS = 20000    # cap on validation chart sets used for early stopping
FT_LR_BACKBONE, FT_LR_HEAD = 3e-5, 1e-3

# ---- Stage 7M: Monte Carlo price simulation (v3), calibrated on validation and checked once on the test years
MC_DRAWS = 20000           # simulation draws used for calibration
MC_TDF_GRID = [3, 4, 5, 7, 10, 30]          # tail shapes of the daily shocks (Student-t degrees of freedom; 30 = nearly normal)
MC_LOWK_GRID = [0.25, 0.5, 0.75, 1.0]       # intraday-low range factors (how far below the closes the daily lows reach)

REPORT_PDF = True          # Stage 11 writes a 5-page plain-language PDF report (saved to Drive, Hugging Face and your Downloads)
