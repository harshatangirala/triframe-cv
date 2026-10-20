# ==================================== CONFIG: edit here, then Runtime > Run all ====================================
import numpy as np

QUICK = True               # True = a fixed sample of QUICK_TICKERS stocks. False = all 979 stocks.
QUICK_TICKERS = 500        # v4: 500 stocks. About 6-8 hours on an A100 with everything switched on (see FT_STUDY / LANDSCAPE below).
SAMPLE_CHAIN = [60, 120, 500]   # nested samples: the pilot's 60 stocks sit inside the 120 of v3, which sit inside these 500
SEED = 11
HF_USER = "KrishnaHarshaTangirala"
DATA_REPO = f"{HF_USER}/triframe-us-equities-chart-ohlcv"   # existing private dataset: charts, features, labels, project code
RUNS_REPO = f"{HF_USER}/triframe-cv-runs"                   # private dataset: embeddings, results, figures, ledger, logs, report
MODEL_REPO = f"{HF_USER}/triframe-cv-model" + (f"-s{QUICK_TICKERS}" if QUICK else "-full")   # private model repo for this run's bundle
PREVIOUS_RUNS = {"pilot, 60 stocks": (f"{HF_USER}/triframe-cv-model", 60),                  # earlier bundles, read only for the
                 "v3, 120 stocks": (f"{HF_USER}/triframe-cv-model-s120", 120)}            # like-for-like comparison in Stage 7
DEPLOY_HF_SPACE = False    # Gradio Spaces need a paid Hugging Face plan; the dashboard runs on Streamlit Cloud (docs/DEPLOY.md)
SPACE_REPO = f"{HF_USER}/triframe-cv-dashboard"             # only used if DEPLOY_HF_SPACE = True
SPACE_PRIVATE = False
DASHBOARD_URL = ""         # the live dashboard address, e.g. "https://triframe-cv.streamlit.app" (fills the website snippet)
UPLOAD = True              # push every stage's outputs to Hugging Face
UPLOAD_EMBEDDINGS = True   # 500 stocks = about 18 GB of embeddings; if your Hugging Face storage is too small, set False
                           # (they are then kept on Google Drive instead, which is what lets a run resume after a disconnect)
GRADIO_VERSION = "6.20.0"

SPLITS = {"train": ("2001-01-01", "2017-01-01"),            # fit the models
          "val":   ("2017-01-01", "2020-01-01"),            # tune, select, calibrate, set the NO TRADE rule and the simulation
          "test":  ("2020-01-01", "2023-01-03")}            # historical diagnostic: these years were already read by v3 (see Stage 0)
HOLDOUT_START = "2023-01-03"                                # already opened twice (pilot and v3): exposed, see Stage 0

BACKBONE_NAMES = None      # None = the 7 pretrained backbones + 3 random-weight controls (tfcv_train.BACKBONES)
PCA_DIM = 128              # per backbone and view, fitted on training rows only
ALPHAS = [float(a) for a in np.logspace(-2, 6, 17)]         # ridge grid (volatility)
CS = [float(c) for c in np.logspace(-4, 1, 6)]              # logistic grid (direction, drawdown)
N_BOOT = 1000              # month-block bootstrap resamples for headline confidence intervals
LB_BOOT = 400              # resamples for the leaderboards (many models)
CHUNK_TICKERS = 40         # embedding work unit: one resumable file per chunk of tickers
GARCH_BASELINE = True      # GARCH(1,1) volatility baseline (needs the raw daily bars in the dataset)

# ---- Stage 5F: fine-tuning (v4 protocol). Early stopping now uses an INNER validation slice inside the training years, so the
#      2017-2019 validation years stay untouched until model selection (in v3 they were used for both).
FINETUNE = True
FT_BACKBONES = "all"       # "all" = the 7 pretrained backbones, or a list such as ["resnet18", "convnext_tiny"]
FT_INNER = ("2015-01-01", "2017-01-01")   # inner validation slice (decides early stopping); fine-tuning learns from rows before it
FT_EPOCHS, FT_PATIENCE = 8, 2              # at most 8 epochs; stop after 2 epochs without improvement on the inner slice
FT_ROWS_PER_EPOCH = 60000  # a fresh random draw of training chart sets each epoch
FT_MAX_STOP_ROWS, FT_MAX_VAL_ROWS = 15000, 20000
FT_LR_BACKBONE, FT_LR_HEAD = 3e-5, 1e-3
FT_STUDY = True            # controlled study on the 2 best backbones: last 25% / last 50% of the network trained, risk-only loss
FT_STUDY_TOP = 2
FT_STUDY_VARIANTS = {"p25": dict(unfreeze=0.25), "p50": dict(unfreeze=0.50), "risk": dict(task_weights=(1.0, 0.0, 1.0))}
FT_SEEDS = [12, 13]        # two extra seeds for the best fine-tuning setup (robustness, not selection)
LANDSCAPE = True           # loss-landscape plots around the best fine-tuned network (Stage 6D)
LANDSCAPE_ROWS, LANDSCAPE_GRID = 512, 13

# ---- Stage 7M: Monte Carlo price simulation, calibrated on validation and checked on the test years
MC_DRAWS = 20000
MC_TDF_GRID = [3, 4, 5, 7, 10, 30]
MC_LOWK_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]             # v3's grid ended at 1.0, which was chosen: extended

REPORT_PDF = True          # Stage 11 writes the plain-language PDF report (Drive, Hugging Face and your Downloads)
