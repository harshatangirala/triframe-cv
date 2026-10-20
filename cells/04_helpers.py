# ==================================== shared helpers (imports, run manifest, metrics, fitting with tuning, provenance) ====================================
import importlib
import tfcv_core as C, tfcv_train as T
importlib.reload(C); importlib.reload(T)
from triframe_loader import ImageStore

RES, FIG = os.path.join(WORK, "results"), os.path.join(WORK, "figures")
EMB_DIR, EMB_FT_DIR = os.path.join(WORK, f"emb_{RUN_TAG}"), os.path.join(WORK, f"emb_ft_{RUN_TAG}")
PCA_DIR = os.path.join(WORK, f"pca_{RUN_TAG}")
os.makedirs(os.path.join(RES, "predictions"), exist_ok=True)


def emb_dir(b):
    """Embedding chunks of a backbone: frozen ones in EMB_DIR, fine-tuned ones ('ft_...') in EMB_FT_DIR."""
    return EMB_FT_DIR if b.startswith(T.FT_PREFIX) else EMB_DIR


TARGETS = {"V": "reg", "O5": "clf", "D10": "clf"}
TARGET_NAME = {"V": "20-day volatility (logrv20)", "O5": "beats the median stock over 5 days (y_out5)",
               "D10": "drawdown of 10 % or more within 20 days (dd20_le_10)"}
TARGET_FORMULA = {"V": "ln sqrt(sum of squared daily close-to-close returns, sessions t+1..t+20) [tf_data.labels v1]",
                  "O5": "ret5 (open t+1 -> close t+5, with dividends) > same-date median of eligible stocks in the 979 universe [labels v1]",
                  "D10": "min(low t+1..t+20) / open t+1 - 1 <= -10% [derive_labels v1]"}
PRIMARY = {"V": ("R2", +1), "O5": ("logloss", -1), "D10": ("brier", -1)}    # +1: higher is better, -1: lower is better
LEDGER, VAL_PRED, FITS = [], {}, {}
BASEX = {}                                                  # numeric baseline feature sets (Stage 3): HAR, ROLL22, EWMA, GARCH

# ---- run manifest: what produced every number below
NOTEBOOK_VERSION = "v4"
CODE_HASH = hashlib.sha256(b"".join(open(os.path.join(WORK, f), "rb").read() for f in
                                    ("tfcv_core.py", "tfcv_train.py", "dashboard_logic.py", "tfcv_report.py"))).hexdigest()[:16]
_cfg_keys = ("QUICK", "QUICK_TICKERS", "SAMPLE_CHAIN", "SEED", "SPLITS", "HOLDOUT_START", "BACKBONE_NAMES", "PCA_DIM", "ALPHAS", "CS",
             "N_BOOT", "LB_BOOT", "FINETUNE", "FT_BACKBONES", "FT_INNER", "FT_EPOCHS", "FT_PATIENCE", "FT_ROWS_PER_EPOCH",
             "FT_LR_BACKBONE", "FT_LR_HEAD", "FT_STUDY", "FT_STUDY_TOP", "FT_STUDY_VARIANTS", "FT_SEEDS", "MC_TDF_GRID", "MC_LOWK_GRID")
CONFIG = {k: globals()[k] for k in _cfg_keys if k in globals()}
CONFIG_HASH = hashlib.sha256(json.dumps(CONFIG, sort_keys=True, default=str).encode()).hexdigest()[:16]
RUN_ID = f"{RUN_TAG}-{NOTEBOOK_VERSION}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M')}"
METRICS_LOG = []


def log_metric(**kw):
    """One machine-readable metric row with its provenance (review brief, section 4)."""
    METRICS_LOG.append(dict(run_id=RUN_ID, code_hash=CODE_HASH, config_hash=CONFIG_HASH, **kw))


def save_preds(name, split, ids, cols):
    """Persist predictions so every metric can be recomputed without rerunning the networks; returns the file's SHA-256 (short)."""
    path = os.path.join(RES, "predictions", f"{name}_{split}.parquet")
    pd.DataFrame({"sample_id": np.asarray(ids), **{k: np.asarray(v) for k, v in cols.items()}}).to_parquet(path, index=False)
    push(path, f"results_{RUN_TAG}/predictions/{os.path.basename(path)}")
    return _sha(path)[:16]


def ledger(**kw):
    LEDGER.append(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), run=RUN_TAG, **kw))


def eval_target(t, y, p, dates, s2=None, xret=None, base=None):
    """All metrics for one target on one evaluation set (no confidence intervals). For volatility, pass s2 = the variance of the
    model's VALIDATION residuals when scoring anything other than validation itself."""
    if TARGETS[t] == "reg":
        c, _ = T.month_codes(dates)
        s2 = float(np.var(y - p)) if s2 is None else s2
        return dict(n=len(y), R2=T.r2(y, p), wmR2=T.wm_r2(y, p, c), QLIKE=T.qlike(y, p, s2), MAE=float(np.mean(np.abs(y - p))),
                    RMSE=T.rmse(y, p))
    out = dict(n=len(y), AUC=T.auc(y, p), logloss=T.logloss(y, p), brier=T.brier(y, p), ECE=T.ece(y, p))
    if base is not None:
        out["BSS"] = 1 - out["brier"] / T.brier(y, np.full(len(y), base))
    if xret is not None:
        out["rankIC"], out["rankIC_t"], out["rankIC_weeks"] = T.rank_ic(dates, p, xret)
    return out


def fit_one(t, fs, X, heads=("linear", "tree"), stage="?"):
    """Fit on TRAIN, tune on VAL (ridge alpha / logistic C / tree grid with early stopping), keep the best per head. The ledger
    records the training score too, so validation curves (train vs validation against regularisation) can be drawn."""
    m_tr, m_va = TR & OK[t], VA & OK[t]
    ytr, yva = Y[t][m_tr], Y[t][m_va]
    reg = TARGETS[t] == "reg"
    score = (lambda y, p: T.r2(y, p)) if reg else (lambda y, p: -T.logloss(y, p))     # higher is better
    if "linear" in heads:
        path = T.ridge_path(X[m_tr], ytr, X[m_va], ALPHAS) if reg else T.logit_path(X[m_tr], ytr, X[m_va], CS, device=DEVICE)
        sc = {k: score(yva, v[2]) for k, v in path.items()}
        for k, s in sc.items():
            w, b, _ = path[k]
            z = np.asarray(X[m_tr], np.float64) @ w + b
            ledger(stage=stage, target=t, feature_set=fs, head="linear", param=float(k), val_score=s,
                   train_score=score(ytr, z if reg else C.sigmoid(z)))
        kbest = max(sc, key=sc.get)
        w, b, p = path[kbest]
        grid = ALPHAS if reg else CS
        FITS[(t, fs, "linear")] = dict(param=float(kbest), coef=w, intercept=float(b),
                                       at_edge=bool(kbest in (grid[0], grid[-1])), n_features=X.shape[1])
        VAL_PRED[(t, fs, "linear")] = p
    if "tree" in heads:
        best = None
        for cfg in T.TREE_GRID:
            m, n, p = T.tree_fit(X[m_tr], ytr, X[m_va], yva, TARGETS[t], cfg, gpu=(DEVICE == "cuda"), seed=SEED)
            s = score(yva, p)
            ledger(stage=stage, target=t, feature_set=fs, head="tree", param=json.dumps({**cfg, "n_estimators": n}), val_score=s)
            if best is None or s > best[0]:
                best = (s, cfg, n, m, p)
        FITS[(t, fs, "tree")] = dict(cfg=best[1], n_estimators=best[2], model=best[3], n_features=X.shape[1],
                                     curve=getattr(best[3], "tfcv_curve", None))
        VAL_PRED[(t, fs, "tree")] = best[4]


def val_table(t, keys=None):
    m = VA & OK[t]
    rows = []
    for (tt, fs, hd), p in VAL_PRED.items():
        if tt != t or (keys is not None and (tt, fs, hd) not in keys):
            continue
        rows.append(dict(target=t, feature_set=fs, head=hd,
                         **eval_target(t, Y[t][m], p, DATES[m], xret=XRET[m] if t == "O5" else None, base=BASE.get(t))))
    return pd.DataFrame(rows)


_IMG_CACHE = {}


def img_matrix(b, views=("W", "D", "H")):
    """PCA'd image features of one backbone (views concatenated), standardised on TRAIN rows and clipped at +-10."""
    key = (b, tuple(views))
    if key not in _IMG_CACHE:
        if len(_IMG_CACHE) > 6:
            _IMG_CACHE.clear(); gc.collect()
        Z = np.hstack([np.load(os.path.join(PCA_DIR, f"{b}__{v}.npy")) for v in views]).astype(np.float32)
        mu, sd = T.standardize(Z[TR])
        _IMG_CACHE[key] = (np.clip((Z - mu) / sd, -10, 10).astype(np.float32), mu, sd)
    return _IMG_CACHE[key]


def design(fs):
    """Design matrix for a feature-set name: NUM | IMG:<backbone> | NUM+IMG:<backbone> | NUM+IMG:<backbone>:<view> | a baseline."""
    if fs == "NUM":
        return XN
    if fs in BASEX:
        return BASEX[fs]
    parts = fs.split(":")
    if parts[0] == "IMG":
        return img_matrix(parts[1])[0]
    views = (parts[2],) if len(parts) > 2 else ("W", "D", "H")
    return np.hstack([XN, img_matrix(parts[1], views)[0]])


def save_df(df_, name, push_it=True):
    path = os.path.join(RES, name)
    df_.to_csv(path, index=False)
    if push_it:
        push(path, f"results_{RUN_TAG}/{name}")
    return path


def split_info(mask):
    """Effective sample units of a set of rows: rows, stocks, distinct dates, distinct weeks."""
    d = df.loc[mask]
    return dict(n_rows=int(mask.sum()), n_stocks=int(d["ticker"].nunique()), n_dates=int(d["date"].nunique()),
                n_weeks=int(pd.to_datetime(d["date"]).dt.to_period("W-FRI").nunique()))


print(f"Modules loaded. Run {RUN_ID} | code {CODE_HASH} | config {CONFIG_HASH}")
print("Targets:", "; ".join(f"{k} = {v}" for k, v in TARGET_NAME.items()))
