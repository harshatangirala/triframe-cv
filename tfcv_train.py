"""TriFrame CV - training-side code for the Colab notebook (loader, embeddings, PCA, heads, metrics, selection).
Data is read only through triframe_loader.ImageStore (sealed); the fast reader below is checked against
ImageStore.triplet on random rows before it is used."""
import os, json, time, math, gc, re
import numpy as np
import pandas as pd
import tf_render as RN

VIEWS = ("W", "D", "H")
_SHAPES = {"W": (128, 3 * 104), "D": (128, 3 * 126), "H": (128, 3 * 140)}
_OFF = {"W": 0, "D": 128 * 312, "H": 128 * (312 + 378)}

# ===================================================================================== 1. fast triplet reader
class FastTriplets:
    """Vectorised version of ImageStore.triplet for whole tickers at a time (same shards, same r1.1 patch overlay)."""

    def __init__(self, st):
        self.st = st

    def read(self, g):
        """g: rows of ONE ticker from the store's index (columns ticker, sample_id, src, shard_row, patched)."""
        t = g["ticker"].iloc[0]
        n = len(g)
        buf = np.empty((n, RN.TRIPLET_BYTES), np.uint8)
        pm = g["patched"].values.astype(bool)
        if (~pm).any():
            src = g["src"].values
            for s in np.unique(src[~pm]):
                sel = (~pm) & (src == s)
                buf[sel] = np.asarray(self.st._shard(s, t)[g["shard_row"].values[sel]])
        if pm.any():
            prow = self.st._patch.loc[g["sample_id"].values[pm], "patch_row"].values.astype(int)
            buf[pm] = np.asarray(self.st._shard("cache_patch_r11", t)[prow])
        bits = np.unpackbits(buf, axis=1)[:, :RN.TRIPLET_BITS]
        return {v: bits[:, _OFF[v]:_OFF[v] + _SHAPES[v][0] * _SHAPES[v][1]].reshape(n, *_SHAPES[v]) for v in VIEWS}

    def parity(self, rows, n=300, seed=0):
        """Assert the fast reader equals ImageStore.triplet on n random rows (patched rows included)."""
        rng = np.random.default_rng(seed)
        pick = rows.iloc[rng.choice(len(rows), min(n, len(rows)), replace=False)]
        patched = rows[rows["patched"]]
        if len(patched):
            pick = pd.concat([pick, patched.iloc[rng.choice(len(patched), min(50, len(patched)), replace=False)]])
        bad = 0
        for t, g in pick.groupby("ticker"):
            fast = self.read(g)
            for j, sid in enumerate(g["sample_id"].values):
                ref = self.st.triplet(sid)
                bad += sum(int(not np.array_equal(fast[v][j], ref[v])) for v in VIEWS)
        assert bad == 0, f"fast reader differs from ImageStore.triplet on {bad} views"
        return len(pick)


# ===================================================================================== 2. backbones and embeddings
BACKBONES = {
    "resnet18": dict(timm="resnet18"),
    "efficientnet_b0": dict(timm="efficientnet_b0"),
    "vit_base_patch16_224": dict(timm="vit_base_patch16_224"),
    "convnext_tiny": dict(timm="convnext_tiny"),
    "swin_tiny_patch4_window7_224": dict(timm="swin_tiny_patch4_window7_224"),
    "vit_small_patch14_dinov2": dict(timm="vit_small_patch14_dinov2", kwargs={"img_size": 224}),
    "mobilenetv3_large_100": dict(timm="mobilenetv3_large_100"),
    "resnet18_random_s0": dict(timm="resnet18", pretrained=False, seed=0),
    "resnet18_random_s1": dict(timm="resnet18", pretrained=False, seed=1),
    "resnet18_random_s2": dict(timm="resnet18", pretrained=False, seed=2),
}
PRETRAINED = [b for b, s in BACKBONES.items() if s.get("pretrained", True)]
RANDOM = [b for b in BACKBONES if b not in PRETRAINED]


def build_backbone(name, device, factory=None):
    """Returns (model, mean, std, record). factory(name, spec) may replace timm (used only for the local CPU test)."""
    import torch
    spec = BACKBONES[name]
    if factory is not None:
        return factory(name, spec, device)
    import timm
    torch.manual_seed(int(spec.get("seed", 0)))
    m = timm.create_model(spec["timm"], pretrained=bool(spec.get("pretrained", True)), num_classes=0, **spec.get("kwargs", {}))
    m = m.to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    pc = getattr(m, "pretrained_cfg", {}) or {}
    tag = pc.get("tag") if spec.get("pretrained", True) else None
    rec = dict(name=name, timm=spec["timm"], timm_name=(f"{spec['timm']}.{tag}" if tag else spec["timm"]),
               pretrained=bool(spec.get("pretrained", True)), seed=int(spec.get("seed", 0)), kwargs=spec.get("kwargs", {}),
               mean=list(cfg["mean"]), std=list(cfg["std"]), hf_hub_id=pc.get("hf_hub_id"), timm_version=timm.__version__)
    return m, torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1), torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1), rec


def extract_embeddings(st, rows, names, out_dir, device, chunk_tickers=40, batch=384, factory=None,
                       on_chunk=None, have_chunk=None, log=print):
    """Embeds every row x view x backbone. Work is cut into chunks of tickers; each finished chunk is one .npz
    (sample_id + one float16 array per 'backbone|view'). Finished chunks are skipped, so the run can resume.
    on_chunk(path) is called after each chunk (e.g. upload to Hugging Face); have_chunk(name) can fetch a chunk saved
    in an earlier session (returns a local path or None)."""
    import torch, torch.nn.functional as F
    os.makedirs(out_dir, exist_ok=True)
    tickers = list(dict.fromkeys(rows["ticker"].tolist()))
    chunks = [tickers[i:i + chunk_tickers] for i in range(0, len(tickers), chunk_tickers)]
    models, records = {}, {}
    for nm in names:
        m, mu, sd, rec = build_backbone(nm, device, factory)
        models[nm] = (m, mu, sd); records[nm] = rec
    reader = FastTriplets(st)
    use_amp = str(device).startswith("cuda")
    checked = set()
    t0 = time.time(); done_rows = 0
    for ci, ch in enumerate(chunks):
        fname = f"chunk_{ci:04d}.npz"; path = os.path.join(out_dir, fname)
        if not os.path.exists(path) and have_chunk is not None:
            got = have_chunk(fname)
            if got and os.path.exists(got) and got != path:
                os.replace(got, path)
        if os.path.exists(path):
            continue
        sub = rows[rows["ticker"].isin(ch)]
        sub = pd.concat([sub[sub["ticker"] == t] for t in ch if (sub["ticker"] == t).any()])
        out = {}
        pos = 0
        for t, g in sub.groupby("ticker", sort=False):
            imgs = reader.read(g)
            n = len(g)
            for v in VIEWS:
                for b0 in range(0, n, batch):
                    x = torch.from_numpy(np.ascontiguousarray(imgs[v][b0:b0 + batch])).to(device)
                    x = F.interpolate(x.float().unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
                    if v not in checked:
                        assert float(x.max()) > 0.5, f"blank chart batch in view {v}"
                    x3 = x.expand(-1, 3, -1, -1)
                    for nm, (m, mu, sd) in models.items():
                        xn = (x3 - mu) / sd
                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                            e = m(xn)
                        e = e.float()
                        if (nm, v) not in checked:
                            assert float(xn.std()) > 0.2, f"{nm}/{v}: input contrast too low after normalisation"
                            assert torch.isfinite(e).all(), f"{nm}/{v}: non-finite embeddings"
                            assert float(e.std(0).mean()) > 1e-4, f"{nm}/{v}: embeddings do not vary across charts"
                            checked.add((nm, v))
                        key = f"{nm}|{v}"
                        if key not in out:
                            out[key] = np.empty((len(sub), e.shape[1]), np.float16)
                        out[key][pos + b0:pos + b0 + e.shape[0]] = e.cpu().numpy().astype(np.float16)
                checked.add(v)
            pos += n
        tmp = path + ".tmp.npz"
        np.savez(tmp, sample_id=np.asarray(sub["sample_id"].tolist(), dtype="U"), **out)    # fixed-width text, no pickle
        os.replace(tmp, path)
        done_rows += len(sub)
        el = time.time() - t0
        log(f"   chunk {ci + 1}/{len(chunks)}: {len(sub):,} rows, {done_rows / max(el, 1e-9):,.0f} rows/s this session")
        if on_chunk is not None:
            on_chunk(path)
    return records


def assemble(out_dir, key, order_ids):
    """Rebuild one 'backbone|view' matrix (float32) in the order of order_ids from the chunk files."""
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("chunk_") and f.endswith(".npz"))
    ids, parts = [], []
    for f in files:
        z = np.load(os.path.join(out_dir, f))
        ids.append(z["sample_id"]); parts.append(z[key])
    ids = np.concatenate(ids); X = np.concatenate(parts).astype(np.float32)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    miss = ~pd.Index(order_ids).isin(pos.index)
    assert not miss.any(), f"{int(miss.sum())} rows have no embedding for {key}"
    return X[pos.loc[order_ids].values]


def pca_fit_transform(X, train_mask, k, seed=0, device="cpu"):
    """PCA fitted on training rows only. Returns (Z float32, mean float64, components float64, explained ratio)."""
    import torch
    mu = X[train_mask].mean(0).astype(np.float64)
    q = int(min(k + 10, X.shape[1]))
    torch.manual_seed(seed)
    Xt = torch.from_numpy((X[train_mask] - mu).astype(np.float32)).to(device)
    U, S, V = torch.pca_lowrank(Xt, q=q, center=False, niter=4)
    comp = V[:, :k].double().cpu().numpy()
    tot = float((Xt.double() ** 2).sum().cpu()); expl = float((S[:k].double() ** 2).sum().cpu()) / max(tot, 1e-12)
    del Xt, U, S, V
    Z = ((X - mu) @ comp).astype(np.float32)
    return Z, mu, comp, expl


# ===================================================================================== 3. heads
def standardize(Xtr):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return mu, sd


def ridge_path(Xtr, ytr, Xva, alphas):
    """Ridge with an unpenalised intercept for each alpha (sklearn convention). Returns {alpha: (coef, intercept, val_pred)}."""
    Xtr = np.asarray(Xtr, np.float64); xm = Xtr.mean(0); ym = float(np.mean(ytr))
    Xc = Xtr - xm
    G = Xc.T @ Xc; b = Xc.T @ (np.asarray(ytr, np.float64) - ym)
    out = {}
    for a in alphas:
        w = np.linalg.solve(G + a * np.eye(G.shape[0]), b)
        out[a] = (w, ym - xm @ w, np.asarray(Xva, np.float64) @ w + (ym - xm @ w))
    return out


def logit_path(Xtr, ytr, Xva, Cs, device="cpu", max_iter=300):
    """L2 logistic regression (sklearn C convention, unpenalised intercept), fitted by full-batch L-BFGS along the C path
    from strong to weak regularisation with warm starts. Returns {C: (coef, intercept, val_prob)}."""
    import torch
    dt = torch.float32 if str(device).startswith("cuda") else torch.float64
    X = torch.as_tensor(np.asarray(Xtr), dtype=dt, device=device); y = torch.as_tensor(np.asarray(ytr), dtype=dt, device=device)
    Xv = torch.as_tensor(np.asarray(Xva), dtype=dt, device=device)
    n, d = X.shape
    w = torch.zeros(d, dtype=dt, device=device, requires_grad=True); b = torch.zeros(1, dtype=dt, device=device, requires_grad=True)
    out = {}
    for C in sorted(Cs):
        lam = 1.0 / (C * n)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * (w * w).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pv = torch.sigmoid(Xv @ w + b).double().cpu().numpy()
        out[C] = (w.detach().double().cpu().numpy().copy(), float(b.detach().cpu()), pv)
    return out


TREE_GRID = [dict(max_depth=3, learning_rate=0.05, min_child_weight=100),
             dict(max_depth=5, learning_rate=0.05, min_child_weight=100),
             dict(max_depth=7, learning_rate=0.05, min_child_weight=300),
             dict(max_depth=5, learning_rate=0.10, min_child_weight=1000)]


def tree_backend():
    try:
        import xgboost  # noqa
        return "xgb"
    except Exception:
        return "lgbm"


def tree_fit(Xtr, ytr, Xva, yva, task, cfg, gpu, seed=11, n_estimators=None):
    """One boosted-tree head. With Xva/yva: early stopping on validation. With n_estimators: fixed size (refit).
    Returns (model, best_n, val_pred or None)."""
    be = tree_backend()
    early = n_estimators is None
    n_est = 2000 if early else int(n_estimators)
    if be == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=n_est, learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"],
                  min_child_weight=cfg["min_child_weight"], subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", device="cuda" if gpu else "cpu", random_state=seed, n_jobs=-1)
        if early:
            kw["early_stopping_rounds"] = 100
        if task == "reg":
            m = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **kw)
        else:
            m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)
        if early:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False); best = int(m.best_iteration) + 1
        else:
            m.fit(Xtr, ytr, verbose=False); best = n_est
    else:
        import lightgbm as lgb
        kw = dict(n_estimators=n_est, learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"], num_leaves=min(2 ** cfg["max_depth"], 127),
                  min_child_samples=int(cfg["min_child_weight"]), subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
        m = lgb.LGBMRegressor(**kw) if task == "reg" else lgb.LGBMClassifier(**kw)
        if early:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
            best = int(m.best_iteration_ or n_est)
        else:
            m.fit(Xtr, ytr); best = n_est
    pv = None
    if Xva is not None:
        pv = m.predict(Xva) if task == "reg" else m.predict_proba(Xva)[:, 1]
    return m, best, pv


def tree_save(model, path):
    if tree_backend() == "xgb":
        model.get_booster().save_model(path); return "xgb"
    model.booster_.save_model(path); return "lgbm"


def tree_predict(model, X, task):
    return model.predict(X) if task == "reg" else model.predict_proba(X)[:, 1]


# ===================================================================================== 4. metrics and month-block bootstrap
def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def month_codes(dates):
    m = pd.to_datetime(pd.Series(dates)).dt.to_period("M").astype(str).values
    u = np.unique(m)
    return np.searchsorted(u, m), len(u)


def wm_r2(y, p, c):
    """Within-month R2: demean target and prediction by calendar month, best scalar rescale (cross-sectional skill only)."""
    y = np.asarray(y, float); p = np.asarray(p, float); K = int(c.max()) + 1
    n = np.maximum(np.bincount(c, minlength=K), 1)
    yd = y - (np.bincount(c, y, minlength=K) / n)[c]; pd_ = p - (np.bincount(c, p, minlength=K) / n)[c]
    ss = (pd_ ** 2).sum(); b = (yd * pd_).sum() / ss if ss > 0 else 0.0
    return float(1 - ((yd - b * pd_) ** 2).sum() / (yd ** 2).sum())


def qlike(y_log, p_log, s2):
    """QLIKE on variance: RV = exp(2 y), forecast F = exp(2 p + 2 s2) (log-normal bias correction, s2 from validation)."""
    rv = np.exp(2 * np.asarray(y_log, float)); f = np.exp(2 * np.asarray(p_log, float) + 2 * s2)
    q = rv / f
    return float(np.mean(q - np.log(q) - 1))


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, int), np.asarray(p, float)))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = np.clip((p * bins).astype(int), 0, bins - 1); tot = 0.0
    for k in range(bins):
        m = e == k
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def rank_ic(dates, p, x, min_names=20):
    """Mean cross-sectional Spearman correlation per decision date, and a month-block t-statistic."""
    df = pd.DataFrame({"d": pd.to_datetime(pd.Series(dates)).values, "p": np.asarray(p, float), "x": np.asarray(x, float)})
    g = df.groupby("d")
    ic = g.apply(lambda z: z["p"].rank().corr(z["x"].rank()) if len(z) >= min_names else np.nan, include_groups=False).dropna()
    if len(ic) < 3:
        return float("nan"), float("nan")
    mon = ic.index.to_period("M"); mm = ic.groupby(mon).mean()
    t = mm.mean() / (mm.std(ddof=1) / np.sqrt(len(mm))) if len(mm) > 2 else float("nan")
    return float(ic.mean()), float(t)


class MonthBoot:
    """Month-block bootstrap on one evaluation set: resample calendar months with replacement, keep all their rows.
    The same resamples are used for every metric and every model comparison on that set."""

    def __init__(self, dates, n=1000, seed=42):
        self.c, self.K = month_codes(dates)
        rows = [np.flatnonzero(self.c == k) for k in range(self.K)]
        rng = np.random.default_rng(seed)
        self.idx = [np.concatenate([rows[k] for k in rng.integers(0, self.K, self.K)]) for _ in range(n)]

    def ci(self, fn, *arrays, uses_months=False):
        v = [fn(*[a[i] for a in arrays], self.c[i]) if uses_months else fn(*[a[i] for a in arrays]) for i in self.idx]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], np.asarray(v)

    def diff(self, fn, y, p1, p2, uses_months=False):
        if uses_months:
            v = [fn(y[i], p1[i], self.c[i]) - fn(y[i], p2[i], self.c[i]) for i in self.idx]
        else:
            v = [fn(y[i], p1[i]) - fn(y[i], p2[i]) for i in self.idx]
        v = np.asarray(v)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], v


def holm(pvals):
    """Holm step-down adjusted p-values (dict in, dict out)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    ph = k / n; den = 1 + z * z / n
    return float((ph + z * z / (2 * n) - z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / den)


def isotonic_arrays(p, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), np.asarray(y, float))
    return {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}


def abstention_table(p, y, thresholds=None, coverages=(1.0, 0.5, 0.25, 0.10)):
    """Accuracy of direction calls among the most confident predictions. thresholds: {coverage: confidence threshold}
    fixed on validation (pass None on validation itself to compute them)."""
    p = np.asarray(p, float); y = np.asarray(y, int); conf = np.abs(p - 0.5); rows = []; thr = {}
    for c in coverages:
        t = 0.0 if c == 1.0 else (float(np.quantile(conf, 1 - c)) if thresholds is None else thresholds[c])
        thr[c] = t
        keep = conf >= t; n = int(keep.sum()); k = int(((p[keep] > 0.5) == y[keep]).sum())
        rows.append(dict(coverage_target=c, coverage_realised=n / len(p), n=n, accuracy=k / max(n, 1), wilson_lower=wilson_lower(k, n)))
    return pd.DataFrame(rows), thr
