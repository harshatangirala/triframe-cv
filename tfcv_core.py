"""TriFrame CV - shared inference code, used by the Colab notebook AND the Hugging Face Space.

Everything the dashboard computes goes through the same functions the notebook used to build training inputs:
  * numeric inputs  -> feat_visible.from_series (the 33 deployable features, last 126 daily bars)
  * chart images    -> tf_render.StockWindows (weekly 104 / daily 126 / hourly 140 bars, binary, scale-free)
  * predictions     -> Bundle.predict (same scalers, PCA, heads, calibrators and NO TRADE rule as the frozen config)
"""
import os, io, json, math
import numpy as np
import pandas as pd
import feat_visible as FV
import tf_render as RN

DEPLOY = ["x_" + f for f in FV.DEPLOY]                 # 33 inputs (x_mean_dvol_log is excluded by project rule)
VIEWS = ("W", "D", "H")
TRADING_DAYS, HORIZON_V = 252, 20


def ann_vol(logrv20):
    """logrv20 = ln sqrt(sum of squared daily returns over the next 20 sessions) -> annualised volatility (fraction)."""
    return np.exp(np.asarray(logrv20, float)) * math.sqrt(TRADING_DAYS / HORIZON_V)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


# ------------------------------------------------------------------------------------------------ live market data
def _clean(df):
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0)
    return df


def fetch_yahoo(ticker, daily_period="3y", hourly_period="60d"):
    """Split-adjusted (not dividend-adjusted) daily and regular-hours hourly bars from Yahoo Finance, in the same layout as
    the training data: daily(date, open, high, low, close, volume) and hourly(timestamp UTC, open, high, low, close, volume)."""
    import yfinance as yf
    d = yf.download(ticker, period=daily_period, interval="1d", auto_adjust=False, actions=False, progress=False,
                    multi_level_index=False, threads=False)
    if d is None or len(d) == 0:
        raise ValueError(f"no daily data for {ticker}")
    di = pd.DatetimeIndex(d.index)
    di = di.tz_localize(None) if di.tz is not None else di
    daily = _clean(pd.DataFrame({"date": di.normalize(), "open": d["Open"].values, "high": d["High"].values,
                                 "low": d["Low"].values, "close": d["Close"].values, "volume": d["Volume"].values}))
    h = yf.download(ticker, period=hourly_period, interval="60m", auto_adjust=False, actions=False, prepost=False,
                    progress=False, multi_level_index=False, threads=False)
    if h is None or len(h) == 0:
        hourly = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        hi = pd.DatetimeIndex(h.index)
        hi = hi.tz_localize("America/New_York") if hi.tz is None else hi
        hourly = _clean(pd.DataFrame({"timestamp": hi.tz_convert("UTC"), "open": h["Open"].values, "high": h["High"].values,
                                      "low": h["Low"].values, "close": h["Close"].values, "volume": h["Volume"].values}))
    return daily.reset_index(drop=True), rth_only(hourly)


def rth_only(hourly, early_close_days=None):
    """Keep regular-session hourly bars stamped at :30 New York time (09:30 ... 15:30), as in the training images.
    early_close_days: optional set of dates (13:00 closes) on which only the first 4 slots are kept (training rule r1.1;
    Yahoo has no bars after an early close, so the dashboard does not need it)."""
    if len(hourly) == 0:
        return hourly.reset_index(drop=True)
    et = pd.DatetimeIndex(hourly["timestamp"]).tz_convert("America/New_York")
    keep = np.asarray((et.minute == 30) & (et.hour >= 9) & (et.hour <= 15))
    if early_close_days:
        day = et.tz_localize(None).normalize()
        keep &= ~(np.asarray(day.isin(list(early_close_days))) & np.asarray(et.hour - 9 >= 4))
    return hourly[keep].sort_values("timestamp").reset_index(drop=True)


def drop_incomplete_session(daily, now=None):
    """Drop today's daily bar while the US session is still open (before 16:05 New York time)."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    if len(daily) and daily["date"].iloc[-1] == now.tz_localize(None).normalize() and (now.hour, now.minute) < (16, 5):
        return daily.iloc[:-1].reset_index(drop=True)
    return daily


# ------------------------------------------------------------------------------------------------ as-of date from a screenshot legend
def _close(a, b, tol):
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def match_daily(daily, ohlc, tol=0.003, lookback=400):
    """Index of the latest daily bar whose O/H/L/C all match the screenshot legend, else None."""
    n = len(daily)
    for k in range(n - 1, max(-1, n - 1 - lookback), -1):
        r = daily.iloc[k]
        if all(_close(r[c], ohlc[c], tol) for c in ("open", "high", "low", "close")):
            return k
    return None


def match_weekly(daily, ohlc, tol=0.003, lookback=60):
    """Index k of the daily bar at which the week-to-date bar equals the weekly legend (handles a week still in progress)."""
    S = RN.StockWindows(daily[["date", "open", "high", "low", "close", "volume"]], _empty_hourly())
    n = len(daily)
    for k in range(n - 1, max(-1, n - 1 - lookback), -1):
        o, h, l, c, _ = S.weekly(k)
        if _close(o[-1], ohlc["open"], tol) and _close(h[-1], ohlc["high"], tol) and _close(l[-1], ohlc["low"], tol) \
                and _close(c[-1], ohlc["close"], tol):
            return k
    return None


def match_hourly(daily, hourly, ohlc, tol=0.015):
    """Daily index of the session that contains an hourly bar matching the hourly legend (close within 1.5 %, see notes)."""
    if len(hourly) == 0:
        return None
    for j in range(len(hourly) - 1, -1, -1):
        r = hourly.iloc[j]
        if all(_close(r[c], ohlc[c], tol) for c in ("open", "high", "low", "close")):
            day = pd.Timestamp(r["timestamp"]).tz_convert("America/New_York").tz_localize(None).normalize()
            k = np.searchsorted(daily["date"].values.astype("datetime64[ns]"), np.datetime64(day, "ns"), side="right") - 1
            return int(k) if k >= 0 else None
    return None


def _empty_hourly():
    return pd.DataFrame({"timestamp": pd.to_datetime([], utc=True), "open": [], "high": [], "low": [], "close": [], "volume": []})


# ------------------------------------------------------------------------------------------------ model inputs
def features_at(daily, k):
    """The 33 numeric inputs at daily index k (needs k >= 125)."""
    if k < FV.W - 1:
        raise ValueError(f"need at least {FV.W} daily bars before the as-of date, have {k + 1}")
    a = [daily[c].values.astype(float) for c in ("open", "high", "low", "close", "volume")]
    return FV.from_series(*a, [k])[DEPLOY].iloc[0].values.astype(float)


def render_at(daily, hourly, k):
    """The three binary chart images at daily index k, exactly as the training renderer draws them."""
    S = RN.StockWindows(daily[["date", "open", "high", "low", "close", "volume"]], hourly if len(hourly) else _empty_hourly())
    (w, d, h), (nw, nd, nh) = S.triplet(int(k))
    return {"W": w, "D": d, "H": h, "n_valid": {"W": int(nw), "D": int(nd), "H": int(nh)}}


def triplet_png(trip, scale=2):
    """One PNG (bytes) with the three model-view charts side by side, white bars on black."""
    from PIL import Image, ImageDraw
    panels = []
    for v, label in (("W", "1W (104 weeks)"), ("D", "1D (126 days)"), ("H", "1H (140 hours)")):
        a = (trip[v] * 255).astype(np.uint8)
        im = Image.fromarray(a, "L").resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST).convert("RGB")
        canvas = Image.new("RGB", (im.width, im.height + 22), (17, 17, 17)); canvas.paste(im, (0, 22))
        ImageDraw.Draw(canvas).text((6, 4), label, fill=(200, 200, 200))
        panels.append(canvas)
    W = sum(p.width for p in panels) + 12 * (len(panels) - 1); H = max(p.height for p in panels)
    out = Image.new("RGB", (W, H), (0, 0, 0)); x = 0
    for p in panels:
        out.paste(p, (x, 0)); x += p.width + 12
    return out


# ------------------------------------------------------------------------------------------------ image embeddings (only if a chosen model uses images)
def to_input(img01, mean, std):
    import torch, torch.nn.functional as F
    t = torch.from_numpy(img01.astype(np.float32))[None, None].repeat(1, 3, 1, 1)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    return (t - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


_BACKBONE_CACHE = {}


def embed_views(trip, spec, root=None):
    """Backbone embeddings of the three views (CPU), with the training preprocessing. A fine-tuned backbone (spec has
    'weights_file') is built from its architecture and loads its weights from the bundle folder `root`."""
    import torch, timm
    key = spec.get("name", spec["timm_name"])
    if key not in _BACKBONE_CACHE:
        torch.manual_seed(int(spec.get("seed", 0)))
        if spec.get("weights_file"):
            from safetensors.torch import load_file
            m = timm.create_model(spec["timm"], pretrained=False, num_classes=0, **spec.get("kwargs", {}))
            m.load_state_dict(load_file(os.path.join(root or ".", spec["weights_file"]), device="cpu"))
        else:
            m = timm.create_model(spec["timm_name"], pretrained=bool(spec.get("pretrained", True)), num_classes=0, **spec.get("kwargs", {}))
        _BACKBONE_CACHE[key] = m.eval()
    m = _BACKBONE_CACHE[key]
    out = {}
    with torch.no_grad():
        for v in VIEWS:
            out[v] = m(to_input(trip[v], spec["mean"], spec["std"])).float().numpy()[0]
    return out


# ------------------------------------------------------------------------------------------------ frozen model bundle
class Bundle:
    """Loads config.json + arrays.npz + per-target tree files written by the notebook, and predicts one row or a matrix."""

    def __init__(self, path):
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        self.A = dict(np.load(os.path.join(path, "arrays.npz")))
        self.heads = {}
        for t, spec in self.cfg["targets"].items():
            if spec["head"] == "xgb":
                import xgboost as xgb
                b = xgb.Booster(); b.load_model(os.path.join(path, spec["file"])); self.heads[t] = b
            elif spec["head"] == "lgbm":
                import lightgbm as lgb
                self.heads[t] = lgb.Booster(model_file=os.path.join(path, spec["file"]))

    # -- design matrix for one target from standardized numbers and (optional) PCA'd image embeddings
    def num_block(self, x_num):
        x = np.asarray(x_num, float).reshape(-1, len(self.cfg["features"]))
        x = np.where(np.isfinite(x), x, self.A["num_fill"])
        return (x - self.A["num_mean"]) / self.A["num_std"]

    @staticmethod
    def parse_fs(fs):
        """'NUM' -> (False, None, ()); 'IMG:b' -> (False, b, WDH); 'NUM+IMG:b' -> (True, b, WDH); 'NUM+IMG:b:W' -> (True, b, (W,))."""
        if fs == "NUM":
            return True, None, ()
        parts = fs.split(":")
        views = (parts[2],) if len(parts) > 2 else VIEWS
        return parts[0] == "NUM+IMG", parts[1], tuple(views)

    def img_block(self, backbone, views, emb_by_view):
        """Raw embeddings {view: (n, dim)} -> PCA (frozen) -> standardised (frozen) -> clipped, exactly as in training."""
        parts = []
        for v in views:
            key = f"{backbone}|{v}"
            e = np.asarray(emb_by_view[v], np.float64).reshape(-1, self.A[f"pca_mean|{key}"].shape[0])
            parts.append((e - self.A[f"pca_mean|{key}"]) @ self.A[f"pca_comp|{key}"])
        tag = f"{backbone}|{''.join(views)}"
        z = (np.hstack(parts) - self.A[f"img_mean|{tag}"]) / self.A[f"img_std|{tag}"]
        return np.clip(z, -self.cfg["img_clip"], self.cfg["img_clip"])

    def design(self, target, x_num_std, emb_by_backbone=None, _cache=None):
        use_num, b, views = self.parse_fs(self.cfg["targets"][target]["feature_set"])
        if b is None:
            return x_num_std
        key = (b, views)
        if _cache is not None and key in _cache:
            z = _cache[key]
        else:
            z = self.img_block(b, views, emb_by_backbone[b])
            if _cache is not None:
                _cache[key] = z
        return np.hstack([x_num_std, z]) if use_num else z

    def raw_predict(self, target, X):
        spec = self.cfg["targets"][target]
        if spec["head"] == "linear":
            z = X @ self.A[f"coef|{target}"] + float(self.A[f"intercept|{target}"])
            return z if spec["task"] == "reg" else sigmoid(z)
        if spec["head"] == "xgb":
            return np.asarray(self.heads[target].inplace_predict(np.ascontiguousarray(X, dtype=np.float32)), float)
        return np.asarray(self.heads[target].predict(X), float)

    def calibrate(self, target, p):
        c = self.cfg["targets"][target].get("calibrator")
        return np.interp(p, c["x"], c["y"]) if c else p

    def needed_backbones(self):
        return sorted({self.parse_fs(s["feature_set"])[1] for s in self.cfg["targets"].values()} - {None})

    def predict_matrix(self, x_num, emb_by_backbone=None):
        """x_num: (n, 33) raw features; emb_by_backbone: {backbone: {view: (n, dim) raw embeddings}}.
        Returns arrays: logrv20, vol_ann, vol_ann_lo, vol_ann_hi, p_out5, p_dd10 (probabilities calibrated)."""
        xs = self.num_block(x_num)
        cache = {}
        v = self.raw_predict("V", self.design("V", xs, emb_by_backbone, cache))
        q = self.cfg["targets"]["V"]
        out = {"logrv20": v, "vol_ann": ann_vol(v), "vol_ann_lo": ann_vol(v + q["res_q10"]), "vol_ann_hi": ann_vol(v + q["res_q90"])}
        for t, k in (("O5", "p_out5"), ("D10", "p_dd10")):
            out[k] = self.calibrate(t, self.raw_predict(t, self.design(t, xs, emb_by_backbone, cache)))
        return out

    def predict_row(self, x_num, emb_by_backbone=None):
        """x_num: 33 raw features; emb_by_backbone: {backbone: {view: raw embedding}} for backbones the bundle needs."""
        m = self.predict_matrix(np.asarray(x_num, float).reshape(1, -1),
                                {b: {v: np.asarray(e).reshape(1, -1) for v, e in d.items()} for b, d in (emb_by_backbone or {}).items()})
        out = {k: float(np.asarray(v)[0]) for k, v in m.items()}
        out["decision"], out["reasons"] = decide(out, self.cfg["rule"])
        return out


# ------------------------------------------------------------------------------------------------ Monte Carlo price paths
MC_DEFAULT = dict(t_df=5.0, low_k=1.0)


def _unit_t(rng, df, size):
    """Student-t shocks scaled to unit variance (df >= 100 means normal)."""
    if df is None or df >= 100:
        return rng.standard_normal(size)
    return rng.standard_t(df, size) * math.sqrt((df - 2.0) / df)


def mc_unit_paths(n, t_df, rng, horizon=HORIZON_V, low_k=1.0):
    """Path shapes with a realised 20-day volatility of exactly 1 (sqrt of the sum of squared daily log returns = 1, the same
    definition as the logrv20 label). Returns (cum, low), both (n, horizon): the cumulative log return after each session,
    and that session's intraday low, drawn from the exact minimum of a Brownian bridge between consecutive closes whose
    variance is low_k times the daily variance (low_k is calibrated on validation data so simulated drawdowns match real ones)."""
    z = _unit_t(rng, t_df, (n, horizon))
    z /= np.sqrt((z ** 2).sum(1, keepdims=True))
    cum = np.cumsum(z, 1)
    prev = np.hstack([np.zeros((n, 1)), cum[:, :-1]])
    low = 0.5 * (prev + cum - np.sqrt((cum - prev) ** 2 - 2.0 * low_k / horizon * np.log(rng.random((n, horizon)))))
    return cum, low


def tilt_weights(hit, ret, p_target, iters=80):
    """The smallest change to equal path weights (minimum relative entropy) such that the weighted share of paths touching a 10%
    drawdown equals p_target while the mean 20-day return stays what it was (no directional view is added).
    Weights are proportional to exp(a * hit + b * ret); (a, b) are found by damped Newton steps. Returns (weights, converged)."""
    h = np.asarray(hit, float); r = np.asarray(ret, float)
    r = (r - r.mean()) / (r.std() + 1e-12)
    target = np.array([float(p_target), 0.0])
    ab = np.zeros(2)

    def moments(ab_):
        z = ab_[0] * h + ab_[1] * r
        w = np.exp(z - z.max()); w /= w.sum()
        return w, np.array([w @ h, w @ r])
    w, m = moments(ab)
    for _ in range(iters):
        g = m - target
        if np.abs(g).max() < 1e-6:
            return w, True
        dh, dr = h - m[0], r - m[1]
        J = np.array([[w @ (dh * dh), w @ (dh * dr)], [w @ (dh * dr), w @ (dr * dr)]]) + 1e-10 * np.eye(2)
        step, t = np.linalg.solve(J, g), 1.0
        while t > 1e-4:
            w2, m2 = moments(ab - t * step)
            if np.abs(m2 - target).sum() < np.abs(g).sum():
                ab, w, m = ab - t * step, w2, m2
                break
            t *= 0.5
        else:
            break
    return w, bool(np.abs(m - target).max() < 1e-4)


def residual_sampler(vspec, mc=None):
    """Draws of the volatility model's error (log scale): from the validation quantile table if the bundle has one, else a
    normal with the validation median and variance."""
    q = (mc or {}).get("res_quantiles") or vspec.get("res_quantiles")
    if q:
        qs = np.asarray(q, float); grid = np.linspace(0.0, 1.0, len(qs))
        return lambda rng, n: np.interp(rng.random(n), grid, qs)
    mid, s = float(vspec.get("res_q50", 0.0)), math.sqrt(float(vspec.get("s2", 0.1)))
    return lambda rng, n: mid + s * rng.standard_normal(n)


def wquantile(x, w, qs):
    """Weighted quantiles of x (n,) or column-wise of x (n, k)."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        return np.column_stack([wquantile(x[:, j], w, qs) for j in range(x.shape[1])])
    o = np.argsort(x); cw = np.cumsum(w[o]); cw /= cw[-1]
    return np.interp(qs, cw - 0.5 * w[o] / w.sum(), x[o])


def simulate_paths(p0, logrv20, vspec, mc=None, n_paths=5000, p_dd_target=None, seed=0, horizon=HORIZON_V, n_show=40):
    """Monte Carlo price paths over the model's 20-session horizon.
    * Each path's volatility = the model's forecast x its validated error (drawn from the validation error distribution), so a
      less accurate model gives a wider spread of paths.
    * Daily shocks are fat-tailed (Student-t, tail shape chosen on validation data) and scaled so the path's realised volatility
      equals its drawn volatility (the label's own definition). No drift: the direction model has no reliable edge.
    * Intraday lows come from the exact Brownian-bridge minimum (range calibrated on validation data), because the drawdown
      label uses daily lows.
    * p_dd_target: reweight the paths (smallest possible change, mean return unchanged) so the share touching a 10% drawdown
      equals the calibrated drawdown model's probability.
    Returns a dict of arrays and summary numbers."""
    mc = {**MC_DEFAULT, **(mc or {})}
    rng = np.random.default_rng(seed)
    eps = residual_sampler(vspec, mc)(rng, n_paths)
    rv = np.exp(float(logrv20) + eps)
    cum, low = mc_unit_paths(n_paths, mc["t_df"], rng, horizon, mc["low_k"])
    prices = float(p0) * np.exp(np.hstack([np.zeros((n_paths, 1)), rv[:, None] * cum]))
    lows = float(p0) * np.exp(rv[:, None] * low)
    worst = lows.min(1) / float(p0) - 1
    hit = worst <= -0.10
    p_sim = float(hit.mean())
    ret = prices[:, -1] / float(p0) - 1
    w = np.full(n_paths, 1.0 / n_paths)
    tilted = False
    if p_dd_target is not None and 0 < p_sim < 1:
        w2, ok = tilt_weights(hit, ret, float(np.clip(p_dd_target, 1e-4, 1 - 1e-4)))
        if ok:
            w, tilted = w2, True
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    fan = wquantile(prices, w, qs)                                  # (5, horizon + 1)
    q05 = float(wquantile(ret, w, [0.05])[0])
    tail = ret <= q05
    show = rng.choice(n_paths, size=min(n_show, n_paths), replace=False, p=w / w.sum())
    return dict(p0=float(p0), days=np.arange(horizon + 1), fan=fan, fan_q=qs, sample_paths=prices[show], ret=ret, w=w,
                p_dd_sim=p_sim, p_dd=float(np.sum(w * hit)), tilted=tilted, ess=float(1.0 / np.sum(w ** 2)),
                ret_q05=q05, ret_q50=float(wquantile(ret, w, [0.5])[0]), ret_q95=float(wquantile(ret, w, [0.95])[0]),
                p_up=float(np.sum(w * (ret > 0))), p_down10=float(np.sum(w * (ret <= -0.10))),
                es95=float(-np.sum(w[tail] * ret[tail]) / max(np.sum(w[tail]), 1e-12)), var95=-q05,
                price_q05=float(p0) * (1 + q05), price_q95=float(p0) * (1 + float(wquantile(ret, w, [0.95])[0])),
                mean_ret=float(np.sum(w * ret)), n_paths=int(n_paths), t_df=float(mc["t_df"]), low_k=float(mc["low_k"]))


def decide(out, rule):
    """Pre-registered NO TRADE rule (frozen on validation data). NO TRADE is the default."""
    reasons = []
    risk_high = out["vol_ann"] >= rule["vol_ann_q80"] or out["p_dd10"] >= rule["p_dd10_q80"]
    if out["vol_ann"] >= rule["vol_ann_q80"]:
        reasons.append(f"Predicted volatility is in the top 20 % of the validation period (>= {rule['vol_ann_q80']:.0%} a year).")
    if out["p_dd10"] >= rule["p_dd10_q80"]:
        reasons.append(f"Chance of a 10 % drawdown is in the top 20 % of the validation period (>= {rule['p_dd10_q80']:.0%}).")
    conf = abs(out["p_out5"] - 0.5)
    if not rule["direction_usable"]:
        reasons.append("The direction model showed no reliable edge on held-out data, so direction is not used for a call.")
        return "NO TRADE", reasons
    if conf < rule["conf_threshold"]:
        reasons.append("Direction confidence is below the threshold chosen on validation data.")
    if risk_high or conf < rule["conf_threshold"]:
        return "NO TRADE", reasons
    side = "above" if out["p_out5"] > 0.5 else "below"
    reasons.append("Direction confidence passes the validated threshold, and risk is not elevated.")
    return f"Risk within normal range; 5-day outlook {side} the median stock (low-confidence signal)", reasons
