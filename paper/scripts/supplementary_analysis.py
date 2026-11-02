"""Supplementary analysis behind the TriFrame paper (Sections V-VII).

Run after the main notebook. It uses only rows of the 500-stock run that are dated before 2023-01-03 (the split file lists
them), re-fits the numbers-only baselines exactly as the notebook does (selection and early stopping on 2017-2019), checks that
they replicate the run, and then computes paired tests on the saved chosen-model test predictions. It chooses nothing: every
test-year number below is descriptive, and the test years were already read by the earlier runs.

Usage: python supplementary_analysis.py <data_root> <run_root> <repo_root> <out_dir>
"""
import json, os, sys, time, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
DATA, RUN, REPO, OUT = sys.argv[1:5]
sys.path.insert(0, REPO)
import tfcv_train as T
import feat_visible as FV

os.makedirs(OUT, exist_ok=True)
t0 = time.time()
RES = os.path.join(RUN, "results_s500")
split = pd.read_parquet(os.path.join(RES, "split_ids.parquet"))
split["date"] = pd.to_datetime(split["date"])
assert split["date"].max() < pd.Timestamp("2023-01-03")
FROZEN = json.load(open(os.path.join(RUN, "frozen", "frozen_config_s500.json")))
FEATS = ["x_" + f for f in FV.DEPLOY]
tickers = sorted(split["ticker"].unique())
print("rows", len(split), "tickers", len(tickers))

# ------------------------------------------------------------------ features, labels (holdout rows never leave these readers)
fv = pd.concat([pd.read_parquet(os.path.join(DATA, "features_visible", f"{t}.parquet"), columns=["sample_id"] + FEATS) for t in tickers])
df = split.merge(fv, on="sample_id", how="left", validate="one_to_one")
lab = pq.read_table(os.path.join(DATA, "final", "meta", "labels.parquet"), columns=["ticker", "date", "ret5", "ret20", "logrv20", "y_out5", "adv20_usd", "sp500_member"],
                    filters=[("date", "<", pd.Timestamp("2023-01-03"))]).to_pandas()
lab["date"] = pd.to_datetime(lab["date"])
df = df.merge(lab, on=["ticker", "date"], how="left", validate="one_to_one")
der = pd.read_parquet(os.path.join(DATA, "derived", "labels_derived_v1.parquet"), columns=["sample_id", "dd20_le_10", "xret5_med"])
df = df.merge(der[der["sample_id"].isin(split["sample_id"])], on="sample_id", how="left", validate="one_to_one")
del fv, lab, der
print(df["split"].value_counts().to_dict(), f"{time.time() - t0:.0f}s")


# ------------------------------------------------------------------ hourly realised-volatility features (mechanism test)
def hourly_feats(tk):
    h = pd.read_parquet(os.path.join(DATA, "final", "1hour", f"{tk}.parquet"), columns=["timestamp", "open", "close"])
    et = pd.to_datetime(h["timestamp"], utc=True).dt.tz_convert("America/New_York")
    h["day"] = et.dt.tz_localize(None).dt.normalize()
    h = h[h["day"] < pd.Timestamp("2023-01-03")]
    if h.empty:
        return None
    c, o = h["close"].to_numpy(float), h["open"].to_numpy(float)
    new = h["day"].to_numpy() != np.roll(h["day"].to_numpy(), 1); new[0] = True
    r = np.full(len(h), np.nan); r[1:] = np.log(c[1:] / c[:-1])
    intra = np.where(new, np.log(c / o), r)                         # first bar of a session: open to close, so no overnight move
    on = np.where(new, np.nan, np.nan); on[1:] = np.where(new[1:], np.log(o[1:] / c[:-1]), np.nan)
    h = h.assign(i2=intra ** 2, ia=np.abs(intra), on2=on ** 2)
    g = h.groupby("day").agg(rv=("i2", "sum"), mx=("ia", "max"), on2=("on2", "sum"))
    f = pd.DataFrame(index=g.index)
    eps = 1e-8
    f["h_rv1"] = 0.5 * np.log(g["rv"] + eps)
    f["h_rv5"] = 0.5 * np.log(g["rv"].rolling(5, min_periods=3).mean() + eps)
    f["h_rv20"] = 0.5 * np.log(g["rv"].rolling(20, min_periods=10).mean() + eps)
    f["h_on20"] = 0.5 * np.log(g["on2"].rolling(20, min_periods=10).mean() + eps)
    f["h_max20"] = np.log(g["mx"].rolling(20, min_periods=10).max() + eps)
    f["h_iofrac20"] = g["rv"].rolling(20, min_periods=10).sum() / (g["rv"].rolling(20, min_periods=10).sum() + g["on2"].rolling(20, min_periods=10).sum() + eps)
    f = f.reset_index().rename(columns={"day": "date"}); f["ticker"] = tk
    return f


hf = pd.concat([x for x in Parallel(n_jobs=10)(delayed(hourly_feats)(t) for t in tickers) if x is not None])
HF = ["h_rv1", "h_rv5", "h_rv20", "h_on20", "h_max20", "h_iofrac20"]
df = df.merge(hf, on=["ticker", "date"], how="left", validate="one_to_one")
del hf
print("hourly features: share finite", float(np.isfinite(df["h_rv20"]).mean()), f"{time.time() - t0:.0f}s")

# ------------------------------------------------------------------ standardisation on training rows only (as the notebook)
TR, VA, TE = (df["split"] == s for s in ("train", "val", "test"))
TR, VA, TE = TR.to_numpy(), VA.to_numpy(), TE.to_numpy()


def std_matrix(cols):
    X = df[cols].to_numpy(np.float64); fill = np.nanmedian(X[TR], 0); fill = np.where(np.isfinite(fill), fill, 0.0)
    X = np.where(np.isfinite(X), X, fill); m = X[TR].mean(0); s = X[TR].std(0); s[s == 0] = 1.0
    return ((X - m) / s).astype(np.float32)


XN, XH = std_matrix(FEATS), std_matrix(FEATS + HF)
Y = {"V": df["logrv20"].to_numpy(float), "D10": df["dd20_le_10"].to_numpy(float), "O5": df["y_out5"].to_numpy(float)}


def fit_tree(X, t):
    task = "reg" if t == "V" else "clf"; ok = np.isfinite(Y[t]); best = None
    for cfg in T.TREE_GRID:
        m, n, _ = T.tree_fit(X[TR & ok], Y[t][TR & ok], X[VA & ok], Y[t][VA & ok], task, cfg, False, seed=11)
        pv = m.predict(X[VA & ok]) if task == "reg" else m.predict_proba(X[VA & ok])[:, 1]
        s = T.r2(Y[t][VA & ok], pv) if task == "reg" else -T.logloss(Y[t][VA & ok], pv)
        if best is None or s > best[0]:
            best = (s, m, cfg, n, pv)
    s, m, cfg, n, pv = best
    pred = m.predict(X) if task == "reg" else m.predict_proba(X)[:, 1]
    return pd.Series(pred, index=df.index), dict(cfg=cfg, n_trees=n)


P, INFO = {}, {}
for name, X in (("NUM", XN), ("NUMH", XH)):
    for t in ("V", "D10"):
        P[(name, t)], INFO[f"{name}_{t}"] = fit_tree(X, t); print("fit", name, t, INFO[f"{name}_{t}"], f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ linear volatility baselines (train-only fit)
def linear(cols):
    X = std_matrix(cols); ok = np.isfinite(Y["V"]) & TR
    return pd.Series(LinearRegression().fit(X[ok], Y["V"][ok]).predict(X), index=df.index)


P[("HAR", "V")] = linear([f"x_{k}_{w}" for k in ("cc", "park", "gk") for w in (1, 5, 22)] + ["x_ewma_win"])
P[("ROLL22", "V")] = linear(["x_cc_22"])
P[("EWMA", "V")] = linear(["x_ewma_win"])
P[("HARH", "V")] = linear(["h_rv1", "h_rv5", "h_rv20", "h_on20"])          # textbook HAR-RV on hourly realised variance


def garch_one(tk):
    try:
        s = T.garch_logrv20(pd.read_parquet(os.path.join(DATA, "final", "1day", f"{tk}.parquet"), columns=["date", "close"]).query("date < '2023-01-03'"), "2017-01-01")
        return None if s is None else s.rename("g").rename_axis("date").reset_index().assign(ticker=tk)
    except Exception:
        return None


gr = [x for x in Parallel(n_jobs=10)(delayed(garch_one)(t) for t in tickers) if x is not None]
g = pd.concat(gr); g["date"] = pd.to_datetime(g["date"])
gcol = df[["ticker", "date"]].merge(g, on=["ticker", "date"], how="left")["g"].to_numpy()
cover = float(np.isfinite(gcol[TE]).mean()); n_garch = len(gr)
ok = np.isfinite(gcol) & TR & np.isfinite(Y["V"])
lr = LinearRegression().fit(gcol[ok][:, None], Y["V"][ok])
P[("GARCH", "V")] = pd.Series(np.where(np.isfinite(gcol), lr.predict(np.nan_to_num(gcol)[:, None]), np.nan), index=df.index)
print("GARCH stocks", n_garch, "test coverage", cover, f"{time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------ the frozen chosen models' saved test predictions
for f, c, k in (("chosen_test", "logrv20_hat", ("IMG", "V")), ("chosen_D10_test", "p_D10", ("IMG", "D10")), ("chosen_O5_test", "p_O5", ("IMG", "O5"))):
    d = pd.read_parquet(os.path.join(RES, "predictions", f + ".parquet"))
    P[k] = df[["sample_id"]].merge(d, on="sample_id", how="left")[c]
s120 = pd.read_parquet(os.path.join(RUN, "results_s120", "stage7_test_predictions.parquet"))
P[("S120", "V")] = df[["sample_id"]].merge(s120[["sample_id", "pred_logrv20"]], on="sample_id", how="left")["pred_logrv20"]

te = df[TE].reset_index(drop=True)
pt = {k: v[TE].to_numpy() for k, v in P.items()}
pv = {k: v[VA].to_numpy() for k, v in P.items()}
yV = te["logrv20"].to_numpy(float); yD = te["dd20_le_10"].to_numpy(float); dates = te["date"].values
years = te["date"].dt.year.to_numpy()
mc, K = T.month_codes(dates, "M")
tab = {}

# ---- 1. replication check against the run
rep = {"NUM_V_val_R2": (T.r2(Y["V"][VA], pv[("NUM", "V")]), 0.5461), "NUM_V_test_R2": (T.r2(yV, pt[("NUM", "V")]), 0.5004),
       "NUM_D10_val_AUC": (roc_auc_score(Y["D10"][VA], pv[("NUM", "D10")]), 0.7257), "NUM_D10_test_AUC": (roc_auc_score(yD, pt[("NUM", "D10")]), 0.6533),
       "HAR_test_R2": (T.r2(yV, pt[("HAR", "V")]), 0.4776), "EWMA_test_R2": (T.r2(yV, pt[("EWMA", "V")]), 0.4484), "ROLL22_test_R2": (T.r2(yV, pt[("ROLL22", "V")]), 0.4161),
       "IMG_V_test_R2": (T.r2(yV, pt[("IMG", "V")]), 0.5031), "IMG_D10_test_AUC": (roc_auc_score(yD, pt[("IMG", "D10")]), 0.6477)}
tab["replication"] = {k: dict(here=float(a), run=b) for k, (a, b) in rep.items()}
print(json.dumps(tab["replication"], indent=0), flush=True)

# ---- 2. volatility: every forecaster with paired month-block CIs against the chosen model
boot = T.MonthBoot(dates, n=1000, seed=42)
s2 = {k[0]: float(np.nanvar(Y["V"][VA] - pv[k])) for k in pv if k[1] == "V" and k[0] not in ("IMG", "S120")}
s2["IMG"] = float(FROZEN["volatility_residuals"].get("s2", FROZEN["volatility_residuals"].get("res_var", np.nan))) if isinstance(FROZEN.get("volatility_residuals"), dict) else np.nan
print("vol residual keys", list(FROZEN["volatility_residuals"].keys())[:8])


def vm(y, p, c):
    return dict(R2=T.r2(y, p), wmR2=T.wm_r2(y, p, c), RMSE=float(np.sqrt(np.mean((y - p) ** 2))), MAE=float(np.mean(np.abs(y - p))), Spearman=float(spearmanr(y, p)[0]))


rows = []
for nm in ("IMG", "NUM", "NUMH", "HARH", "HAR", "GARCH", "EWMA", "ROLL22"):
    p = pt[(nm, "V")]; ok = np.isfinite(p) & np.isfinite(yV); r = dict(model=nm, n=int(ok.sum()), val_R2=(T.r2(Y["V"][VA], pv[(nm, "V")][:]) if nm not in ("IMG", "GARCH") else np.nan), **vm(yV[ok], p[ok], mc[ok]))
    for yr in (2020, 2021, 2022):
        m = ok & (years == yr); r[f"R2_{yr}"] = T.r2(yV[m], p[m])
    rows.append(r)
tab["vol_models"] = rows


def paired(a, b, metric):
    """a minus b on the same resamples (month blocks)."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(yV); out = []
    for i in boot.idx:
        i = i[ok[i]]
        out.append(metric(yV[i], a[i], mc[i]) - metric(yV[i], b[i], mc[i]))
    pt_est = metric(yV[ok], a[ok], mc[ok]) - metric(yV[ok], b[ok], mc[ok])
    return dict(delta=float(pt_est), lo=float(np.percentile(out, 2.5)), hi=float(np.percentile(out, 97.5)), n=int(ok.sum()))


METRICS = {"R2": lambda y, p, c: T.r2(y, p), "wmR2": lambda y, p, c: T.wm_r2(y, p, c), "RMSE": lambda y, p, c: float(np.sqrt(np.mean((y - p) ** 2))),
           "MAE": lambda y, p, c: float(np.mean(np.abs(y - p))), "Spearman": lambda y, p, c: float(spearmanr(y, p)[0])}
tab["paired"] = {}
for nm_a, nm_b in (("IMG", "NUM"), ("NUMH", "NUM"), ("IMG", "NUMH"), ("IMG", "HAR"), ("IMG", "GARCH"), ("NUMH", "HARH"), ("IMG", "S120")):
    tab["paired"][f"{nm_a}-{nm_b}"] = {m: paired(pt[(nm_a, "V")], pt[(nm_b, "V")], f) for m, f in METRICS.items()}
    print(nm_a, nm_b, {m: round(v["delta"], 4) for m, v in tab["paired"][f"{nm_a}-{nm_b}"].items()}, f"{time.time() - t0:.0f}s", flush=True)

# ---- by-year paired gain of images over numbers, and of hourly numbers over numbers
tab["by_year"] = []
for yr in (2020, 2021, 2022):
    m = years == yr; bb = T.MonthBoot(dates[m], n=1000, seed=42); mm, _ = T.month_codes(dates[m], "M")
    for a, b in (("IMG", "NUM"), ("NUMH", "NUM")):
        pa, pb, y = pt[(a, "V")][m], pt[(b, "V")][m], yV[m]
        for mn in ("R2", "wmR2", "Spearman"):
            f = METRICS[mn]; d = [f(y[i], pa[i], mm[i]) - f(y[i], pb[i], mm[i]) for i in bb.idx]
            tab["by_year"].append(dict(year=yr, a=a, b=b, metric=mn, delta=f(y, pa, mm) - f(y, pb, mm), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))


# ---- 3. Diebold-Mariano on monthly mean loss differentials (Newey-West, 3 lags)
def dm(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(yV); d = (yV - a) ** 2 - (yV - b) ** 2
    s = pd.Series(d[ok]).groupby(mc[ok]).mean().to_numpy()
    return dict(mean_diff=float(s.mean()), t_nw=float(T.nw_tstat(s, lags=3)), months=len(s))


tab["dm"] = {f"{a}-{b}": dm(pt[(a, "V")], pt[(b, "V")]) for a, b in (("IMG", "NUM"), ("NUMH", "NUM"), ("IMG", "NUMH"), ("IMG", "HAR"), ("IMG", "GARCH"), ("NUM", "HAR"), ("NUM", "GARCH"))}

# ---- 4. decomposition: how much of the squared error is a market-wide level miss? (month-mean component)
rows = []
for nm in ("IMG", "NUM", "NUMH", "HAR"):
    p = pt[(nm, "V")]
    for yr in (None, 2020, 2021, 2022):
        m = np.ones(len(yV), bool) if yr is None else years == yr
        e = (yV - p)[m]; c = mc[m]; K_ = int(c.max()) + 1; n = np.bincount(c, minlength=K_); mean_e = np.bincount(c, e, minlength=K_) / np.maximum(n, 1)
        total = float((e ** 2).sum()); between = float((n * mean_e ** 2).sum())
        rows.append(dict(model=nm, year=yr or "all", mse=total / len(e), mean_error=float(e.mean()), share_market_level=between / total, mse_cross_section=(total - between) / len(e)))
tab["decomp"] = rows


# ---- 5. Mincer-Zarnowitz and encompassing regressions with month-clustered standard errors
def ols_cl(y, X, c):
    X = np.column_stack([np.ones(len(y)), X]); XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b; meat = np.zeros((X.shape[1],) * 2)
    for k in np.unique(c):
        s = X[c == k].T @ u[c == k]; meat += np.outer(s, s)
    G = len(np.unique(c)); V = XtXi @ meat @ XtXi * G / (G - 1)
    return b, np.sqrt(np.diag(V)), 1 - (u ** 2).sum() / ((y - y.mean()) ** 2).sum()


rows = []
for nm in ("IMG", "NUM", "NUMH", "HAR"):
    for yr in (None, 2020, 2021, 2022):
        m = np.ones(len(yV), bool) if yr is None else years == yr
        b, se, r2_ = ols_cl(yV[m], pt[(nm, "V")][m], mc[m]); rows.append(dict(model=nm, year=yr or "all", a=b[0], a_se=se[0], b=b[1], b_se=se[1], R2=r2_))
tab["mz"] = rows
rows = []
for a, b_ in (("IMG", "NUM"), ("NUMH", "NUM"), ("IMG", "NUMH")):
    b, se, r2_ = ols_cl(yV, np.column_stack([pt[(a, "V")], pt[(b_, "V")]]), mc)
    rows.append(dict(f1=a, f2=b_, b1=b[1], se1=se[1], t1=b[1] / se[1], b2=b[2], se2=se[2], t2=b[2] / se[2], R2=r2_))
    # the same with month fixed effects: purely cross-sectional information
    yd = yV - pd.Series(yV).groupby(mc).transform("mean").to_numpy()
    d1 = pt[(a, "V")] - pd.Series(pt[(a, "V")]).groupby(mc).transform("mean").to_numpy(); d2 = pt[(b_, "V")] - pd.Series(pt[(b_, "V")]).groupby(mc).transform("mean").to_numpy()
    b, se, r2_ = ols_cl(yd, np.column_stack([d1, d2]), mc)
    rows[-1].update(fe_b1=b[1], fe_se1=se[1], fe_t1=b[1] / se[1], fe_b2=b[2], fe_se2=se[2], fe_t2=b[2] / se[2], fe_R2=r2_)
tab["encompass"] = rows

# ---- 6. drawdown: paired AUC, within-month AUC and by year
def wm_auc(y, p, c):
    v, w = [], []
    for k in np.unique(c):
        m = c == k
        if 0 < y[m].sum() < m.sum():
            v.append(roc_auc_score(y[m], p[m])); w.append(m.sum())
    return float(np.average(v, weights=w))


D = {}
for nm in ("IMG", "NUM", "NUMH"):
    p = pt[(nm, "D10")]
    D[nm] = dict(AUC=float(roc_auc_score(yD, p)), wmAUC=wm_auc(yD, p, mc), **{f"AUC_{y}": float(roc_auc_score(yD[years == y], p[years == y])) for y in (2020, 2021, 2022)},
                 val_AUC=(float(roc_auc_score(Y["D10"][VA], pv[(nm, "D10")])) if nm != "IMG" else np.nan))
tab["dd_models"] = D
tab["dd_paired"] = {}
for a, b in (("IMG", "NUM"), ("NUMH", "NUM"), ("IMG", "NUMH")):
    pa, pb = pt[(a, "D10")], pt[(b, "D10")]
    d = [(roc_auc_score(yD[i], pa[i]) - roc_auc_score(yD[i], pb[i])) for i in boot.idx]
    dw = [wm_auc(yD[i], pa[i], mc[i]) - wm_auc(yD[i], pb[i], mc[i]) for i in boot.idx[:300]]
    tab["dd_paired"][f"{a}-{b}"] = dict(AUC=D[a]["AUC"] - D[b]["AUC"], AUC_lo=float(np.percentile(d, 2.5)), AUC_hi=float(np.percentile(d, 97.5)),
                                        wmAUC=D[a]["wmAUC"] - D[b]["wmAUC"], wm_lo=float(np.percentile(dw, 2.5)), wm_hi=float(np.percentile(dw, 97.5)))
print("drawdown", tab["dd_paired"], f"{time.time() - t0:.0f}s", flush=True)

# ---- 7. economic content: deciles of predicted risk, flag overlay (gross of costs, survivors, test years only)
ann = np.sqrt(252 / 20)
te["pv"] = pt[("IMG", "V")]; te["pd"] = pt[("IMG", "D10")]; te["pnum"] = pt[("NUM", "V")]
te["rv_ann"] = np.exp(te["logrv20"]) * ann; te["pv_ann"] = np.exp(te["pv"]) * ann
te["month"] = te["date"].dt.to_period("M").astype(str)
te["dec"] = te.groupby("month")["pv"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
dec = te.groupby("dec").agg(n=("pv", "size"), pred_vol=("pv_ann", "mean"), real_vol=("rv_ann", "mean"), dd=("dd20_le_10", "mean"), ret20=("ret20", "mean")).reset_index()
msd = te.groupby(["month", "dec"])["ret20"].mean().unstack().dropna()
spread = (msd[9] - msd[0]).to_numpy()
tab["deciles"] = dec.to_dict("records")
tab["decile_spread"] = dict(ret20_hi_lo=float(spread.mean()), t=float(T.nw_tstat(spread, 3)), months=len(spread),
                            rv_ratio=float(dec["real_vol"].iloc[-1] / dec["real_vol"].iloc[0]), dd_hi=float(dec["dd"].iloc[-1]), dd_lo=float(dec["dd"].iloc[0]))
rule = FROZEN["rule"]; print("rule", rule)
vthr, dthr = float(rule["vol_ann_q80"]), float(rule["p_dd10_q80"])
te["flag"] = (te["pv_ann"] >= vthr) | (te["pd"] >= dthr)
ms = te.groupby(["month", "flag"])["ret20"].mean().unstack()
allm = te.groupby("month")["ret20"].mean()


def stats(s):
    s = s.dropna().to_numpy(); return dict(mean=float(s.mean()), sd=float(s.std(ddof=1)), ratio_ann=float(s.mean() / s.std(ddof=1) * np.sqrt(252 / 20)), worst=float(s.min()), months=len(s), share_below_m10=float((s < -0.10).mean()))


tab["overlay"] = dict(flag_rate=float(te["flag"].mean()), vthr=vthr, dthr=dthr, all=stats(allm), unflagged=stats(ms[False]), flagged=stats(ms[True]),
                      rv_flag=float(te.loc[te["flag"], "rv_ann"].mean()), rv_noflag=float(te.loc[~te["flag"], "rv_ann"].mean()),
                      dd_flag=float(te.loc[te["flag"], "dd20_le_10"].mean()), dd_noflag=float(te.loc[~te["flag"], "dd20_le_10"].mean()),
                      spread_t=float(T.nw_tstat((ms[False] - ms[True]).dropna().to_numpy(), 3)), spread=float((ms[False] - ms[True]).mean()))

# ---- 8. heterogeneity: liquidity terciles (by month) and S&P membership
te["liq"] = te.groupby("month")["adv20_usd"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
rows = []
for name, mask in [(f"ADV tercile {i + 1}", (te["liq"] == i).to_numpy()) for i in range(3)] + [("S&P 500 member", te["sp500_member"].fillna(False).astype(bool).to_numpy()), ("Non-member", ~te["sp500_member"].fillna(False).astype(bool).to_numpy())]:
    idx = np.flatnonzero(mask); y = yV[idx]; c = mc[idx]; a = pt[("IMG", "V")][idx]; b = pt[("NUM", "V")][idx]; h = pt[("NUMH", "V")][idx]
    d, dw = [], []
    bb = T.MonthBoot(dates[idx], n=500, seed=42); cc, _ = T.month_codes(dates[idx], "M")
    for i in bb.idx:
        d.append(T.r2(y[i], a[i]) - T.r2(y[i], b[i])); dw.append(T.wm_r2(y[i], a[i], cc[i]) - T.wm_r2(y[i], b[i], cc[i]))
    rows.append(dict(group=name, n=len(idx), R2_img=T.r2(y, a), R2_num=T.r2(y, b), R2_numh=T.r2(y, h), d=T.r2(y, a) - T.r2(y, b), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                     wm_img=T.wm_r2(y, a, cc), wm_num=T.wm_r2(y, b, cc), wm_d=T.wm_r2(y, a, cc) - T.wm_r2(y, b, cc), wm_lo=float(np.percentile(dw, 2.5)), wm_hi=float(np.percentile(dw, 97.5))))
tab["hetero"] = rows

# ---- 9. does validation rank predict test rank across candidates?
lv = pd.read_csv(os.path.join(RES, "stage6L_leaderboard_validation.csv")); lt = pd.read_csv(os.path.join(RES, "stage7_leaderboard_test.csv"))
rows = {}
COND = next(c for c in lv["condition"].unique() if "image" in str(c).lower() and "number" in str(c).lower() and "only" not in str(c).lower())
print("using condition:", COND, "of", list(lv["condition"].unique()))
for t in ("V", "D10", "O5"):
    print(t, "conditions", lv["condition"].unique()[:8], lt["condition"].unique()[:8])
    a = lv[(lv["target"] == t) & (lv["condition"] == COND)][["model", "kind", "primary_value"]].rename(columns={"primary_value": "val"})
    b = lt[(lt["target"] == t) & (lt["condition"] == COND)][["model", "primary_value"]].rename(columns={"primary_value": "test"})
    j = a.merge(b, on="model"); j = j[j["kind"].isin(["frozen", "fine-tuned"])]
    rows[t] = dict(n=len(j), spearman=float(spearmanr(j["val"], j["test"])[0]), val_spread=float(j["val"].max() - j["val"].min()), test_spread=float(j["test"].max() - j["test"].min()),
                   ft_val=float(j[j.kind == "fine-tuned"]["val"].mean()), fr_val=float(j[j.kind == "frozen"]["val"].mean()), ft_test=float(j[j.kind == "fine-tuned"]["test"].mean()), fr_test=float(j[j.kind == "frozen"]["test"].mean()))
    j.to_csv(os.path.join(OUT, f"val_vs_test_{t}.csv"), index=False)
tab["val_test_rank"] = rows
tab["info"] = dict(trees=INFO, garch_stocks=n_garch, garch_test_coverage=cover, hourly_finite=float(np.isfinite(df["h_rv20"]).mean()))


# ---- 10. stress tests of the cross-sectional result (images minus numbers, test years)
pi, pn = pt[("IMG", "V")], pt[("NUM", "V")]
tick = te["ticker"].to_numpy()


def demean(x, codes):
    K_ = int(codes.max()) + 1
    return x - (np.bincount(codes, x, minlength=K_) / np.maximum(np.bincount(codes, minlength=K_), 1))[codes]


def xs_mse(y, p, c):
    return float(np.mean(demean(y - p, c) ** 2))


def level_mse(y, p, c):
    e = y - p; return float(np.mean(e ** 2) - np.mean(demean(e, c) ** 2))


def twoway_r2(y, p, c, s):
    yy, pp = y.copy(), p.copy()
    for _ in range(8):
        yy, pp = demean(demean(yy, c), s), demean(demean(pp, c), s)
    b = (yy * pp).sum() / (pp * pp).sum(); return float(1 - ((yy - b * pp) ** 2).sum() / (yy ** 2).sum())


st = {}
wm = METRICS["wmR2"]
d = np.array([wm(yV[i], pi[i], mc[i]) - wm(yV[i], pn[i], mc[i]) for i in boot.idx])
st["wmR2_month_block"] = dict(delta=wm(yV, pi, mc) - wm(yV, pn, mc), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], ci99=[float(np.percentile(d, 0.5)), float(np.percentile(d, 99.5))], ci_bonf5=[float(np.percentile(d, 0.5)), float(np.percentile(d, 99.5))], share_le0=float((d <= 0).mean()))
bq = T.MonthBoot(dates, n=1000, seed=7, freq="Q")
d = np.array([wm(yV[i], pi[i], mc[i]) - wm(yV[i], pn[i], mc[i]) for i in bq.idx]); st["wmR2_quarter_block"] = dict(ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
tk_u, tk_c = np.unique(tick, return_inverse=True); rows_tk = [np.flatnonzero(tk_c == k) for k in range(len(tk_u))]; rng = np.random.default_rng(11)
d = []
for _ in range(500):
    i = np.concatenate([rows_tk[k] for k in rng.integers(0, len(tk_u), len(tk_u))]); d.append(wm(yV[i], pi[i], mc[i]) - wm(yV[i], pn[i], mc[i]))
st["wmR2_stock_block"] = dict(ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
wk, _ = T.month_codes(dates, "W-FRI")
st["within_week_R2"] = dict(img=T.wm_r2(yV, pi, wk), num=T.wm_r2(yV, pn, wk))
d = [T.wm_r2(yV[i], pi[i], wk[i]) - T.wm_r2(yV[i], pn[i], wk[i]) for i in boot.idx[:500]]; st["within_week_R2"]["ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
st["within_date_note"] = "rows of one stock are not on the same date as others, so month and week are the cross-sections"
for nm, f in (("xs_mse", xs_mse), ("level_mse", level_mse)):
    d = [f(yV[i], pi[i], mc[i]) - f(yV[i], pn[i], mc[i]) for i in boot.idx]
    st[nm] = dict(img=f(yV, pi, mc), num=f(yV, pn, mc), delta=f(yV, pi, mc) - f(yV, pn, mc), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
st["twoway"] = dict(img=twoway_r2(yV, pi, mc, tk_c), num=twoway_r2(yV, pn, mc, tk_c))
d = []
for i in boot.idx[:300]:
    _, s_i = np.unique(tick[i], return_inverse=True); _, m_i = np.unique(mc[i], return_inverse=True)
    d.append(twoway_r2(yV[i], pi[i], m_i, s_i) - twoway_r2(yV[i], pn[i], m_i, s_i))
st["twoway"]["ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
# month-by-month cross-sectional R2: a simple time series the reader can inspect
rows = []
for k in np.unique(mc):
    m = mc == k; yd = demean(yV[m], np.zeros(m.sum(), int))
    f = lambda p: float(np.corrcoef(yV[m], p[m])[0, 1] ** 2)
    rows.append(dict(month=str(te["date"][m].iloc[0].to_period("M")), n=int(m.sum()), real=float(yV[m].mean()), pred_img=float(pi[m].mean()), pred_num=float(pn[m].mean()), xs_r2_img=f(pi), xs_r2_num=f(pn), err_level_img=float((yV - pi)[m].mean()), err_level_num=float((yV - pn)[m].mean())))
mt = pd.DataFrame(rows); mt.to_csv(os.path.join(OUT, "test_monthly_series.csv"), index=False)
dd_ = (mt["xs_r2_img"] - mt["xs_r2_num"]).to_numpy()
st["monthly_xs"] = dict(mean_img=float(mt["xs_r2_img"].mean()), mean_num=float(mt["xs_r2_num"].mean()), mean_diff=float(dd_.mean()), t_nw=float(T.nw_tstat(dd_, 3)), share_pos=float((dd_ > 0).mean()), months=len(dd_))
print("stress", json.dumps(st)[:1500], flush=True)
tab["stress"] = st


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


json.dump(clean(tab), open(os.path.join(OUT, "supplementary_results.json"), "w"), indent=1)
print("DONE", f"{time.time() - t0:.0f}s")
