# ==================================== STAGE 0: audit, provenance and metric unit tests (v4) ====================================
# The review brief asks that every number can be traced, that exposed data is labelled as such, and that the metric code is
# tested before any model is trained. This cell does that; it trains nothing.
print(f"Run {RUN_ID}: notebook {NOTEBOOK_VERSION}, code hash {CODE_HASH}, config hash {CONFIG_HASH}, seed {SEED}")
print("Versions:", {m: __import__(m).__version__ for m in ("numpy", "pandas", "torch", "timm")}, "| device", DEVICE)

# ---- 1. holdout and test-year history (hash-chained access log, kept on Drive and Hugging Face)
glog = os.path.join(GOV, "holdout_access_log.csv")
HOLDOUT_OPENINGS = pd.read_csv(glog) if os.path.exists(glog) else pd.DataFrame()
if len(HOLDOUT_OPENINGS):
    print(f"\nHoldout ({HOLDOUT_START} onward) access log: {len(HOLDOUT_OPENINGS)} earlier opening(s)")
    print(HOLDOUT_OPENINGS[["utc_time", "who", "rows_returned"]].to_string(index=False))
    print("=> The holdout is EXPOSED: it has been looked at, so this run treats it as development data and uses it for no choice.")
else:
    print("\nNo holdout access log found (expected two earlier openings: pilot and v3). Check that Drive / Hugging Face are reachable.")
TEST_SEEN_BY = []
if UPLOAD:
    try:
        TEST_SEEN_BY = sorted(f for f in api.list_repo_files(RUNS_REPO, repo_type="dataset") if f.startswith("frozen/test_opened_"))
    except Exception:
        pass
print(f"Test years {SPLITS['test'][0][:4]}-{int(SPLITS['test'][1][:4]) - 1} already read by: {TEST_SEEN_BY or 'unknown (no listing)'} "
      "=> in this run they are a HISTORICAL DIAGNOSTIC, never used to choose anything.")

# ---- 2. unit tests of the metric code (fail loudly instead of reporting a wrong number)
_rng = np.random.default_rng(0)
_y = _rng.normal(-2.5, 0.5, 2000); _p = _y + _rng.normal(0, 0.3, 2000)
assert abs(T.qlike(_y, _y, 0.0)) < 1e-12, "QLIKE of a perfect variance forecast must be 0"
_rv, _f = np.exp(2 * _y), np.exp(2 * _p + 2 * 0.09)                     # QLIKE is computed on VARIANCE, from log-volatility inputs
assert abs(T.qlike(_y, _p, 0.09) - np.mean(_rv / _f - np.log(_rv / _f) - 1)) < 1e-12, "QLIKE must equal the variance-scale formula"
try:
    T.board_metrics("reg", _y, _p, pd.date_range("2018-01-01", periods=2000, freq="D").values)
    raise AssertionError("board_metrics must refuse to compute QLIKE without the validation residual variance")
except ValueError:
    pass
_yb = (_rng.random(2000) < 0.3).astype(float); _pb = np.clip(0.3 + 0.2 * (_yb - 0.3) + _rng.normal(0, 0.1, 2000), 0.01, 0.99)
_bss = 1 - T.brier(_yb, _pb) / T.brier(_yb, np.full(2000, 0.25))      # Brier skill against a FIXED reference rate (the training base rate)
assert abs(T.board_metrics("clf", _yb, _pb, None, base=0.25)["BSS"] - _bss) < 1e-12, "BSS must use the given reference rate"
_cm = T.class_metrics_at(np.array([1, 1, 0, 0, 1.0]), np.array([0.9, 0.2, 0.8, 0.1, 0.7]), 0.5)
assert (_cm["TP"], _cm["FP"], _cm["FN"], _cm["TN"]) == (2, 1, 1, 1) and abs(_cm["precision"] - 2 / 3) < 1e-12, "confusion counts"
_d = np.repeat(pd.date_range("2018-01-05", periods=30, freq="7D").values, 25); _x = _rng.normal(size=len(_d))
assert abs(T.rank_ic(_d, _x, _x, min_names=20)[0] - 1.0) < 1e-12, "weekly rank IC of a perfect ranking must be 1"


def assert_finite(d, name):
    """A NaN or infinite metric is a failure to explain, not a number to report."""
    bad = [k for k, v in d.items() if isinstance(v, float) and not np.isfinite(v) and not k.startswith("rankIC")]
    assert not bad, f"{name}: non-finite metrics {bad}"


print("\nMetric unit tests passed: QLIKE on the variance scale (and refused without validation residuals), Brier skill against a "
      "fixed reference rate, confusion counts, weekly rank IC.")
print("Other checks run later: purge across split boundaries (Stage 2), preprocessing fitted on training rows only (Stages 3-5),"
      " dashboard features and charts identical to training (Stage 8).")
