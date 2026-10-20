"""Runs the notebook's code cells locally on CPU (QUICK, 20 stocks, tiny stand-in backbones, no uploads)."""
import json, os, sys, time

NB, WORK = sys.argv[1], sys.argv[2]
os.environ.update(TFCV_LOCAL_TEST="1", TFCV_DATA_ROOT="C:/Users/harsh/Downloads/us_stocks_ohlcv", TFCV_WORK=WORK,
                  MPLBACKEND="Agg")                                 # no chart windows when run as a script (Colab draws inline)
nb = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "__main__"}


def stub_factory(name, spec, device):
    """Tiny random CNN standing in for a timm backbone (local plumbing test only)."""
    import torch, torch.nn as nn
    torch.manual_seed(abs(hash(name)) % 1000)
    m = nn.Sequential(nn.Conv2d(3, 8, 5, stride=4), nn.ReLU(), nn.Conv2d(8, 16, 3, stride=2), nn.ReLU(),
                      nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(256, 32)).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rec = dict(name=name, timm=spec["timm"], timm_name=spec["timm"], pretrained=bool(spec.get("pretrained", True)),
               seed=int(spec.get("seed", 0)), kwargs=spec.get("kwargs", {}), mean=mean.flatten().tolist(), std=std.flatten().tolist(),
               hf_hub_id=None, timm_version="stub")
    return m, mean, std, rec


code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(code_cells):
    src = "".join(c["source"])
    t0 = time.time()
    exec(compile(src, f"cell{i}", "exec"), ns)
    if i == 0:                                    # local overrides after the config cell
        ns.update(QUICK_TICKERS=20, SAMPLE_CHAIN=[8, 14, 20], CHUNK_TICKERS=8, N_BOOT=200, LB_BOOT=100, PCA_DIM=16, FACTORY=stub_factory,
                  FT_BACKBONES=["resnet18", "convnext_tiny"], FT_EPOCHS=3, FT_PATIENCE=2, FT_ROWS_PER_EPOCH=800, FT_MAX_STOP_ROWS=400,
                  FT_MAX_VAL_ROWS=400, FT_STUDY_TOP=1, FT_SEEDS=[12], LANDSCAPE_ROWS=64, LANDSCAPE_GRID=5,
                  MC_DRAWS=4000, MC_TDF_GRID=[4, 30], MC_LOWK_GRID=[0.75, 1.0, 1.5],
                  PREVIOUS_RUNS={"pilot bundle (local stand-in)": (os.environ.get("TFCV_PILOT_BUNDLE", ""), 8)})
    if os.environ.get("TFCV_FORCE_FT") and "STAGE 6: compare, select" in src:
        # test hook: pretend a fine-tuned backbone won the volatility selection, so every later stage runs the fine-tuned path
        fs = f"NUM+IMG:{os.environ['TFCV_FORCE_FT']}"
        fit = {k: v for k, v in ns["FITS"][("V", fs, "tree")].items() if k not in ("model", "coef", "intercept")}
        fz = json.load(open(ns["FROZEN_PATH"]))
        fz["selection"]["V"].update(feature_set=fs, head="tree", fit=fit)
        json.dump(fz, open(ns["FROZEN_PATH"], "w"), indent=1, default=float)
        ns["SEL"]["V"].update(feature_set=fs, head="tree", fit=fit)
        print(f"TEST HOOK: volatility selection forced to {fs} / tree")
    print(f"--- cell {i} ok ({time.time() - t0:.0f}s)", flush=True)
print("ALL CELLS RAN")
