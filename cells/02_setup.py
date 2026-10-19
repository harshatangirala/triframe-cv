# ==================================== SETUP: packages, Drive, Hugging Face, data, integrity ====================================
import os, sys, json, time, math, glob, shutil, hashlib, random, subprocess, warnings, gc, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

IN_COLAB = "google.colab" in sys.modules
LOCAL_TEST = os.environ.get("TFCV_LOCAL_TEST") == "1"
RUN_TAG = "quick" if QUICK else "full"
DST = os.environ.get("TFCV_DATA_ROOT", "/content/triframe").rstrip("/")
WORK = os.environ.get("TFCV_WORK", "/content/tfcv")
for sub in ("", "results", "figures", "frozen", "bundle", "space", "web", f"emb_{RUN_TAG}", f"pca_{RUN_TAG}"):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)

if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "xgboost", "lightgbm", "yfinance",
                    "huggingface_hub>=0.25", "pyarrow", "pandas_market_calendars"], check=True)
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/triframe_cv"

    def _secret(name):
        try:
            return userdata.get(name)
        except Exception:
            return None
    HF_TOKEN, HF_READ_TOKEN = _secret("HF_TOKEN"), _secret("HF_READ_TOKEN")
else:
    DRIVE_DIR = os.path.join(WORK, "drive_mirror")
    HF_TOKEN, HF_READ_TOKEN = os.environ.get("HF_TOKEN"), None
os.makedirs(DRIVE_DIR, exist_ok=True)
if LOCAL_TEST:
    UPLOAD = False

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0.0
BATCH = 384 if GPU_GB > 30 else 160
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}, {GPU_GB:.0f} GB) | run: {RUN_TAG} | upload: {UPLOAD}")
if DEVICE != "cuda" and not LOCAL_TEST:
    print("WARNING: no GPU. Runtime > Change runtime type > A100 (or T4). The embedding stage needs a GPU.")

from huggingface_hub import HfApi, snapshot_download, hf_hub_download
api = HfApi(token=HF_TOKEN) if HF_TOKEN else None
if UPLOAD:
    assert HF_TOKEN, "Add a Hugging Face token with WRITE access as the Colab secret HF_TOKEN (key icon on the left)."
    print("Hugging Face user:", api.whoami()["name"])
    for rid, rtype in ((RUNS_REPO, "dataset"), (MODEL_REPO, "model")):
        api.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)

FAILED_UPLOADS = []


def push(local, path_in_repo, repo_id=None, repo_type="dataset", msg=None):
    """Upload a file or folder to Hugging Face. Failures are recorded and retried in Stage 11, never silently dropped."""
    if not UPLOAD:
        return
    repo_id = repo_id or RUNS_REPO
    try:
        if os.path.isdir(local):
            api.upload_folder(folder_path=local, path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type,
                              commit_message=msg or f"update {path_in_repo}")
        else:
            api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo_id, repo_type=repo_type,
                            commit_message=msg or f"update {path_in_repo}")
    except Exception as e:
        FAILED_UPLOADS.append((local, path_in_repo, repo_id, repo_type))
        print(f"   upload failed for {path_in_repo}: {e} (will be retried in Stage 11)")


# ---- data: download once (about 13 GB), then verify sizes and sample checksums against the manifests
if IN_COLAB:
    print("Downloading / refreshing the TriFrame dataset from Hugging Face (first run takes 10-20 minutes)...")
    snapshot_download(DATA_REPO, repo_type="dataset", local_dir=DST, token=HF_TOKEN, max_workers=8)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


bad, n_files = [], 0
for man in ("cache/MANIFEST_cache.json", "cache_2001_2009/MANIFEST_cache_2001_2009.json",
            "cache_patch_r11/MANIFEST_cache_patch_r11.json", "features_visible/MANIFEST_features_visible.json"):
    m = json.load(open(f"{DST}/{man}"))["manifest"]
    for k, v in m.items():
        n_files += 1
        if not os.path.exists(f"{DST}/{k}") or os.path.getsize(f"{DST}/{k}") != v["bytes"]:
            bad.append(k)
    for k in random.Random(7).sample(sorted(m), min(8, len(m))):
        if k not in bad and _sha(f"{DST}/{k}") != m[k]["sha256"]:
            bad.append(k)
assert not bad, f"integrity check failed for {len(bad)} files, e.g. {bad[:3]}"
print(f"Integrity: {n_files:,} files match their manifests (sizes + sampled SHA-256).")

# ---- governance: the holdout access log must survive Colab resets (Drive + Hugging Face copy)
GOV = os.path.join(DRIVE_DIR, "governance")
os.makedirs(GOV, exist_ok=True)
os.environ["TRIFRAME_GOV_DIR"] = GOV
if UPLOAD and not os.path.exists(os.path.join(GOV, "holdout_access_log.csv")):
    try:
        hf_hub_download(RUNS_REPO, "governance/holdout_access_log.csv", repo_type="dataset", local_dir=DRIVE_DIR, token=HF_TOKEN)
        print("Restored the holdout access log from Hugging Face.")
    except Exception:
        pass
for p in (os.path.join(DST, "scripts"), WORK):
    if p not in sys.path:
        sys.path.insert(0, p)
