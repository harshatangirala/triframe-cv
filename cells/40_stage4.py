# ==================================== STAGE 4: frozen-backbone embeddings for every chart (the GPU-heavy step) ====================================
t0 = time.time()
NAMES = BACKBONE_NAMES or list(T.BACKBONES)
EMB_DIR = os.path.join(WORK, f"emb_{RUN_TAG}")
rows_e = st._idx.set_index("sample_id").loc[df["sample_id"].values].reset_index()      # store rows in df order
reader = T.FastTriplets(st)
n_chk = reader.parity(rows_e, n=200)
print(f"Fast chart reader matches ImageStore.triplet on {n_chk} rows (including patched hourly charts).")

manifest = dict(names=NAMES, chunk_tickers=CHUNK_TICKERS, n_rows=len(rows_e),
                tickers_sha=hashlib.sha256(",".join(dict.fromkeys(rows_e["ticker"])).encode()).hexdigest())
mpath = os.path.join(EMB_DIR, "manifest.json")
if os.path.exists(mpath):
    assert json.load(open(mpath)) == manifest, "embedding manifest changed (backbones, rows or chunk size); use a new WORK folder"
json.dump(manifest, open(mpath, "w"), indent=1)
push(mpath, f"emb_{RUN_TAG}/manifest.json")


EMB_DRIVE = os.path.join(DRIVE_DIR, f"emb_{RUN_TAG}")


def have_chunk(fname):
    """Fetch a chunk finished in an earlier Colab session (Drive copy first, then Hugging Face) so the run resumes."""
    if os.path.exists(os.path.join(EMB_DRIVE, fname)):
        shutil.copy(os.path.join(EMB_DRIVE, fname), os.path.join(EMB_DIR, fname))
        return os.path.join(EMB_DIR, fname)
    if UPLOAD and UPLOAD_EMBEDDINGS:
        try:
            return hf_hub_download(RUNS_REPO, f"emb_{RUN_TAG}/{fname}", repo_type="dataset", local_dir=WORK, token=HF_TOKEN)
        except Exception:
            return None
    return None


def keep_chunk(path):
    if UPLOAD and UPLOAD_EMBEDDINGS:
        push(path, f"emb_{RUN_TAG}/{os.path.basename(path)}")
    else:
        os.makedirs(EMB_DRIVE, exist_ok=True)
        shutil.copy(path, os.path.join(EMB_DRIVE, os.path.basename(path)))


records = T.extract_embeddings(st, rows_e, NAMES, EMB_DIR, DEVICE, chunk_tickers=CHUNK_TICKERS, batch=BATCH,
                               factory=globals().get("FACTORY"), on_chunk=keep_chunk, have_chunk=have_chunk)
bpath = os.path.join(RES, "backbones.json")
json.dump(records, open(bpath, "w"), indent=1)
push(bpath, f"results_{RUN_TAG}/backbones.json")
print(f"Embeddings ready for {len(NAMES)} backbones x 3 views x {len(rows_e):,} charts in {(time.time() - t0) / 60:.1f} min.")
print("Backbones:", ", ".join(f"{k} ({v['timm_name']})" for k, v in records.items()))
