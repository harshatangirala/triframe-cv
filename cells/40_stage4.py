# ==================================== STAGE 4: frozen-backbone embeddings for every chart (the GPU-heavy step) ====================================
t0 = time.time()
NAMES = BACKBONE_NAMES or list(T.BACKBONES)
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


def chunk_store(local_dir, repo_dir):
    """(have_chunk, keep_chunk) for one embedding folder: fetch a chunk finished in an earlier Colab session (Drive copy first,
    then Hugging Face) so the run resumes; keep each new chunk on Hugging Face (or Drive if embeddings are not uploaded)."""
    drive = os.path.join(DRIVE_DIR, repo_dir)

    def have_chunk(fname):
        if os.path.exists(os.path.join(drive, fname)):
            shutil.copy(os.path.join(drive, fname), os.path.join(local_dir, fname))
            return os.path.join(local_dir, fname)
        if UPLOAD and UPLOAD_EMBEDDINGS:
            try:
                got = hf_hub_download(RUNS_REPO, f"{repo_dir}/{fname}", repo_type="dataset", local_dir=os.path.join(WORK, "_dl"), token=HF_TOKEN)
                shutil.move(got, os.path.join(local_dir, fname))
                return os.path.join(local_dir, fname)
            except Exception:
                return None
        return None

    def keep_chunk(path):
        if UPLOAD and UPLOAD_EMBEDDINGS:
            push(path, f"{repo_dir}/{os.path.basename(path)}")
        else:
            os.makedirs(drive, exist_ok=True)
            shutil.copy(path, os.path.join(drive, os.path.basename(path)))
    return have_chunk, keep_chunk


have_chunk, keep_chunk = chunk_store(EMB_DIR, f"emb_{RUN_TAG}")
records = T.extract_embeddings(st, rows_e, NAMES, EMB_DIR, DEVICE, chunk_tickers=CHUNK_TICKERS, batch=BATCH,
                               factory=globals().get("FACTORY"), on_chunk=keep_chunk, have_chunk=have_chunk)
bpath = os.path.join(RES, "backbones.json")
json.dump(records, open(bpath, "w"), indent=1)
push(bpath, f"results_{RUN_TAG}/backbones.json")
print(f"Embeddings ready for {len(NAMES)} backbones x 3 views x {len(rows_e):,} charts in {(time.time() - t0) / 60:.1f} min.")
print(pd.DataFrame([{"backbone": k, "timm weights": v["timm_name"], "parameters (M)": v["n_params"] / 1e6, "embedding size": v["emb_dim"],
                     "images/s on this GPU": v["infer_img_per_s"]} for k, v in records.items()]).round(1).to_string(index=False))
