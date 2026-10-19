# ==================================== STAGE 9: the dashboard files and the krishnaharsha.com page ====================================
# The dashboard (streamlit_app.py in the GitHub repo) loads the model bundle from MODEL_REPO at start-up. Deploy it on Streamlit Cloud (docs/DEPLOY.md);
# an optional Hugging Face Space is built only if DEPLOY_HF_SPACE = True (it needs a paid Hugging Face plan).
SPACE_DIR = os.path.join(WORK, "space")
shutil.rmtree(SPACE_DIR, ignore_errors=True); os.makedirs(SPACE_DIR)
for f in ("tf_render.py", "feat_visible.py"):
    shutil.copy(os.path.join(DST, "scripts", f), SPACE_DIR)
for f in ("tfcv_core.py", "screen_reader.py", "dashboard_logic.py", "app.py"):
    shutil.copy(os.path.join(WORK, f), SPACE_DIR)
req = [f"gradio=={GRADIO_VERSION}", "numpy", "pandas", "pillow", "huggingface_hub", "yfinance", "rapidocr_onnxruntime==1.4.4", "onnxruntime"]
heads = {s["head"] for s in BUNDLE_CFG["targets"].values()}
if "xgb" in heads:
    req.append("xgboost-cpu")                                 # reads model JSON written by the same or older xgboost
if "lgbm" in heads:
    req.append("lightgbm")
if BUNDLE_CFG["backbones"]:                                   # only if a chosen model uses chart images
    req += ["--extra-index-url https://download.pytorch.org/whl/cpu", "torch", f"timm=={BUNDLE_CFG['versions'].get('timm', '')}".rstrip("=")]
open(os.path.join(SPACE_DIR, "requirements.txt"), "w").write("\n".join(req) + "\n")
open(os.path.join(SPACE_DIR, "packages.txt"), "w").write("libgl1\nlibglib2.0-0\n")
open(os.path.join(SPACE_DIR, "README.md"), "w").write(
    "---\ntitle: Computer Vision with Deep Learning Project Dashboard\nemoji: 📈\ncolorFrom: gray\ncolorTo: blue\nsdk: gradio\n"
    f"sdk_version: {GRADIO_VERSION}\npython_version: \"3.11\"\napp_file: app.py\npinned: false\n"
    "short_description: Three chart screenshots to volatility and risk.\n---\n\n"
    "TriFrame dashboard. The model bundle is loaded from a private model repository at start-up.\n")
SPACE_URL = f"https://{SPACE_REPO.replace('/', '-').replace('_', '-').replace('.', '-').lower()}.hf.space"
if UPLOAD and DEPLOY_HF_SPACE:
    try:
        api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", private=SPACE_PRIVATE, exist_ok=True)
        api.upload_folder(folder_path=SPACE_DIR, repo_id=SPACE_REPO, repo_type="space", commit_message=f"dashboard ({RUN_TAG} bundle)")
        api.add_space_variable(SPACE_REPO, "MODEL_REPO", MODEL_REPO)
        if HF_READ_TOKEN:
            api.add_space_secret(SPACE_REPO, "HF_TOKEN", HF_READ_TOKEN)
            print("Space secret HF_TOKEN set from the Colab secret HF_READ_TOKEN.")
        else:
            print("ACTION NEEDED: in the Space settings add a secret HF_TOKEN = a READ-only token (so it can load the private model).")
        print(f"Space pushed: https://huggingface.co/spaces/{SPACE_REPO}  (app URL {SPACE_URL}; first build takes 5-10 min)")
    except Exception as e:
        FAILED_UPLOADS.append((SPACE_DIR, "", SPACE_REPO, "space")); print("Space upload failed:", e)

# ---- the page for krishnaharsha.com (paste into a Custom HTML / Embed block, or upload as its own page)
tm = BUNDLE_CFG["metrics"]["test"]
LIVE_URL = DASHBOARD_URL or (SPACE_URL if (UPLOAD and DEPLOY_HF_SPACE) else "")
IFRAME = (f'<iframe src="{LIVE_URL}{"&" if "?" in LIVE_URL else "?"}embed=true" title="TriFrame dashboard" style="width:100%;height:1700px;border:0;border-radius:8px" loading="lazy"></iframe>'
          if LIVE_URL else "<!-- set DASHBOARD_URL in the config cell once the dashboard is deployed -->")
page = f"""<!-- Computer Vision with Deep Learning Project Dashboard: paste into a Custom HTML / Embed code block -->
<section style="max-width:1100px;margin:0 auto;padding:24px 16px;font-family:system-ui,Segoe UI,Arial,sans-serif;color:#1d2533">
  <h1 style="font-size:32px;margin:0 0 12px">Computer Vision with Deep Learning Project Dashboard</h1>
  <p style="font-size:16px;line-height:1.6">TriFrame tests whether weekly, daily and hourly chart images add information about a US stock's
  next-20-day volatility, drawdown risk and 5-day relative direction beyond plain numbers from the same price history. Seven frozen,
  ImageNet-pretrained vision models were compared with a numerical baseline on 979 US stocks, with chronological training,
  validation and test periods and a sealed final holdout.</p>
  <p style="font-size:16px;line-height:1.6"><strong>How to use:</strong> upload three TradingView screenshots of one stock (1W, 1D and 1H).
  The app reads the ticker and the last bar of each picture, checks them against market data, rebuilds the exact inputs the model
  was trained on and returns expected volatility, the chance of a 10% drawdown, the chance of beating the median stock over five
  days, and a NO TRADE decision (the default).</p>
  <p style="font-size:14px;color:#555">Test-period results (2020-2022): volatility R&sup2; {tm['V'].get('R2', float('nan')):.2f};
  drawdown AUC {tm['D10'].get('AUC', float('nan')):.2f}; direction AUC {tm['O5'].get('AUC', float('nan')):.3f}.
  Research output, not investment advice.</p>
  {IFRAME}
</section>
"""
WEB = os.path.join(WORK, "web", "krishnaharsha_cv_dashboard.html")
open(WEB, "w").write(page)
shutil.copy(WEB, os.path.join(DRIVE_DIR, os.path.basename(WEB)))
push(WEB, "web/krishnaharsha_cv_dashboard.html")
print(f"Website snippet: {WEB} (also on Drive: {DRIVE_DIR}).")

print(f"Model bundle the dashboard must load: {MODEL_REPO}")
print("Next: deploy streamlit_app.py from the GitHub repo on Streamlit Cloud (docs/DEPLOY.md), with secrets HF_TOKEN (read-only) and "
      f"MODEL_REPO = \"{MODEL_REPO}\".")
