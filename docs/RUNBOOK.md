# TriFrame CV: how to run the end-to-end notebook

**File:** `TriFrame_CV_EndToEnd.ipynb`. It is self-contained: it writes its own code modules and needs no other files.

## 1. One-time setup (about 5 minutes)
1. Open https://colab.research.google.com, then File → Upload notebook → `TriFrame_CV_EndToEnd.ipynb`.
2. Runtime → Change runtime type → **A100 GPU**, and switch on **High-RAM**. A T4 also works, about 3× slower.
3. Click the key icon (Secrets) in the left bar and add:
   - `HF_TOKEN`: a Hugging Face token with **write** access. Use a new one; the two old tokens were leaked.
   - Switch on "Notebook access" for both.

## 2. First run: QUICK = True (60 stocks, about 45-90 minutes)
- Leave the config cell as it is and choose Runtime → Run all. Accept the Google Drive pop-up.
- Lines to check in the output:
  - Stage 1: `Universe: 979 US stocks ... 25.8 years`.
  - Stage 4: `Fast chart reader matches ImageStore.triplet ...` (the run stops if it does not).
  - Stage 8: `Bundle vs notebook test predictions ...` and `Dashboard input pipeline vs training inputs ...: feature mismatches 0, chart mismatches 0`.
  - Stage 9: the website snippet and the model repo name the dashboard must load. Then follow `docs/DEPLOY.md` to put the dashboard online.

## 3. Full run: QUICK = False (all 979 stocks, about 2.5-4 hours on an A100)
- Set `QUICK = False` in the config cell and Run all again. The pilot took 3.4 minutes to embed 27,313 charts and 4.2 minutes to fit
  the models; the full run is about 16 times that data.
- The full run writes its model to its own repo, `triframe-cv-model-full`, so the live dashboard keeps using the pilot model until
  you switch it (see `docs/DEPLOY.md`).
- Embeddings are saved chunk by chunk, so a disconnect loses only the chunk in progress. Run all again and it resumes.
- The full run stores about 20 GB of embeddings on Hugging Face. If that is too much, set `UPLOAD_EMBEDDINGS = False`; they then go to
  Google Drive (which also needs about 20 GB free).
- The test years (2020-2022) are read once per run type. A second evaluation needs `os.environ['TFCV_TEST_REUSE_REASON'] = '<why>'`.
- The sealed holdout (Stage 10) stays off until you set `RUN_HOLDOUT = True` and paste the frozen config's SHA-256. The full run has
  its own frozen config, so it gets its own single look.

## 4. Decisions that are yours
| Decision | Default | Notes |
|---|---|---|
| Open the 2023+ holdout (Stage 10) | off | Only when you are happy with the frozen config. Compute its SHA-256 yourself (`!sha256sum <path>`) and paste it when asked. It can be opened once per config, and every opening is logged. |
| Live data source | Yahoo Finance (yfinance) | Yahoo's terms limit commercial use. For a public product, consider a licensed data API. |
| Directional calls | shown only if both validation and the test years show an edge | Otherwise the dashboard gives risk outputs and NO TRADE. |

## 5. Put the dashboard on krishnaharsha.com
1. Deploy the dashboard on Streamlit Community Cloud (`docs/DEPLOY.md`), which gives a public address such as `https://triframe-cv.streamlit.app`.
2. The project page on the site embeds it with `<iframe src=".../?embed=true">`.

## 6. What the dashboard does
1. You upload three TradingView screenshots of one stock (1W, 1D, 1H).
2. It reads the ticker, timeframe and last-bar O/H/L/C from each screenshot. It was tested on 66 real screenshots: timeframe 100%, legend 100%, and the true ticker was among the candidates 66/66.
3. It confirms the ticker by matching the legend to Yahoo bars. That also pins the analysis to the screenshot's date.
4. It rebuilds the exact training inputs: 33 numbers and the three charts. Verified to reproduce the training data exactly.
5. It returns:
   - expected 20-day volatility with a range;
   - the chance of a 10% drawdown;
   - the chance of beating the median stock over 5 days, with a note on whether that edge is real;
   - a NO TRADE / risk decision with reasons.

## 7. After the full run, send back
`REPORT_full.md` (Drive: `MyDrive/triframe_cv/`) and the notebook with its outputs, for review.
