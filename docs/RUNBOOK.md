# TriFrame CV: how to run the end-to-end notebook

**File:** `TriFrame_CV_EndToEnd.ipynb`. It is self-contained: it writes its own code modules and needs no other files.

## 1. One-time setup (about 5 minutes)
1. Open https://colab.research.google.com, then File → Upload notebook → `TriFrame_CV_EndToEnd.ipynb`.
2. Runtime → Change runtime type → **A100 GPU**, and switch on **High-RAM**. A T4 also works, about 3× slower.
3. Click the key icon (Secrets) in the left bar and add:
   - `HF_TOKEN`: a Hugging Face token with **write** access. Use a new one; the two old tokens were leaked.
   - `HF_READ_TOKEN`: a **read-only** token. The dashboard uses it to load the private model. If you skip it, add it later by hand in the Space settings (Stage 9 prints a reminder).
   - Switch on "Notebook access" for both.

## 2. First run: QUICK = True (60 stocks, about 45-90 minutes)
- Leave the config cell as it is and choose Runtime → Run all. Accept the Google Drive pop-up.
- Lines to check in the output:
  - Stage 1: `Universe: 979 US stocks ... 25.8 years`.
  - Stage 4: `Fast chart reader matches ImageStore.triplet ...` (the run stops if it does not).
  - Stage 8: `Bundle vs notebook test predictions ...` and `Dashboard input pipeline vs training inputs ...: feature mismatches 0, chart mismatches 0`.
  - Stage 9: the Space link. The first build takes 5-10 minutes; then try it with three screenshots.

## 3. Full run: QUICK = False (all 979 stocks, about 4-8 hours on an A100)
- Set `QUICK = False` in the config cell and Run all again.
- Embeddings are saved chunk by chunk, so a disconnect loses only the chunk in progress. Run all again and it resumes.
- The full run stores about 20 GB of embeddings on Hugging Face. If your Hugging Face storage is too small, set `UPLOAD_EMBEDDINGS = False`, and they go to Google Drive instead.
- The test years (2020-2022) are read once per run type. A second evaluation needs `os.environ['TFCV_TEST_REUSE_REASON'] = '<why>'`.

## 4. Decisions that are yours
| Decision | Default | Notes |
|---|---|---|
| Open the 2023+ holdout (Stage 10) | off | Only when you are happy with the frozen config. Compute its SHA-256 yourself (`!sha256sum <path>`) and paste it when asked. It can be opened once per config, and every opening is logged. |
| Public Space | public | The website iframe needs a public Space. The model files stay in the private model repo. |
| Live data source | Yahoo Finance (yfinance) | Yahoo's terms limit commercial use. For a public product, consider a licensed data API. |
| Directional calls | shown only if both validation and the test years show an edge | Otherwise the dashboard gives risk outputs and NO TRADE. |

## 5. Put the dashboard on krishnaharsha.com
1. After Stage 9, open `MyDrive/triframe_cv/krishnaharsha_cv_dashboard.html`. It is also in the Hugging Face runs repo under `web/`.
2. On your site, add a new page titled **Computer Vision with Deep Learning Project Dashboard**.
   - Hostinger Website Builder: Add element → **Embed code**.
   - WordPress: add a **Custom HTML** block.
3. Paste the whole HTML file in and publish. The page shows the project description and embeds the live dashboard.

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

If the dashboard reports "no daily data" for a ticker that exists, Yahoo is rate-limiting the Space's server. Try again a few minutes later. For a public product, switch `fetch_yahoo` in `tfcv_core.py` to a licensed data API.

## 7. After the full run, send back
`REPORT_full.md` (Drive: `MyDrive/triframe_cv/`) and the notebook with its outputs, for review.
