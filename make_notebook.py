"""Assembles TriFrame_CV_EndToEnd.ipynb (v3) from the tested cell files and modules."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "TriFrame_CV_EndToEnd.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def cell_file(name):
    return open(os.path.join(HERE, "cells", name), encoding="utf-8").read()


def module_cell(fname):
    src = open(os.path.join(HERE, fname), encoding="utf-8").read()
    assert "'''" not in src, f"{fname} contains triple single quotes"
    return code(f"# writes {fname} (used by this notebook and, later, by the dashboard)\n"
                f"open(os.path.join(WORK, {fname!r}), 'w', encoding='utf-8').write(r'''{src}''')\nprint('wrote', {fname!r})")


INTRO = r"""
# TriFrame: Computer Vision with Deep Learning, end to end (v3)

**What is new in v3:** (1) 120 stocks instead of 60, and the sample contains the pilot's 60, so the two runs compare like for like;
(2) a leaderboard of every pretrained model on six metrics per target, with charts; (3) a 5-page plain-language PDF report;
(4) fine-tuning of all seven pretrained networks, scored against their frozen versions; (5) a clear table of the models the
dashboard uses; (6)-(9) the dashboard (in the GitHub repo) now analyses up to four companies at once, with charts, a risk ranking,
detailed plain-language explanations and a Monte Carlo simulation calibrated here in Stage 7M.

| Stage | Your step | What happens |
|---|---|---|
| 1 | Load the data | 979 US stocks, 25.8 years of daily/hourly bars (2001-01-02 to 2026-10-09), as chart sets: weekly 104, daily 126 and hourly 140 bars, plus 33 numeric features and the labels. This run: 120 stocks (the pilot's 60 + 60 more). Rows from 2023-01-03 on stay sealed. |
| 2 | Split | Train 2001-2016, validation 2017-2019, test 2020-2022, each with a 20-day purge at the boundary. Holdout 2023 onward. |
| 3 | Baselines | Numbers only: constant, HAR volatility model, linear and boosted-tree heads on the 33 features. |
| 4 | Pretrained models | Frozen embeddings of every chart from 7 ImageNet backbones (ResNet-18, EfficientNet-B0, ViT-B/16, ConvNeXt-Tiny, Swin-T, DINOv2-S, MobileNetV3) plus 3 random-weight controls; size and speed of each. Resumable. |
| 5 | Combine timeframes and tune | PCA per view, then image-only, numbers + images (all three views) and single-view heads; ridge/logistic paths and a tree grid tuned on validation. Every configuration is logged. |
| 5F | **Fine-tuning (new)** | Each pretrained backbone is trained end to end on the training years (early stopping on validation); its embeddings compete with the frozen ones. Frozen vs fine-tuned report with confidence intervals and training curves. |
| 6 | Pick the best model | Pre-registered rule; H2 test; calibration; the NO TRADE thresholds; everything frozen to a config file. |
| 6L | **Leaderboard (new)** | Model cards (what each network is, pretraining data, size, speed) and every model on six metrics per target, with bars, a heatmap and accuracy-vs-size charts. |
| 7 | Test | One look at 2020-2022: metrics with month-block confidence intervals, H1 and H3, what NO TRADE avoids, survivorship sensitivity, the test leaderboard, and 120 vs 60 stocks on the same rows. |
| 7M | **Monte Carlo (new)** | The dashboard's price simulation: tail shape and intraday range chosen on validation, coverage checked on test. |
| 8 | Model bundle | Saved with the simulation settings and risk percentiles, checked against the notebook's predictions, the training inputs and the dashboard's own embedding code, pushed to a private model repo. |
| 9 | Dashboard | The HTML block for krishnaharsha.com, then **which models the dashboard uses** and an example of its explanation. |
| 10 | Holdout (owner, once) | Off by default. Opens 2023 onward through the project's holdout guard with your key. |
| 11 | Report | REPORT.md and the **5-page PDF report** from computed values only (saved to Drive, Hugging Face and your Downloads). |

**Before you run**
1. Runtime > Change runtime type > **A100 GPU** and **High-RAM** (T4 works, about 3x slower).
2. Colab secrets (key icon): **HF_TOKEN** = a Hugging Face token with *write* access. Optional **HF_READ_TOKEN** = a *read-only* token for the dashboard.
3. Runtime > Run all. The 120-stock run takes about 2-3 hours on an A100 (fine-tuning is about half of it) and resumes after a disconnect.

**Rules this notebook enforces:** data only through the sealed `ImageStore`; preprocessing fitted on training rows only; hyper-parameters (and fine-tuning's early stopping, and the simulation's settings) chosen on validation only; the test years are read once (a second run needs a written reason); the holdout opens only with your key; trained weights stay in private repositories. Expect chart images to add little or nothing to the numbers: the renderer scales each chart to fit its panel, so pictures cannot show the volatility level. The notebook measures this rather than assuming it.
"""

CELLS = [md(INTRO), code(cell_file("01_config.py")), code(cell_file("02_setup.py")),
         md("## Code modules\nThe next cells write the modules: shared inference code (including the Monte Carlo engine), training code "
            "(including fine-tuning), the screenshot reader, the dashboard logic, the PDF report and the Gradio app."),
         module_cell("tfcv_core.py"), module_cell("tfcv_train.py"), module_cell("screen_reader.py"), module_cell("dashboard_logic.py"),
         module_cell("tfcv_report.py"), module_cell("app.py"),
         code(cell_file("04_helpers.py")),
         md("## Stage 1 - load the data\nSealed development rows (2001-10-22 to 2022-12-01) with features and labels. In QUICK mode a fixed "
            "random sample of stocks is used: the pilot's 60 plus 60 more."),
         code(cell_file("10_stage1.py")),
         md("## Stage 2 - split\nA row joins a split only if its 20-day label window ends inside that split, so no label leaks across a boundary."),
         code(cell_file("20_stage2.py")),
         md("## Stage 3 - numeric baselines\nThe bar every image model has to clear. Volatility is scored by R² on log volatility (with QLIKE "
            "and within-month R² alongside); direction and drawdown by log loss and Brier score (with AUC, rank IC and calibration error)."),
         code(cell_file("30_stage3.py")),
         md("## Stage 4 - frozen-backbone embeddings (GPU)\nThe fast chart reader is first checked against `ImageStore.triplet`. Each finished "
            "chunk of stocks is saved and pushed to Hugging Face, so a disconnect only loses the chunk in progress. Size and speed of every "
            "network are recorded for the model cards."),
         code(cell_file("40_stage4.py")),
         md("## Stage 5 - combine the timeframes and tune\nPCA per backbone and view (fitted on training rows), then: images only; numbers + "
            "all three views (linear and trees); numbers + one view (the single-timeframe condition for H1)."),
         code(cell_file("50_stage5.py")),
         md("## Stage 5F - fine-tuning (new in v3)\n**Until v2 every backbone was frozen.** Here each of the 7 pretrained backbones is "
            "fine-tuned end to end: one network embeds the weekly, daily and hourly chart, a small head adds the 33 numbers and predicts "
            "volatility, direction and drawdown together. Training years only, AdamW with a cosine schedule, early stopping on the "
            "validation loss, no augmentation (a flipped chart means something else). The fine-tuned embeddings then go through the same "
            "PCA and heads as the frozen ones and compete under the same selection rule. The cell reports, per backbone, frozen vs "
            "fine-tuned validation scores with month-block confidence intervals, the network's own predictions, the training curves and "
            "the GPU time, so *is fine-tuning worth it?* gets a measured answer. Resumable per backbone."),
         code(cell_file("55_stage5f.py")),
         md("## Stage 6 - pick the model, calibrate, freeze\nThe selection rule and the hypothesis tests were written before any test data "
            "is read. The frozen config is the only thing later stages use."),
         code(cell_file("60_stage6.py")),
         md("## Stage 6L - the model leaderboard (new in v3)\nWhich pretrained models were used and what they are (model cards), then every "
            "model, frozen and fine-tuned, on six metrics per target with 95% confidence intervals: R², within-month R², QLIKE, MAE, RMSE "
            "and Spearman for volatility; AUC, average precision, accuracy, log loss, Brier skill and calibration error for drawdown and "
            "direction. Charts: ranked bars, an all-metrics heatmap, and accuracy against model size and speed. Validation years only, so "
            "nothing here can change the frozen choices."),
         code(cell_file("65_stage6l.py")),
         md("## Stage 7 - one honest test (2020-2022)\nRefits the frozen choices on train + validation and reads the test years once. Also: "
            "the same leaderboard on the test years (descriptive), and the 60 vs 120 comparison on the pilot's own stocks and test rows."),
         code(cell_file("70_stage7.py")),
         md("## Stage 7M - Monte Carlo price simulation (new in v3)\nThe dashboard simulates the next 20 trading days from the frozen "
            "volatility forecast and its measured error. Here its tail shape and intraday-low range are chosen on validation, and its "
            "bands are checked once against the real test-year outcomes."),
         code(cell_file("75_stage7m.py")),
         md("## Stage 8 - model bundle and parity checks\nThe bundle must reproduce the notebook's test predictions, the dashboard's input "
            "code must reproduce the training inputs exactly, and the dashboard's embedding code must reproduce the notebook's embeddings."),
         code(cell_file("80_stage8.py")),
         md("## Stage 9 - dashboard files and website page\nThe dashboard is deployed from the GitHub repo on Streamlit Cloud (docs/DEPLOY.md). "
            "This stage writes the HTML block for krishnaharsha.com and names the model repo the dashboard must load."),
         code(cell_file("90_stage9.py")),
         md("### Which models the dashboard uses\nThe exact frozen models behind each dashboard output, why each was chosen, and an example "
            "of the dashboard's plain-language explanation on a real test-year row."),
         code(cell_file("92_models_in_use.py")),
         md("## Stage 10 - holdout (owner only, once)\nLeave `RUN_HOLDOUT = False` until you are satisfied with the frozen config. Opening it "
            "is logged and single-use."),
         code(cell_file("95_stage10.py")),
         md("## Stage 11 - report, PDF and final upload\nREPORT.md and a 5-page plain-language PDF (what was done, step by step; the "
            "pretrained models; the architecture; results; fine-tuning; the Monte Carlo check; the dashboard's models), all from the values "
            "computed above. The PDF is saved to Google Drive (My Drive > triframe_cv), to Hugging Face, and downloaded to your computer."),
         code(cell_file("99_stage11.py"))]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", OUT, "with", len(CELLS), "cells")
