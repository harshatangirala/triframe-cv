"""Assembles TriFrame_CV_EndToEnd.ipynb from the tested cell files and modules."""
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
# TriFrame: Computer Vision with Deep Learning, end to end

**What this notebook does, in order** (each stage saves its outputs to Hugging Face before the next one starts):

| Stage | Your step | What happens |
|---|---|---|
| 1 | Load the data | 979 US stocks, 25.8 years of daily/hourly bars (2001-01-02 to 2026-10-09), as chart sets: weekly 104, daily 126 and hourly 140 bars, plus 33 numeric features and the labels. Rows from 2023-01-03 on stay sealed. |
| 2 | Split | Train 2001-2016, validation 2017-2019, test 2020-2022, each with a 20-day purge at the boundary. Holdout 2023 onward. |
| 3 | Baselines | Numbers only: constant, HAR volatility model, linear and boosted-tree heads on the 33 features. |
| 4 | Pretrained models | Frozen embeddings of every chart from 7 ImageNet backbones (ResNet-18, EfficientNet-B0, ViT-B/16, ConvNeXt-Tiny, Swin-T, DINOv2-S, MobileNetV3) plus 3 random-weight controls. Resumable. |
| 5 | Combine timeframes and tune | PCA per view, then image-only, numbers + images (all three views) and single-view heads; ridge/logistic paths and a tree grid tuned on validation. Every configuration is logged. |
| 6 | Pick the best model | Pre-registered rule; H2 test; calibration; the NO TRADE thresholds; everything frozen to a config file. |
| 7 | Test | One look at 2020-2022: metrics with month-block confidence intervals, H1 and H3, what NO TRADE avoids, survivorship sensitivity. |
| 8 | Model bundle | Saved, checked against the notebook's own predictions and against the training inputs, pushed to a private model repo. |
| 9 | Dashboard | Gradio app on a Hugging Face Space + the HTML page for krishnaharsha.com ("Computer Vision with Deep Learning Project Dashboard"). |
| 10 | Holdout (owner, once) | Off by default. Opens 2023 onward through the project's holdout guard with your key. |
| 11 | Report | REPORT.md from computed values only; retries any failed upload. |

**Before you run**
1. Runtime > Change runtime type > **A100 GPU** and **High-RAM** (T4 works, about 3x slower).
2. Colab secrets (key icon): **HF_TOKEN** = a Hugging Face token with *write* access. Optional **HF_READ_TOKEN** = a *read-only* token for the dashboard.
3. First run with `QUICK = True` (60 stocks, roughly 45-90 minutes). If everything passes, set `QUICK = False` and run again (several hours; it resumes after a disconnect).

**Rules this notebook enforces:** data only through the sealed `ImageStore`; preprocessing fitted on training rows only; hyper-parameters chosen on validation only; the test years are read once (a second run needs a written reason); the holdout opens only with your key. Expect chart images to add little or nothing to the numbers: the renderer scales each chart to fit its panel, so pictures cannot show the volatility level. The notebook measures this rather than assuming it.
"""

CELLS = [md(INTRO), code(cell_file("01_config.py")), code(cell_file("02_setup.py")),
         md("## Code modules\nThe next cells write the four modules: shared inference code, training code, the screenshot reader and the dashboard app."),
         module_cell("tfcv_core.py"), module_cell("tfcv_train.py"), module_cell("screen_reader.py"), module_cell("app.py"),
         code(cell_file("04_helpers.py")),
         md("## Stage 1 - load the data\nSealed development rows (2001-10-22 to 2022-12-01) with features and labels. In QUICK mode a fixed random set of stocks is used."),
         code(cell_file("10_stage1.py")),
         md("## Stage 2 - split\nA row joins a split only if its 20-day label window ends inside that split, so no label leaks across a boundary."),
         code(cell_file("20_stage2.py")),
         md("## Stage 3 - numeric baselines\nThe bar every image model has to clear. Volatility is scored by R² on log volatility (with QLIKE and within-month R² alongside); direction and drawdown by log loss and Brier score (with AUC, rank IC and calibration error)."),
         code(cell_file("30_stage3.py")),
         md("## Stage 4 - frozen-backbone embeddings (GPU)\nThe fast chart reader is first checked against `ImageStore.triplet`. Each finished chunk of stocks is saved and pushed to Hugging Face, so a disconnect only loses the chunk in progress. Full run: about 1.5-3 hours on an A100."),
         code(cell_file("40_stage4.py")),
         md("## Stage 5 - combine the timeframes and tune\nPCA per backbone and view (fitted on training rows), then: images only; numbers + all three views (linear and trees); numbers + one view (the single-timeframe condition for H1)."),
         code(cell_file("50_stage5.py")),
         md("## Stage 6 - pick the model, calibrate, freeze\nThe selection rule and the hypothesis tests were written before any test data is read. The frozen config is the only thing later stages use."),
         code(cell_file("60_stage6.py")),
         md("## Stage 7 - one honest test (2020-2022)\nRefits the frozen choices on train + validation and reads the test years once."),
         code(cell_file("70_stage7.py")),
         md("## Stage 8 - model bundle and parity checks\nThe bundle must reproduce the notebook's test predictions, and the dashboard's input code must reproduce the training inputs exactly."),
         code(cell_file("80_stage8.py")),
         md("## Stage 9 - dashboard and website page\nPushes the Gradio app to a Hugging Face Space and writes the HTML block for a new page on krishnaharsha.com."),
         code(cell_file("90_stage9.py")),
         md("## Stage 10 - holdout (owner only, once)\nLeave `RUN_HOLDOUT = False` until you are satisfied with the frozen config. Opening it is logged and single-use."),
         code(cell_file("95_stage10.py")),
         md("## Stage 11 - report and final upload"),
         code(cell_file("99_stage11.py"))]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", OUT, "with", len(CELLS), "cells")
