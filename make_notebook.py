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
# TriFrame: Computer Vision with Deep Learning, end to end (v4)

**What is new in v4** (the professor's requests and an external review, both answered in the notebook):
1. **500 stocks** (the 60 of the pilot and the 120 of v3 are inside them, so all three runs compare on the same rows).
2. **"Did training reach the global minimum?"** (Stage 6D): the ridge and logistic heads are convex, so their global minimum is
   *certified* (zero gradient, identical answers from random starts, a single bowl); boosted trees keep the minimum of their
   validation curve; for the fine-tuned networks, which are not convex, no method can certify a global minimum: the notebook shows
   convergence, the minimum of the held-out loss, loss-landscape plots and seed-to-seed stability instead.
3. **Precision, recall, F1, ROC-AUC, R², RMSE with visualisations** (Stages 6L and 7E): ROC and precision-recall curves, confusion
   matrices, threshold curves, reliability plots, predicted-vs-realised, residuals, by-year bars.
4. **Loss, training and validation curves** (Stages 5F and 6D): per epoch and per step for every fine-tuned network, boosting curves,
   regularisation curves and learning curves (score vs number of training stocks).
5. **Higher R² where it can honestly come from**: a local test (validation years only) showed that more stocks is the lever that
   works (numbers-only R² 0.487 with 120 stocks, 0.526 with 500, 0.531 with all 979) while a wider tree search and market-wide
   features did not help; v4 also fixes fine-tuning so it no longer uses the selection years, and adds a controlled fine-tuning study.
6. **The external review**: an audit stage (Stage 0), the exposed holdout and the already-read test years labelled as such, volatility
   baselines (rolling, EWMA, GARCH), corrected QLIKE and rank IC, thresholds fixed on validation, and a provenance record for every metric.

| Stage | What happens |
|---|---|
| 0 | Audit: run manifest (code and config hashes), the holdout's access history, metric unit tests. |
| 1-2 | Load 500 stocks (nested with the earlier runs), split 2001-2016 / 2017-2019 / 2020-2022 with a 20-day purge; effective sample sizes and event rates by year. |
| 3 | Baselines: constant, HAR, rolling 22-day volatility, EWMA, GARCH(1,1), and the 33 numbers (linear and boosted trees). |
| 4-5 | Frozen embeddings from 7 ImageNet backbones + 3 random controls; PCA; image, numbers + images and single-view heads tuned on validation. |
| 5F | Fine-tuning with early stopping on 2015-2016 (inside the training years); controlled study (last 25% / 50% / risk-only); extra seeds; learning curves. |
| 6 | Pre-registered selection (gain CI above zero AND a gain in every validation year), calibration, thresholds, freeze. |
| 6D | Did training reach the minimum? Certificates, curves, loss landscape. |
| 6L | Model cards and the validation leaderboard (all metrics incl. precision / recall / F1). |
| 7 | The frozen choices on 2020-2022 (historical diagnostic), test leaderboard, comparison with the 60- and 120-stock runs on the same rows. |
| 7E | Evaluation charts: ROC, precision-recall, confusion matrix, reliability, predicted vs realised. |
| 7D | Direction re-examined as a continuous ranking (weekly rank IC with a Newey-West t-statistic). |
| 7M | Monte Carlo calibration and coverage by risk decile and year. |
| 8-9 | Model bundle with parity checks; website snippet; the models the dashboard uses. |
| 10 | The 2023+ period: exposed, off by default. |
| 11 | REPORT.md, the PDF report, metric provenance, final upload. |

**Before you run**: Runtime > Change runtime type > **A100 GPU** + **High-RAM** (a T4 would take about three times as long).
Colab secret **HF_TOKEN** with write access. Then Runtime > Run all. With everything on, the 500-stock run takes about 6-8 hours;
it resumes after a disconnect (embeddings and fine-tuned weights are saved as they finish). `FT_STUDY = False` saves about 1.5 hours.

**Honest expectations**: R² on volatility should rise mainly because of the extra stocks; chart images have so far added little or
nothing beyond the numbers on the test years, and the 5-day direction has been a coin flip. The notebook measures all of this rather
than assuming it, and labels every result with the data it came from.
"""

CELLS = [md(INTRO), code(cell_file("01_config.py")), code(cell_file("02_setup.py")),
         md("## Code modules\nThe next cells write the modules: shared inference code (including the Monte Carlo engine), training code "
            "(fine-tuning, metrics, baselines, optimisation checks), the screenshot reader, the dashboard logic, the PDF report and the Gradio app."),
         module_cell("tfcv_core.py"), module_cell("tfcv_train.py"), module_cell("screen_reader.py"), module_cell("dashboard_logic.py"),
         module_cell("tfcv_report.py"), module_cell("app.py"),
         code(cell_file("04_helpers.py")),
         md("## Stage 0 - audit, provenance and metric tests (v4)\nThe run's code and configuration hashes, the access history of the 2023+ "
            "period and of the test years, and unit tests of the metric code (QLIKE on the variance scale, Brier skill reference, confusion "
            "counts, weekly rank IC). Nothing is trained here."),
         code(cell_file("05_audit.py")),
         md("## Stage 1 - load the data\nSealed development rows (2001-10-22 to 2022-12-01) with features and labels. The 500 stocks contain "
            "the pilot's 60 and v3's 120, drawn in the same order with the same seeds."),
         code(cell_file("10_stage1.py")),
         md("## Stage 2 - split\nA row joins a split only if its 20-day label window ends inside that split. Effective sample sizes (dates, "
            "weeks, events) and event rates by year are printed, because the rows are not independent."),
         code(cell_file("20_stage2.py")),
         md("## Stage 3 - baselines\nThe bar every image model has to clear, including the volatility baselines the review asked for: rolling "
            "22-day volatility, EWMA and GARCH(1,1) (parameters estimated on the training years only)."),
         code(cell_file("30_stage3.py")),
         md("## Stage 4 - frozen-backbone embeddings (GPU)\nThe fast chart reader is first checked against `ImageStore.triplet`. Each finished "
            "chunk of stocks is saved, so a disconnect only loses the chunk in progress."),
         code(cell_file("40_stage4.py")),
         md("## Stage 5 - combine the timeframes and tune\nPCA per backbone and view (fitted on training rows), then images only, numbers + all "
            "three views, numbers + one view. The ledger logs training and validation scores of every setting."),
         code(cell_file("50_stage5.py")),
         md("## Stage 5F - fine-tuning (v4 protocol)\nEvery pretrained backbone is fine-tuned end to end. **Early stopping now uses 2015-2016, "
            "inside the training years**, so the 2017-2019 validation years stay clean for selection (in v3 they were used for both, which "
            "flattered fine-tuning). Up to 8 epochs, patience 2, a fresh draw of training rows every epoch. A controlled study on the two "
            "best backbones trains only the last 25% or 50% of the network, or uses a risk-only loss; extra seeds show run-to-run variation. "
            "Learning curves (per epoch and per step) and frozen-vs-fine-tuned comparisons with confidence intervals. Resumable per run."),
         code(cell_file("55_stage5f.py")),
         md("## Stage 6 - pick the model, calibrate, fix thresholds, freeze\nThe pre-registered rule now also requires an image model's gain to "
            "hold in every validation year. Thresholds for precision / recall / F1 are fixed here, on validation."),
         code(cell_file("60_stage6.py")),
         md("## Stage 6D - did training reach the minimum?\nConvex heads: certified global minimum. Boosted trees: training vs validation loss "
            "with the kept minimum. Regularisation curves. Learning curves vs the number of training stocks. The fine-tuned network: loss "
            "landscape (straight line from the starting point, random planes around the solution) and seeds. A global minimum cannot be "
            "certified for a neural network by any method; this stage shows what can be shown."),
         code(cell_file("63_stage6d.py")),
         md("## Stage 6L - the model leaderboard (validation)\nModel cards, then every model on every metric (including precision, recall and "
            "F1 at each model's validation threshold), with the volatility baselines as references."),
         code(cell_file("65_stage6l.py")),
         md("## Stage 7 - the frozen choices on 2020-2022 (historical diagnostic)\nThese years were read by earlier runs, so nothing here can "
            "change a choice. Test metrics with month-block CIs (and a quarter-block check), per-year results, the test leaderboard, and the "
            "comparison with the 60- and 120-stock runs on their own stocks and rows."),
         code(cell_file("70_stage7.py")),
         md("## Stage 7E - evaluation metrics, visually\nPrecision, recall, F1, ROC-AUC, R² and RMSE for the chosen models and the numbers-only "
            "model, on validation and test: ROC and precision-recall curves, threshold curves, confusion matrices, reliability plots, "
            "predicted vs realised volatility, residuals, decile calibration and R² by year."),
         code(cell_file("72_stage7e.py")),
         md("## Stage 7D - the direction signal re-examined\nA separate, pre-declared experiment: the continuous 5-day excess return, scored as a "
            "weekly cross-sectional ranking (rank IC with a Newey-West t-statistic)."),
         code(cell_file("74_stage7d.py")),
         md("## Stage 7M - Monte Carlo price simulation\nTail shape and intraday range chosen on validation; coverage checked on the test years, "
            "also by predicted-risk decile and by year, with the width of the bands."),
         code(cell_file("75_stage7m.py")),
         md("## Stage 8 - model bundle and parity checks\nThe bundle must reproduce the notebook's predictions, the dashboard's input code must "
            "reproduce the training inputs, and the dashboard's embedding code must reproduce the notebook's embeddings."),
         code(cell_file("80_stage8.py")),
         md("## Stage 9 - dashboard files and website page"),
         code(cell_file("90_stage9.py")),
         md("### Which models the dashboard uses\nThe exact frozen models behind each dashboard output, why each was chosen, and an example of "
            "the dashboard's plain-language explanation on a real test-year row."),
         code(cell_file("92_models_in_use.py")),
         md("## Stage 10 - the 2023+ period (exposed)\nOpened for the pilot and for v3, so it is no longer an untouched test. Off by default."),
         code(cell_file("95_stage10.py")),
         md("## Stage 11 - report, PDF, provenance and final upload\nREPORT.md, the plain-language PDF (with the new sections on minima, curves "
            "and metrics), and metrics_provenance.csv (every metric with its run, code and config hash, rows, calibration status and CI method)."),
         code(cell_file("99_stage11.py"))]

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", OUT, "with", len(CELLS), "cells")
