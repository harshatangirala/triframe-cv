# Working paper: What Do Charts Add?

`TriFrame_paper_SSRN_draft.pdf` is the draft. `paper_body.html` holds the text; `build_paper.py` fills the tables from result files and writes `TriFrame_paper.html`, which is printed to PDF with a headless browser.

| Folder | Contents |
|---|---|
| `tables/` | `supplementary_results.json` (every supplementary statistic), monthly series, validation-against-test scores |
| `fig/` | All figures (SVG from `scripts/make_figures.py`; PNG from the executed notebook) |
| `scripts/` | `supplementary_analysis.py` (re-fits and paired tests, uses no rows dated 2023 or later), `make_figures.py` |

To reproduce: run the notebook, then `supplementary_analysis.py <data_root> <run_root> <repo_root> <out_dir>`, `make_figures.py <run_root> <tab_dir> <out_dir>` and `build_paper.py <run_results_s500_dir> <run_results_root>`. The price data are not redistributed; `docs/DATA_CARD.md` describes them. Aggregate run results are in `results/s500_run/`.

Before submitting to SSRN, the author adds an affiliation and email, and confirms the funding and conflict-of-interest statement on the title page.
