"""Functional test of streamlit_app.py with Streamlit's AppTest (no browser). Needs BUNDLE_DIR (local bundle folder).
Two companies by typed ticker: comparison tab, per-company tabs, Monte Carlo, explanations."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from streamlit.testing.v1 import AppTest

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
at = AppTest.from_file(os.path.join(root, "streamlit_app.py"), default_timeout=400).run()
assert not at.exception, at.exception
assert at.title[0].value == "Computer Vision with Deep Learning Project Dashboard"
print("models table rows:", len(at.dataframe[0].value))
at.radio(key="n_companies").set_value(2).run()
assert not at.exception, at.exception
print("file uploaders for 2 companies:", len(at.get("file_uploader")))
assert len(at.get("file_uploader")) == 6
at.text_input(key="ticker_0").set_value("NVDA")
at.text_input(key="ticker_1").set_value("KO")
t0 = time.time()
at.button[0].click().run()
assert not at.exception, at.exception
body = " ".join(m.value for m in at.markdown)
print(f"analyse 2 companies: {time.time() - t0:.0f}s")
assert "Highest risk:" in body and "Lowest risk:" in body, body[:500]
assert "### NVDA" in body and "### KO" in body
for k in ("1. Expected volatility", "2. Chance of a 10%+ drawdown", "3. Chance of beating the median stock", "4. Monte Carlo simulation"):
    assert k in body, k
assert sum("Primary takeaway" in i.value for i in at.info) == 2
print("tabs:", [t.label for t in at.tabs])
print("plotly charts:", len(at.get("plotly_chart")))
assert len(at.get("plotly_chart")) >= 2 * 6 + 5
print([l for l in body.split("**") if l.startswith("Highest risk") or l.startswith("Lowest risk")])
# changing a Monte Carlo setting re-simulates without re-running the analysis
at.toggle(key="mc_match").set_value(False).run()
assert not at.exception, at.exception
print("STREAMLIT APP TEST PASSED")
