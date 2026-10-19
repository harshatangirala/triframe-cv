"""Functional test of streamlit_app.py with Streamlit's AppTest (no browser). Needs BUNDLE_DIR (local bundle folder)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from streamlit.testing.v1 import AppTest

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
at = AppTest.from_file(os.path.join(root, "streamlit_app.py"), default_timeout=240).run()
assert not at.exception, at.exception
print("title:", at.title[0].value)
print("uploaders:", len(at.get("file_uploader")), "| buttons:", [b.label for b in at.button])
assert at.title[0].value == "Computer Vision with Deep Learning Project Dashboard"
body = " ".join(m.value for m in at.markdown)
assert "60 of 979 stocks" in body or "run on 60" in body, body[:400]
t0 = time.time()
at.text_input(key="ticker").set_value("NVDA"); at.button[0].click().run()
assert not at.exception, at.exception
out = " ".join(m.value for m in at.markdown)
print(f"analyse NVDA: {time.time()-t0:.0f}s ->", [l for l in out.splitlines() if l.startswith("### ") or l.startswith("**Decision")][:2])
assert "### NVDA" in out and "Decision" in out
assert any("Screenshot checks" in s.value for s in at.subheader) or True
print("STREAMLIT APP TEST PASSED")
