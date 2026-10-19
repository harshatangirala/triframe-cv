"""Reads ticker candidates, timeframe and the last-bar O/H/L/C legend from a TradingView-style chart screenshot.
Tested on 66 real TradingView dark-theme screenshots (22 tickers x 1W/1D/1H): timeframe 100 %, legend O/H/L/C 100 %,
true ticker among the candidates 66/66. The app confirms a candidate by matching the legend against market data."""
import re
import numpy as np
from PIL import Image

_OCR = None
TF_MAP = {"1W": "1W", "W": "1W", "1D": "1D", "D": "1D", "1H": "1H", "60": "1H"}
STOP = {"USD", "SELL", "BUY", "VOL", "SMA", "EMA", "NASDAQ", "NYSE", "AMEX", "ARCA", "BATS", "OTC", "TRADE", "SAVE", "ALERT",
        "PRE", "POST", "MACD", "RSI", "LOG", "AUTO", "INC", "CORP", "LTD", "CO", "PLC", "THE", "AND", "OF", "CLOSE", "OPEN",
        "HIGH", "LOW", "ADJ", "EXT", "UTC"}


def _ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def _num(s):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _rows(items):
    """Group OCR items (box, text, score) into text rows by vertical centre, each row sorted left to right."""
    boxes = []
    for box, text, _ in items:
        ys = [p[1] for p in box]; xs = [p[0] for p in box]
        boxes.append((sum(ys) / 4, max(ys) - min(ys), min(xs), text.strip()))
    boxes.sort()
    rows, cur = [], []
    for b in boxes:
        if cur and abs(b[0] - cur[-1][0]) > 0.6 * max(b[1], cur[-1][1], 1):
            rows.append(sorted(cur, key=lambda z: z[2])); cur = []
        cur.append(b)
    if cur:
        rows.append(sorted(cur, key=lambda z: z[2]))
    return [[z[3] for z in r if z[3]] for r in rows]


def _toolbar_candidates(rows):
    out = []
    for r in rows[:3]:
        for j, t in enumerate(r):
            if re.search(r"[OHLC0][\d,]+\.\d", t) or "·" in t:            # legend values or the name line
                continue
            for tok in re.findall(r"[A-Za-z][A-Za-z\.\-]{0,5}", t):
                up = tok.upper().strip(".-")
                if not (1 <= len(up) <= 5) or up in STOP or up in TF_MAP:
                    continue
                if sum(c.islower() for c in tok) > 1:                      # ordinary words ("Replay", "Indicators")
                    continue
                if len(up) == 1 and j == 0 and len(r) > 1:                 # the account avatar at the far left
                    continue
                up = up.replace(".", "-")
                if up not in out:
                    out.append(up)
    return out


def _price_tag_candidates(img):
    """The ticker in the price-axis tag on the right ("BRKR 58.84"), for screenshots taken without the toolbar."""
    w, h = img.size
    right = img.crop((int(w * 0.72), 0, w, h))
    if right.size[0] < 800:
        right = right.resize((right.size[0] * 2, right.size[1] * 2), Image.LANCZOS)
    res, _ = _ocr()(np.asarray(right))
    out = []
    for box, text, _ in sorted(res or [], key=lambda r: min(p[1] for p in r[0])):
        tok = text.strip()
        if re.fullmatch(r"[A-Z][A-Z\.\-]{0,5}", tok) and tok not in STOP and tok.replace(".", "-") not in out:
            out.append(tok.replace(".", "-"))
    return out


def read_header(img, deep=False):
    """img: PIL.Image or path. Returns dict(candidates, timeframe, ohlc, name, header_text).
    deep=True also reads the price-axis tag (slower; used when the toolbar candidates cannot be confirmed)."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    top = img.crop((0, 0, int(w * 0.75), max(60, int(h * 0.12))))
    if top.size[0] < 1600:
        top = top.resize((top.size[0] * 2, top.size[1] * 2), Image.LANCZOS)
    res, _ = _ocr()(np.asarray(top))
    rows = _rows(res or [])
    joined = " | ".join(" ".join(r) for r in rows)

    tf = None
    m = re.search(r"[·•]\s*(1W|1D|1H|60)\s*[·•]", joined, re.I)
    if m:
        tf = TF_MAP.get(m.group(1).upper())
    if tf is None and rows:
        for t in rows[0]:
            if t.strip().upper() in TF_MAP:
                tf = TF_MAP[t.strip().upper()]; break

    ohlc = None
    flat = re.sub(r"\s+", "", joined)
    m = re.search(r"[O0]([\d,]*\d\.\d+)H([\d,]+\.\d+)L([\d,]+\.\d+)C([\d,]+\.\d+)", flat)
    if m:
        vals = [_num(x) for x in m.groups()]
        if all(v is not None for v in vals) and vals[1] >= max(vals[0], vals[3]) * 0.999 and vals[2] <= min(vals[0], vals[3]) * 1.001:
            ohlc = dict(zip(("open", "high", "low", "close"), vals))

    name = None
    m = re.search(r"([A-Za-z][^|·•]{2,80}?)\s*[·•]\s*(?:1W|1D|1H|60)\b", joined, re.I)
    if m:
        name = re.sub(r"^[^A-Za-z]+", "", m.group(1)).strip()

    cands = _toolbar_candidates(rows)
    if deep or not cands:
        cands += [c for c in _price_tag_candidates(img) if c not in cands]
    return dict(candidates=cands, timeframe=tf, ohlc=ohlc, name=name, header_text=joined[:300])
