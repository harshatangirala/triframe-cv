"""TriFrame canonical renderer r1 (spec 7.3): text-free, binary, deterministic.
Each view is 128 x (3*bars): price panel 100 px, 2 px gap, volume panel 26 px. OHLC bars, 3 px per bar
(open tick left, high-low line centre, close tick right). Log-price scaling inside the window (min low -> bottom,
max high -> top), linear volume 0..max. Left-padded when fewer bars exist (the number of valid bars is kept)."""
import hashlib
import numpy as np

VERSION = "r1"
H, PRICE_H, GAP, VOL_H = 128, 100, 2, 26
VIEW_BARS = {"W": 104, "D": 126, "H": 140}
TRIPLET_BITS = H * 3 * sum(VIEW_BARS.values())          # 142,080
TRIPLET_BYTES = TRIPLET_BITS // 8                       # 17,760


def render(o, h, l, c, v, n_bars):
    """o,h,l,c,v: arrays of the m <= n_bars valid bars, oldest first. Returns (uint8 image [128, 3*n_bars], m)."""
    m = len(o); W = 3 * n_bars
    img = np.zeros((H, W), np.uint8)
    if m == 0:
        return img, 0
    o, h, l, c, v = (np.asarray(a, float) for a in (o, h, l, c, v))
    lo, hi = np.log(l.min()), np.log(h.max()); span = hi - lo
    if span > 0:
        f = lambda a: np.rint((np.log(a) - lo) / span * (PRICE_H - 1)).astype(int)
        yo, yh, yl, yc = f(o), f(h), f(l), f(c)
    else:
        yo = yh = yl = yc = np.full(m, PRICE_H // 2)
    x0 = 3 * (n_bars - m) + 3 * np.arange(m)
    rows = np.arange(PRICE_H)[:, None]
    price = np.zeros((PRICE_H, W), np.uint8)
    price[:, x0 + 1] = ((rows >= yl[None, :]) & (rows <= yh[None, :])).astype(np.uint8)
    price[yo, x0] = 1
    price[yc, x0 + 2] = 1
    img[:PRICE_H] = price[::-1]
    vmax = v.max()
    if vmax > 0:
        hv = np.rint(v / vmax * (VOL_H - 1)).astype(int) + (v > 0)
        vol = np.zeros((VOL_H, W), np.uint8)
        vol[:, x0 + 1] = (np.arange(VOL_H)[:, None] < hv[None, :]).astype(np.uint8)
        img[PRICE_H + GAP:] = vol[::-1]
    return img, m


def pack_triplet(wimg, dimg, himg):
    return np.packbits(np.concatenate([wimg.ravel(), dimg.ravel(), himg.ravel()]))


def unpack_triplet(buf):
    bits = np.unpackbits(buf)[:TRIPLET_BITS]
    out, p = {}, 0
    for k, nb in VIEW_BARS.items():
        n = H * 3 * nb
        out[k] = bits[p:p + n].reshape(H, 3 * nb); p += n
    return out


def sha(img):
    return hashlib.sha256(np.packbits(img).tobytes()).hexdigest()


class StockWindows:
    """As-of window builder for one stock. Everything returned for decision day t uses bars dated <= t only."""
    def __init__(self, daily, hourly):
        d = daily.sort_values("date").reset_index(drop=True)
        self.d = d
        self.o, self.h, self.l, self.c, self.v = (d[k].values.astype(float) for k in ("open", "high", "low", "close", "volume"))
        self.dates = d.date.values.astype("datetime64[ns]")
        ws = (d.date - pd_to_td(d.date.dt.dayofweek)).values
        self.wid = np.cumsum(np.r_[0, (ws[1:] != ws[:-1]).astype(int)])
        self.first = np.flatnonzero(np.r_[True, self.wid[1:] != self.wid[:-1]])
        self.Wo, self.Wc = self.o[self.first], self.c[np.r_[self.first[1:] - 1, len(d) - 1]]
        self.Wh = np.maximum.reduceat(self.h, self.first); self.Wl = np.minimum.reduceat(self.l, self.first)
        self.Wv = np.add.reduceat(self.v, self.first)
        hh = hourly.sort_values("timestamp").reset_index(drop=True)
        et = hh.timestamp.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        self.hdate = et.values.astype("datetime64[ns]")
        self.ho, self.hh, self.hl, self.hc, self.hv = (hh[k].values.astype(float) for k in ("open", "high", "low", "close", "volume"))

    def daily(self, k):
        s = max(0, k - VIEW_BARS["D"] + 1)
        return self.o[s:k + 1], self.h[s:k + 1], self.l[s:k + 1], self.c[s:k + 1], self.v[s:k + 1]

    def weekly(self, k):
        w = self.wid[k]; f = self.first[w]; n = VIEW_BARS["W"]; s = max(0, w - (n - 1))
        o = np.r_[self.Wo[s:w], self.o[f]]; h = np.r_[self.Wh[s:w], self.h[f:k + 1].max()]
        l = np.r_[self.Wl[s:w], self.l[f:k + 1].min()]; c = np.r_[self.Wc[s:w], self.c[k]]
        v = np.r_[self.Wv[s:w], self.v[f:k + 1].sum()]
        return o, h, l, c, v

    def hourly(self, k):
        e = np.searchsorted(self.hdate, self.dates[k], side="right"); s = max(0, e - VIEW_BARS["H"])
        return self.ho[s:e], self.hh[s:e], self.hl[s:e], self.hc[s:e], self.hv[s:e]

    def triplet(self, k):
        w, nw = render(*self.weekly(k), VIEW_BARS["W"])
        d, nd = render(*self.daily(k), VIEW_BARS["D"])
        h, nh = render(*self.hourly(k), VIEW_BARS["H"])
        return (w, d, h), (nw, nd, nh)


def pd_to_td(x):
    import pandas as pd
    return pd.to_timedelta(x, unit="D")
