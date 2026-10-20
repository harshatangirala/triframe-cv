"""Fine-tuning round trip with a REAL pretrained timm backbone (CPU, tiny): fine-tune ResNet-18 on a few real chart sets, save
the weights, rebuild the backbone as 'ft_resnet18' (notebook side) and through tfcv_core.embed_views with the bundle record
(dashboard side), and check both give the same embeddings - and that they differ from the frozen ImageNet backbone.
Usage: python tests/test_finetune_weights.py <dataset root> <scratch folder>"""
import os, sys, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT, OUT = sys.argv[1].rstrip("/") + "/", sys.argv[2]
sys.path.insert(1, ROOT + "scripts")
import numpy as np, torch
from triframe_loader import ImageStore
import tfcv_core as C, tfcv_train as T

st = ImageStore(who="claude_v3_ft_roundtrip", root=ROOT)
idx = st.join_features(st.index())
g = idx[idx.ticker == idx.ticker.iloc[0]].iloc[:48]
packed = T.FastTriplets(st).read_packed(g)
raw = T.views_from_packed(packed, "cpu", resize=False); ref = T.FastTriplets(st).read(g)
assert all(np.array_equal(raw[v][:, 0].numpy().astype(np.uint8), ref[v]) for v in T.VIEWS), "packed decoder"
rng = np.random.default_rng(0)
xnum = rng.normal(size=(len(g), 33)).astype(np.float32)
ys = (g["logrv20"].to_numpy(float), g["y_out5"].to_numpy(float), rng.integers(0, 2, len(g)).astype(float))
m, mu, sd, rec, hist, e2e = T.finetune("resnet18", packed, xnum, ys, np.arange(32), np.arange(32, 48), "cpu", epochs=1, batch=8)
print(hist.round(4).to_string(index=False)); print("record:", {k: rec[k] for k in ("name", "timm_name", "n_params")}, rec["ft"])
os.makedirs(OUT, exist_ok=True)
wpath = os.path.join(OUT, "ft_resnet18.safetensors"); T.save_backbone(m, wpath)
T.FT_WEIGHTS["ft_resnet18"] = wpath
m2, mu2, sd2, rec2 = T.build_backbone("ft_resnet18", "cpu")
assert rec2["weights_file"] == "ft_resnet18.safetensors" and rec2["name"] == "ft_resnet18"
trip = {v: ref[v][0] for v in T.VIEWS}
vs = T.views_from_packed(packed[:1], "cpu")
with torch.no_grad():
    e_nb = {v: m2((vs[v].expand(-1, 3, -1, -1) - mu2) / sd2).numpy()[0] for v in T.VIEWS}
spec = json.loads(json.dumps(rec2))                                  # what the bundle's config.json stores
e_dash = C.embed_views(trip, spec, OUT)
diff = max(float(np.abs(e_dash[v] - e_nb[v]).max()) for v in T.VIEWS)
m0, mu0, sd0, _ = T.build_backbone("resnet18", "cpu")
with torch.no_grad():
    e_frozen = m0((vs["D"].expand(-1, 3, -1, -1) - mu0) / sd0).numpy()[0]
moved = float(np.abs(e_frozen - e_nb["D"]).max())
print(f"dashboard vs notebook embeddings of the fine-tuned backbone: max |diff| {diff:.2e}; change vs frozen ImageNet weights: {moved:.3f}")
assert diff < 1e-4 and moved > 1e-3
print("FINE-TUNE WEIGHTS ROUND TRIP PASSED")
