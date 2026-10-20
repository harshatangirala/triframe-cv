"""TriFrame CV - training-side code for the Colab notebook (loader, embeddings, PCA, heads, metrics, selection).
Data is read only through triframe_loader.ImageStore (sealed); the fast reader below is checked against
ImageStore.triplet on random rows before it is used."""
import os, json, time, math, gc, re
import numpy as np
import pandas as pd
import tf_render as RN

VIEWS = ("W", "D", "H")
_SHAPES = {"W": (128, 3 * 104), "D": (128, 3 * 126), "H": (128, 3 * 140)}
_OFF = {"W": 0, "D": 128 * 312, "H": 128 * (312 + 378)}

# ===================================================================================== 1. fast triplet reader
class FastTriplets:
    """Vectorised version of ImageStore.triplet for whole tickers at a time (same shards, same r1.1 patch overlay)."""

    def __init__(self, st):
        self.st = st

    def read_packed(self, g):
        """g: rows of ONE ticker from the store's index (columns ticker, sample_id, src, shard_row, patched).
        Returns the packed 1-bit triplets (n, TRIPLET_BYTES) in g's order."""
        t = g["ticker"].iloc[0]
        n = len(g)
        buf = np.empty((n, RN.TRIPLET_BYTES), np.uint8)
        pm = g["patched"].values.astype(bool)
        if (~pm).any():
            src = g["src"].values
            for s in np.unique(src[~pm]):
                sel = (~pm) & (src == s)
                buf[sel] = np.asarray(self.st._shard(s, t)[g["shard_row"].values[sel]])
        if pm.any():
            prow = self.st._patch.loc[g["sample_id"].values[pm], "patch_row"].values.astype(int)
            buf[pm] = np.asarray(self.st._shard("cache_patch_r11", t)[prow])
        return buf

    def read(self, g):
        """g: rows of ONE ticker from the store's index. Returns {view: (n, 128, width) uint8 0/1 images}."""
        n = len(g)
        bits = np.unpackbits(self.read_packed(g), axis=1)[:, :RN.TRIPLET_BITS]
        return {v: bits[:, _OFF[v]:_OFF[v] + _SHAPES[v][0] * _SHAPES[v][1]].reshape(n, *_SHAPES[v]) for v in VIEWS}

    def parity(self, rows, n=300, seed=0):
        """Assert the fast reader equals ImageStore.triplet on n random rows (patched rows included)."""
        rng = np.random.default_rng(seed)
        pick = rows.iloc[rng.choice(len(rows), min(n, len(rows)), replace=False)]
        patched = rows[rows["patched"]]
        if len(patched):
            pick = pd.concat([pick, patched.iloc[rng.choice(len(patched), min(50, len(patched)), replace=False)]])
        bad = 0
        for t, g in pick.groupby("ticker"):
            fast = self.read(g)
            for j, sid in enumerate(g["sample_id"].values):
                ref = self.st.triplet(sid)
                bad += sum(int(not np.array_equal(fast[v][j], ref[v])) for v in VIEWS)
        assert bad == 0, f"fast reader differs from ImageStore.triplet on {bad} views"
        return len(pick)


# ===================================================================================== 2. backbones and embeddings
BACKBONES = {
    "resnet18": dict(timm="resnet18"),
    "efficientnet_b0": dict(timm="efficientnet_b0"),
    "vit_base_patch16_224": dict(timm="vit_base_patch16_224"),
    "convnext_tiny": dict(timm="convnext_tiny"),
    "swin_tiny_patch4_window7_224": dict(timm="swin_tiny_patch4_window7_224"),
    "vit_small_patch14_dinov2": dict(timm="vit_small_patch14_dinov2", kwargs={"img_size": 224}),
    "mobilenetv3_large_100": dict(timm="mobilenetv3_large_100"),
    "resnet18_random_s0": dict(timm="resnet18", pretrained=False, seed=0),
    "resnet18_random_s1": dict(timm="resnet18", pretrained=False, seed=1),
    "resnet18_random_s2": dict(timm="resnet18", pretrained=False, seed=2),
}
PRETRAINED = [b for b, s in BACKBONES.items() if s.get("pretrained", True)]
RANDOM = [b for b in BACKBONES if b not in PRETRAINED]

# plain-language facts for the model cards (family, what the network is, who introduced it)
BACKBONE_INFO = {
    "resnet18": ("CNN", "ResNet-18", "18-layer residual convolutional network (He et al., 2016)"),
    "efficientnet_b0": ("CNN", "EfficientNet-B0", "compound-scaled convolutional network built for efficiency (Tan & Le, 2019)"),
    "vit_base_patch16_224": ("Transformer", "ViT-B/16", "Vision Transformer: the image is cut into 16x16 patches that attend to each other (Dosovitskiy et al., 2021)"),
    "convnext_tiny": ("CNN", "ConvNeXt-Tiny", "a ResNet modernised with transformer-era design choices (Liu et al., 2022)"),
    "swin_tiny_patch4_window7_224": ("Transformer", "Swin-T", "hierarchical transformer that attends within shifted windows (Liu et al., 2021)"),
    "vit_small_patch14_dinov2": ("Transformer", "DINOv2 ViT-S/14", "Vision Transformer trained without labels by self-distillation (Oquab et al., 2023)"),
    "mobilenetv3_large_100": ("CNN", "MobileNetV3-Large", "lightweight network designed for phones (Howard et al., 2019)"),
    "resnet18_random_s0": ("Control", "ResNet-18 (random weights, seed 0)", "same architecture as ResNet-18 but never trained: a control"),
    "resnet18_random_s1": ("Control", "ResNet-18 (random weights, seed 1)", "same architecture as ResNet-18 but never trained: a control"),
    "resnet18_random_s2": ("Control", "ResNet-18 (random weights, seed 2)", "same architecture as ResNet-18 but never trained: a control"),
}


def pretrain_data(timm_name, pretrained=True):
    """The pretraining dataset behind a timm weight tag, in words."""
    if not pretrained:
        return "none (random weights)"
    tag = timm_name.split(".", 1)[1] if "." in timm_name else ""
    if "lvd142m" in tag:
        return "LVD-142M, 142 million unlabelled images (self-supervised)"
    if "in21k_ft_in1k" in tag:
        return "ImageNet-21k, then fine-tuned on ImageNet-1k"
    if "in12k_ft_in1k" in tag:
        return "ImageNet-12k, then fine-tuned on ImageNet-1k"
    if "in1k" in tag:
        return "ImageNet-1k (1.28 million labelled photos, 1,000 classes)"
    return tag or "ImageNet"


def display_name(name):
    """'ft_convnext_tiny' -> 'ConvNeXt-Tiny (fine-tuned)'; 'ft_p50.resnet18' -> 'ResNet-18 (fine-tuned, last 50%)'."""
    base = base_name(name)
    nm = BACKBONE_INFO.get(base, ("", base, ""))[1]
    v = ft_variant(name)
    return nm if v is None else f"{nm} ({FT_VARIANTS.get(v, 'fine-tuned ' + v)})"


# Fine-tuned backbones are named "ft_<backbone>" (full fine-tuning on all three targets) or "ft_<variant>.<backbone>" for the
# controlled study in Stage 5F: "p25" / "p50" = only the last 25% / 50% of the backbone's parameters train; "risk" = full
# fine-tuning on the two risk targets only (volatility and drawdown). Weights load from FT_WEIGHTS (filled by Stage 5F).
FT_PREFIX = "ft_"
FT_WEIGHTS = {}
FT_VARIANTS = {"": "fine-tuned", "p25": "fine-tuned, last 25%", "p50": "fine-tuned, last 50%", "risk": "fine-tuned, risk targets only"}


def ft_name(base, variant=""):
    return FT_PREFIX + (f"{variant}.{base}" if variant else base)


def ft_variant(name):
    """None for a frozen backbone, "" for full fine-tuning, else the study variant tag."""
    if not name.startswith(FT_PREFIX):
        return None
    s = name[len(FT_PREFIX):]
    return s.split(".", 1)[0] if "." in s else ""


def base_name(name):
    if not name.startswith(FT_PREFIX):
        return name
    s = name[len(FT_PREFIX):]
    return s.split(".", 1)[1] if "." in s else s


def build_backbone(name, device, factory=None):
    """Returns (model, mean, std, record). factory(name, spec) may replace timm (used only for the local CPU test).
    A name 'ft_<backbone>' builds <backbone> and loads its fine-tuned weights from FT_WEIGHTS[name]."""
    import torch
    base = base_name(name)
    ft = name != base
    spec = BACKBONES[base]
    if factory is not None:
        m, mu, sd, rec = factory(base, spec, device)
    else:
        import timm
        torch.manual_seed(int(spec.get("seed", 0)))
        m = timm.create_model(spec["timm"], pretrained=bool(spec.get("pretrained", True)) and not ft, num_classes=0, **spec.get("kwargs", {}))
        m = m.to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=m)
        pc = getattr(m, "pretrained_cfg", {}) or {}
        tag = pc.get("tag") if spec.get("pretrained", True) else None
        rec = dict(name=base, timm=spec["timm"], timm_name=(f"{spec['timm']}.{tag}" if tag else spec["timm"]),
                   pretrained=bool(spec.get("pretrained", True)), seed=int(spec.get("seed", 0)), kwargs=spec.get("kwargs", {}),
                   mean=list(cfg["mean"]), std=list(cfg["std"]), hf_hub_id=pc.get("hf_hub_id"), timm_version=timm.__version__)
        mu = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        sd = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    if ft:
        from safetensors.torch import load_file
        m.load_state_dict(load_file(FT_WEIGHTS[name], device="cpu"))
        m = m.to(device).eval()
        rec = dict(rec, name=name, base=base, finetuned=True, weights_file=os.path.basename(FT_WEIGHTS[name]))
    rec["n_params"] = int(sum(p.numel() for p in m.parameters()))
    return m, mu, sd, rec


def throughput(m, mu, sd, device, n_img=256, batch=64):
    """Inference speed (images per second) of a backbone on this device, with the same preprocessing and precision as the
    embedding stage."""
    import torch
    x = torch.rand(batch, 3, 224, 224, device=device)
    cuda = str(device).startswith("cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
        m((x - mu) / sd)                                           # warm-up
        if cuda:
            torch.cuda.synchronize()
        t0 = time.time(); done = 0
        while done < n_img:
            m((x - mu) / sd); done += batch
        if cuda:
            torch.cuda.synchronize()
    return done / max(time.time() - t0, 1e-9)


def extract_embeddings(st, rows, names, out_dir, device, chunk_tickers=40, batch=384, factory=None,
                       on_chunk=None, have_chunk=None, log=print):
    """Embeds every row x view x backbone. Work is cut into chunks of tickers; each finished chunk is one .npz
    (sample_id + one float16 array per 'backbone|view'). Finished chunks are skipped, so the run can resume.
    on_chunk(path) is called after each chunk (e.g. upload to Hugging Face); have_chunk(name) can fetch a chunk saved
    in an earlier session (returns a local path or None)."""
    import torch, torch.nn.functional as F
    os.makedirs(out_dir, exist_ok=True)
    tickers = list(dict.fromkeys(rows["ticker"].tolist()))
    chunks = [tickers[i:i + chunk_tickers] for i in range(0, len(tickers), chunk_tickers)]
    models, records = {}, {}
    for nm in names:
        m, mu, sd, rec = build_backbone(nm, device, factory)
        with torch.no_grad():
            rec["emb_dim"] = int(m(torch.zeros(2, 3, 224, 224, device=device)).shape[1])
        rec["infer_img_per_s"] = float(throughput(m, mu, sd, device))
        models[nm] = (m, mu, sd); records[nm] = rec
    reader = FastTriplets(st)
    use_amp = str(device).startswith("cuda")
    checked = set()
    t0 = time.time(); done_rows = 0
    for ci, ch in enumerate(chunks):
        fname = f"chunk_{ci:04d}.npz"; path = os.path.join(out_dir, fname)
        if not os.path.exists(path) and have_chunk is not None:
            got = have_chunk(fname)
            if got and os.path.exists(got) and got != path:
                os.replace(got, path)
        if os.path.exists(path):
            continue
        sub = rows[rows["ticker"].isin(ch)]
        sub = pd.concat([sub[sub["ticker"] == t] for t in ch if (sub["ticker"] == t).any()])
        out = {}
        pos = 0
        for t, g in sub.groupby("ticker", sort=False):
            imgs = reader.read(g)
            n = len(g)
            for v in VIEWS:
                for b0 in range(0, n, batch):
                    x = torch.from_numpy(np.ascontiguousarray(imgs[v][b0:b0 + batch])).to(device)
                    x = F.interpolate(x.float().unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
                    if v not in checked:
                        assert float(x.max()) > 0.5, f"blank chart batch in view {v}"
                    x3 = x.expand(-1, 3, -1, -1)
                    for nm, (m, mu, sd) in models.items():
                        xn = (x3 - mu) / sd
                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                            e = m(xn)
                        e = e.float()
                        if (nm, v) not in checked:
                            assert float(xn.std()) > 0.2, f"{nm}/{v}: input contrast too low after normalisation"
                            assert torch.isfinite(e).all(), f"{nm}/{v}: non-finite embeddings"
                            assert float(e.std(0).mean()) > 1e-4, f"{nm}/{v}: embeddings do not vary across charts"
                            checked.add((nm, v))
                        key = f"{nm}|{v}"
                        if key not in out:
                            out[key] = np.empty((len(sub), e.shape[1]), np.float16)
                        out[key][pos + b0:pos + b0 + e.shape[0]] = e.cpu().numpy().astype(np.float16)
                checked.add(v)
            pos += n
        tmp = path + ".tmp.npz"
        np.savez(tmp, sample_id=np.asarray(sub["sample_id"].tolist(), dtype="U"), **out)    # fixed-width text, no pickle
        os.replace(tmp, path)
        done_rows += len(sub)
        el = time.time() - t0
        log(f"   chunk {ci + 1}/{len(chunks)}: {len(sub):,} rows, {done_rows / max(el, 1e-9):,.0f} rows/s this session")
        if on_chunk is not None:
            on_chunk(path)
    return records


# ===================================================================================== 2b. fine-tuning (end to end, multi-task)
def packed_rows(st, rows):
    """Packed 1-bit chart triplets (n, TRIPLET_BYTES) for index rows, in the order of rows (about 17.8 kB per chart set)."""
    reader = FastTriplets(st)
    out = np.empty((len(rows), RN.TRIPLET_BYTES), np.uint8)
    tick = rows["ticker"].values
    for t in pd.unique(tick):
        sel = np.flatnonzero(tick == t)
        out[sel] = reader.read_packed(rows.iloc[sel])
    return out


def views_from_packed(buf, device, resize=True):
    """Packed triplets (B, TRIPLET_BYTES) -> {view: (B, 1, 224, 224) float tensor in [0, 1]} on device, resized exactly as
    in extract_embeddings (bilinear, antialiased). resize=False returns the raw 0/1 images (B, 1, 128, width)."""
    import torch, torch.nn.functional as F
    b = torch.from_numpy(np.ascontiguousarray(buf)).to(device)
    shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
    bits = ((b.unsqueeze(-1) >> shifts) & 1).reshape(b.shape[0], -1)[:, :RN.TRIPLET_BITS]
    out = {}
    for v in VIEWS:
        h, w = _SHAPES[v]
        x = bits[:, _OFF[v]:_OFF[v] + h * w].reshape(-1, 1, h, w).float()
        out[v] = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True) if resize else x
    return out


def _ft_net(backbone, emb_dim, n_num, hidden=256, drop=0.2):
    """The fine-tuning network: ONE backbone embeds each of the three views; the three embeddings and the 33 standardised
    numbers feed a small shared head with three outputs (volatility, direction logit, drawdown logit)."""
    import torch, torch.nn as nn

    class TriFrameNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.trunk = nn.Sequential(nn.LayerNorm(3 * emb_dim + n_num), nn.Linear(3 * emb_dim + n_num, hidden), nn.GELU(), nn.Dropout(drop))
            self.out = nn.Linear(hidden, 3)

        def forward(self, xs, xnum):
            n = xnum.shape[0]
            e = self.backbone(torch.cat(xs, 0)).float()
            e = torch.cat(e.split(n, 0), 1)
            return self.out(self.trunk(torch.cat([e, xnum], 1)))
    return TriFrameNet()


def _ft_losses(out, ys):
    """Per-task losses with missing labels masked: MSE on standardised volatility, plain binary cross-entropy on the two
    probability targets (no label smoothing, no focal loss: project rule)."""
    import torch, torch.nn.functional as F
    terms = []
    for k, y in enumerate(ys):
        m = torch.isfinite(y)
        if not bool(m.any()):
            terms.append(out[:, k].sum() * 0.0); continue
        p = out[m, k].float()
        terms.append(F.mse_loss(p, y[m]) if k == 0 else F.binary_cross_entropy_with_logits(p, y[m]))
    return terms


def _set_trainable(m, frac):
    """Train only the last part of a backbone (timm registers parameters roughly from input to output): every parameter tensor
    that reaches into the last `frac` share of the parameter count trains, the rest is frozen, so at least `frac` of the
    network trains. frac = 1 trains everything. Returns the number of trainable parameters."""
    params = list(m.parameters())
    total = sum(p.numel() for p in params); start = total * (1.0 - float(frac)); seen = 0; n = 0
    for p in params:
        p.requires_grad = seen + p.numel() > start + 0.5 or frac >= 1.0
        seen += p.numel(); n += p.numel() if p.requires_grad else 0
    return n


def _freeze_bn(module):
    """Keep batch-norm layers whose parameters are frozen in eval mode, so frozen layers really stay as they were."""
    import torch
    for mod in module.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in mod.parameters(recurse=False)):
            mod.eval()


def finetune(name, packed, xnum, ys, train_idx, stop_idx, device, val_idx=None, epochs=8, patience=2, rows_per_epoch=None,
             batch=64, lr_backbone=3e-5, lr_head=1e-3, weight_decay=0.05, unfreeze=1.0, task_weights=(1.0, 1.0, 1.0),
             seed=11, factory=None, log=print, log_every=25, train_probe=4096):
    """Fine-tunes one pretrained backbone end to end.
    * train_idx: rows it learns from (a fresh random draw of rows_per_epoch of them every epoch);
    * stop_idx: an inner validation slice whose loss decides early stopping (stop after `patience` epochs without improvement);
    * val_idx: rows that are only scored and logged, never used for any decision (so the selection years stay clean).
    packed: (n, TRIPLET_BYTES) chart sets; xnum: (n, 33) standardised numbers; ys: (yV, yO5, yD10) arrays (NaN = missing).
    unfreeze: share of the backbone's parameters (the last ones) that train; task_weights: loss weights for (volatility,
    direction, drawdown). One backbone embeds the weekly, daily and hourly chart; a small head adds the numbers. AdamW, short
    warm-up then cosine decay, gradient clipping, mixed precision, no augmentation.
    Returns (backbone at the best epoch, mean, std, record, epoch history, step history, the network's own predictions on
    val_idx at the best epoch, head states {"init", "best"})."""
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m, mu, sd, rec = build_backbone(name, device, factory)
    with torch.no_grad():
        dim = int(m(torch.zeros(2, 3, 224, 224, device=device)).shape[1])
    net = _ft_net(m, dim, xnum.shape[1]).to(device)
    n_trainable = _set_trainable(m, unfreeze)
    head_init = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if not k.startswith("backbone.")}
    yv = np.asarray(ys[0], float)
    v_mu, v_sd = float(np.nanmean(yv[train_idx])), float(np.nanstd(yv[train_idx]))
    Yt = [(yv - v_mu) / v_sd, np.asarray(ys[1], float), np.asarray(ys[2], float)]
    tw = [float(w) for w in task_weights]
    cuda = str(device).startswith("cuda")
    amp_dtype = torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and amp_dtype == torch.float16)
    head_params = list(net.trunk.parameters()) + list(net.out.parameters())
    bb_params = [p for p in m.parameters() if p.requires_grad]
    groups = ([{"params": bb_params, "lr": lr_backbone}] if bb_params else []) + [{"params": head_params, "lr": lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    n_epoch = int(min(len(train_idx), rows_per_epoch or len(train_idx)))
    steps = epochs * math.ceil(n_epoch / batch); warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    probe = np.sort(rng.choice(train_idx, min(train_probe, len(train_idx)), replace=False))   # fixed rows: training loss in eval mode

    def prep(ii):
        vs = views_from_packed(packed[ii], device)
        xs = [(vs[v].expand(-1, 3, -1, -1) - mu) / sd for v in VIEWS]
        yb = [torch.as_tensor(Yt[k][ii], dtype=torch.float32, device=device) for k in range(3)]
        return xs, torch.as_tensor(xnum[ii], dtype=torch.float32, device=device), yb

    def predict(idx):
        net.eval(); outs = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cuda):
            for b0 in range(0, len(idx), 2 * batch):
                xs, xn, _ = prep(idx[b0:b0 + 2 * batch])
                outs.append(net(xs, xn).float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 3))

    def scores(o, idx):
        """Weighted loss (the training objective) and plain-language metrics of the network's outputs on rows idx."""
        tot, parts = 0.0, []
        for k in range(3):
            y = Yt[k][idx]; ok = np.isfinite(y); p = o[ok, k]
            l_ = (float(np.mean((p - y[ok]) ** 2)) if k == 0 else float(np.mean(np.logaddexp(0, p) - y[ok] * p))) if ok.any() else 0.0
            parts.append(l_); tot += tw[k] * l_
        yvv = yv[idx]; okv = np.isfinite(yvv)
        res = dict(loss=tot, loss_V=parts[0], loss_O5=parts[1], loss_D10=parts[2], risk_loss=parts[0] + parts[2],
                   R2_V=r2(yvv[okv], o[okv, 0] * v_sd + v_mu) if okv.sum() > 2 else float("nan"))
        for k, nm in ((2, "AUC_D10"), (1, "AUC_O5")):
            y = Yt[k][idx]; ok = np.isfinite(y)
            res[nm] = auc(y[ok], o[ok, k]) if ok.sum() > 2 and 0 < np.mean(y[ok]) < 1 else float("nan")
        return res

    hist, steps_log, best, bad, best_state, best_head, best_out = [], [], np.inf, 0, None, None, None
    t_start = time.time(); n_img = 0; step = 0
    for ep in range(1, epochs + 1):
        net.train(); _freeze_bn(m); t0 = time.time(); run = []
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        order = rng.choice(train_idx, n_epoch, replace=False)
        for b0 in range(0, len(order), batch):
            ii = order[b0:b0 + batch]
            if len(ii) < 2:
                continue
            xs, xn, yb = prep(ii)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cuda):
                out = net(xs, xn)
            loss = sum(w * t for w, t in zip(tw, _ft_losses(out, yb)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            run.append(float(loss.detach())); n_img += 3 * len(ii)
            if step % log_every == 0:
                steps_log.append(dict(backbone=name, step=step, epoch=ep, loss=float(np.mean(run[-log_every:])), grad_norm=float(gn),
                                      lr_backbone=float(opt.param_groups[0]["lr"]) if bb_params else 0.0,
                                      lr_head=float(opt.param_groups[-1]["lr"])))
        s_tr, s_st = scores(predict(probe), probe), scores(predict(stop_idx), stop_idx)
        o_val = predict(val_idx) if val_idx is not None and len(val_idx) else None
        s_va = scores(o_val, val_idx) if o_val is not None else {}
        h = dict(backbone=name, epoch=ep, train_loss_running=float(np.mean(run)) if run else float("nan"), train_loss=s_tr["loss"],
                 stop_loss=s_st["loss"], val_loss=s_va.get("loss", float("nan")), train_risk_loss=s_tr["risk_loss"],
                 stop_risk_loss=s_st["risk_loss"], val_risk_loss=s_va.get("risk_loss", float("nan")), train_R2_V=s_tr["R2_V"], stop_R2_V=s_st["R2_V"],
                 val_R2_V=s_va.get("R2_V", float("nan")), stop_AUC_D10=s_st["AUC_D10"], val_AUC_D10=s_va.get("AUC_D10", float("nan")),
                 val_AUC_O5=s_va.get("AUC_O5", float("nan")), lr_backbone=float(opt.param_groups[0]["lr"]) if bb_params else 0.0,
                 seconds=time.time() - t0, peak_gpu_gb=(torch.cuda.max_memory_allocated() / 1e9) if cuda else 0.0, seed=seed)
        hist.append(h)
        log(f"   {name} epoch {ep}/{epochs}: train loss {h['train_loss']:.4f} | early-stop loss {h['stop_loss']:.4f} | "
            f"val loss {h['val_loss']:.4f} | val R2 (volatility) {h['val_R2_V']:.3f} | {h['seconds']:.0f}s")
        if h["stop_loss"] < best - 1e-5:
            best, bad, best_out = h["stop_loss"], 0, o_val
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in net.state_dict().items() if not k.startswith("backbone.")}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    for p in m.parameters():
        p.requires_grad = True
    m.eval()
    best_ep = int(min(hist, key=lambda h_: h_["stop_loss"])["epoch"])
    secs = time.time() - t_start
    rec = dict(rec, ft=dict(epochs_run=len(hist), best_epoch=best_ep, minutes=secs / 60, train_img_per_s=n_img / max(secs, 1e-9),
                            n_train=int(len(train_idx)), rows_per_epoch=n_epoch, n_stop=int(len(stop_idx)),
                            n_val=int(len(val_idx)) if val_idx is not None else 0, lr_backbone=lr_backbone, lr_head=lr_head, batch=batch,
                            weight_decay=weight_decay, unfreeze=float(unfreeze), task_weights=tw, n_trainable_backbone=int(n_trainable),
                            patience=patience, max_epochs=epochs, seed=seed, v_mean=v_mu, v_std=v_sd, best_stop_loss=float(best), dim=dim,
                            best_stop_risk_loss=float([h_["stop_risk_loss"] for h_ in hist if h_["epoch"] == best_ep][0])))
    e2e = None if best_out is None else {"V": best_out[:, 0] * v_sd + v_mu, "O5": 1 / (1 + np.exp(-best_out[:, 1])),
                                         "D10": 1 / (1 + np.exp(-best_out[:, 2]))}
    return m, mu, sd, rec, pd.DataFrame(hist), pd.DataFrame(steps_log), e2e, {"init": head_init, "best": best_head}


def ft_landscape(name, weights_path, heads, dim, packed, xnum, ys, sets, device, v_mu, v_sd, task_weights=(1.0, 1.0, 1.0),
                 alphas=None, grid=None, seed=0, batch=128, factory=None):
    """Loss landscape of a fine-tuned network around its solution.
    1-D (Goodfellow et al., 2015): the loss along the straight line from the starting point (pretrained backbone + the head's
    random initial weights, alpha = 0) to the trained network (alpha = 1), and beyond on both sides.
    2-D (Li et al., 2018): the loss on a plane through the trained network spanned by two random directions, each scaled filter
    by filter to the size of the network's own weights (biases and norm layers left out).
    sets: {"train": rows, "validation": rows}. A minimum shows up as the lowest point sitting at alpha = 1 / at the centre."""
    import torch
    from safetensors.torch import load_file
    alphas = np.linspace(-0.5, 1.5, 21) if alphas is None else np.asarray(alphas)
    grid = np.linspace(-1.0, 1.0, 13) if grid is None else np.asarray(grid)
    m0, mu, sd, _ = build_backbone(base_name(name), device, factory)          # the pretrained starting point
    net = _ft_net(m0, dim, xnum.shape[1]).to(device).eval()
    start = {f"backbone.{k}": v.detach().cpu().clone() for k, v in m0.state_dict().items()}
    start.update(heads["init"])
    end = {f"backbone.{k}": v for k, v in load_file(weights_path, device="cpu").items()}
    end.update(heads["best"])
    yv = np.asarray(ys[0], float)
    Yt = [(yv - v_mu) / v_sd, np.asarray(ys[1], float), np.asarray(ys[2], float)]
    cuda = str(device).startswith("cuda")
    amp_dtype = torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else torch.float16
    inputs = {}
    for nm, idx in sets.items():
        vs_ = views_from_packed(packed[idx], "cpu")
        inputs[nm] = ([((vs_[v].expand(-1, 3, -1, -1) - mu.cpu()) / sd.cpu()) for v in VIEWS],
                      torch.as_tensor(xnum[idx], dtype=torch.float32), idx)

    def loss_of(state):
        net.load_state_dict({k: (v.to(start[k].dtype) if v.is_floating_point() else v) for k, v in state.items()})
        net.eval(); out = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cuda):
            for nm, (xs, xn, idx) in inputs.items():
                o = []
                for b0 in range(0, len(idx), batch):
                    o.append(net([x[b0:b0 + batch].to(device) for x in xs], xn[b0:b0 + batch].to(device)).float().cpu().numpy())
                o = np.concatenate(o); tot = 0.0
                for k in range(3):
                    y = Yt[k][idx]; ok = np.isfinite(y); p = o[ok, k]
                    if ok.any():
                        tot += task_weights[k] * (float(np.mean((p - y[ok]) ** 2)) if k == 0 else float(np.mean(np.logaddexp(0, p) - y[ok] * p)))
                out[nm] = tot
        return out

    floats = [k for k in end if end[k].is_floating_point()]
    line = {nm: [] for nm in sets}
    for al in alphas:
        st_ = {k: ((1.0 - al) * start[k].float() + al * end[k].float()) if k in floats else end[k] for k in end}
        r = loss_of(st_)
        for nm in sets:
            line[nm].append(r[nm])
    gen = torch.Generator().manual_seed(int(seed))

    def direction():
        d = {}
        for k in floats:
            w = end[k].float()
            if w.dim() <= 1:
                d[k] = torch.zeros_like(w); continue
            r_ = torch.randn(w.shape, generator=gen)
            wf, rf = w.reshape(w.shape[0], -1), r_.reshape(w.shape[0], -1)
            d[k] = (rf * (wf.norm(dim=1, keepdim=True) / (rf.norm(dim=1, keepdim=True) + 1e-10))).reshape(w.shape)
        return d
    d1, d2 = direction(), direction()
    surf = {nm: np.zeros((len(grid), len(grid))) for nm in sets}
    for i, a in enumerate(grid):
        for j, b in enumerate(grid):
            st_ = {k: (end[k].float() + a * d1[k] + b * d2[k]) if k in floats else end[k] for k in end}
            r = loss_of(st_)
            for nm in sets:
                surf[nm][i, j] = r[nm]
    return dict(alphas=[float(a) for a in alphas], line={k: [float(x) for x in v] for k, v in line.items()},
                grid=[float(g) for g in grid], surface={k: v.tolist() for k, v in surf.items()})


def save_backbone(m, path):
    """Backbone weights as .safetensors (plain tensors, no pickle)."""
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in m.state_dict().items()}, path)


def assemble(out_dir, key, order_ids):
    """Rebuild one 'backbone|view' matrix (float32) in the order of order_ids from the chunk files."""
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("chunk_") and f.endswith(".npz"))
    ids, parts = [], []
    for f in files:
        z = np.load(os.path.join(out_dir, f))
        ids.append(z["sample_id"]); parts.append(z[key])
    ids = np.concatenate(ids); X = np.concatenate(parts).astype(np.float32)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    miss = ~pd.Index(order_ids).isin(pos.index)
    assert not miss.any(), f"{int(miss.sum())} rows have no embedding for {key}"
    return X[pos.loc[order_ids].values]


def pca_fit_transform(X, train_mask, k, seed=0, device="cpu"):
    """PCA fitted on training rows only. Returns (Z float32, mean float64, components float64, explained ratio)."""
    import torch
    mu = X[train_mask].mean(0).astype(np.float64)
    q = int(min(k + 10, X.shape[1]))
    torch.manual_seed(seed)
    Xt = torch.from_numpy((X[train_mask] - mu).astype(np.float32)).to(device)
    U, S, V = torch.pca_lowrank(Xt, q=q, center=False, niter=4)
    comp = V[:, :k].double().cpu().numpy()
    tot = float((Xt.double() ** 2).sum().cpu()); expl = float((S[:k].double() ** 2).sum().cpu()) / max(tot, 1e-12)
    del Xt, U, S, V
    Z = ((X - mu) @ comp).astype(np.float32)
    return Z, mu, comp, expl


# ===================================================================================== 3. heads
def standardize(Xtr):
    mu = np.nanmean(Xtr, 0); sd = np.nanstd(Xtr, 0); sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    return mu, sd


def ridge_path(Xtr, ytr, Xva, alphas):
    """Ridge with an unpenalised intercept for each alpha (sklearn convention). Returns {alpha: (coef, intercept, val_pred)}."""
    Xtr = np.asarray(Xtr, np.float64); xm = Xtr.mean(0); ym = float(np.mean(ytr))
    Xc = Xtr - xm
    G = Xc.T @ Xc; b = Xc.T @ (np.asarray(ytr, np.float64) - ym)
    out = {}
    for a in alphas:
        w = np.linalg.solve(G + a * np.eye(G.shape[0]), b)
        out[a] = (w, ym - xm @ w, np.asarray(Xva, np.float64) @ w + (ym - xm @ w))
    return out


def logit_path(Xtr, ytr, Xva, Cs, device="cpu", max_iter=300):
    """L2 logistic regression (sklearn C convention, unpenalised intercept), fitted by full-batch L-BFGS along the C path
    from strong to weak regularisation with warm starts. Returns {C: (coef, intercept, val_prob)}."""
    import torch
    dt = torch.float32 if str(device).startswith("cuda") else torch.float64
    X = torch.as_tensor(np.asarray(Xtr), dtype=dt, device=device); y = torch.as_tensor(np.asarray(ytr), dtype=dt, device=device)
    Xv = torch.as_tensor(np.asarray(Xva), dtype=dt, device=device)
    n, d = X.shape
    w = torch.zeros(d, dtype=dt, device=device, requires_grad=True); b = torch.zeros(1, dtype=dt, device=device, requires_grad=True)
    out = {}
    for C in sorted(Cs):
        lam = 1.0 / (C * n)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-9, tolerance_change=1e-12,
                                history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * (w * w).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pv = torch.sigmoid(Xv @ w + b).double().cpu().numpy()
        out[C] = (w.detach().double().cpu().numpy().copy(), float(b.detach().cpu()), pv)
    return out


TREE_GRID = [dict(max_depth=3, learning_rate=0.05, min_child_weight=100),
             dict(max_depth=5, learning_rate=0.05, min_child_weight=100),
             dict(max_depth=7, learning_rate=0.05, min_child_weight=300),
             dict(max_depth=5, learning_rate=0.10, min_child_weight=1000)]


def tree_backend():
    try:
        import xgboost  # noqa
        return "xgb"
    except Exception:
        return "lgbm"


def tree_fit(Xtr, ytr, Xva, yva, task, cfg, gpu, seed=11, n_estimators=None):
    """One boosted-tree head. With Xva/yva: early stopping on validation. With n_estimators: fixed size (refit).
    Returns (model, best_n, val_pred or None)."""
    be = tree_backend()
    early = n_estimators is None
    n_est = 2000 if early else int(n_estimators)
    if be == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=n_est, learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"],
                  min_child_weight=cfg["min_child_weight"], subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", device="cuda" if gpu else "cpu", random_state=seed, n_jobs=-1)
        if early:
            kw["early_stopping_rounds"] = 100
        if task == "reg":
            m = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **kw)
        else:
            m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kw)
        if early:                                              # the last eval set (validation) drives early stopping
            m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False); best = int(m.best_iteration) + 1
            ev = m.evals_result(); met = "rmse" if task == "reg" else "logloss"
            m.tfcv_curve = dict(metric=met, train=list(map(float, ev["validation_0"][met])), val=list(map(float, ev["validation_1"][met])), best=best)
        else:
            m.fit(Xtr, ytr, verbose=False); best = n_est
    else:
        import lightgbm as lgb
        kw = dict(n_estimators=n_est, learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"], num_leaves=min(2 ** cfg["max_depth"], 127),
                  min_child_samples=int(cfg["min_child_weight"]), subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)
        m = lgb.LGBMRegressor(**kw) if task == "reg" else lgb.LGBMClassifier(**kw)
        if early:
            m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["training", "valid"],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            best = int(m.best_iteration_ or n_est)
            ev = m.evals_result_; met = list(ev["valid"])[0]
            m.tfcv_curve = dict(metric=met, train=list(map(float, ev["training"][met])), val=list(map(float, ev["valid"][met])), best=best)
        else:
            m.fit(Xtr, ytr); best = n_est
    pv = None
    if Xva is not None:
        pv = m.predict(Xva) if task == "reg" else m.predict_proba(Xva)[:, 1]
    return m, best, pv


def tree_save(model, path):
    if tree_backend() == "xgb":
        model.get_booster().save_model(path); return "xgb"
    model.booster_.save_model(path); return "lgbm"


def tree_predict(model, X, task):
    return model.predict(X) if task == "reg" else model.predict_proba(X)[:, 1]


# ===================================================================================== 3b. baselines and optimisation checks
def garch_logrv20(daily, fit_end, horizon=20):
    """GARCH(1,1) volatility baseline with Student-t errors on the label's own returns (daily close-to-close). Parameters are
    estimated on returns BEFORE fit_end only and then held fixed; at each date the forecast uses returns up to that date only.
    Returns a Series indexed by date: ln sqrt(sum of the next `horizon` forecast daily variances), the logrv20 scale."""
    from arch import arch_model
    d = daily.sort_values("date")
    r = d.set_index("date")["close"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna() * 100.0
    tr = r[r.index < pd.Timestamp(fit_end)]
    if len(tr) < 500:
        return None
    res = arch_model(tr, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False)
    om, al, be = float(res.params["omega"]), float(res.params["alpha[1]"]), float(res.params["beta[1]"])
    rv = r.values; s2 = np.empty(len(rv) + 1); s2[0] = float(np.var(tr.values))
    for i in range(len(rv)):                                # sigma^2_{t+1} = omega + alpha r_t^2 + beta sigma^2_t
        s2[i + 1] = om + al * rv[i] ** 2 + be * s2[i]
    phi = min(al + be, 0.9999); lr_var = om / (1 - phi)    # long-run variance; E[sigma^2_{t+h}] = lr + phi^(h-1) (sigma^2_{t+1} - lr)
    tot = horizon * lr_var + (s2[1:] - lr_var) * (1 - phi ** horizon) / (1 - phi)
    return pd.Series(0.5 * np.log(np.maximum(tot, 1e-12) / 1e4), index=pd.DatetimeIndex(r.index))


def ridge_global_check(X, y, alpha, n_starts=5, iters=80, seed=0):
    """Ridge regression is strictly convex, so it has exactly one minimum, which has a closed form. This computes it, checks that
    the gradient there is zero, runs conjugate gradient from n_starts random starting points (recording the gap to the
    closed-form minimum after every iteration), and evaluates the loss along random lines and on a random plane through it."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xc, yc = X - X.mean(0), y - y.mean(); n, k = X.shape
    A = (Xc.T @ Xc) / n + (alpha / n) * np.eye(k); bvec = (Xc.T @ yc) / n
    w_star = np.linalg.solve(A, bvec)
    J = lambda w: float(np.mean((Xc @ w - yc) ** 2) + (alpha / n) * (w @ w))
    J_star = J(w_star)
    rng = np.random.default_rng(seed); traces = []
    for _ in range(n_starts):
        w = rng.normal(0, 3, k); r = bvec - A @ w; pdir = r.copy(); tr = [J(w) - J_star]
        for _ in range(iters):
            Ap = A @ pdir; a = (r @ r) / max(pdir @ Ap, 1e-300); w = w + a * pdir; r_new = r - a * Ap
            tr.append(max(J(w) - J_star, 1e-16))
            if np.linalg.norm(r_new) < 1e-13:
                break
            pdir = r_new + ((r_new @ r_new) / max(r @ r, 1e-300)) * pdir; r = r_new
        traces.append(tr)
    scale = max(float(np.linalg.norm(w_star)), 1e-3); ts = np.linspace(-1, 1, 41)
    dirs = [d / np.linalg.norm(d) * scale for d in rng.normal(size=(3, k))]
    slices = [[J(w_star + t * d) for t in ts] for d in dirs]
    g = np.linspace(-1, 1, 25); d1, d2 = dirs[0], dirs[1]
    surf = [[J(w_star + a * d1 + b * d2) for b in g] for a in g]
    return dict(J_star=J_star, grad_norm=float(np.linalg.norm(2 * (A @ w_star - bvec))), cond=float(np.linalg.cond(A)),
                traces=traces, ts=ts.tolist(), slices=slices, grid=g.tolist(), surface=surf, w_star=w_star.tolist())


def logistic_global_check(X, y, C, n_starts=5, max_iter=500, seed=0):
    """L2-regularised logistic regression (the same objective as logit_path) is strictly convex: one global minimum. Fits it with
    L-BFGS from n_starts random starting points and reports that they reach the same loss and the same weights, the gradient
    norm at the solution, and every start's loss after each function evaluation."""
    import torch
    Xt = torch.as_tensor(np.asarray(X, float), dtype=torch.float64); yt = torch.as_tensor(np.asarray(y, float), dtype=torch.float64)
    n, k = Xt.shape; lam = 1.0 / (C * n)
    gen = torch.Generator().manual_seed(seed)
    finals, weights, traces = [], [], []
    for _ in range(n_starts):
        w = (torch.randn(k, generator=gen, dtype=torch.float64) * 2).requires_grad_(True)
        b = (torch.randn(1, generator=gen, dtype=torch.float64) * 2).requires_grad_(True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-12, tolerance_change=1e-15, history_size=30,
                                line_search_fn="strong_wolfe")
        tr = []

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 0.5 * lam * (w * w).sum()
            loss.backward(); tr.append(float(loss.detach()))
            return loss
        for _ in range(5):
            opt.step(closure)
        finals.append(tr[-1]); weights.append(torch.cat([w.detach(), b.detach()]).numpy()); traces.append(tr)
    w, b = torch.as_tensor(weights[int(np.argmin(finals))][:-1]).requires_grad_(True), torch.as_tensor(weights[int(np.argmin(finals))][-1:]).requires_grad_(True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt) + 0.5 * lam * (w * w).sum(); loss.backward()
    W = np.stack(weights)
    return dict(final_losses=finals, max_weight_spread=float(np.max(W.max(0) - W.min(0))), loss_spread=float(max(finals) - min(finals)),
                grad_norm=float(torch.cat([w.grad, b.grad]).norm()), traces=traces)


# ===================================================================================== 4. metrics and month-block bootstrap
def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def month_codes(dates, freq="M"):
    """Integer code of each row's calendar period (month by default; "Q" = quarter, "W-FRI" = week) and the number of periods."""
    m = pd.to_datetime(pd.Series(dates)).dt.to_period(freq).astype(str).values
    u = np.unique(m)
    return np.searchsorted(u, m), len(u)


def wm_r2(y, p, c):
    """Within-month R2: demean target and prediction by calendar month, best scalar rescale (cross-sectional skill only)."""
    y = np.asarray(y, float); p = np.asarray(p, float); K = int(c.max()) + 1
    n = np.maximum(np.bincount(c, minlength=K), 1)
    yd = y - (np.bincount(c, y, minlength=K) / n)[c]; pd_ = p - (np.bincount(c, p, minlength=K) / n)[c]
    ss = (pd_ ** 2).sum(); b = (yd * pd_).sum() / ss if ss > 0 else 0.0
    return float(1 - ((yd - b * pd_) ** 2).sum() / (yd ** 2).sum())


def qlike(y_log, p_log, s2):
    """QLIKE on variance: RV = exp(2 y), forecast F = exp(2 p + 2 s2) (log-normal bias correction, s2 from validation)."""
    rv = np.exp(2 * np.asarray(y_log, float)); f = np.exp(2 * np.asarray(p_log, float) + 2 * s2)
    q = rv / f
    return float(np.mean(q - np.log(q) - 1))


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.asarray(y, int), np.asarray(p, float)))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def spearman(y, p):
    return float(pd.Series(np.asarray(y, float)).rank().corr(pd.Series(np.asarray(p, float)).rank()))


def avg_precision(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y, int), np.asarray(p, float)))


def accuracy(y, p):
    return float(((np.asarray(p, float) > 0.5) == (np.asarray(y, float) > 0.5)).mean())


def best_f1_threshold(y, p, n_grid=200):
    """The probability threshold with the highest F1 score; meant to be computed on VALIDATION rows and then fixed."""
    y = np.asarray(y, float) > 0.5; p = np.asarray(p, float)
    cand = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n_grid)))
    best = (-1.0, 0.5)
    for t in cand:
        pred = p >= t; tp = float((pred & y).sum())
        f1 = 2 * tp / max(float(pred.sum() + y.sum()), 1e-12)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def class_metrics_at(y, p, thr):
    """Threshold metrics: precision, recall (sensitivity), F1, specificity, balanced accuracy, accuracy, the share of rows flagged
    (alert rate), the event rate (prevalence) and the accuracy of always predicting the majority class."""
    y = np.asarray(y, float) > 0.5; pred = np.asarray(p, float) >= thr
    tp, fp = float((pred & y).sum()), float((pred & ~y).sum()); fn, tn = float((~pred & y).sum()), float((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan"); rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else float("nan")
    prev = float(y.mean())
    return dict(threshold=float(thr), precision=prec, recall=rec, F1=f1, specificity=spec, balanced_acc=(rec + spec) / 2,
                accuracy=(tp + tn) / len(y), alert_rate=float(pred.mean()), prevalence=prev, majority_acc=max(prev, 1 - prev),
                TP=tp, FP=fp, FN=fn, TN=tn)


def calibration_line(y, p):
    """Calibration intercept and slope: logistic regression of the outcome on logit(p). Perfect calibration = (0, 1)."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y, int))
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def board_metrics(task, y, p, dates, base=None, s2=None, thr=None):
    """The leaderboard's metric set. Regression: R2, within-month R2, QLIKE, MAE, RMSE, Spearman; s2 (the variance of the
    model's VALIDATION residuals) is required for QLIKE, so no test label ever enters the metric. Classification: AUC, average
    precision, accuracy at 0.5, log loss, Brier skill score vs `base` (training base rate), ECE, prevalence; with thr (a
    threshold fixed on validation) also precision, recall, F1, specificity, balanced accuracy and alert rate."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    if task == "reg":
        if s2 is None:
            raise ValueError("board_metrics needs s2 = the variance of the model's validation residuals (QLIKE bias correction)")
        c, _ = month_codes(dates)
        return dict(R2=r2(y, p), wmR2=wm_r2(y, p, c), QLIKE=qlike(y, p, s2), MAE=float(np.mean(np.abs(y - p))), RMSE=rmse(y, p),
                    Spearman=spearman(y, p))
    base = float(np.mean(y)) if base is None else base
    out = dict(AUC=auc(y, p), AP=avg_precision(y, p), ACC=accuracy(y, p), logloss=logloss(y, p),
               BSS=1 - brier(y, p) / brier(y, np.full(len(y), base)), ECE=ece(y, p), prevalence=float(np.mean(y)))
    if thr is not None:
        cm = class_metrics_at(y, p, thr)
        out.update({k: cm[k] for k in ("precision", "recall", "F1", "specificity", "balanced_acc", "alert_rate")})
    return out


# ===================================================================================== 4b. Monte Carlo calibration (notebook side)
def mc_universal(n, t_df, res_sampler, seed=0, horizon=20, low_k=1.0):
    """Volatility-free draws of the dashboard's Monte Carlo model (tfcv_core.simulate_paths):
    U = 20-day log return per unit of forecast volatility, D = lowest intraday log price per unit of forecast volatility.
    For a forecast x_hat (log 20-day volatility): log return = exp(x_hat) * U and log(low / start) = exp(x_hat) * D."""
    import tfcv_core as C
    rng = np.random.default_rng(seed)
    eps = res_sampler(rng, n)
    cum, low = C.mc_unit_paths(n, t_df, rng, horizon, low_k)
    s = np.exp(eps)
    return s * cum[:, -1], s * low.min(1)


def _tilt_rows(H, R, p, iters=40):
    """tfcv_core.tilt_weights for many rows at once: H (rows, draws) drawdown hits, R (rows, draws) 20-day returns,
    p (rows,) target drawdown chances. Keeps each row's mean return; returns the weights (rows, draws)."""
    h = H.astype(np.float64)
    r = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-12)
    a = np.zeros(len(p)); b = np.zeros(len(p))
    for _ in range(iters):
        z = a[:, None] * h + b[:, None] * r
        w = np.exp(z - z.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        mh, mr = (w * h).sum(1), (w * r).sum(1)
        g1, g2 = mh - p, mr
        if max(np.abs(g1).max(), np.abs(g2).max()) < 1e-6:
            break
        dh, dr = h - mh[:, None], r - mr[:, None]
        j11, j12, j22 = (w * dh * dh).sum(1) + 1e-10, (w * dh * dr).sum(1), (w * dr * dr).sum(1) + 1e-10
        det = j11 * j22 - j12 ** 2
        a = np.clip(a - np.clip((j22 * g1 - j12 * g2) / det, -2, 2), -30, 30)
        b = np.clip(b - np.clip((-j12 * g1 + j11 * g2) / det, -2, 2), -30, 30)
    return w


def mc_scores(U, D, xhat, logret, dd_hit, p_model=None, levels=(0.5, 0.8, 0.9, 0.95), chunk=200, n_tilt=6000):
    """How well the simulation describes real outcomes. For each row: the probability integral transform (PIT) of the realised
    20-day log return, and the simulated chance of a 10% drawdown. With p_model (the calibrated drawdown model), the paths
    are reweighted per row exactly as the dashboard's 'match the drawdown model' option does (drawdown share = p_model, mean
    return unchanged). Returns {coverage at each level, mean absolute coverage error, PIT histogram, drawdown Brier}."""
    xhat = np.asarray(xhat, float); scale = np.exp(xhat)
    u_row = np.asarray(logret, float) / scale
    c_row = math.log(0.9) / scale
    if p_model is None:
        pit = np.searchsorted(np.sort(U), u_row, side="right") / len(U)
        p_dd = np.searchsorted(np.sort(D), c_row, side="right") / len(D)
    else:
        Ut, Dt = U[:n_tilt], D[:n_tilt]
        pit = np.empty(len(xhat)); p_dd = np.asarray(p_model, float).copy()
        for a in range(0, len(xhat), chunk):
            sl = slice(a, a + chunk)
            H = Dt[None, :] <= c_row[sl, None]
            R = np.expm1(scale[sl, None] * Ut[None, :])
            w = _tilt_rows(H, R, np.clip(p_dd[sl], 1e-4, 1 - 1e-4))
            pit[sl] = (w * (Ut[None, :] <= u_row[sl, None])).sum(1)
    cov = {f"{int(l * 100)}%": float(np.mean((pit >= (1 - l) / 2) & (pit <= 1 - (1 - l) / 2))) for l in levels}
    err = float(np.mean([abs(cov[f"{int(l * 100)}%"] - l) for l in levels]))
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / max(len(pit), 1)
    ok = np.isfinite(dd_hit)
    return dict(coverage=cov, coverage_error=err, pit_hist=hist.tolist(), mean_pit=float(np.mean(pit)),
                dd_brier=brier(np.asarray(dd_hit)[ok], p_dd[ok]), dd_rate=float(np.mean(np.asarray(dd_hit)[ok])), dd_mean_pred=float(np.mean(p_dd[ok])))


def ece(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = np.clip((p * bins).astype(int), 0, bins - 1); tot = 0.0
    for k in range(bins):
        m = e == k
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def nw_tstat(x, lags=4):
    """t-statistic of the mean of a time series with a Newey-West (HAC) standard error, allowing for serial correlation up to
    `lags` periods (overlapping 5-day labels make neighbouring weekly ICs correlated)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5:
        return float("nan")
    e = x - x.mean(); s = float(e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - L / (lags + 1)) * float(e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(max(s, 1e-18) / n))


def rank_ic(dates, p, x, min_names=20, freq="W-FRI", lags=4):
    """Rank IC: the Spearman correlation between predictions and outcomes across stocks, computed per calendar WEEK (a sample of
    stocks rarely shares the exact same decision date), then averaged over weeks. Returns (mean IC, Newey-West t-statistic,
    number of weeks). Weeks with fewer than min_names stocks are skipped."""
    df = pd.DataFrame({"w": pd.to_datetime(pd.Series(dates)).dt.to_period(freq).values, "p": np.asarray(p, float), "x": np.asarray(x, float)})
    df = df[np.isfinite(df.p) & np.isfinite(df.x)]
    ic = df.groupby("w").apply(lambda z: z["p"].rank().corr(z["x"].rank()) if len(z) >= min_names else np.nan, include_groups=False).dropna()
    if len(ic) < 5:
        return float("nan"), float("nan"), int(len(ic))
    return float(ic.mean()), nw_tstat(ic.sort_index().values, lags), int(len(ic))


class MonthBoot:
    """Block bootstrap on one evaluation set: resample calendar blocks (months by default, "Q" for quarters) with replacement and
    keep all their rows, so stocks that move together within a block stay together. The same resamples are used for every
    metric and every paired model comparison on that set."""

    def __init__(self, dates, n=1000, seed=42, freq="M"):
        self.c, self.K = month_codes(dates, freq)
        self.mc, _ = month_codes(dates, "M")                     # within-month R2 always uses calendar months
        rows = [np.flatnonzero(self.c == k) for k in range(self.K)]
        rng = np.random.default_rng(seed)
        self.idx = [np.concatenate([rows[k] for k in rng.integers(0, self.K, self.K)]) for _ in range(n)]

    def ci(self, fn, *arrays, uses_months=False):
        v = [fn(*[a[i] for a in arrays], self.mc[i]) if uses_months else fn(*[a[i] for a in arrays]) for i in self.idx]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], np.asarray(v)

    def diff(self, fn, y, p1, p2, uses_months=False):
        if uses_months:
            v = [fn(y[i], p1[i], self.mc[i]) - fn(y[i], p2[i], self.mc[i]) for i in self.idx]
        else:
            v = [fn(y[i], p1[i]) - fn(y[i], p2[i]) for i in self.idx]
        v = np.asarray(v)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], v


def holm(pvals):
    """Holm step-down adjusted p-values (dict in, dict out)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj


def wilson_lower(k, n, z=1.96):
    if n == 0:
        return 0.0
    ph = k / n; den = 1 + z * z / n
    return float((ph + z * z / (2 * n) - z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / den)


def isotonic_arrays(p, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), np.asarray(y, float))
    return {"x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist()}


def abstention_table(p, y, thresholds=None, coverages=(1.0, 0.5, 0.25, 0.10)):
    """Accuracy of direction calls among the most confident predictions. thresholds: {coverage: confidence threshold}
    fixed on validation (pass None on validation itself to compute them)."""
    p = np.asarray(p, float); y = np.asarray(y, int); conf = np.abs(p - 0.5); rows = []; thr = {}
    for c in coverages:
        t = 0.0 if c == 1.0 else (float(np.quantile(conf, 1 - c)) if thresholds is None else thresholds[c])
        thr[c] = t
        keep = conf >= t; n = int(keep.sum()); k = int(((p[keep] > 0.5) == y[keep]).sum())
        rows.append(dict(coverage_target=c, coverage_realised=n / len(p), n=n, accuracy=k / max(n, 1), wilson_lower=wilson_lower(k, n)))
    return pd.DataFrame(rows), thr
