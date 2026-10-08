#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cae_pipeline.py - Pipeline CAE one-class untuk ROI latent image "BI" Rupiah
===========================================================================

Satu skrip, hanya butuh:  numpy, pillow   (matplotlib opsional, untuk gambar)
    pip install numpy pillow matplotlib

Alur:
  1. muat dataset ROI uang ASLI  -> split per kelompok (lembar/sesi), bukan per crop
  2. latih CAE (float) murni NumPy: Conv(stride 2) x3 + ConvTranspose(stride 2) x3
  3. kuantisasi int8 pangkat-dua (bobot, aktivasi, requant hanya dengan shift)
  4. model integer bit-exact = golden reference untuk simulasi RTL
  5. threshold T lewat split conformal (SSE <= T  ->  AUTHENTIC, selain itu NON_AUTHENTIC)
  6. evaluasi AUROC terhadap anomali digital + baseline PCA dan template-SAD
  7. (opsional) ekspor bobot .hex, golden vector, dan digest SHA-256

KELUARAN SISTEM = 2 STATUS (v2.2):
  AUTHENTIC      hanya bila SEMUA syarat terpenuhi sekaligus: integritas SHA-256 lulus, input tepat
                 N piksel, selesai sebelum watchdog, dan SSE <= T.
  NON_AUTHENTIC  segala kondisi lain (SSE > T, input salah ukuran, integritas gagal, timeout, reset).
  NON_AUTHENTIC berarti "tidak terbukti asli oleh penyaring ini", BUKAN "pasti palsu".

Contoh (Windows, PowerShell):
  python cae_pipeline.py --data D:\\dataset\\BI --out runs\\k4 --enc-k 4 --dec-k 4 ^
         --group-regex "^(roi_latent_?\\d+)" --anomaly-dir D:\\dataset\\anomali
  python cae_pipeline.py --data D:\\dataset\\BI --out runs\\k4 --export
  python cae_pipeline.py --synthetic-demo --out runs\\demo --epochs 10     # cek instalasi

Struktur dataset yang dianggap benar:
  DATA/lembar_001/foto1.png, foto2.png ...   <- tiap subfolder = satu kelompok
  atau file datar + --group-regex "^(lembar\\d+)_"  <- grup diambil dari nama file
  Jika tidak ada info grup, tiap file jadi grupnya sendiri DAN skrip memperingatkan
  bahwa hasil bisa terlalu optimistis (kebocoran antar crop dari lembar yang sama).

Semua tensor channels-last (N, H, W, C). Bobot (k, k, Cin, Cout).
Definisi operasi (sama dengan PyTorch Conv2d / ConvTranspose2d, stride 2):
  conv   : cross-correlation, padding 1 sisi, keluaran H/2
  convT  : y = 2*i - p + ky, p = (k-2)//2, keluaran 2H   (k = 2 atau 4)
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(*a):
    print(*a, flush=True)


LANES = 8              # P pada spesifikasi
IN_EXP = -7            # skala input & rekonstruksi = 2**-7  (x_q = piksel - 128)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Keluaran keputusan: 2 status. Kondisi gagal apa pun jatuh ke NON_AUTHENTIC (fail-secure).
AUTHENTIC = "AUTHENTIC"
NON_AUTHENTIC = "NON_AUTHENTIC"
ANOM_GROUP_RX = r"^(.*?)_anom_\d+"        # nama crop anomali -> id foto sumber


def decide(sse, T, integrity_ok=True, input_ok=True, in_time=True):
    """Logika keputusan 2 status. AUTHENTIC hanya bila keempat syarat benar sekaligus."""
    return AUTHENTIC if (integrity_ok and input_ok and in_time and sse <= T) else NON_AUTHENTIC


def wilson(k, n, z=1.96):
    """Interval kepercayaan Wilson 95% untuk proporsi k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# =============================================================================
# 1. Arsitektur dan statistik (sumber tunggal untuk angka di spesifikasi)
# =============================================================================
def build_layers(channels, enc_k, dec_k, c_in=1):
    c1, c2, c3 = channels
    return [
        dict(name="E1", kind="conv",  k=enc_k, ci=c_in, co=c1,   relu=True),
        dict(name="E2", kind="conv",  k=enc_k, ci=c1,   co=c2,   relu=True),
        dict(name="E3", kind="conv",  k=enc_k, ci=c2,   co=c3,   relu=True),
        dict(name="D1", kind="convT", k=dec_k, ci=c3,   co=c2,   relu=True),
        dict(name="D2", kind="convT", k=dec_k, ci=c2,   co=c1,   relu=True),
        dict(name="D3", kind="convT", k=dec_k, ci=c1,   co=c_in, relu=False),
    ]


def layer_stats(layers, S, lanes=LANES):
    """MAC, bobot, siklus (pemetaan dasar dan lanjutan), memori per lapis."""
    h, rows = S, []
    for l in layers:
        k, ci, co = l["k"], l["ci"], l["co"]
        if l["kind"] == "conv":
            ho = h // 2
            taps = k * k * ci
            mac = ho * ho * co * taps
            cyc_base = ho * ho * taps * math.ceil(co / lanes)
            cyc_adv = cyc_base
        else:                                   # bentuk gather / sub-pixel
            ho = 2 * h
            taps = (k // 2) ** 2 * ci
            mac = ho * ho * co * taps
            cyc_base = ho * ho * taps * math.ceil(co / lanes)
            cyc_adv = h * h * taps * math.ceil(4 * co / lanes)
        rows.append(dict(name=l["name"], kind=l["kind"], k=k, ci=ci, co=co,
                         in_hw=h, out_hw=ho, mac=mac, weights=k * k * ci * co,
                         bias=co, taps=taps, cyc_base=cyc_base, cyc_adv=cyc_adv,
                         rom_words=math.ceil(co / lanes) * k * k * ci,
                         fm_bytes=ho * ho * co))
        h = ho
    return rows


def model_summary(layers, S, lanes=LANES):
    rows = layer_stats(layers, S, lanes)
    tot = lambda key: sum(r[key] for r in rows)
    sizes = {r["name"]: r["fm_bytes"] for r in rows}
    a = max(sizes["E1"], sizes["E3"], sizes["D2"])
    b = max(sizes["E2"], sizes["D1"])
    max_terms = max(r["taps"] for r in rows)
    acc_bound = max_terms * 128 * 128
    n_layers = len(rows)
    payload = tot("weights") + 4 * tot("bias") + n_layers + 4
    return dict(
        rows=rows, S=S, mac=tot("mac"), weights=tot("weights"), bias=tot("bias"),
        params=tot("weights") + tot("bias"),
        enc_mac=sum(r["mac"] for r in rows[:3]), dec_mac=sum(r["mac"] for r in rows[3:]),
        cyc_base=tot("cyc_base"), cyc_adv=tot("cyc_adv"),
        util_base=tot("mac") / (lanes * tot("cyc_base")),
        util_adv=tot("mac") / (lanes * tot("cyc_adv")),
        latent=rows[2]["fm_bytes"], inbuf=S * S, mem_A=a, mem_B=b,
        mem_min=S * S + a + b, mem_sym=S * S + 2 * max(a, b),
        rom_words=tot("rom_words"), rom_bytes=8 * tot("rom_words"),
        max_terms=max_terms, acc_bound=acc_bound,
        acc_bits=math.ceil(math.log2(acc_bound + 1)) + 1,
        sse_max=S * S * 255 * 255, sad_max=S * S * 255,
        sse_bits=math.ceil(math.log2(S * S * 255 * 255 + 1)),
        sad_bits=math.ceil(math.log2(S * S * 255 + 1)),
        payload_bytes=payload, sha_blocks=math.ceil((payload + 9) / 64),
    )


# =============================================================================
# 2. Operasi float (forward + backward) - channels-last
# =============================================================================
def conv_fwd(x, W, b):
    k = W.shape[0]
    N, H, Wd, _ = x.shape
    Ho, Wo = H // 2, Wd // 2
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros((N, Ho, Wo, W.shape[3]), dtype=x.dtype)
    for ky in range(k):
        for kx in range(k):
            xs = xp[:, ky:ky + 2 * Ho:2, kx:kx + 2 * Wo:2, :]
            out += np.tensordot(xs, W[ky, kx], axes=([3], [0]))
    return out + b, xp


def conv_bwd(dout, xp, W):
    k = W.shape[0]
    N, Ho, Wo, Co = dout.shape
    dW = np.zeros_like(W)
    dxp = np.zeros_like(xp)
    d2 = dout.reshape(-1, Co)
    for ky in range(k):
        for kx in range(k):
            xs = xp[:, ky:ky + 2 * Ho:2, kx:kx + 2 * Wo:2, :]
            dW[ky, kx] = xs.reshape(-1, xs.shape[3]).T @ d2
            dxp[:, ky:ky + 2 * Ho:2, kx:kx + 2 * Wo:2, :] += np.tensordot(dout, W[ky, kx], axes=([3], [1]))
    return dxp[:, 1:-1, 1:-1, :], dW, d2.sum(0)


def convT_fwd(x, W, b):
    k = W.shape[0]
    p = (k - 2) // 2
    N, H, Wd, _ = x.shape
    full = np.zeros((N, 2 * (H - 1) + k, 2 * (Wd - 1) + k, W.shape[3]), dtype=x.dtype)
    for ky in range(k):
        for kx in range(k):
            full[:, ky:ky + 2 * H:2, kx:kx + 2 * Wd:2, :] += np.tensordot(x, W[ky, kx], axes=([3], [0]))
    return full[:, p:p + 2 * H, p:p + 2 * Wd, :] + b


def convT_bwd(dout, x, W):
    k = W.shape[0]
    p = (k - 2) // 2
    N, H, Wd, Ci = x.shape
    Co = W.shape[3]
    dfull = np.zeros((N, 2 * (H - 1) + k, 2 * (Wd - 1) + k, Co), dtype=x.dtype)
    dfull[:, p:p + 2 * H, p:p + 2 * Wd, :] = dout
    dW = np.zeros_like(W)
    dx = np.zeros_like(x)
    x2 = x.reshape(-1, Ci)
    for ky in range(k):
        for kx in range(k):
            ds = dfull[:, ky:ky + 2 * H:2, kx:kx + 2 * Wd:2, :]
            dW[ky, kx] = x2.T @ ds.reshape(-1, Co)
            dx += np.tensordot(ds, W[ky, kx], axes=([3], [1]))
    return dx, dW, dout.reshape(-1, Co).sum(0)


def forward(params, layers, x):
    caches, h = [], x
    for l, (W, b) in zip(layers, params):
        if l["kind"] == "conv":
            z, c = conv_fwd(h, W, b)
        else:
            z, c = convT_fwd(h, W, b), h
        caches.append((c, z))
        h = np.maximum(z, 0) if l["relu"] else z
    return h, caches


def loss_and_grads(params, layers, x):
    out, caches = forward(params, layers, x)
    diff = out - x
    loss = float(np.mean(diff ** 2))
    d = 2.0 * diff / diff.size
    grads = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        l = layers[i]
        c, z = caches[i]
        if l["relu"]:
            d = d * (z > 0)
        if l["kind"] == "conv":
            d, dW, db = conv_bwd(d, c, params[i][0])
        else:
            d, dW, db = convT_bwd(d, c, params[i][0])
        grads[i] = (dW, db)
    return loss, grads


def init_params(layers, rng, dtype=np.float32):
    params = []
    for l in layers:
        k, ci, co = l["k"], l["ci"], l["co"]
        fan_in = ci * k * k if l["kind"] == "conv" else ci * (k // 2) ** 2
        W = rng.standard_normal((k, k, ci, co)) * math.sqrt(2.0 / fan_in)
        params.append((W.astype(dtype), np.zeros(co, dtype=dtype)))
    return params


def eval_loss(params, layers, X, bs=256):
    tot = 0.0
    for i in range(0, len(X), bs):
        xb = X[i:i + bs]
        out, _ = forward(params, layers, xb)
        tot += float(np.sum((out - xb) ** 2))
    return tot / X.size


def train_cae(layers, Xtr, Xva, epochs, batch, lr, seed, patience=20):
    rng = np.random.default_rng(seed)
    if batch > len(Xtr):
        log(f"  PERINGATAN: batch {batch} > jumlah data latih {len(Xtr)}; batch diturunkan ke {len(Xtr)}.")
        batch = len(Xtr)
    steps_ep = math.ceil(len(Xtr) / batch)
    log(f"  data latih {len(Xtr)}, batch {batch}, {steps_ep} langkah/epoch, maksimum {steps_ep * epochs} langkah")
    if steps_ep * epochs < 300:
        log("  PERINGATAN: total langkah gradien < 300; model kemungkinan kurang terlatih. "
            "Turunkan --batch atau naikkan --epochs.")
    params = init_params(layers, rng)
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    best, best_params, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        lr_ep = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * ep / max(1, epochs))))
        perm = rng.permutation(len(Xtr))
        tl, nb = 0.0, 0
        for i in range(0, len(perm), batch):
            xb = Xtr[perm[i:i + batch]]
            loss, grads = loss_and_grads(params, layers, xb)
            t += 1
            for j, (gW, gb) in enumerate(grads):
                for idx, g in enumerate((gW, gb)):
                    m[j][idx][...] = b1 * m[j][idx] + (1 - b1) * g
                    v[j][idx][...] = b2 * v[j][idx] + (1 - b2) * g * g
                    mh = m[j][idx] / (1 - b1 ** t)
                    vh = v[j][idx] / (1 - b2 ** t)
                    params[j][idx][...] -= lr_ep * mh / (np.sqrt(vh) + eps)
            tl += loss
            nb += 1
        va = eval_loss(params, layers, Xva)
        mark = ""
        if va < best - 1e-9:
            best, bad = va, 0
            best_params = [(W.copy(), b.copy()) for W, b in params]
            mark = " *"
        else:
            bad += 1
        log(f"  epoch {ep + 1:3d}/{epochs}  train {tl / nb:.6f}  val {va:.6f}  "
            f"lr {lr_ep:.5f}  {time.time() - t0:6.0f}s{mark}")
        if bad >= patience:
            log(f"  early stopping (tidak membaik {patience} epoch)")
            break
    return best_params, best


# =============================================================================
# 3. Model integer bit-exact (golden reference)
#    Penjumlahan dilakukan di float64: semua nilai bilangan bulat < 2**53,
#    jadi hasilnya EXACT berapapun urutan penjumlahan BLAS.
# =============================================================================
def conv_acc(h, Wq):
    k = Wq.shape[0]
    N, H, Wd, _ = h.shape
    Ho, Wo = H // 2, Wd // 2
    xp = np.pad(h, ((0, 0), (1, 1), (1, 1), (0, 0)))
    out = np.zeros((N, Ho, Wo, Wq.shape[3]))
    for ky in range(k):
        for kx in range(k):
            out += np.tensordot(xp[:, ky:ky + 2 * Ho:2, kx:kx + 2 * Wo:2, :], Wq[ky, kx], axes=([3], [0]))
    return out


def convT_acc(h, Wq):
    return convT_fwd(h, Wq, np.zeros(Wq.shape[3]))


def requant(acc, shift):
    if shift > 0:
        return (acc + (1 << (shift - 1))) >> shift       # geser aritmetika + pembulatan
    if shift == 0:
        return acc
    return acc << (-shift)


def int_forward(qparams, layers, xq, keep=False, bs=128):
    """xq: int (N,S,S,C) dalam [-128,127].  Return (xhat int, max|acc| per lapis, [keluaran per lapis])."""
    outs_all, accmax = [], np.zeros(len(layers), dtype=np.int64)
    result = []
    for s in range(0, len(xq), bs):
        h = xq[s:s + bs].astype(np.float64)
        outs = []
        for i, (l, q) in enumerate(zip(layers, qparams)):
            Wf = q["Wq"].astype(np.float64)
            acc = conv_acc(h, Wf) if l["kind"] == "conv" else convT_acc(h, Wf)
            acc = np.rint(acc).astype(np.int64) + q["bq"]
            accmax[i] = max(accmax[i], int(np.abs(acc).max()))
            if l["relu"]:
                acc = np.maximum(acc, 0)
            out = np.clip(requant(acc, q["shift"]), 0 if l["relu"] else -128, 127)
            if keep:
                outs.append(out.astype(np.int16))
            h = out.astype(np.float64)
        result.append(h.astype(np.int64))
        if keep:
            outs_all.append(outs)
    xhat = np.concatenate(result, axis=0)
    layer_outs = None
    if keep:
        layer_outs = [np.concatenate([o[i] for o in outs_all], axis=0) for i in range(len(layers))]
    return xhat, accmax, layer_outs


def quantize_model(params, layers, Xcal, pct):
    """Skala pangkat dua. Return daftar dict(Wq, bq, shift, e_w, e_in, e_out)."""
    acts, h = [], Xcal
    for l, (W, b) in zip(layers, params):
        z = conv_fwd(h, W, b)[0] if l["kind"] == "conv" else convT_fwd(h, W, b)
        h = np.maximum(z, 0) if l["relu"] else z
        acts.append(h)
    q, e_in = [], IN_EXP
    for i, (l, (W, b)) in enumerate(zip(layers, params)):
        wmax = float(np.abs(W).max()) or 1e-8
        e_w = int(math.ceil(math.log2(wmax / 127.0)))
        Wq = np.clip(np.rint(W / 2.0 ** e_w), -127, 127).astype(np.int64)
        if i == len(layers) - 1:
            e_out = IN_EXP                                  # rekonstruksi satu skala dengan input
        else:
            a = float(np.percentile(acts[i], pct)) or float(acts[i].max())
            e_out = int(math.ceil(math.log2(max(a, 1e-8) / 127.0)))
        bq = np.rint(b / 2.0 ** (e_in + e_w)).astype(np.int64)
        assert np.abs(bq).max() < 2 ** 31, f"bias {l['name']} tidak muat int32"
        q.append(dict(Wq=Wq, bq=bq, shift=e_out - e_in - e_w, e_w=e_w, e_in=e_in, e_out=e_out))
        e_in = e_out
    return q


# =============================================================================
# 4. Skor, baseline, metrik
# =============================================================================
def avg_ranks(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    c = np.cumsum(cnt)
    return ((c - cnt + 1 + c) / 2.0)[inv]


def auroc(neg, pos):
    """P(skor anomali > skor asli) + 0.5 P(sama). Skor lebih besar = lebih anomali."""
    s = np.concatenate([neg, pos])
    r = avg_ranks(s)
    n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def fpr_at_tpr(neg, pos, tpr=0.95):
    thr = np.sort(pos)[int(math.floor((1 - tpr) * len(pos)))]
    return float(np.mean(neg >= thr))


def conformal_threshold(scores, alpha):
    s = np.sort(scores)
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(s[-1]), False
    return float(s[k - 1]), True


def pca_fit(X, k, seed):
    mu = X.mean(0)
    Xc = X - mu
    rng = np.random.default_rng(seed)
    Y = Xc @ rng.standard_normal((Xc.shape[1], k + 10))
    for _ in range(3):
        Y, _ = np.linalg.qr(Xc @ (Xc.T @ Y))
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return mu, Vt[:k]


def pca_score(X, mu, V):
    Xc = X - mu
    R = Xc - (Xc @ V.T) @ V
    return (R ** 2).sum(1)


class Scorer:
    """Hitung skor semua metode untuk satu himpunan ROI uint8 (N,S,S)."""

    def __init__(self, layers, params, qparams, mu_p, V_p, mu_t):
        self.layers, self.params, self.qparams = layers, params, qparams
        self.mu_p, self.V_p, self.mu_t = mu_p, V_p, mu_t

    def __call__(self, Xu8, keep=False):
        xq = (Xu8.astype(np.int64) - 128)[..., None]
        flat = xq.reshape(len(xq), -1).astype(np.float32)
        out = {}
        sf = []
        for i in range(0, len(xq), 256):
            xb = (xq[i:i + 256] / 128.0).astype(np.float32)
            rec, _ = forward(self.params, self.layers, xb)
            sf.append((((xb - rec) * 128.0) ** 2).sum((1, 2, 3)))
        out["cae_float_sse"] = np.concatenate(sf)
        xhat, accmax, louts = int_forward(self.qparams, self.layers, xq, keep=keep)
        d = xq - xhat
        out["cae_int8_sse"] = (d * d).sum((1, 2, 3)).astype(np.float64)
        out["cae_int8_sad"] = np.abs(d).sum((1, 2, 3)).astype(np.float64)
        out["pca"] = pca_score(flat, self.mu_p, self.V_p).astype(np.float64)
        out["template_sad"] = np.abs(flat - self.mu_t).sum(1).astype(np.float64)
        # Baseline sepele: apakah statistik piksel sederhana sudah cukup memisahkan?
        out["std_rendah"] = (-flat.std(1)).astype(np.float64)                     # kontras rendah = anomali
        out["mean_dev"] = np.abs(flat.mean(1) - float(self.mu_t.mean())).astype(np.float64)
        out["_accmax"] = accmax
        if keep:
            out["_xhat"], out["_louts"] = xhat, louts
        return out


SHORT = {"pca": "PCA", "template_sad": "Templ.", "cae_float_sse": "CAEflt", "cae_int8_sse": "CAEi8",
         "cae_int8_sad": "CAEi8sad", "std_rendah": "Kontras", "mean_dev": "Cerah"}
METHODS = ["pca", "template_sad", "cae_float_sse", "cae_int8_sse", "cae_int8_sad", "std_rendah", "mean_dev"]


# =============================================================================
# 5. Data: muat, split per kelompok, anomali digital
# =============================================================================
def pool2x2(a):
    a = a.astype(np.uint16)
    return ((a[:, 0::2, 0::2] + a[:, 0::2, 1::2] + a[:, 1::2, 0::2] + a[:, 1::2, 1::2] + 2) >> 2).astype(np.uint8)


def load_images(folder, size, group_regex=None, with_groups=True, limit=None):
    from PIL import Image
    root = Path(folder)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"Tidak ada gambar ({', '.join(sorted(EXTS))}) di {root}")
    if limit:
        files = files[:limit]
    arrs, groups, warned, any_group = [], [], False, False
    rx = re.compile(group_regex) if group_regex else None
    for p in files:
        im = Image.open(p).convert("L")
        if im.size != (64, 64):
            if not warned:
                log(f"  PERINGATAN: ukuran gambar {im.size}, diubah ke 64x64 (contoh: {p.name})")
                warned = True
            im = im.resize((64, 64), Image.BOX if im.size[0] > 64 else Image.BILINEAR)
        arrs.append(np.asarray(im, dtype=np.uint8))
        if rx is not None:
            mt = rx.search(p.name)
            groups.append(mt.group(1) if mt and mt.groups() else (mt.group(0) if mt else p.stem))
            any_group = True
        elif p.parent != root:
            groups.append(str(p.parent.relative_to(root)))
            any_group = True
        else:
            groups.append(p.stem)
    X = np.stack(arrs)
    if size == 32:
        X = pool2x2(X)
    return X, np.array(groups), any_group


def group_split(groups, seed, frac=(("train", .60), ("val", .10), ("calib", .15), ("test", .15))):
    rng = np.random.default_rng(seed)
    ug, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(ug))
    total = counts.sum()
    cur = {n: 0 for n, _ in frac}
    assign = {}
    for gi in order:
        name = max(frac, key=lambda nf: (nf[1] * total - cur[nf[0]]) / (nf[1] * total))[0]
        assign[ug[gi]] = name
        cur[name] += counts[gi]
    idx = {n: np.where(np.array([assign[g] for g in groups]) == n)[0] for n, _ in frac}
    for n, i in idx.items():
        if len(i) == 0:
            raise SystemExit(f"Split '{n}' kosong: terlalu sedikit kelompok ({len(ug)}). "
                             f"Tambah data atau kurangi pembagian.")
    return idx, len(ug)


def _rng_for(name, seed):
    return np.random.default_rng(seed + zlib.crc32(name.encode()) % 100000)


def gblur(X, sigma):
    r = int(math.ceil(3 * sigma))
    xs = np.arange(-r, r + 1)
    ker = np.exp(-xs ** 2 / (2 * sigma ** 2))
    ker /= ker.sum()
    Y = X.astype(np.float32)
    n = Y.shape[2]
    Yp = np.pad(Y, ((0, 0), (0, 0), (r, r)), mode="edge")
    Y = sum(ker[i] * Yp[:, :, i:i + n] for i in range(len(ker)))
    n = Y.shape[1]
    Yp = np.pad(Y, ((0, 0), (r, r), (0, 0)), mode="edge")
    Y = sum(ker[i] * Yp[:, i:i + n, :] for i in range(len(ker)))
    return np.clip(np.rint(Y), 0, 255).astype(np.uint8)


def shift_img(X, dx, dy):
    S = X.shape[1]
    pad = max(abs(dx), abs(dy))
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    return Xp[:, pad - dy:pad - dy + S, pad - dx:pad - dx + S].copy()


def rotate_deg(X, deg):
    from PIL import Image
    out = np.empty_like(X)
    for i, a in enumerate(X):
        im = Image.fromarray(a).rotate(deg, resample=Image.BILINEAR, fillcolor=int(np.median(a)))
        out[i] = np.asarray(im)
    return out


def make_perturbations(S, seed):
    """Urutan: (nama, kelompok, fungsi). G1 = variasi capture UANG ASLI (idealnya tetap lolos).
    A1 = anomali struktural (idealnya terdeteksi). Penetapan kelompok adalah keputusan tim."""
    t = max(1, S // 4)

    def faded(X, rng, alpha=0.15):
        m = X.mean((1, 2), keepdims=True)
        return np.clip(np.rint(m + alpha * (X - m)), 0, 255).astype(np.uint8)

    def erase(X, rng):
        Y = X.copy()
        for i in range(len(Y)):
            y, x = rng.integers(0, S - t + 1, 2)
            Y[i, y:y + t, x:x + t] = int(np.median(Y[i]))
        return Y

    def noise(X, rng):
        return np.clip(np.rint(X + rng.normal(0, 25, X.shape)), 0, 255).astype(np.uint8)

    def shuffle(X, rng):
        Y = X.copy()
        n = S // t
        for i in range(len(Y)):
            tiles = [X[i, r * t:(r + 1) * t, c * t:(c + 1) * t] for r in range(n) for c in range(n)]
            for j, pj in enumerate(rng.permutation(len(tiles))):
                r, c = divmod(j, n)
                Y[i, r * t:(r + 1) * t, c * t:(c + 1) * t] = tiles[pj]
        return Y

    gain = lambda g: (lambda X, rng: np.clip(np.rint(X * g), 0, 255).astype(np.uint8))
    return [
        ("G1_blur_s1",   "G1", lambda X, r: gblur(X, 1.0)),
        ("G1_blur_s2",   "G1", lambda X, r: gblur(X, 2.0)),
        ("G1_terang+25", "G1", gain(1.25)),
        ("G1_gelap-25",  "G1", gain(0.75)),
        ("G1_geser_2px", "G1", lambda X, r: shift_img(X, 2, 2)),
        ("G1_rotasi_3d", "G1", lambda X, r: rotate_deg(X, 3.0)),
        ("A1_pudar_15",  "A1", faded),
        ("A1_hapus_blok", "A1", erase),
        ("A1_noise_25",  "A1", noise),
        ("A1_cermin",    "A1", lambda X, r: X[:, :, ::-1].copy()),
        ("A1_rotasi_90", "A1", lambda X, r: np.rot90(X, 1, axes=(1, 2)).copy()),
        ("A1_acak_tile", "A1", shuffle),
    ]


# -----------------------------------------------------------------------------
# Dataset sintetis (HANYA untuk memeriksa pipeline; bukan latent image sungguhan)
# -----------------------------------------------------------------------------
def make_synthetic(out_dir, n_sheets=24, per_sheet=40, seed=0):
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for s in range(n_sheets):
        d = out / f"lembar_{s:03d}"
        d.mkdir(parents=True, exist_ok=True)
        sdx, sdy = rng.integers(-1, 2, 2)
        sgain, soff = rng.uniform(0.92, 1.08), rng.uniform(-6, 6)
        for i in range(per_sheet):
            im = Image.new("L", (64, 64), 150)
            dr = ImageDraw.Draw(im)
            ink = 85
            dr.rectangle((14, 16, 19, 48), fill=ink)                       # B
            dr.ellipse((14, 16, 34, 32), outline=ink, width=5)
            dr.ellipse((14, 32, 36, 48), outline=ink, width=5)
            dr.rectangle((43, 16, 48, 48), fill=ink)                       # I
            dr.rectangle((38, 16, 53, 20), fill=ink)
            dr.rectangle((38, 44, 53, 48), fill=ink)
            a = np.asarray(im, dtype=np.float32)
            a = np.roll(a, (int(sdy + rng.integers(-1, 2)), int(sdx + rng.integers(-1, 2))), (0, 1))
            gx, gy = rng.uniform(-8, 8, 2)
            yy, xx = np.mgrid[0:64, 0:64]
            a = a * sgain * rng.uniform(0.95, 1.05) + soff + gx * (xx - 32) / 32 + gy * (yy - 32) / 32
            a = a + rng.normal(0, rng.uniform(2, 5), a.shape)
            a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
            if rng.random() < 0.3:
                a = gblur(a[None], rng.uniform(0.3, 0.8))[0]
            Image.fromarray(a).save(d / f"roi_{i:03d}.png")
    return out


# =============================================================================
# 6. Ekspor untuk RTL / testbench
# =============================================================================
def _hex8(a):
    return [f"{int(v) & 0xFF:02x}" for v in np.asarray(a).reshape(-1)]


def _write(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def payload_bytes(qparams, T):
    b = bytearray()
    for q in qparams:
        b += q["Wq"].astype(np.int8).tobytes()                           # W[ky][kx][ci][co]
        b += q["bq"].astype("<i4").tobytes()
    b += np.array([q["shift"] for q in qparams], dtype=np.int8).tobytes()
    b += np.array([int(T)], dtype="<u4").tobytes()
    return bytes(b)


def export_all(out, layers, qparams, T, scorer, test_u8, perturbed, n_golden):
    ex = Path(out) / "export"
    (ex / "golden").mkdir(parents=True, exist_ok=True)
    for l, q in zip(layers, qparams):
        W, b = q["Wq"], q["bq"]
        _write(ex / f"{l['name']}_w.hex", _hex8(W))
        _write(ex / f"{l['name']}_b.hex", [f"{int(v) & 0xFFFFFFFF:08x}" for v in b])
        words = []
        for g in range(math.ceil(l["co"] / LANES)):
            for ky in range(l["k"]):
                for kx in range(l["k"]):
                    for ci in range(l["ci"]):
                        lanes = np.zeros(LANES, dtype=np.int64)
                        sl = W[ky, kx, ci, g * LANES:(g + 1) * LANES]
                        lanes[:len(sl)] = sl
                        word = sum((int(v) & 0xFF) << (8 * i) for i, v in enumerate(lanes))
                        words.append(f"{word:016x}")
        _write(ex / f"{l['name']}_rom64.hex", words)
    _write(ex / "tau.hex", [f"{int(T):08x}"])
    pay = payload_bytes(qparams, T)
    (ex / "payload.bin").write_bytes(pay)
    dig = hashlib.sha256(pay).hexdigest()
    _write(ex / "payload.sha256", [f"{dig}  payload.bin  ({len(pay)} bytes)"])

    samples = [("clean", test_u8[i]) for i in range(max(1, n_golden // 2))]
    names = [n for n in perturbed if n.startswith("A1")] + [n for n in perturbed if n.startswith("G1")] \
        + [n for n in perturbed if n == "NYATA"]
    j = 0
    while len(samples) < n_golden and names:
        n = names[j % len(names)]
        samples.append((n, perturbed[n][(j // len(names)) % len(perturbed[n])]))
        j += 1
    index = []
    for si, (nm, img) in enumerate(samples):
        sc = scorer(img[None], keep=True)
        tag = f"s{si:02d}"
        _write(ex / "golden" / f"{tag}_in.hex", _hex8(img))
        files = {"input": f"{tag}_in.hex"}
        for l, o in zip(layers, sc["_louts"]):
            fn = f"{tag}_{l['name']}.hex"
            _write(ex / "golden" / fn, _hex8(o[0]))
            files[l["name"]] = fn
        sse = int(sc["cae_int8_sse"][0])
        index.append(dict(sample=tag, source=nm, files=files, sse=sse, sad=int(sc["cae_int8_sad"][0]),
                          decision=decide(sse, T)))
    (ex / "golden" / "golden_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    # Tabel kebenaran keputusan (16 kombinasi) untuk testbench: hanya 1111 -> AUTHENTIC.
    tt = [dict(integrity_ok=a, input_ok=b, in_time=c, sse_le_T=d, result=decide(0 if d else 1, 0, a, b, c))
          for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]
    assert sum(r["result"] == AUTHENTIC for r in tt) == 1
    (ex / "decision_truth_table.json").write_text(json.dumps(tt, indent=1), encoding="utf-8")
    (ex / "README_export.txt").write_text(EXPORT_README, encoding="utf-8")
    return dig, len(pay)


EXPORT_README = """FORMAT EKSPOR (usulan; RTL harus mengikuti byte yang sama)
- *_w.hex      : bobot int8, urutan W[ky][kx][ci][co] (co paling cepat), 1 byte per baris.
- *_b.hex      : bias int32 (domain akumulator), 1 nilai per baris, komplemen dua.
- *_rom64.hex  : ROM lebar 64-bit untuk $readmemh. 1 kata = 8 lane keluaran (lane 0 = byte
                 paling rendah, sisa lane diisi 0). Alamat = grup*(k*k*ci) + (ky*k+kx)*ci + ci.
- tau.hex      : threshold T (uint32). Keputusan 2 status: SSE <= T (dan syarat lain) -> AUTHENTIC,
                 selain itu NON_AUTHENTIC. Register hasil 1 bit: 1 = AUTHENTIC, nilai reset 0.
- decision_truth_table.json : 16 kombinasi (integrity_ok, input_ok, in_time, sse<=T). Hanya 1111 -> AUTHENTIC.
- payload.bin  : bobot, bias (int32 LE), shift per lapis (int8), T (uint32 LE), berurutan
                 per lapis E1..D3. payload.sha256 = SHA-256 byte tersebut.
- golden/      : untuk tiap sampel: input uint8 (baris-per-baris), keluaran tiap lapis int8
                 urutan (H, W, C) channels-last, plus SSE/SAD/keputusan di golden_index.json.
Operasi: conv = cross-correlation stride 2, pad 1. convT = y = 2*i - p + ky, p=(k-2)//2.
Requant: acc -> ReLU -> (acc + (1<<(s-1))) >> s -> saturasi [0,127]; lapis terakhir linear,
saturasi [-128,127]. Piksel masuk: x_q = piksel - 128. SSE = jumlah (x_q - x_hat_q)^2.
"""


# =============================================================================
# 7. Self-test (selalu dijalankan di awal)
# =============================================================================
def _naive_conv(x, W):
    H, Wd, Ci = x.shape
    k = W.shape[0]
    xp = np.zeros((H + 2, Wd + 2, Ci), dtype=np.int64)
    xp[1:-1, 1:-1] = x
    out = np.zeros((H // 2, Wd // 2, W.shape[3]), dtype=np.int64)
    for i in range(H // 2):
        for j in range(Wd // 2):
            for ky in range(k):
                for kx in range(k):
                    for ci in range(Ci):
                        out[i, j] += xp[2 * i + ky, 2 * j + kx, ci] * W[ky, kx, ci]
    return out


def _naive_convT(x, W):
    H, Wd, Ci = x.shape
    k = W.shape[0]
    p = (k - 2) // 2
    out = np.zeros((2 * H, 2 * Wd, W.shape[3]), dtype=np.int64)
    for i in range(H):
        for j in range(Wd):
            for ky in range(k):
                for kx in range(k):
                    y, xx = 2 * i - p + ky, 2 * j - p + kx
                    if 0 <= y < 2 * H and 0 <= xx < 2 * Wd:
                        out[y, xx] += x[i, j] @ W[ky, kx]
    return out


def selftest():
    rng = np.random.default_rng(123)
    # (a) operasi integer vs implementasi naif
    for k in (3, 4):
        x = rng.integers(-128, 128, (1, 8, 8, 2))
        W = rng.integers(-127, 128, (k, k, 2, 3))
        assert np.array_equal(np.rint(conv_acc(x.astype(float), W.astype(float))[0]).astype(np.int64),
                              _naive_conv(x[0], W)), f"conv k={k} tidak cocok"
    for k in (2, 4):
        x = rng.integers(0, 128, (1, 4, 4, 3))
        W = rng.integers(-127, 128, (k, k, 3, 2))
        assert np.array_equal(np.rint(convT_acc(x.astype(float), W.astype(float))[0]).astype(np.int64),
                              _naive_convT(x[0], W)), f"convT k={k} tidak cocok"
    # (b) gradien analitik vs beda-hingga (float64)
    for ek, dk in ((4, 4), (3, 2)):
        layers = build_layers((2, 3, 3), ek, dk)
        params = [(W.astype(np.float64), (rng.standard_normal(b.shape) * 0.1)) for W, b in init_params(layers, rng)]
        x = rng.standard_normal((2, 16, 16, 1))
        _, grads = loss_and_grads(params, layers, x)
        worst = 0.0
        for li in range(len(layers)):
            for which in (0, 1):
                arr = params[li][which]
                for _ in range(3):
                    idx = tuple(int(rng.integers(0, s)) for s in arr.shape)
                    old = arr[idx]
                    eps = 1e-6
                    arr[idx] = old + eps
                    lp = loss_and_grads(params, layers, x)[0]
                    arr[idx] = old - eps
                    lm = loss_and_grads(params, layers, x)[0]
                    arr[idx] = old
                    num, ana = (lp - lm) / (2 * eps), grads[li][which][idx]
                    worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-10))
        assert worst < 1e-4, f"gradien salah (k={ek}/{dk}): galat relatif {worst:.2e}"
    # (c) AUROC: kasus tertutup (ada tie)
    assert abs(auroc(np.array([1., 2., 2.]), np.array([2., 3.])) - 5.0 / 6.0) < 1e-12   # ada tie
    assert abs(auroc(np.array([0., 1.]), np.array([2., 3.])) - 1.0) < 1e-12
    assert abs(auroc(np.array([2., 3.]), np.array([0., 1.])) - 0.0) < 1e-12
    assert abs(auroc(np.array([1., 2.]), np.array([1., 2.])) - 0.5) < 1e-12
    try:
        from sklearn.metrics import roc_auc_score
        a, b = rng.integers(0, 20, 200).astype(float), rng.integers(5, 30, 150).astype(float)
        ref = roc_auc_score(np.r_[np.zeros(200), np.ones(150)], np.r_[a, b])
        assert abs(auroc(a, b) - ref) < 1e-12
    except ImportError:
        pass
    # (d) conformal: n=19, alpha=0.05 -> skor terbesar
    assert conformal_threshold(np.arange(19.), 0.05) == (18.0, True)
    # (e) keputusan 2 status: hanya satu dari 16 kombinasi yang AUTHENTIC; batas SSE == T masih AUTHENTIC
    n_auth = 0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for d in (0, 1):
                    n_auth += decide(0 if d else 2, 1, a, b, c) == AUTHENTIC
    assert n_auth == 1 and decide(5, 5) == AUTHENTIC and decide(6, 5) == NON_AUTHENTIC
    lo, hi = wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95 and wilson(0, 0)[0] != wilson(0, 0)[0]
    return True


# =============================================================================
# 8. Laporan, gambar
# =============================================================================
def fmt_table(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def make_plots(out, sc_clean, sc_by, T, auc_a1, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  (matplotlib tidak ada, gambar dilewati)")
        return []
    ink, ink2, muted, grid, base, surf = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"         # slot kategorikal 1-3 (tervalidasi all-pairs)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": base,
                         "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
                         "axes.facecolor": surf, "figure.facecolor": surf, "text.color": ink})
    files = []
    # --- Gambar 1: distribusi SSE int8
    a1 = np.concatenate([v["cae_int8_sse"] for k, v in sc_by.items() if k.startswith("A1")])
    g1 = np.concatenate([v["cae_int8_sse"] for k, v in sc_by.items() if k.startswith("G1")])
    cl = sc_clean["cae_int8_sse"]
    rl = sc_by["NYATA"]["cae_int8_sse"] if "NYATA" in sc_by else None
    lo = np.log10(max(1.0, min(cl.min(), a1.min(), g1.min())))
    hi = np.log10(max(cl.max(), a1.max(), g1.max()) + 1)
    if rl is not None:
        lo, hi = min(lo, np.log10(max(1.0, rl.min()))), max(hi, np.log10(rl.max() + 1))
    bins = np.linspace(lo, hi, 50)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    series = [(cl, c1, "Uang asli (uji)"), (g1, c3, "Asli, variasi capture (G1)"),
              (a1, c2, "Anomali digital (A1)")]
    if rl is not None:
        series.append((rl, "#7a4fd0", "Anomali nyata (crop non-asli)"))
    for data, col, lab in series:
        ax.hist(np.log10(np.maximum(data, 1)), bins=bins, density=True, histtype="stepfilled",
                color=col, alpha=0.28, linewidth=0)
        ax.hist(np.log10(np.maximum(data, 1)), bins=bins, density=True, histtype="step",
                color=col, linewidth=1.8, label=lab)
    ax.axvline(math.log10(max(T, 1)), color=ink, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(math.log10(max(T, 1)), ax.get_ylim()[1] * 0.97, f"  T = {int(T):,}".replace(",", "."),
            va="top", ha="left", fontsize=9, color=ink)
    ax.set_xlabel("SSE rekonstruksi, CAE int8 (skala log10)")
    ax.set_ylabel("Kepadatan")
    ax.set_title(f"Distribusi SSE pada himpunan uji (alpha = {str(args.alpha).replace('.', ',')})",
                 loc="left", fontsize=11, color=ink)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88), fontsize=9)
    fig.tight_layout()
    p = Path(out) / "fig_sse_distribution.png"
    fig.savefig(p, facecolor=surf)
    plt.close(fig)
    files.append(p)
    # --- Gambar 2: AUROC per metode (A1 gabungan)
    labs = {"pca": "PCA", "template_sad": "Template-SAD", "cae_float_sse": "CAE float",
            "cae_int8_sse": "CAE int8 (SSE)", "cae_int8_sad": "CAE int8 (SAD)",
            "std_rendah": "Kontras piksel (sepele)", "mean_dev": "Kecerahan rata-rata (sepele)"}
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    vals = [auc_a1[m] for m in METHODS]
    y = np.arange(len(METHODS))
    ax.barh(y, vals, color=c1, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([labs[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)                       # batang selalu mulai dari nol
    ax.axvline(0.5, color=ink2, linewidth=1, linestyle=(0, (4, 3)))
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}".replace(".", ","), va="center", fontsize=9, color=ink)
    ax.set_xlabel("AUROC (asli vs anomali struktural A1 gabungan; 0,5 = acak)")
    ax.set_title("Perbandingan metode pada anomali struktural", loc="left", fontsize=11, color=ink)
    ax.grid(axis="x", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = Path(out) / "fig_auroc_methods.png"
    fig.savefig(p, facecolor=surf)
    plt.close(fig)
    files.append(p)
    return files


# =============================================================================
# 9. Main
# =============================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pipeline CAE one-class latent image BI (NumPy)")
    ap.add_argument("--data", help="folder ROI uang ASLI (rekursif)")
    ap.add_argument("--out", default="runs/exp", help="folder keluaran")
    ap.add_argument("--size", type=int, default=64, choices=(32, 64), help="64 = asli, 32 = pool 2x2")
    ap.add_argument("--enc-k", type=int, default=4, choices=(3, 4))
    ap.add_argument("--dec-k", type=int, default=4, choices=(2, 4))
    ap.add_argument("--channels", default="4,8,8", help="kanal E1,E2,E3 (latent = E3)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8, help="otomatis diturunkan bila > jumlah data latih")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--alpha", type=float, default=0.05, help="target laju false-anomaly (conformal)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-regex", default=None, help="regex; group(1) = id kelompok dari nama file")
    ap.add_argument("--anomaly-dir", default=None, help="folder anomali NYATA (sampel resmi), opsional")
    ap.add_argument("--anomaly-group-regex", default=None,
                    help="regex; group(1) = id foto sumber crop anomali (default: nama sebelum _anom_NN)")
    ap.add_argument("--pca-k", type=int, default=16)
    ap.add_argument("--act-percentile", type=float, default=99.99)
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah gambar (debug)")
    ap.add_argument("--export", action="store_true", help="tulis .hex, golden vector, digest SHA-256")
    ap.add_argument("--n-golden", type=int, default=16)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--synthetic-demo", action="store_true", help="buat data sintetis untuk uji pipeline")
    ap.add_argument("--selftest-only", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    log("== Self-test ==")
    selftest()
    log("  OK: operasi integer == implementasi naif; gradien == beda-hingga; AUROC; conformal; keputusan 2 status")
    if args.selftest_only:
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic_demo:
        data_dir = make_synthetic(out / "synthetic_data", seed=args.seed)
        log(f"== Data sintetis dibuat di {data_dir} (BUKAN latent image sungguhan) ==")
    elif args.data:
        data_dir = args.data
    else:
        raise SystemExit("Beri --data FOLDER atau --synthetic-demo")
    S = args.size
    channels = tuple(int(c) for c in args.channels.split(","))
    assert len(channels) == 3
    layers = build_layers(channels, args.enc_k, args.dec_k)
    summ = model_summary(layers, S)

    log("== Data ==")
    X, groups, has_groups = load_images(data_dir, S, args.group_regex, limit=args.limit)
    idx, n_groups = group_split(groups, args.seed)
    log(f"  {len(X)} gambar, {n_groups} kelompok, ukuran {S}x{S}; "
        + ", ".join(f"{n}={len(i)}" for n, i in idx.items()))
    if not has_groups:
        log("  PERINGATAN: tidak ada info kelompok -> tiap file jadi grup sendiri. Crop dari lembar yang sama")
        log("              bisa tersebar di train dan test, sehingga hasil terlalu optimistis. Gunakan subfolder")
        log("              per lembar/sesi atau --group-regex.")
    elif n_groups < 8:
        log(f"  PERINGATAN: hanya {n_groups} kelompok; estimasi threshold dan AUROC kurang stabil.")

    to_f = lambda u8: ((u8.astype(np.float32) - 128.0) / 128.0)[..., None]
    Xtr, Xva = to_f(X[idx["train"]]), to_f(X[idx["val"]])
    Xcal_u8, Xte_u8 = X[idx["calib"]], X[idx["test"]]

    log(f"== Model: kernel enc {args.enc_k}x{args.enc_k}, dec {args.dec_k}x{args.dec_k}, kanal {channels} ==")
    log(f"  MAC {summ['mac']:,} | parameter {summ['params']:,} | latent {summ['latent']} nilai".replace(",", "."))
    log("== Latih CAE (float) ==")
    params, best_val = train_cae(layers, Xtr, Xva, args.epochs, args.batch, args.lr, args.seed)

    log("== Kuantisasi int8 pangkat-dua ==")
    sub = Xtr[np.random.default_rng(args.seed).permutation(len(Xtr))[:2000]]
    qparams = quantize_model(params, layers, sub, args.act_percentile)
    log(fmt_table([[l["name"], q["e_w"], q["e_in"], q["e_out"], q["shift"],
                    int(np.abs(q["Wq"]).max()), int(np.abs(q["bq"]).max()) if len(q["bq"]) else 0]
                   for l, q in zip(layers, qparams)],
                  ["lapis", "e_w", "e_in", "e_out", "shift", "|Wq|max", "|bq|max"]))
    if any(q["shift"] < 0 for q in qparams):
        log("  PERINGATAN: ada shift negatif (geser kiri). Spesifikasi mengasumsikan geser kanan.")

    log("== Baseline dan skor ==")
    flat = lambda u8: (u8.astype(np.int64) - 128).reshape(len(u8), -1).astype(np.float32)
    rngp = np.random.default_rng(args.seed)
    ptr = flat(X[idx["train"]][rngp.permutation(len(idx["train"]))[:4000]])
    mu_p, V_p = pca_fit(ptr, args.pca_k, args.seed)
    mu_t = flat(X[idx["train"]]).mean(0)
    scorer = Scorer(layers, params, qparams, mu_p, V_p, mu_t)

    sc_cal = scorer(Xcal_u8)
    sc_clean = scorer(Xte_u8)
    T, ok = conformal_threshold(sc_cal["cae_int8_sse"], args.alpha)
    if not ok:
        log(f"  PERINGATAN: kalibrasi {len(Xcal_u8)} sampel terlalu sedikit untuk alpha={args.alpha}; "
            f"T diset ke skor maksimum (tanpa jaminan).")
    T = int(T)
    N = S * S
    log(f"  T = {T}  (tau = T/N = {T / N:.3f} per piksel; butuh {max(1, T.bit_length())} bit; SSE maks {summ['sse_max']})")
    assert T <= summ["sse_max"]

    pert_defs = make_perturbations(S, args.seed)
    sc_by, perturbed = {}, {}
    for name, grp, fn in pert_defs:
        Xp = fn(Xte_u8, _rng_for(name, args.seed))
        perturbed[name] = Xp
        sc_by[name] = scorer(Xp)
    if args.anomaly_dir:
        Xa, ga, _ = load_images(args.anomaly_dir, S, args.anomaly_group_regex or ANOM_GROUP_RX)
        perturbed["NYATA"] = Xa
        sc_by["NYATA"] = scorer(Xa)
        pert_defs.append(("NYATA", "REAL", None))
        log(f"  anomali nyata: {len(Xa)} gambar dari {len(np.unique(ga))} sumber foto")

    accmax = np.maximum.reduce([sc_clean["_accmax"], sc_cal["_accmax"]] + [v["_accmax"] for v in sc_by.values()])
    acc_bits_obs = [int(a).bit_length() + 1 for a in accmax]
    log("  |acc| maksimum teramati per lapis: " + ", ".join(f"{l['name']}={int(a)}({b}b)"
                                                          for l, a, b in zip(layers, accmax, acc_bits_obs)))
    assert max(acc_bits_obs) <= 24, "akumulator 24-bit TIDAK cukup untuk data ini"

    # ---- AUROC
    names = [n for n, _, _ in pert_defs]
    auc = {n: {m: auroc(sc_clean[m], sc_by[n][m]) for m in METHODS} for n in names}
    pool = lambda grp: {m: np.concatenate([sc_by[n][m] for n, g, _ in pert_defs if g == grp]) for m in METHODS}
    pa1, pg1 = pool("A1"), pool("G1")                      # HANYA perturbasi digital; anomali nyata terpisah
    auc_a1 = {m: auroc(sc_clean[m], pa1[m]) for m in METHODS}
    auc_g1 = {m: auroc(sc_clean[m], pg1[m]) for m in METHODS}
    fpr95 = {m: fpr_at_tpr(sc_clean[m], pa1[m]) for m in METHODS}
    has_real = "NYATA" in sc_by
    # ---- titik operasi pada T: keputusan 2 status
    sse_clean, sse_cal = sc_clean["cae_int8_sse"], sc_cal["cae_int8_sse"]
    fa_clean = float(np.mean(sse_clean > T))
    flagged = {n: float(np.mean(sc_by[n]["cae_int8_sse"] > T)) for n in names}
    fa_cal = float(np.mean(sse_cal > T))
    # ---- paritas float vs int8
    a, b = avg_ranks(sc_clean["cae_float_sse"]), avg_ranks(sc_clean["cae_int8_sse"])
    spearman = float(np.corrcoef(a, b)[0, 1])
    dauc = auc_a1["cae_float_sse"] - auc_a1["cae_int8_sse"]

    # ---- evaluasi keputusan 2 status (tiap himpunan: berapa AUTHENTIC / NON_AUTHENTIC, benar atau salah)
    def dec_row(label, sse, expect):
        n = len(sse)
        n_auth = int(np.sum(sse <= T))
        k = n_auth if expect == AUTHENTIC else n - n_auth
        lo, hi = wilson(k, n)
        return [label, n, n_auth, n - n_auth, expect, f"{k / n:.3f}", f"[{lo:.2f}-{hi:.2f}]"], k, n
    eval_rows, dec_stats = [], {}
    r, k, n = dec_row("ASLI uji (G0)", sse_clean, AUTHENTIC); eval_rows.append(r); dec_stats["G0"] = (k, n)
    for nm, g, _ in pert_defs:
        if g == "G1":
            r, k, n = dec_row(nm, sc_by[nm]["cae_int8_sse"], AUTHENTIC)
        elif g == "A1":
            r, k, n = dec_row(nm, sc_by[nm]["cae_int8_sse"], NON_AUTHENTIC)
        else:
            r, k, n = dec_row("ANOMALI NYATA", sc_by[nm]["cae_int8_sse"], NON_AUTHENTIC)
        eval_rows.append(r); dec_stats[nm] = (k, n)
    bal = None
    if has_real:
        tar = dec_stats["G0"][0] / dec_stats["G0"][1]
        trr = dec_stats["NYATA"][0] / dec_stats["NYATA"][1]
        bal = 0.5 * (tar + trr)
    # ---- pembanding pada tingkat keputusan: tiap metode dengan T-nya sendiri (conformal pada kalibrasi)
    cmp_rows = []
    if has_real:
        for m in METHODS:
            Tm, okm = conformal_threshold(sc_cal[m], args.alpha)
            far_m = float(np.mean(sc_clean[m] > Tm))                  # asli ditolak
            trr_m = float(np.mean(sc_by["NYATA"][m] > Tm))            # anomali nyata ditolak
            cmp_rows.append([SHORT[m], f"{auroc(sc_clean[m], sc_by['NYATA'][m]):.3f}",
                             f"{1 - far_m:.3f}", f"{trr_m:.3f}", f"{0.5 * ((1 - far_m) + trr_m):.3f}"])
    # ---- peringatan otomatis
    warns = []
    if len(Xcal_u8) < math.ceil(1 / args.alpha) - 1:
        warns.append(f"n_calib = {len(Xcal_u8)} < {math.ceil(1 / args.alpha) - 1}: T hanyalah skor maksimum kalibrasi; "
                     f"JANGAN menulis 'false-reject <= {args.alpha:.0%} terjamin'.")
    if len(Xte_u8) < 30:
        warns.append(f"Uji asli hanya {len(Xte_u8)} gambar; satu penolakan = {1 / len(Xte_u8):.1%}. Lihat interval kepercayaan, "
                     f"dan jalankan evaluasi.py (validasi silang antar kelompok) untuk angka yang lebih kokoh.")
    if not has_groups:
        warns.append("Tidak ada info kelompok: kemungkinan kebocoran antar crop dari lembar/sesi yang sama.")
    if has_real:
        sd_auc = auc["NYATA"]["std_rendah"]
        if sd_auc >= 0.9:
            warns.append(f"Baseline sepele 'kontras piksel' sudah mencapai AUROC {sd_auc:.3f} pada anomali nyata: "
                         f"himpunan ini kemungkinan didominasi area datar/non-latent, sehingga AUROC CAE belum "
                         f"membuktikan model mengenali pola 'BI'.")
        if auc["NYATA"]["template_sad"] > auc["NYATA"]["cae_int8_sse"]:
            warns.append("Template rata-rata mengalahkan CAE pada anomali nyata. Laporkan apa adanya sebagai pembanding.")
    struct = [auc[n]["cae_int8_sse"] for n in ("A1_cermin", "A1_rotasi_90", "A1_acak_tile") if n in auc]
    if struct and max(abs(v - 0.5) for v in struct) < 0.1:
        warns.append("Cermin/rotasi 90/acak tile ~ AUROC 0,5: model belum terbukti memakai bentuk 'BI'. "
                     "Hindari klaim 'mendeteksi struktur BI'.")
    if auc.get("A1_pudar_15", {}).get("cae_int8_sse", 1) < 0.5 or auc.get("G1_blur_s2", {}).get("cae_int8_sse", 1) < 0.5:
        warns.append("ROI pudar/buram mendapat error LEBIH KECIL daripada asli (AUROC < 0,5): cetakan buram dapat lolos. "
                     "Masukkan ke bagian keterbatasan.")

    # ---- laporan
    L = []
    L += ["LAPORAN PIPELINE CAE", "=" * 60, f"argumen: {vars(args)}", ""]
    L += [f"Data: {len(X)} gambar, {n_groups} kelompok; train/val/calib/test = "
          + "/".join(str(len(idx[k])) for k in ("train", "val", "calib", "test")),
          f"Model: enc k={args.enc_k}, dec k={args.dec_k}, kanal {channels}, S={S}",
          f"  MAC {summ['mac']}, parameter {summ['params']} (bobot {summ['weights']}, bias {summ['bias']}), "
          f"latent {summ['latent']}",
          f"  siklus 8 lane: dasar {summ['cyc_base']} (util {summ['util_base']:.3f}), "
          f"lanjutan {summ['cyc_adv']} (util {summ['util_adv']:.3f})",
          f"  batas akumulator teori {summ['acc_bound']} ({summ['acc_bits']} bit); teramati {max(acc_bits_obs)} bit",
          f"Val MSE terbaik (float): {best_val:.6f}", ""]
    L += [f"Threshold (split conformal, alpha={args.alpha}, n_calib={len(Xcal_u8)}): T = {T}",
          f"  asli ditolak (NON_AUTHENTIC) pada kalibrasi {fa_cal:.3f}; pada UJI asli bersih {fa_clean:.3f}",
          "  (jaminan hanya berlaku bila kondisi capture uji sama dengan kalibrasi DAN n_calib cukup)", ""]
    L += ["KEPUTUSAN 2 STATUS pada T (CAE int8, SSE <= T -> AUTHENTIC; selain itu NON_AUTHENTIC)",
          "  'benar' = fraksi keputusan yang sesuai harapan; interval = Wilson 95% (mengasumsikan sampel independen,",
          "  padahal crop dari lembar sama berkorelasi, jadi interval sebenarnya lebih lebar)",
          fmt_table(eval_rows, ["himpunan", "n", "AUTH", "NON_AUTH", "harapan", "benar", "CI95"]), ""]
    if has_real:
        tp, nn = dec_stats["NYATA"]
        ga_, gn = dec_stats["G0"]
        L += ["Matriks konfusi (asli uji vs anomali nyata):",
              fmt_table([["ASLI (uji)", ga_, gn - ga_], ["ANOMALI NYATA", nn - tp, tp]],
                        ["kebenaran \\ keputusan", "AUTHENTIC", "NON_AUTHENTIC"]),
              f"  TAR (asli diterima) = {ga_ / gn:.3f}  | TRR (anomali ditolak) = {tp / nn:.3f}  | "
              f"akurasi seimbang = {bal:.3f}", ""]
    L += ["AUROC: asli(uji) vs set anomali; kolom = metode; lebih tinggi = lebih mudah dipisahkan",
          "G1 = asli dengan variasi capture (idealnya MENDEKATI 0,5); A1 = anomali digital (idealnya tinggi)",
          "NYATA = crop non-asli dari --anomaly-dir (baca CATATAN di bawah tentang isinya)"]
    rows = [[n if n != "NYATA" else "NYATA (terpisah)"] + [f"{auc[n][m]:.3f}" for m in METHODS] + [f"{flagged[n]:.3f}"]
            for n in names]
    rows.append(["G1 gabungan"] + [f"{auc_g1[m]:.3f}" for m in METHODS] + ["-"])
    rows.append(["A1 digital gabungan"] + [f"{auc_a1[m]:.3f}" for m in METHODS] + ["-"])
    rows.append(["FPR@95%TPR (A1 digital)"] + [f"{fpr95[m]:.3f}" for m in METHODS] + ["-"])
    L += [fmt_table(rows, ["set"] + [SHORT[m] for m in METHODS] + ["NON_AUTH@T"]), ""]
    if cmp_rows:
        L += ["Pembanding tingkat keputusan (tiap metode memakai T conformal-nya sendiri; set anomali = NYATA):",
              fmt_table(cmp_rows, ["metode", "AUROC", "TAR", "TRR", "akurasi seimbang"]), ""]
    L += [f"Paritas float vs int8: Spearman skor SSE = {spearman:.4f}; selisih AUROC A1 (float - int8) = {dauc:+.4f}",
          "  (target awal spesifikasi: selisih <= 0,01)", ""]
    L += ["Kolom 'NON_AUTH@T' = fraksi ROI yang divonis NON_AUTHENTIC oleh CAE int8 pada T.",
          "  Untuk G1 nilai kecil diinginkan (asli tidak boleh ditolak); untuk A1 dan NYATA nilai besar diinginkan.", ""]
    if warns:
        L += ["PERINGATAN OTOMATIS"] + [f"  {i + 1}. {w}" for i, w in enumerate(warns)] + [""]
    L += ["CATATAN: anomali A1/G1 adalah perturbasi DIGITAL. AUROC tinggi pada A1 tidak membuktikan",
          "  deteksi uang palsu cetakan. 'NYATA' hanya setara 'bukan ROI latent asli' kecuali sumbernya",
          "  terbukti uang palsu; sebutkan isi dan sumbernya di proposal."]
    report = "\n".join(L)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    log("\n" + report + "\n")

    # ---- JSON
    res = dict(args=vars(args), T=T, tau=T / N, alpha=args.alpha, auroc=auc, auroc_A1=auc_a1, auroc_G1=auc_g1,
               fpr_at_95tpr_A1=fpr95, false_anomaly_test=fa_clean, flagged=flagged,
               decisions={k: dict(benar=v[0], n=v[1]) for k, v in dec_stats.items()}, balanced_accuracy=bal,
               warnings=warns, spearman=spearman,
               acc_bits_observed=acc_bits_obs, best_val_mse=best_val,
               model={k: v for k, v in summ.items() if k != "rows"},
               shifts=[q["shift"] for q in qparams],
               exps=[dict(layer=l["name"], e_w=q["e_w"], e_in=q["e_in"], e_out=q["e_out"])
                     for l, q in zip(layers, qparams)])
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    np.savez(out / "model_float.npz", **{f"{l['name']}_{n}": a for l, (W, b) in zip(layers, params)
                                         for n, a in (("W", W), ("b", b))})
    if not args.no_plots:
        for f in make_plots(out, sc_clean, sc_by, T, auc_a1, args):
            log(f"  gambar: {f}")
    if args.export:
        dig, nb = export_all(out, layers, qparams, T, scorer, Xte_u8, perturbed, args.n_golden)
        log(f"== Ekspor ke {out / 'export'} ==")
        log(f"  payload {nb} byte, SHA-256 = {dig}")
    log(f"Selesai dalam {time.time() - t_start:.0f} detik. Keluaran di: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
