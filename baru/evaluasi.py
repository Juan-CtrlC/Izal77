#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluasi.py - Evaluasi keputusan 2 status (AUTHENTIC / NON_AUTHENTIC)
=====================================================================
Validasi silang ANTAR KELOMPOK (lembar/sesi), bukan antar crop. Setiap fold:
  latih CAE pada kelompok lain -> kalibrasi T (split conformal) pada kelompok lain ->
  nilai kelompok yang ditahan (out-of-fold). Semua 106 ROI asli ikut dinilai tepat sekali
  oleh model yang TIDAK pernah melihat kelompoknya. Anomali dinilai oleh setiap model fold.

Yang dilaporkan (per metode, termasuk baseline sepele):
  TAR  = fraksi uang asli yang diputuskan AUTHENTIC           (1 - FRR)
  TRR  = fraksi anomali nyata yang diputuskan NON_AUTHENTIC   (1 - FAR)
  akurasi seimbang = (TAR + TRR) / 2, AUROC rata-rata antar fold
  Interval 95% = bootstrap KLASTER: asli diacak per kelompok, anomali diacak per foto sumber
  (crop dari foto/sesi yang sama tidak dianggap independen).

Contoh:
  python evaluasi.py --data dataset_BI --anomaly-dir dataset_anomaly_roi ^
         --group-regex "^(roi_latent_?\\d+)" --out runs/eval_cv
  python evaluasi.py ... --norm --out runs/eval_cv_norm     # normalisasi per-ROI (spesifikasi bagian 4, [UJI])
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import cae_pipeline as C

METHODS = C.METHODS
SHORT = C.SHORT
LABEL = {"pca": "PCA", "template_sad": "Template-SAD", "cae_float_sse": "CAE float",
         "cae_int8_sse": "CAE int8 (SSE)", "cae_int8_sad": "CAE int8 (SAD)",
         "std_rendah": "Kontras piksel (sepele)", "mean_dev": "Kecerahan rata-rata (sepele)"}


# ----------------------------------------------------------------------------- fold
def make_folds(groups, k, seed):
    """Tetapkan tiap KELOMPOK ke satu fold; seimbang menurut jumlah gambar."""
    rng = np.random.default_rng(seed)
    ug, cnt = np.unique(groups, return_counts=True)
    order = np.lexsort((rng.random(len(ug)), -cnt))          # besar dulu, seri diacak
    size = np.zeros(k, dtype=int)
    assign = {}
    for gi in order:
        f = int(np.argmin(size))
        assign[ug[gi]] = f
        size[f] += cnt[gi]
    return np.array([assign[g] for g in groups])


def split_rest(groups_rest, seed, min_calib):
    """group_split train/val/calib pada sisa kelompok; coba beberapa seed agar n_calib >= min_calib."""
    best = None
    for a in range(30):
        try:
            idx, _ = C.group_split(groups_rest, seed * 100 + a,
                                   frac=(("train", .60), ("val", .10), ("calib", .30)))
        except SystemExit:
            continue
        if best is None or len(idx["calib"]) > len(best["calib"]):
            best = idx
        if len(idx["calib"]) >= min_calib and len(idx["train"]) >= 30:
            return idx
    if best is None:
        raise SystemExit("Split train/val/calib gagal: terlalu sedikit kelompok.")
    return best


def run_fold(f, X, groups, fold_of, Xa, layers, args, nrm):
    te = np.where(fold_of == f)[0]
    rest = np.where(fold_of != f)[0]
    min_calib = math.ceil(1 / args.alpha) - 1
    sp = split_rest(groups[rest], args.seed + f, min_calib)
    tr, va, ca = rest[sp["train"]], rest[sp["val"]], rest[sp["calib"]]

    tf = lambda u8: ((nrm(u8).astype(np.float32) - 128.0) / 128.0)[..., None]
    flat = lambda u8: (u8.astype(np.int64) - 128).reshape(len(u8), -1).astype(np.float32)
    Xtr, Xva = tf(X[tr]), tf(X[va])

    log0 = C.log
    C.log = lambda *a, **k: None                                  # senyapkan log per-epoch
    try:
        params, best_val = C.train_cae(layers, Xtr, Xva, args.epochs, args.batch, args.lr, args.seed + f,
                                       patience=args.patience)
    finally:
        C.log = log0
    qparams = C.quantize_model(params, layers, Xtr, 99.99)
    ptr = flat(nrm(X[tr]))
    rng = np.random.default_rng(args.seed + f)
    mu_p, V_p = C.pca_fit(ptr[rng.permutation(len(ptr))[:4000]], 16, args.seed + f)
    scorer = C.Scorer(layers, params, qparams, mu_p, V_p, ptr.mean(0))
    sc = lambda u8: scorer(nrm(u8))

    s_cal, s_te, s_an = sc(X[ca]), sc(X[te]), sc(Xa)
    T = {m: C.conformal_threshold(s_cal[m], args.alpha)[0] for m in METHODS}
    pert = {}
    for name, grp, fn in C.make_perturbations(X.shape[1], args.seed):
        s = sc(fn(X[te], C._rng_for(name, args.seed)))
        pert[name] = (grp, {m: s[m] for m in METHODS})
    return dict(te=te, T=T, n_train=len(tr), n_val=len(va), n_calib=len(ca), val_mse=best_val,
                calib_groups=sorted(set(groups[ca])), test_groups=sorted(set(groups[te])),
                s_te={m: s_te[m] for m in METHODS}, s_an={m: s_an[m] for m in METHODS}, pert=pert,
                T_cae=T["cae_int8_sse"])


# ----------------------------------------------------------------------------- metrik
class Pooled:
    """Skor dan keputusan out-of-fold, siap dihitung ulang pada sampel bootstrap."""

    def __init__(self, res, K, groups, photos, X, Xa):
        self.K = K
        order = np.concatenate([r["te"] for r in res])
        self.g_fold = np.concatenate([np.full(len(r["te"]), f) for f, r in enumerate(res)])
        self.g_group = groups[order]
        self.g_idx = order
        self.g_score = {m: np.concatenate([r["s_te"][m] for r in res]) for m in METHODS}
        self.g_rej = {m: np.concatenate([r["s_te"][m] > r["T"][m] for r in res]) for m in METHODS}
        self.a_score = {m: np.stack([r["s_an"][m] for r in res]) for m in METHODS}          # (K, nA)
        self.a_rej = {m: np.stack([r["s_an"][m] > r["T"][m] for r in res]) for m in METHODS}
        self.a_photo = photos
        self.ug, self.g_members = np.unique(self.g_group, return_inverse=False), None
        self.g_members = [np.where(self.g_group == g)[0] for g in self.ug]
        self.up = np.unique(photos)
        self.p_members = [np.where(photos == p)[0] for p in self.up]

    def metrics(self, gi, ai, methods=METHODS):
        out = {}
        for m in methods:
            frr = float(self.g_rej[m][gi].mean())
            trr = float(self.a_rej[m][:, ai].mean())
            aucs = []
            for f in range(self.K):
                gf = gi[self.g_fold[gi] == f]
                if len(gf) and len(ai):
                    aucs.append(C.auroc(self.g_score[m][gf], self.a_score[m][f, ai]))
            out[m] = dict(tar=1 - frr, trr=trr, bal=0.5 * ((1 - frr) + trr),
                          auroc=float(np.mean(aucs)) if aucs else float("nan"))
        return out

    def bootstrap(self, B, seed, ai_mask=None, methods=METHODS):
        rng = np.random.default_rng(seed)
        acc = {m: {k: [] for k in ("tar", "trr", "bal", "auroc")} for m in methods}
        for _ in range(B):
            sg = rng.integers(0, len(self.ug), len(self.ug))
            sp = rng.integers(0, len(self.up), len(self.up))
            gi = np.concatenate([self.g_members[i] for i in sg])
            ai = np.concatenate([self.p_members[i] for i in sp])
            if ai_mask is not None:
                ai = ai[ai_mask[ai]]
                if len(ai) == 0:
                    continue
            r = self.metrics(gi, ai, methods)
            for m in methods:
                for k in acc[m]:
                    acc[m][k].append(r[m][k])
        return {m: {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
                    for k, v in acc[m].items() if v} for m in methods}


def pct(x):
    return f"{100 * x:.1f}%".replace(".", ",")


def ci(c):
    return f"{100 * c[0]:.0f}–{100 * c[1]:.0f}%"


# ----------------------------------------------------------------------------- gambar
def make_figures(out, pooled, res, summary, X, Xa, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    ink, ink2, muted, grid, base, surf = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": base,
                         "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
                         "axes.facecolor": surf, "figure.facecolor": surf, "text.color": ink})
    files = []
    meths = ["cae_int8_sse", "template_sad", "pca", "std_rendah", "mean_dev"]
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=160)
    y = np.arange(len(meths))
    h = 0.26
    for j, (key, col, lab) in enumerate((("tar", c1, "TAR: asli diterima (AUTHENTIC)"),
                                         ("trr", c2, "TRR: anomali ditolak (NON_AUTHENTIC)"),
                                         ("bal", c3, "Akurasi seimbang"))):
        v = [summary["all"][m][key] for m in meths]
        lo = [summary["all"][m][key] - summary["ci"][m][key][0] for m in meths]
        hi = [summary["ci"][m][key][1] - summary["all"][m][key] for m in meths]
        ax.barh(y + (j - 1) * h, v, height=h, color=col, label=lab, xerr=[lo, hi],
                error_kw=dict(ecolor=ink2, lw=0.9, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in meths])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color=ink2, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Out-of-fold, validasi silang antar kelompok (batang galat = bootstrap klaster 95%)")
    ax.set_title("Keputusan 2 status: seberapa benar tiap metode", loc="left", fontsize=11, color=ink)
    ax.grid(axis="x", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    p = Path(out) / "fig_eval_keputusan.png"
    fig.savefig(p, facecolor=surf)
    plt.close(fig)
    files.append(p)

    # contoh keputusan salah (CAE int8)
    m = "cae_int8_sse"
    acc_a = np.where(pooled.a_rej[m].mean(0) < 0.5)[0]
    rej_g = np.where(pooled.g_rej[m])[0]
    rng = np.random.default_rng(0)
    acc_a = rng.permutation(acc_a)[:12]
    rej_g = rng.permutation(rej_g)[:12]
    if len(acc_a) or len(rej_g):
        fig, axs = plt.subplots(2, 12, figsize=(12, 2.9), dpi=140)
        for row, (ids, src, ttl) in enumerate(((acc_a, Xa, "Anomali yang LOLOS (diputuskan AUTHENTIC)"),
                                               (rej_g, X, "Asli yang DITOLAK (diputuskan NON_AUTHENTIC)"))):
            for c in range(12):
                a = axs[row, c]
                a.axis("off")
                if c < len(ids):
                    img = src[ids[c]] if row == 0 else X[pooled.g_idx[ids[c]]]
                    a.imshow(img, cmap="gray", vmin=0, vmax=255)
            axs[row, 0].set_title(ttl, loc="left", fontsize=9, color=ink)
        fig.tight_layout()
        p = Path(out) / "fig_contoh_salah.png"
        fig.savefig(p, facecolor=surf)
        plt.close(fig)
        files.append(p)
    return files


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluasi keputusan 2 status, validasi silang antar kelompok")
    ap.add_argument("--data", required=True)
    ap.add_argument("--anomaly-dir", required=True)
    ap.add_argument("--group-regex", default=None)
    ap.add_argument("--anomaly-group-regex", default=None)
    ap.add_argument("--out", default="runs/eval_cv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--enc-k", type=int, default=4, choices=(3, 4))
    ap.add_argument("--dec-k", type=int, default=4, choices=(2, 4))
    ap.add_argument("--channels", default="4,8,8")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", action="store_true",
                    help="normalisasi per-ROI: x_q = piksel - rata-rata ROI (spesifikasi bagian 4, opsi [UJI])")
    args = ap.parse_args(argv)
    t0 = time.time()
    C.selftest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.norm:
        def nrm(u8):
            m = np.rint(u8.reshape(len(u8), -1).mean(1)).astype(np.int64)[:, None, None]
            return np.clip(u8.astype(np.int64) - m + 128, 0, 255).astype(np.uint8)
    else:
        nrm = lambda u8: u8

    X, groups, has_groups = C.load_images(args.data, 64, args.group_regex)
    Xa, photos, _ = C.load_images(args.anomaly_dir, 64, args.anomaly_group_regex or C.ANOM_GROUP_RX)
    channels = tuple(int(c) for c in args.channels.split(","))
    layers = C.build_layers(channels, args.enc_k, args.dec_k)
    fold_of = make_folds(groups, args.folds, args.seed)
    print(f"{len(X)} ROI asli, {len(np.unique(groups))} kelompok, {len(Xa)} anomali dari {len(np.unique(photos))} foto; "
          f"{args.folds} fold; norm={args.norm}", flush=True)

    res = []
    for f in range(args.folds):
        r = run_fold(f, X, groups, fold_of, Xa, layers, args, nrm)
        res.append(r)
        print(f"  fold {f}: uji={len(r['te'])} ({','.join(r['test_groups'])}) | latih={r['n_train']} val={r['n_val']} "
              f"calib={r['n_calib']} | val MSE {r['val_mse']:.4f} | T={int(r['T_cae'])}", flush=True)

    pooled = Pooled(res, args.folds, groups, photos, X, Xa)
    allg, alla = np.arange(len(pooled.g_idx)), np.arange(len(Xa))
    point = pooled.metrics(allg, alla)
    cis = pooled.bootstrap(args.boot, args.seed + 1)

    # himpunan bagian: anomali "berpola" (kontras >= persentil-10 kontras uang asli) -> bukan sekadar kertas polos
    sd_g = X.reshape(len(X), -1).astype(np.float32).std(1)
    sd_a = Xa.reshape(len(Xa), -1).astype(np.float32).std(1)
    thr = float(np.percentile(sd_g, 10))
    mask = sd_a >= thr
    sub_point = pooled.metrics(allg, np.where(mask)[0]) if mask.any() else None
    sub_ci = pooled.bootstrap(args.boot, args.seed + 2, ai_mask=mask) if mask.sum() >= 5 else None

    # perturbasi digital (CAE int8 dan template), laju ditolak
    pert_rows = []
    for name in res[0]["pert"]:
        grp = res[0]["pert"][name][0]
        for m in ("cae_int8_sse", "template_sad"):
            rej = np.concatenate([r["pert"][name][1][m] > r["T"][m] for r in res])
            pert_rows.append((name, grp, m, float(rej.mean()), len(rej)))

    # diagnosis per kelompok (CAE int8)
    m = "cae_int8_sse"
    ratio = np.concatenate([r["s_te"][m] / max(1.0, r["T"][m]) for r in res])
    diag = []
    for g in pooled.ug:
        ii = np.where(pooled.g_group == g)[0]
        px = X[pooled.g_idx[ii]].reshape(len(ii), -1).astype(np.float32)
        diag.append((g, len(ii), float(pooled.g_rej[m][ii].mean()), float(np.median(ratio[ii])),
                     float(px.mean()), float(px.std(1).mean())))

    summary = dict(all=point, ci=cis, sub=sub_point, sub_ci=sub_ci)
    figs = make_figures(out, pooled, res, summary, X, Xa, args)

    # ------------------------------------------------------------------ laporan Markdown
    n_calib = [r["n_calib"] for r in res]
    L = ["# Evaluasi keputusan 2 status (AUTHENTIC / NON_AUTHENTIC)", "",
         f"Validasi silang antar kelompok, {args.folds} fold. Normalisasi per-ROI: **{'ya' if args.norm else 'tidak'}**. "
         f"Dibuat otomatis oleh `evaluasi.py`.", "",
         "## 1. Data dan protokol", "",
         f"- ROI asli: **{len(X)}** gambar dalam **{len(np.unique(groups))}** kelompok "
         f"({'dari --group-regex' if has_groups else 'TIDAK ADA info kelompok'}).",
         f"- Anomali: **{len(Xa)}** crop dari **{len(np.unique(photos))}** foto sumber (diacak per foto saat menghitung interval).",
         f"- Per fold: kelompok uji ditahan penuh; sisanya dibagi per kelompok menjadi latih/val/kalibrasi "
         f"(n_calib per fold: {', '.join(map(str, n_calib))}). Syarat jaminan konformal α={args.alpha}: n_calib ≥ "
         f"{math.ceil(1 / args.alpha) - 1}.",
         "- T dikalibrasi per fold (conformal, model int8 bit-exact). Keputusan: SSE ≤ T → AUTHENTIC, selain itu NON_AUTHENTIC.",
         "- Setiap ROI asli dinilai **sekali** oleh model yang tidak pernah melihat kelompoknya. Setiap anomali dinilai oleh "
         "kelima model; angka TRR adalah rata-rata kelima fold.", "",
         "## 2. Hasil utama (anomali nyata = seluruh crop)", "",
         "| Metode | TAR (asli diterima) | TRR (anomali ditolak) | Akurasi seimbang | AUROC (rata-rata fold) |",
         "|---|---|---|---|---|"]
    for mm in ("cae_int8_sse", "cae_int8_sad", "template_sad", "pca", "std_rendah", "mean_dev"):
        p, c = point[mm], cis[mm]
        L.append(f"| {LABEL[mm]} | {pct(p['tar'])} ({ci(c['tar'])}) | {pct(p['trr'])} ({ci(c['trr'])}) | "
                 f"{pct(p['bal'])} ({ci(c['bal'])}) | {p['auroc']:.3f} ({c['auroc'][0]:.2f}–{c['auroc'][1]:.2f}) |")
    L += ["", "Kurung = interval 95% bootstrap klaster (asli per kelompok, anomali per foto). "
          "Dua baris terakhir adalah **baseline sepele** (hanya kontras atau kecerahan rata-rata ROI).", ""]
    if sub_point:
        L += [f"## 3. Anomali yang berpola saja (kontras ≥ {thr:.1f}, persentil-10 kontras uang asli)", "",
              f"{int(mask.sum())} dari {len(Xa)} crop anomali ({len(np.unique(photos[mask]))} foto). "
              "Ini menyingkirkan kertas polos yang mudah dibedakan lewat kontras saja.", "",
              "| Metode | TRR | AUROC |", "|---|---|---|"]
        for mm in ("cae_int8_sse", "template_sad", "pca", "std_rendah", "mean_dev"):
            c = sub_ci[mm] if sub_ci else None
            trr = f"{pct(sub_point[mm]['trr'])}" + (f" ({ci(c['trr'])})" if c else "")
            au = f"{sub_point[mm]['auroc']:.3f}" + (f" ({c['auroc'][0]:.2f}–{c['auroc'][1]:.2f})" if c else "")
            L.append(f"| {LABEL[mm]} | {trr} | {au} |")
        L.append("")
        if int(mask.sum()) < 30:
            L += [f"> Hanya {int(mask.sum())} crop berpola: angka himpunan bagian ini kasar.", ""]
    L += ["## 4. Per kelompok (CAE int8): di mana uang asli ditolak", "",
          "| Kelompok | n | ditolak | median SSE/T | rata-rata piksel | rata-rata kontras |", "|---|---|---|---|---|---|"]
    for g, n, rj, md, mu, sd in sorted(diag, key=lambda d: -d[2]):
        L.append(f"| {g} | {n} | {pct(rj)} | {md:.2f} | {mu:.0f} | {sd:.1f} |")
    L += ["", "SSE/T > 1 berarti melewati ambang. Kelompok dengan penolakan tinggi biasanya berbeda kecerahan atau kontras "
          "dari kelompok lain; itu menunjukkan **pergeseran antar sesi capture**, bukan kerusakan uang.", ""]
    L += ["## 5. Variasi capture dan anomali digital (laju ROI DITOLAK, out-of-fold)", "",
          "| Perturbasi | Jenis | CAE int8 | Template-SAD |", "|---|---|---|---|"]
    names = list(res[0]["pert"])
    for name in names:
        grp = res[0]["pert"][name][0]
        vals = {mm: [r for r in pert_rows if r[0] == name and r[2] == mm][0][3] for mm in ("cae_int8_sse", "template_sad")}
        want = "diharapkan rendah" if grp == "G1" else "diharapkan tinggi"
        L.append(f"| {name} | {grp} ({want}) | {pct(vals['cae_int8_sse'])} | {pct(vals['template_sad'])} |")
    L += [""]

    # ------------------------------------------------------------------ peringatan & kesimpulan otomatis
    warns = []
    if min(n_calib) < math.ceil(1 / args.alpha) - 1:
        warns.append(f"n_calib minimum {min(n_calib)} < {math.ceil(1 / args.alpha) - 1} pada sebagian fold: "
                     f"T pada fold itu tanpa jaminan statistik.")
    if len(np.unique(groups)) < 15:
        warns.append(f"Hanya {len(np.unique(groups))} kelompok asli; interval bootstrap klaster lebar dan "
                     f"fold bergantung pada beberapa kelompok besar.")
    cae, tem = point["cae_int8_sse"], point["template_sad"]
    if tem["bal"] > cae["bal"] + 0.02:
        warns.append(f"Template rata-rata (akurasi seimbang {pct(tem['bal'])}) mengungguli CAE int8 ({pct(cae['bal'])}). "
                     f"Laporkan sebagai pembanding, jangan disembunyikan.")
    triv = max(point["std_rendah"]["bal"], point["mean_dev"]["bal"])
    if triv >= cae["bal"] - 0.02:
        warns.append(f"Baseline sepele (kontras/kecerahan) mencapai akurasi seimbang {pct(triv)}, setara atau lebih baik daripada "
                     f"CAE ({pct(cae['bal'])}): himpunan anomali ini belum membuktikan model memakai pola 'BI'.")
    if cae["tar"] < 0.8:
        warns.append(f"TAR CAE hanya {pct(cae['tar'])} antar kelompok: terlalu banyak uang asli ditolak. "
                     f"Penyebab utama kemungkinan pergeseran capture antar sesi (lihat bagian 4).")
    L += ["## 6. Peringatan otomatis", ""] + [f"{i + 1}. {w}" for i, w in enumerate(warns)] + [""]
    L += ["## 7. Cara membaca dan menulis hasil ini di proposal", "",
          "- NON_AUTHENTIC berarti **tidak lolos penyaring**, bukan 'pasti palsu'. AUTHENTIC berarti **lolos penyaring**, bukan 'pasti asli'.",
          "- Anomali di `--anomaly-dir` adalah crop dari foto non-latent atau area lain, **bukan** terbukti uang palsu cetakan, "
          "sehingga TRR di atas berarti 'membedakan ROI latent asli dari area lain', bukan 'mendeteksi uang palsu'.",
          "- Jangan menulis jaminan false-reject ≤ α. Tulis TAR out-of-fold beserta intervalnya.",
          "- Paritas int8 vs float adalah klaim hardware yang paling kuat; ukur ulang dengan model final.", ""]
    L += ["Gambar: " + ", ".join(f"`{f.name}`" for f in figs), ""]
    (out / "evaluasi_report.md").write_text("\n".join(L), encoding="utf-8")

    js = dict(args=vars(args), point=point, ci=cis, subset=dict(threshold_std=thr, n=int(mask.sum()),
              point=sub_point, ci=sub_ci), per_group=[dict(group=g, n=n, reject=rj, median_sse_over_T=md, mean_px=mu, mean_std=sd)
                                                      for g, n, rj, md, mu, sd in diag],
              perturbations=[dict(name=a, kind=b, method=c, reject_rate=d, n=e) for a, b, c, d, e in pert_rows],
              folds=[dict(test_groups=r["test_groups"], n_test=len(r["te"]), n_train=r["n_train"], n_calib=r["n_calib"],
                          T=r["T_cae"], val_mse=r["val_mse"]) for r in res], warnings=warns)
    (out / "evaluasi_results.json").write_text(json.dumps(js, indent=1, default=float), encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"Selesai dalam {time.time() - t0:.0f} detik -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
