# Evaluasi keputusan 2 status (AUTHENTIC / NON_AUTHENTIC)

Validasi silang antar kelompok, 5 fold. Normalisasi per-ROI: **ya**. Dibuat otomatis oleh `evaluasi.py`.

## 1. Data dan protokol

- ROI asli: **74** gambar dalam **10** kelompok (dari --group-regex).
- Anomali: **185** crop dari **37** foto sumber (diacak per foto saat menghitung interval).
- Per fold: kelompok uji ditahan penuh; sisanya dibagi per kelompok menjadi latih/val/kalibrasi (n_calib per fold: 22, 20, 22, 22, 22). Syarat jaminan konformal α=0.05: n_calib ≥ 19.
- T dikalibrasi per fold (conformal, model int8 bit-exact). Keputusan: SSE ≤ T → AUTHENTIC, selain itu NON_AUTHENTIC.
- Setiap ROI asli dinilai **sekali** oleh model yang tidak pernah melihat kelompoknya. Setiap anomali dinilai oleh kelima model; angka TRR adalah rata-rata kelima fold.

## 2. Hasil utama (anomali nyata = seluruh crop)

| Metode | TAR (asli diterima) | TRR (anomali ditolak) | Akurasi seimbang | AUROC (rata-rata fold) |
|---|---|---|---|---|
| CAE int8 (SSE) | 97,3% (91–100%) | 9,2% (3–17%) | 53,2% (49–58%) | 0.139 (0.07–0.23) |
| CAE int8 (SAD) | 97,3% (91–100%) | 8,1% (3–16%) | 52,7% (49–57%) | 0.127 (0.06–0.22) |
| Template-SAD | 95,9% (90–100%) | 8,0% (3–15%) | 52,0% (48–56%) | 0.140 (0.07–0.23) |
| PCA | 95,9% (90–100%) | 9,1% (3–17%) | 52,5% (48–57%) | 0.170 (0.09–0.26) |
| Kontras piksel (sepele) | 95,9% (89–100%) | 81,1% (71–89%) | 88,5% (82–93%) | 0.855 (0.77–0.92) |
| Kecerahan rata-rata (sepele) | 93,2% (85–99%) | 6,7% (4–9%) | 50,0% (46–53%) | 0.536 (0.46–0.60) |

Kurung = interval 95% bootstrap klaster (asli per kelompok, anomali per foto). Dua baris terakhir adalah **baseline sepele** (hanya kontras atau kecerahan rata-rata ROI).

## 3. Anomali yang berpola saja (kontras ≥ 14.5, persentil-10 kontras uang asli)

34 dari 185 crop anomali (18 foto). Ini menyingkirkan kertas polos yang mudah dibedakan lewat kontras saja.

| Metode | TRR | AUROC |
|---|---|---|
| CAE int8 (SSE) | 50,0% (25–66%) | 0.757 (0.59–0.88) |
| Template-SAD | 43,5% (25–57%) | 0.755 (0.61–0.87) |
| PCA | 49,4% (26–65%) | 0.804 (0.68–0.91) |
| Kontras piksel (sepele) | 1,2% (0–4%) | 0.217 (0.11–0.36) |
| Kecerahan rata-rata (sepele) | 7,1% (2–13%) | 0.548 (0.43–0.63) |

## 4. Per kelompok (CAE int8): di mana uang asli ditolak

| Kelompok | n | ditolak | median SSE/T | rata-rata piksel | rata-rata kontras |
|---|---|---|---|---|---|
| roi_latent_00100 | 10 | 20,0% | 0.67 | 92 | 25.7 |
| roi_latent1 | 12 | 0,0% | 0.52 | 81 | 18.5 |
| roi_latent2 | 14 | 0,0% | 0.22 | 93 | 17.6 |
| roi_latent4 | 4 | 0,0% | 0.27 | 89 | 13.6 |
| roi_latent50 | 10 | 0,0% | 0.69 | 99 | 31.6 |
| roi_latent_00010 | 2 | 0,0% | 0.32 | 42 | 18.9 |
| roi_latent_003 | 6 | 0,0% | 0.59 | 89 | 28.1 |
| roi_latent_04 | 2 | 0,0% | 0.32 | 91 | 14.6 |
| roi_latent_12 | 6 | 0,0% | 0.68 | 95 | 28.6 |
| roi_latent_15 | 8 | 0,0% | 0.46 | 101 | 23.1 |

SSE/T > 1 berarti melewati ambang. Kelompok dengan penolakan tinggi biasanya berbeda kecerahan atau kontras dari kelompok lain; itu menunjukkan **pergeseran antar sesi capture**, bukan kerusakan uang.

## 5. Variasi capture dan anomali digital (laju ROI DITOLAK, out-of-fold)

| Perturbasi | Jenis | CAE int8 | Template-SAD |
|---|---|---|---|
| G1_blur_s1 | G1 (diharapkan rendah) | 0,0% | 0,0% |
| G1_blur_s2 | G1 (diharapkan rendah) | 0,0% | 0,0% |
| G1_terang+25 | G1 (diharapkan rendah) | 27,0% | 28,4% |
| G1_gelap-25 | G1 (diharapkan rendah) | 0,0% | 2,7% |
| G1_geser_2px | G1 (diharapkan rendah) | 2,7% | 4,1% |
| G1_rotasi_3d | G1 (diharapkan rendah) | 0,0% | 0,0% |
| A1_pudar_15 | A1 (diharapkan tinggi) | 0,0% | 0,0% |
| A1_hapus_blok | A1 (diharapkan tinggi) | 2,7% | 2,7% |
| A1_noise_25 | A1 (diharapkan tinggi) | 77,0% | 36,5% |
| A1_cermin | A1 (diharapkan tinggi) | 12,2% | 4,1% |
| A1_rotasi_90 | A1 (diharapkan tinggi) | 9,5% | 4,1% |
| A1_acak_tile | A1 (diharapkan tinggi) | 4,1% | 4,1% |

## 6. Peringatan otomatis

1. Hanya 10 kelompok asli; interval bootstrap klaster lebar dan fold bergantung pada beberapa kelompok besar.
2. Baseline sepele (kontras/kecerahan) mencapai akurasi seimbang 88,5%, setara atau lebih baik daripada CAE (53,2%): himpunan anomali ini belum membuktikan model memakai pola 'BI'.

## 7. Cara membaca dan menulis hasil ini di proposal

- NON_AUTHENTIC berarti **tidak lolos penyaring**, bukan 'pasti palsu'. AUTHENTIC berarti **lolos penyaring**, bukan 'pasti asli'.
- Anomali di `--anomaly-dir` adalah crop dari foto non-latent atau area lain, **bukan** terbukti uang palsu cetakan, sehingga TRR di atas berarti 'membedakan ROI latent asli dari area lain', bukan 'mendeteksi uang palsu'.
- Jangan menulis jaminan false-reject ≤ α. Tulis TAR out-of-fold beserta intervalnya.
- Paritas int8 vs float adalah klaim hardware yang paling kuat; ukur ulang dengan model final.

Gambar: `fig_eval_keputusan.png`, `fig_contoh_salah.png`
