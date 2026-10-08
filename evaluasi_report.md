# Evaluasi keputusan 2 status (AUTHENTIC / NON_AUTHENTIC)

Validasi silang antar kelompok, 5 fold. Normalisasi per-ROI: **tidak**. Dibuat otomatis oleh `evaluasi.py`.

## 1. Data dan protokol

- ROI asli: **106** gambar dalam **11** kelompok (dari --group-regex).
- Anomali: **185** crop dari **37** foto sumber (diacak per foto saat menghitung interval).
- Per fold: kelompok uji ditahan penuh; sisanya dibagi per kelompok menjadi latih/val/kalibrasi (n_calib per fold: 20, 40, 36, 32, 32). Syarat jaminan konformal α=0.05: n_calib ≥ 19.
- T dikalibrasi per fold (conformal, model int8 bit-exact). Keputusan: SSE ≤ T → AUTHENTIC, selain itu NON_AUTHENTIC.
- Setiap ROI asli dinilai **sekali** oleh model yang tidak pernah melihat kelompoknya. Setiap anomali dinilai oleh kelima model; angka TRR adalah rata-rata kelima fold.

## 2. Hasil utama (anomali nyata = seluruh crop)

| Metode | TAR (asli diterima) | TRR (anomali ditolak) | Akurasi seimbang | AUROC (rata-rata fold) |
|---|---|---|---|---|
| CAE int8 (SSE) | 74,5% (43–100%) | 67,6% (59–75%) | 71,0% (54–86%) | 0.858 (0.79–0.97) |
| CAE int8 (SAD) | 73,6% (41–100%) | 75,2% (66–83%) | 74,4% (57–90%) | 0.880 (0.81–0.97) |
| Template-SAD | 78,3% (60–100%) | 95,9% (90–99%) | 87,1% (78–98%) | 0.978 (0.93–1.00) |
| PCA | 81,1% (58–100%) | 7,1% (2–14%) | 44,1% (31–56%) | 0.406 (0.26–0.66) |
| Kontras piksel (sepele) | 72,6% (45–90%) | 83,8% (75–91%) | 78,2% (64–88%) | 0.869 (0.78–0.93) |
| Kecerahan rata-rata (sepele) | 82,1% (71–97%) | 95,5% (91–99%) | 88,8% (82–96%) | 0.976 (0.95–0.99) |

Kurung = interval 95% bootstrap klaster (asli per kelompok, anomali per foto). Dua baris terakhir adalah **baseline sepele** (hanya kontras atau kecerahan rata-rata ROI).

## 3. Anomali yang berpola saja (kontras ≥ 14.9, persentil-10 kontras uang asli)

34 dari 185 crop anomali (18 foto). Ini menyingkirkan kertas polos yang mudah dibedakan lewat kontras saja.

| Metode | TRR | AUROC |
|---|---|---|
| CAE int8 (SSE) | 46,5% (20–66%) | 0.798 (0.67–0.96) |
| Template-SAD | 84,7% (67–97%) | 0.935 (0.85–0.99) |
| PCA | 38,8% (13–57%) | 0.790 (0.67–0.94) |
| Kontras piksel (sepele) | 16,5% (10–26%) | 0.288 (0.12–0.42) |
| Kecerahan rata-rata (sepele) | 80,6% (66–93%) | 0.895 (0.79–0.97) |

## 4. Per kelompok (CAE int8): di mana uang asli ditolak

| Kelompok | n | ditolak | median SSE/T | rata-rata piksel | rata-rata kontras |
|---|---|---|---|---|---|
| roi_latent3 | 32 | 84,4% | 1.16 | 82 | 35.2 |
| roi_latent1 | 12 | 0,0% | 0.18 | 81 | 18.5 |
| roi_latent2 | 14 | 0,0% | 0.15 | 93 | 17.6 |
| roi_latent4 | 4 | 0,0% | 0.12 | 89 | 13.6 |
| roi_latent50 | 10 | 0,0% | 0.36 | 99 | 31.6 |
| roi_latent_00010 | 2 | 0,0% | 0.23 | 42 | 18.9 |
| roi_latent_00100 | 10 | 0,0% | 0.23 | 92 | 25.7 |
| roi_latent_003 | 6 | 0,0% | 0.31 | 89 | 28.1 |
| roi_latent_04 | 2 | 0,0% | 0.13 | 91 | 14.6 |
| roi_latent_12 | 6 | 0,0% | 0.42 | 95 | 28.6 |
| roi_latent_15 | 8 | 0,0% | 0.30 | 101 | 23.1 |

SSE/T > 1 berarti melewati ambang. Kelompok dengan penolakan tinggi biasanya berbeda kecerahan atau kontras dari kelompok lain; itu menunjukkan **pergeseran antar sesi capture**, bukan kerusakan uang.

## 5. Variasi capture dan anomali digital (laju ROI DITOLAK, out-of-fold)

| Perturbasi | Jenis | CAE int8 | Template-SAD |
|---|---|---|---|
| G1_blur_s1 | G1 (diharapkan rendah) | 0,0% | 5,7% |
| G1_blur_s2 | G1 (diharapkan rendah) | 0,0% | 5,7% |
| G1_terang+25 | G1 (diharapkan rendah) | 34,0% | 59,4% |
| G1_gelap-25 | G1 (diharapkan rendah) | 0,0% | 32,1% |
| G1_geser_2px | G1 (diharapkan rendah) | 25,5% | 21,7% |
| G1_rotasi_3d | G1 (diharapkan rendah) | 0,0% | 5,7% |
| A1_pudar_15 | A1 (diharapkan tinggi) | 0,0% | 5,7% |
| A1_hapus_blok | A1 (diharapkan tinggi) | 21,7% | 19,8% |
| A1_noise_25 | A1 (diharapkan tinggi) | 31,1% | 41,5% |
| A1_cermin | A1 (diharapkan tinggi) | 16,0% | 21,7% |
| A1_rotasi_90 | A1 (diharapkan tinggi) | 24,5% | 21,7% |
| A1_acak_tile | A1 (diharapkan tinggi) | 26,4% | 21,7% |

## 6. Peringatan otomatis

1. Hanya 11 kelompok asli; interval bootstrap klaster lebar dan fold bergantung pada beberapa kelompok besar.
2. Template rata-rata (akurasi seimbang 87,1%) mengungguli CAE int8 (71,0%). Laporkan sebagai pembanding, jangan disembunyikan.
3. Baseline sepele (kontras/kecerahan) mencapai akurasi seimbang 88,8%, setara atau lebih baik daripada CAE (71,0%): himpunan anomali ini belum membuktikan model memakai pola 'BI'.
4. TAR CAE hanya 74,5% antar kelompok: terlalu banyak uang asli ditolak. Penyebab utama kemungkinan pergeseran capture antar sesi (lihat bagian 4).

## 7. Cara membaca dan menulis hasil ini di proposal

- NON_AUTHENTIC berarti **tidak lolos penyaring**, bukan 'pasti palsu'. AUTHENTIC berarti **lolos penyaring**, bukan 'pasti asli'.
- Anomali di `--anomaly-dir` adalah crop dari foto non-latent atau area lain, **bukan** terbukti uang palsu cetakan, sehingga TRR di atas berarti 'membedakan ROI latent asli dari area lain', bukan 'mendeteksi uang palsu'.
- Jangan menulis jaminan false-reject ≤ α. Tulis TAR out-of-fold beserta intervalnya.
- Paritas int8 vs float adalah klaim hardware yang paling kuat; ukur ulang dengan model final.

Gambar: `fig_eval_keputusan.png`, `fig_contoh_salah.png`
