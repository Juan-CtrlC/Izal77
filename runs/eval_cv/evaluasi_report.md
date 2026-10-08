# Evaluasi keputusan 2 status (AUTHENTIC / NON_AUTHENTIC)

Validasi silang antar kelompok, 5 fold. Normalisasi per-ROI: **tidak**. Dibuat otomatis oleh `evaluasi.py`.

## 1. Data dan protokol

- ROI asli: **74** gambar dalam **10** kelompok (dari --group-regex).
- Anomali: **185** crop dari **37** foto sumber (diacak per foto saat menghitung interval).
- Per fold: kelompok uji ditahan penuh; sisanya dibagi per kelompok menjadi latih/val/kalibrasi (n_calib per fold: 22, 20, 22, 22, 22). Syarat jaminan konformal α=0.05: n_calib ≥ 19.
- T dikalibrasi per fold (conformal, model int8 bit-exact). Keputusan: SSE ≤ T → AUTHENTIC, selain itu NON_AUTHENTIC.
- Setiap ROI asli dinilai **sekali** oleh model yang tidak pernah melihat kelompoknya. Setiap anomali dinilai oleh kelima model; angka TRR adalah rata-rata kelima fold.

## 2. Hasil utama (anomali nyata = seluruh crop)

| Metode | TAR (asli diterima) | TRR (anomali ditolak) | Akurasi seimbang | AUROC (rata-rata fold) |
|---|---|---|---|---|
| CAE int8 (SSE) | 98,6% (96–100%) | 76,5% (71–81%) | 87,6% (84–90%) | 0.835 (0.76–0.99) |
| CAE int8 (SAD) | 97,3% (94–100%) | 78,7% (73–83%) | 88,0% (85–91%) | 0.849 (0.78–0.99) |
| Template-SAD | 93,2% (80–100%) | 94,9% (89–99%) | 94,1% (87–99%) | 0.980 (0.94–1.00) |
| PCA | 95,9% (90–100%) | 26,5% (21–33%) | 61,2% (57–65%) | 0.537 (0.35–0.76) |
| Kontras piksel (sepele) | 95,9% (89–100%) | 80,9% (71–89%) | 88,4% (82–93%) | 0.854 (0.76–0.92) |
| Kecerahan rata-rata (sepele) | 91,9% (79–100%) | 92,5% (86–97%) | 92,2% (85–97%) | 0.974 (0.94–0.99) |

Kurung = interval 95% bootstrap klaster (asli per kelompok, anomali per foto). Dua baris terakhir adalah **baseline sepele** (hanya kontras atau kecerahan rata-rata ROI).

## 3. Anomali yang berpola saja (kontras ≥ 14.5, persentil-10 kontras uang asli)

34 dari 185 crop anomali (18 foto). Ini menyingkirkan kertas polos yang mudah dibedakan lewat kontras saja.

| Metode | TRR | AUROC |
|---|---|---|
| CAE int8 (SSE) | 71,8% (52–86%) | 0.864 (0.75–0.98) |
| Template-SAD | 81,2% (63–94%) | 0.946 (0.84–0.99) |
| PCA | 60,6% (39–75%) | 0.875 (0.77–0.96) |
| Kontras piksel (sepele) | 0,0% (0–0%) | 0.209 (0.10–0.35) |
| Kecerahan rata-rata (sepele) | 67,6% (54–78%) | 0.885 (0.77–0.96) |

## 4. Per kelompok (CAE int8): di mana uang asli ditolak

| Kelompok | n | ditolak | median SSE/T | rata-rata piksel | rata-rata kontras |
|---|---|---|---|---|---|
| roi_latent50 | 10 | 10,0% | 0.74 | 99 | 31.6 |
| roi_latent1 | 12 | 0,0% | 0.30 | 81 | 18.5 |
| roi_latent2 | 14 | 0,0% | 0.23 | 93 | 17.6 |
| roi_latent4 | 4 | 0,0% | 0.10 | 89 | 13.6 |
| roi_latent_00010 | 2 | 0,0% | 0.50 | 42 | 18.9 |
| roi_latent_00100 | 10 | 0,0% | 0.24 | 92 | 25.7 |
| roi_latent_003 | 6 | 0,0% | 0.58 | 89 | 28.1 |
| roi_latent_04 | 2 | 0,0% | 0.21 | 91 | 14.6 |
| roi_latent_12 | 6 | 0,0% | 0.68 | 95 | 28.6 |
| roi_latent_15 | 8 | 0,0% | 0.55 | 101 | 23.1 |

SSE/T > 1 berarti melewati ambang. Kelompok dengan penolakan tinggi biasanya berbeda kecerahan atau kontras dari kelompok lain; itu menunjukkan **pergeseran antar sesi capture**, bukan kerusakan uang.

## 5. Variasi capture dan anomali digital (laju ROI DITOLAK, out-of-fold)

| Perturbasi | Jenis | CAE int8 | Template-SAD |
|---|---|---|---|
| G1_blur_s1 | G1 (diharapkan rendah) | 0,0% | 5,4% |
| G1_blur_s2 | G1 (diharapkan rendah) | 0,0% | 5,4% |
| G1_terang+25 | G1 (diharapkan rendah) | 29,7% | 40,5% |
| G1_gelap-25 | G1 (diharapkan rendah) | 0,0% | 9,5% |
| G1_geser_2px | G1 (diharapkan rendah) | 1,4% | 6,8% |
| G1_rotasi_3d | G1 (diharapkan rendah) | 0,0% | 5,4% |
| A1_pudar_15 | A1 (diharapkan tinggi) | 0,0% | 5,4% |
| A1_hapus_blok | A1 (diharapkan tinggi) | 0,0% | 5,4% |
| A1_noise_25 | A1 (diharapkan tinggi) | 39,2% | 29,7% |
| A1_cermin | A1 (diharapkan tinggi) | 6,8% | 6,8% |
| A1_rotasi_90 | A1 (diharapkan tinggi) | 8,1% | 8,1% |
| A1_acak_tile | A1 (diharapkan tinggi) | 4,1% | 8,1% |

## 6. Peringatan otomatis

1. Hanya 10 kelompok asli; interval bootstrap klaster lebar dan fold bergantung pada beberapa kelompok besar.
2. Template rata-rata (akurasi seimbang 94,1%) mengungguli CAE int8 (87,6%). Laporkan sebagai pembanding, jangan disembunyikan.
3. Baseline sepele (kontras/kecerahan) mencapai akurasi seimbang 92,2%, setara atau lebih baik daripada CAE (87,6%): himpunan anomali ini belum membuktikan model memakai pola 'BI'.

## 7. Cara membaca dan menulis hasil ini di proposal

- NON_AUTHENTIC berarti **tidak lolos penyaring**, bukan 'pasti palsu'. AUTHENTIC berarti **lolos penyaring**, bukan 'pasti asli'.
- Anomali di `--anomaly-dir` adalah crop dari foto non-latent atau area lain, **bukan** terbukti uang palsu cetakan, sehingga TRR di atas berarti 'membedakan ROI latent asli dari area lain', bukan 'mendeteksi uang palsu'.
- Jangan menulis jaminan false-reject ≤ α. Tulis TAR out-of-fold beserta intervalnya.
- Paritas int8 vs float adalah klaim hardware yang paling kuat; ukur ulang dengan model final.

Gambar: `fig_eval_keputusan.png`, `fig_contoh_salah.png`
