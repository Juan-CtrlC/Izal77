# Spesifikasi Sistem v2.2: Akselerator Edge-AI Latent Image "BI" Rupiah

PERURI Chip Hackathon 2026, kategori AI / Edge Accelerator
Versi: 2.2 (8 Oktober 2026). Menggantikan v2.1: keluaran 4 status diganti **2 status** (AUTHENTIC / NON_AUTHENTIC). Status: draf untuk proposal (batas pengumpulan 8 Oktober 2026).

**Cara membaca label:**
- **[PUTUSAN]** keputusan desain yang diambil.
- **[HITUNG]** angka hasil aritmetika, dihasilkan oleh fungsi `model_summary` di `cae_pipeline.py` (satu sumber dengan laporan skrip).
- **[ESTIMASI]** perkiraan yang baru bisa dipastikan oleh laporan sintesis Quartus.
- **[UJI]** belum diputuskan, menunggu eksperimen pada dataset tim.

---

## 1. Ringkasan parameter kunci

| Aspek | Spesifikasi v2.2 |
|---|---|
| Target hardware | Terasic DE10-Nano, Cyclone V SE 5CSEBA6U23I7 (110K LE, 112 DSP, HPS dual-core Cortex-A9 800 MHz) |
| Input | ROI latent image "BI", **64×64**, grayscale 8-bit [PUTUSAN]. Varian 32×32 (pool 2×2) dan RGB dibandingkan di DSE |
| Model | CAE 6 lapis, **kernel 4×4**: 3 konvolusi stride 2 + 3 transposed-conv stride 2. 3.233 parameter, 524.288 MAC per inferensi [HITUNG] |
| Latent | 8×8×8 = 512 nilai (kompresi 8:1 dari 4.096 piksel) [HITUNG] |
| Kuantisasi | bobot dan aktivasi int8, akumulator 24-bit, requantization hanya dengan shift |
| Compute | 1 array MAC 8 lane, dipakai ulang bergantian untuk setiap lapis |
| Latensi compute | 147.456 siklus = 1.474,6 µs pada 100 MHz (klok asumsi) [HITUNG]. Pemetaan lanjutan: 81.920 siklus = 819,2 µs |
| Error engine | SSE streaming, akumulator 28-bit, dibandingkan ke T = N·τ (tanpa pembagi) |
| Keputusan | **2 status**: AUTHENTIC dan NON_AUTHENTIC. AUTHENTIC hanya bila integritas lulus, input tepat 4.096 piksel, selesai sebelum watchdog, dan SSE ≤ T, semuanya sekaligus. Selain itu NON_AUTHENTIC (fail-secure). Register hasil 1 bit, nilai reset 0 |
| Security | bobot dan τ sebagai ROM tanpa jalur tulis, SHA-256 integritas, validasi input, fail-secure |
| Toolchain | Python (`cae_pipeline.py`: latih, kuantisasi, golden model), SystemVerilog manual, Quartus Prime Lite |

---

## 2. Cakupan dan asumsi

**Dalam cakupan:** inferensi akselerator untuk ROI yang sudah diambil, quantization, RTL, simulasi, sintesis dan analisis timing/resource pada Cyclone V.

**Di luar cakupan:** kamera, deteksi objek uang, perspective correction, deteksi ROI otomatis, fitur keamanan Rupiah lain, cloud/mobile, fabrikasi ASIC.

**Asumsi yang harus ditulis eksplisit di proposal:**
1. ROI diambil pada protokol capture tetap (sudut, jarak, pencahayaan terdefinisi). ROI yang menyimpang dapat menghasilkan NON_AUTHENTIC pada uang asli (false reject), dan itu diukur, bukan disembunyikan.
2. Keluaran sistem adalah **penyaring** yang mengarahkan pemeriksaan lanjutan, bukan penentu tunggal keaslian uang.
3. Dataset tim: ROI latent image "BI" 64×64. **Isi:** jumlah sampel, pecahan, tahun emisi, kondisi capture, dan apakah grayscale atau RGB. Spesifikasi ini mengasumsikan grayscale.

---

## 3. Perubahan dari v2

| v2 | v2.1 | Alasan |
|---|---|---|
| Baseline 32×32 | **64×64** | resolusi asli dataset tim. 32×32 menjadi varian di DSE |
| Konvolusi 3×3, ConvT 2×2 | **Kernel 4×4** pada encoder dan decoder (stride 2, pad 1) | keputusan tim |
| "ConvT 2×2 tidak punya tumpang-tindih" | ConvT 4×4 dihitung dalam bentuk **gather (sub-pixel)**: tiap piksel keluaran memakai 4 tap dari jendela masukan 2×2 | argumen lama tidak berlaku. Tetap tanpa buffer overlap-add (bagian 5.2) |
| 53.248 MAC (32×32) | **524.288 MAC** | 64×64 (×4) dan kernel 4×4 (×2,46 dari 212.992 pada 64×64) |
| Decoder sekitar 31% MAC | **50% MAC** | ConvT 4×4 sama mahalnya dengan konvolusi 4×4 |
| Batas akumulator 22-bit | **23-bit**, akumulator tetap 24-bit | 128 suku pada E3 (8 kanal × 16 tap). Margin 24-bit terhadap batas: 4× untuk bias |
| SSE 26-bit | **28-bit** | N = 4.096 |
| Memori aktivasi 3.072 B | **12.288 B** (minimum 10.240 B) | 64×64 |
| Payload hash sekitar 1.500 B (24 blok) | **3.342 B (53 blok)** | bobot 3.200 + bias 132 + shift 6 + T 4, format persis dari skrip |
| DSP ≤ 8 | **DSP ≤ 9** | koreksi v2: pengkuadrat SSE membutuhkan satu pengali di luar 8 lane |
| Perhitungan manual | **`cae_pipeline.py`** | latih, kuantisasi, model integer bit-exact, ekspor golden vector dan SHA-256 |

---

## 4. Input dan preprocessing

- ROI 64×64 piksel, 1 kanal, 8-bit (uint8).
- Preprocessing di hardware hanya satu operasi: **x_q = piksel − 128** (menjadi int8). Rekonstruksi memakai skala yang sama (2⁻⁷), sehingga selisih input dan rekonstruksi langsung bermakna.
- Padding konvolusi: nol pada domain x_q. Python golden model dan RTL **harus** memakai padding yang sama, karena perbedaan di sini merusak kecocokan bit-exact.
- Validasi: jumlah piksel yang diterima harus tepat N = 4.096. Selain itu, NON_AUTHENTIC (ROI tidak diproses; ambil ulang).
- *Opsi [UJI]:* normalisasi per-ROI (mengurangi rata-rata, yaitu jumlah piksel digeser 12 bit karena N = 4.096) untuk mengurangi sensitivitas terhadap kecerahan global. Ukur dengan set G1_terang dan G1_gelap (bagian 12) sebelum diputuskan.
- *Varian 32×32:* pool 2×2 dengan pembulatan, `(a+b+c+d+2) >> 2`, dapat dilakukan di hardware atau di sisi pengirim.

---

## 5. Model CAE baseline

### 5.1 Lapis

| Lapis | Operasi | Input | Output | Bobot | MAC |
|---|---|---|---|---|---|
| E1 | Conv 4×4, stride 2, pad 1, ReLU | 64×64×1 | 32×32×4 | 64 | 65.536 |
| E2 | Conv 4×4, stride 2, pad 1, ReLU | 32×32×4 | 16×16×8 | 512 | 131.072 |
| E3 | Conv 4×4, stride 2, pad 1, ReLU | 16×16×8 | 8×8×8 | 1.024 | 65.536 |
| D1 | ConvT 4×4, stride 2, pad 1, ReLU | 8×8×8 | 16×16×8 | 1.024 | 65.536 |
| D2 | ConvT 4×4, stride 2, pad 1, ReLU | 16×16×8 | 32×32×4 | 512 | 131.072 |
| D3 | ConvT 4×4, stride 2, pad 1, linear | 32×32×4 | 64×64×1 | 64 | 65.536 |
| **Total** | | | | **3.200 (+33 bias = 3.233)** | **524.288** |

Encoder 262.144 MAC, decoder 262.144 MAC (masing-masing 50%) [HITUNG]. Jumlah kanal 4-8-8 adalah titik awal DSE dan perlu dibuktikan lewat training.

### 5.2 ConvT 4×4 dalam bentuk gather

Definisi ConvT stride 2, pad 1: keluaran y = 2i − 1 + ky. Dalam bentuk gather, piksel keluaran (2m+a, 2n+b) dengan a, b ∈ {0, 1} memakai masukan (i, j) dengan **i ∈ {m−1+a, m+a}** dan **j ∈ {n−1+b, n+b}**, dan tap bobot **ky = 2(m−i) + a + 1**, **kx = 2(n−j) + b + 1**. Masukan di luar batas dianggap 0. Rumus ini sudah dicocokkan dengan definisi scatter pada uji numerik [HITUNG].

Akibatnya setiap piksel keluaran membutuhkan jendela masukan 2×2 dan 4 tap × Cin perkalian, tanpa buffer overlap-add. Biaya tambahannya ada di address generator (pembangkit alamat fase dan pemeriksaan batas), bukan di memori.

### 5.3 Variasi DSE (8 lane)

| Varian | MAC | Parameter | Latent | Siklus (dasar) | µs @100 MHz | Siklus (lanjutan) | µs @100 MHz |
|---|---|---|---|---|---|---|---|
| 64×64 gray, kernel 4×4 (baseline) | 524.288 | 3.233 | 512 | 147.456 | 1.474,6 | 81.920 | 819,2 |
| 32×32 gray (pool 2×2), kernel 4×4 | 131.072 | 3.233 | 128 | 36.864 | 368,6 | 20.480 | 204,8 |
| 64×64 RGB, kernel 4×4 | 786.432 | 3.491 | 512 | 180.224 | 1.802,2 | 131.072 | 1.310,7 |
| 64×64 gray, kernel 3×3 / ConvT 2×2 (baseline v2) | 212.992 | 1.333 | 512 | 49.664 | 496,6 | 33.280 | 332,8 |

Kernel 4×4 menaikkan MAC sekitar 2,46× dibanding 3×3/2×2 pada resolusi yang sama. **Manfaatnya harus dibuktikan dengan AUROC**, bukan diasumsikan. Hipotesisnya: ConvT 4×4 yang saling tumpang-tindih menghasilkan rekonstruksi lebih halus, sedangkan ConvT 2×2 memetakan tiap piksel masukan ke blok 2×2 yang independen. Skrip mendukung pembandingan langsung lewat `--enc-k 3 --dec-k 2` dan `--enc-k 4 --dec-k 4`. Varian RGB hanya dihitung secara analitik (skrip saat ini grayscale).

---

## 6. Arsitektur hardware

### 6.1 Blok

```
HPS (Cortex-A9, tidak tepercaya)
   |  Avalon-MM, lightweight HPS-to-FPGA bridge
   v
[Register Interface (read-mostly)] --> [Input Validator] --> [Input Buffer 4.096 B]
                                                                 |
 [Weight ROM + Layer Descriptors + T (ROM, tanpa write port)]    v
            |                                        [Controller FSM + Address Generator]
            v                                                    |
      [MAC Array, 8 lane] <--> [Feature RAM A / B, ping-pong]
            |
            v
 [Requant (shift) + ReLU/Saturate] --> (D3) --> [Error Engine: (x_q - x_hat)^2, akumulasi]
                                                          |
                                                          v
                        [Comparator SSE <= T] --> [Decision Logic] --> hasil 2 status (1 bit)
 [SHA-256 Integrity Checker] --> integrity_ok (menggerbang Decision Logic)
```

### 6.2 Jadwal komputasi (layer-sequential)

E1: Input → A. E2: A → B. E3: B → A. D1: A → B. D2: B → A. D3: A → langsung ke Error Engine (tidak disimpan).

Feature map yang harus disimpan: E1 4.096 B, E2 2.048 B, E3 512 B, D1 2.048 B, D2 4.096 B. Buffer A menampung maksimum 4.096 B, buffer B 2.048 B. Minimum total (Input + A + B) adalah **10.240 B**. Spesifikasi memakai A dan B simetris (total **12.288 B = 98.304 bit**) sebagai batas konservatif yang menyederhanakan address generator [HITUNG]. Input Buffer dipertahankan sampai D3 selesai karena dibutuhkan untuk menghitung selisih.

### 6.3 MAC array dan memori

- **Dataflow:** output-stationary. Satu aktivasi di-broadcast ke 8 lane, setiap lane memakai bobotnya sendiri (word ROM lebar 64-bit). Satu lane = satu kanal keluaran.
- **Setiap lane:** pengali int8×int8, akumulator 24-bit, penambah bias, requant shift dengan pembulatan, ReLU/saturasi.
- **Weight ROM:** 528 kata × 64-bit = 4.224 B, termasuk lane kosong (E1 memakai 4 dari 8 lane, D3 hanya 1). Isi sebenarnya 3.200 B bobot + 132 B bias. Diinisialisasi saat konfigurasi dari file `.hex`, tanpa write port [HITUNG].
- **Activation RAM:** 12.288 B. Dengan packing word 64-bit (satu posisi piksel per alamat, 8 kanal), perkiraannya 112,0–160,0 Kbit tergantung B dibuat minimum atau simetris [ESTIMASI]. Bersama ROM (33,0 Kbit), totalnya sekitar 145,0–193,0 Kbit.

**Siklus per lapis (8 lane):**

| Lapis | MAC | Siklus (dasar) | Siklus (lanjutan) | Kata ROM |
|---|---|---|---|---|
| E1 | 65.536 | 16.384 | 16.384 | 16 |
| E2 | 131.072 | 16.384 | 16.384 | 64 |
| E3 | 65.536 | 8.192 | 8.192 | 128 |
| D1 | 65.536 | 8.192 | 8.192 | 128 |
| D2 | 131.072 | 32.768 | 16.384 | 128 |
| D3 | 65.536 | 65.536 | 16.384 | 64 |
| **Total** | **524.288** | **147.456** | **81.920** | **528** |

- **Pemetaan dasar:** lane dipetakan ke kanal keluaran. Utilisasi 0,444. **D3 (hanya 1 kanal keluaran) menghabiskan 44% dari seluruh siklus** (65.536 siklus) karena 7 dari 8 lane menganggur [HITUNG].
- **Pemetaan lanjutan:** pada ConvT, 4 fase keluaran dan kanal diparalelkan ke seluruh lane (lane = fase × kanal). D3 turun menjadi 16.384 siklus, total 81.920 siklus, utilisasi 0,800. Susunan ROM berubah (subset tap 2×2 per fase). Dikerjakan hanya bila inti dasar sudah lulus verifikasi.
- **Throughput** pada 100 MHz (asumsi): sekitar 678 inferensi/detik (dasar) dan 1221 (lanjutan). Fmax sebenarnya baru diketahui setelah sintesis.

**Sweep jumlah lane P (64×64 gray, 100 MHz):**

| P | Siklus (dasar) | µs | Utilisasi | Siklus (lanjutan) | µs | Utilisasi |
|---|---|---|---|---|---|---|
| 1 | 524.288 | 5.242,9 | 1,000 | 524.288 | 5.242,9 | 1,000 |
| 2 | 294.912 | 2.949,1 | 0,889 | 262.144 | 2.621,4 | 1,000 |
| 4 | 180.224 | 1.802,2 | 0,727 | 131.072 | 1.310,7 | 1,000 |
| 8 | 147.456 | 1.474,6 | 0,444 | 81.920 | 819,2 | 0,800 |
| 16 | 147.456 | 1.474,6 | 0,222 | 69.632 | 696,3 | 0,471 |

Pada pemetaan dasar, **P > 8 tidak menambah kecepatan** karena kanal keluaran maksimum model ini adalah 8. Itu menjadikan P = 8 titik jenuh yang bisa dijelaskan. Angka latensi hanya siklus compute. Waktu memuat 4.096 piksel dari HPS bergantung pada kecepatan bridge dan tidak termasuk.

---

## 7. Kuantisasi dan datapath

| Besaran | Format | Dasar |
|---|---|---|
| Piksel masuk | uint8 → x_q = piksel − 128, int8 | |
| Bobot | int8, per-tensor, simetris, skala 2^e_w | |
| Aktivasi | int8 (ReLU: 0…127), skala 2^e_out per lapis | |
| Hasil kali | int16 | \|a\|·\|w\| ≤ 128·128 = 16.384 |
| Akumulator | 24-bit signed | batas terburuk 128 suku × 16.384 = 2.097.152, butuh 23-bit [HITUNG] |
| Requant | akumulator → ReLU → `(acc + (1<<(s-1))) >> s` → saturasi | shift s = e_out − e_in − e_w, bilangan bulat karena semua skala pangkat dua |
| Rekonstruksi x̂_q | int8, skala 2⁻⁷ (sama dengan input) | |
| Selisih d = x_q − x̂_q | int9 (−255…255) | |
| d² | uint16 | maksimum 65.025 |
| SSE (N = 4.096) | uint28 | maksimum 266.342.400 [HITUNG] |
| SSE (N = 1.024, varian 32×32) | uint26 | maksimum 66.585.600 |

**Verifikasi di data nyata:** skrip melaporkan |acc| maksimum teramati per lapis dan menghentikan proses bila akumulator 24-bit tidak cukup.

**[UJI]** Skala pangkat dua membuat bobot hanya memakai sekitar 6–7 bit efektif. Bila selisih AUROC int8 terhadap float melebihi 1 poin persentase, alternatifnya quantization-aware training atau requant dengan pengali tetap plus shift.

---

## 8. Error engine dan keputusan

- Selisih, kuadrat, dan akumulasi berjalan **streaming** saat D3 mengeluarkan piksel rekonstruksi, sehingga tidak menambah siklus di luar D3.
- Keputusan: **SSE ≤ T** menghasilkan AUTHENTIC (dengan syarat lain di bagian 10). **SSE > T** menghasilkan NON_AUTHENTIC. Makna: AUTHENTIC = lolos penyaring, NON_AUTHENTIC = tidak lolos penyaring; **bukan** "pasti asli" atau "pasti palsu". T = N·τ dihitung sekali saat desain, jadi tidak ada pembagi di hardware.
- **Pilihan SSE vs SAD [UJI]:** SSE butuh satu pengali. SAD tidak butuh pengali dan maksimumnya 1.044.480 (20-bit) untuk 64×64 [HITUNG]. Skrip melaporkan AUROC keduanya.
- **Jangan menjual "dividerless" sebagai kontribusi utama.** Membandingkan SSE dengan N·τ adalah optimasi standar yang bernilai kecil.
- **Titik lemah reconstruction error:** ROI yang sangat rata atau pudar dapat direkonstruksi dengan error lebih rendah daripada ROI asli yang kaya detail. Set `A1_pudar_15` di skrip khusus mengukur ini. Pada uji pipeline dengan data sintetis, kasus ini memang menghasilkan skor *lebih rendah* dari uang asli. Periksa pada dataset tim.

---

## 9. Antarmuka

**Build utama (SoC):** Avalon-MM slave 32-bit melalui lightweight HPS-to-FPGA bridge (dibangun dengan Platform Designer).

| Offset | Nama | Akses | Fungsi |
|---|---|---|---|
| 0x00 | CTRL | W | start, soft reset |
| 0x04 | STATUS | R | busy, done, dan bit diagnostik opsional (integrity_ok, input_ok, timeout). Bit diagnostik **bukan** keputusan |
| 0x08 | PIXEL | W | 4 piksel per word, 1.024 kali tulis per ROI |
| 0x0C | RESULT | R | bit 0: 1 = AUTHENTIC, 0 = NON_AUTHENTIC (nilai reset 0) |
| 0x10 | SSE | R | nilai SSE (untuk logging dan kalibrasi) |
| 0x14 | CYCLES | R | jumlah siklus inferensi |
| 0x18–0x34 | DIGEST[0..7] | R | hasil hash terhitung (untuk verifikator eksternal) |

Tidak ada register yang dapat menulis bobot, bias, atau T. Peta register ini adalah usulan [PUTUSAN].

**Build demo mandiri (opsional, disarankan):** beberapa ROI uji disimpan di ROM, tombol memulai inferensi, hasil tampil di 8 LED. Template proposal meminta pengujian di board, dan varian ini memungkinkan demo on-board tanpa menyiapkan Linux HPS.

---

## 10. Security-by-design

**Aset:** bobot model, T, keputusan akhir.
**Batas kepercayaan:** inti akselerator di fabric tepercaya, HPS dan sumber ROI tidak tepercaya.

| Ancaman | Kontrol | Sisa risiko |
|---|---|---|
| Bobot atau T diubah saat runtime | ROM tanpa write port, tanpa register tulis | Penggantian seluruh bitstream |
| Korupsi memori atau fault injection | SHA-256 atas bobot, bias, shift, dan T. Dicek saat boot dan berkala | Digest ikut diganti jika bitstream diganti |
| Input rusak atau salah ukuran | Validasi jumlah piksel, handshake valid/ready, watchdog (batas 2× siklus nominal) | Input valid format tetapi berbahaya |
| Bit-flip memalsukan keputusan | Bit hasil disimpan redundan (mis. dua flip-flop komplementer), nilai reset = NON_AUTHENTIC | Pembalikan banyak bit sekaligus |
| Overflow akumulator | Lebar bit dibuktikan cukup, aritmetika saturasi | Hampir tidak ada |

**Aturan fail-secure:** AUTHENTIC hanya keluar bila integritas lulus, input valid, komputasi selesai tepat waktu, dan SSE ≤ T, **semuanya sekaligus**. Reset, timeout, input salah ukuran, dan kegagalan integritas selalu menghasilkan NON_AUTHENTIC.

**Payload hash (format dari `cae_pipeline.py --export`):** bobot int8 + bias int32 (little-endian) per lapis E1…D3, lalu 6 byte shift, lalu T (uint32). Ukurannya **3.342 B** (3.200 + 132 + 6 + 4), sekitar **53 blok** SHA-256 [HITUNG]. Pada inti iteratif ~64 siklus per blok, perkiraannya sekitar 3.392 siklus [ESTIMASI]. Digest acuan dihitung offline oleh skrip (`payload.sha256`). RTL harus meng-hash byte yang persis sama.

**Opsional:** hasil keputusan ditandatangani HMAC-SHA-256 dengan nonce agar HPS tidak dapat memutar ulang hasil lama. Kunci di ROM hanya melindungi dari HPS, bukan dari pihak yang bisa membaca bitstream.

**Batas yang harus ditulis jujur:**
- Bitstream dan root of trust (secure boot, manajemen kunci) berada di luar cakupan.
- **HPS yang dikompromikan masih bisa memasukkan ROI uang asli ke akselerator.** Sistem tidak melindungi keaslian jalur input.
- Cetakan palsu yang meniru pola "BI" dengan baik bisa lolos. Autoencoder juga kadang merekonstruksi input di luar distribusi dengan baik.
- Side-channel dan power analysis tidak ditangani.

---

## 11. Penentuan threshold

Gunakan **split conformal** agar τ punya dasar statistik (sudah diimplementasikan di skrip):
1. Pisahkan ROI uang asli menjadi train, validasi, kalibrasi, dan uji. **Pisahkan berdasarkan lembar uang dan sesi capture**, bukan per crop, supaya tidak terjadi kebocoran. Skrip memakai subfolder atau `--group-regex` untuk ini, dan memperingatkan bila info kelompok tidak ada.
2. Hitung SSE dari **model int8 bit-exact** (bukan model float) untuk n sampel kalibrasi.
3. T = skor SSE ke-⌈(n+1)(1−α)⌉ terkecil. Dengan data yang dapat dipertukarkan (exchangeable), laju penolakan uang asli yang diharapkan tidak melebihi α.
4. Pilih α dari target operasional (misalnya 5%). Dengan α = 0,05, kalibrasi butuh minimal 19 sampel.

Jaminan di atas hanya berlaku bila kondisi capture saat uji menyerupai kalibrasi. Karena variasi capture adalah sumber error utama, sampel kalibrasi harus mencakup variasi yang diizinkan protokol.

---

## 12. Dataset dan evaluasi AI

| Himpunan | Isi | Peran |
|---|---|---|
| G0 | Uang asli, capture nominal | train / validasi / kalibrasi / uji |
| G1 | Uang asli dengan variasi capture: blur, terang/gelap, geser, rotasi kecil | ukur laju false anomaly |
| A1 | Anomali sintetis **digital**: pola dipudarkan, blok dihapus, noise, cermin, rotasi 90°, tile diacak | uji deteksi |
| A2 | Crop dari area non-latent atau pecahan lain | kontrol negatif |
| A3 | Sampel palsu asli **hanya dari sumber resmi** (BI, PERURI, kepolisian), lewat `--anomaly-dir` | uji paling bermakna, minta lewat panitia |

Skrip sudah membuat G1 dan A1 secara otomatis dari himpunan uji. **Penetapan sebuah perturbasi sebagai G1 atau A1 adalah keputusan tim**: misalnya pemudaran 15% saya golongkan anomali, tetapi pemudaran ringan bisa saja kondisi capture yang sah.

**Peringatan hukum:** jangan mencetak atau menggandakan uang fisik untuk membuat sampel uji. Memalsukan Rupiah adalah tindak pidana menurut UU No. 7 Tahun 2011 tentang Mata Uang. Gunakan perturbasi digital dan sampel resmi.

**Metrik (positif = anomali):**
- AUROC antara G0 dan setiap himpunan anomali, serta gabungan A1 dan G1.
- Laju penolakan (NON_AUTHENTIC) per himpunan pada T hasil kalibrasi.
- Laju false anomaly pada G0 dan G1.
- FPR@95%TPR pada A1 gabungan.
- Selisih AUROC int8 terhadap float (target awal ≤ 1 poin persentase) dan korelasi Spearman skor.
- Baseline pembanding wajib: PCA (k komponen) dan SAD terhadap template rata-rata. **Jangan menjanjikan CAE mengalahkan PCA sebelum diukur.** Pada uji pipeline dengan data sintetis, PCA justru lebih baik daripada CAE kecil. Itu bukan temuan tentang dataset tim, tetapi alasan baseline ini wajib ada.

**Catatan base rate:** pada prevalensi uang palsu sekitar 4 per sejuta lembar (data BI 2024–2025), FPR 1% menghasilkan sekitar 10.000 alarm palsu untuk 4 uang palsu per sejuta lembar [HITUNG]. Karena itu keluaran diposisikan sebagai penyaring.

---

## 13. Verifikasi

**Level 1:** model float (Python, `cae_pipeline.py`), baseline AI.
**Level 2:** model integer bit-exact di skrip (semantik shift, pembulatan, saturasi, dan padding tertulis jelas). Ini golden reference. Skrip menjalankan *self-test* di setiap eksekusi: operasi integer dibandingkan dengan implementasi naif (loop), dan gradien dibandingkan dengan beda-hingga.
**Level 3:** simulasi RTL. Bandingkan **keluaran setiap lapis dan SSE akhir** dengan golden vector hasil `--export`, target **0 mismatch** pada seluruh ROI uji dan vektor terarah.

**Vektor terarah:** semua 0, semua 255, papan catur, kasus SSE maksimum, stres saturasi, SSE tepat sama dengan T (harus AUTHENTIC karena kriterianya ≤), inferensi beruntun, serta tepi gambar (padding) pada ConvT.

**Uji keamanan:**
- Balik 1 bit di tiap region ROM → harus NON_AUTHENTIC.
- Jumlah piksel kurang atau lebih → NON_AUTHENTIC.
- Simulasi timeout watchdog → NON_AUTHENTIC.
- Perilaku setelah reset → NON_AUTHENTIC. Tabel kebenaran 16 kombinasi (`decision_truth_table.json` dari `--export`): hanya 1111 yang AUTHENTIC.

**Tool:** cocotb + Verilator (di WSL2 pada Windows) atau simulator bawaan Quartus. Sintesis dan analisis timing dengan Quartus Prime Lite (Timing Analyzer, Power Analyzer).

---

## 14. Metrik hardware dan DSE

Laporkan untuk setiap konfigurasi: ALM, register, DSP, M10K, Fmax, siklus per inferensi, inferensi per detik, utilisasi MAC, dan energi per inferensi (estimasi Power Analyzer). Bandingkan lewat produk area × latensi.

**Sumbu DSE:** resolusi (32/64), kanal (gray/RGB), kernel (3×3 + 2×2 / 4×4 + 4×4), jumlah lane P (1/2/4/8), pemetaan (dasar/lanjutan), metrik error (SSE/SAD), lebar bit.

**Baseline perangkat lunak:** ukur atau estimasi inferensi model yang sama pada Cortex-A9 di HPS (latensi dan energi). Juri akan bertanya mengapa model sekecil ini butuh accelerator, dan jawabannya harus berupa angka perbandingan.

---

## 15. Estimasi resource DE10-Nano [ESTIMASI]

| Komponen | Estimasi |
|---|---|
| DSP | ≤ 9 blok (8 lane + 1 pengkuadrat). Bisa lebih sedikit bila Quartus mengemas beberapa pengali 9×9 dalam satu blok DSP. Jika SAD dipilih, tidak ada pengali tambahan |
| Memori | sekitar 145,0–193,0 Kbit (ROM + activation RAM), kira-kira 3% dari kapasitas M10K. Jumlah blok M10K bergantung lebar dan kedalaman word, perkirakan sekitar 15–25 blok |
| ALM | orde besar di bawah 10% dari kapasitas, belum dipastikan |

Semua angka ini perkiraan kasar. **Hanya fitter report yang menentukan angka final.** Angka kapasitas di template (41.910 ALM, 5.570 Kbit M10K, dan 415.000 register) belum saya konfirmasi dari sumber, dan angka register tampak tidak konsisten dengan jumlah ALM. Cek di *Cyclone V Device Overview* Intel sebelum mengisi tabel.

---

## 16. Risiko dan bagian yang kemungkinan dikritik juri

**Fundamental:**
1. **Akurasi belum terbukti.** CAE one-class pada latent image yang bergantung sudut belum diukur. Bila AUROC rendah, klaim deteksi runtuh. Mitigasi: jalankan `cae_pipeline.py` pada dataset tim sebelum 8 Oktober, dan bingkai kontribusi sebagai accelerator beserta DSE-nya.
2. **Relevansi masalah.** Data BI menunjukkan uang palsu jarang dan kualitasnya relatif rendah (1–4 ppm). Jawaban: pemeriksaan manual itu subjektif, sampel palsu langka (alasan one-class), dan sistem diposisikan sebagai penyaring.
3. **"Mengapa tidak di ARM?"** Model hanya 524.288 MAC. Butuh baseline perangkat lunak pada HPS (bagian 14).
4. **"Mengapa kernel 4×4?"** Biayanya 2,46× MAC dan 2,97× siklus dibanding 3×3/2×2. Jawabannya harus berupa AUROC dari skrip, bukan preferensi.

**Dapat diperbaiki:**
5. "Dividerless" adalah optimasi kecil, jangan jadi sorotan utama.
6. Pengujian on-board diminta template, siapkan build demo mandiri (bagian 9).
7. Batas security (bitstream, jalur input) harus tertulis jujur (bagian 10).
8. **Sensitivitas pencahayaan global:** model menerima x_q mentah tanpa normalisasi. Variasi kecerahan besar dapat memunculkan false anomaly. Ukur dengan G1_terang dan G1_gelap, dan pertimbangkan opsi normalisasi (bagian 4).

**Sudah kuat:**
- Fokus sempit pada satu fitur dan satu accelerator yang bisa disintesis.
- Verifikasi tiga level dengan golden model bit-exact, vektor golden, dan digest yang bisa dihasilkan dari skrip.
- Security yang menyatu di arsitektur (ROM tanpa write port, fail-secure) dan konsisten dengan buku panduan.

---

## 17. Belum diputuskan [UJI]

Kernel final (3/2 vs 4/4), jumlah kanal per lapis, resolusi final (64 vs 32), grayscale vs RGB, SSE vs SAD, normalisasi per-ROI, nilai α, pecahan dan tahun emisi, protokol capture, Fmax target, apakah HMAC dikerjakan, apakah pemetaan lanjutan dikerjakan.

---

## 18. Rencana kerja

**Sebelum bootcamp (jika lolos Top 5, pengumuman 13 Oktober):** jalankan `cae_pipeline.py` pada dataset 64×64 (bandingkan kernel dan resolusi), pilih konfigurasi, simpan hasil `--export`.

| Hari bootcamp (18–20 Okt) | Fokus | Target |
|---|---|---|
| 1 | RTL MAC array + FSM + address generator (termasuk gather ConvT) | Simulasi satu lapis cocok dengan golden vector |
| 2 | Seluruh lapis, error engine, SHA-256, validasi, antarmuka | Simulasi level 3: 0 mismatch, uji fault injection lulus |
| 3 | Sintesis, DSE (P, resolusi, kernel), build demo, dokumentasi | Laporan resource/timing, demo LED, repositori |

---

## 19. Referensi

- Terasic DE10-Nano Kit, Intel: https://www.intel.com/content/www/us/en/developer/topic-technology/edge-5g/hardware/fpga-de10-nano.html
- Terasic DE10-Nano, Arrow: https://arrow.com/en/products/p0496/terasic-technologies
- Siaran pers Bank Indonesia, pemusnahan Rupiah palsu: https://www.bi.go.id/id/publikasi/ruang-media/news-release/Pages/sp_2810426.aspx
- Baseline panitia: TT07 SHA-256, TT07 Iterative MAC, TinyTPU (Buku Panduan Peserta, topik 03)

---

## 20. Hasil evaluasi awal (8 Oktober 2026) dan status klaim

Sumber: `evaluasi.py` (validasi silang 5 fold antar kelompok, 106 ROI asli dalam 11 kelompok, 185 crop anomali dari 37 foto). Rincian di `evaluasi_report.md`.

| Klaim | Status | Bukti |
|---|---|---|
| int8 setara float | **Terbukti pada data ini** | Spearman 0,9994–1,0000; selisih AUROC ≤ 0,0013 |
| Akumulator 24-bit cukup | **Terbukti pada data ini** | teramati 17–18 bit, batas teori 23 bit |
| CAE memberi sinyal keputusan yang andal | **Belum terbukti** | lihat di bawah |
| Model mengenali bentuk "BI" | **Belum terbukti** | cermin/rotasi 90°/acak tile: AUROC 0,5–0,64 |

Angka kunci (CAE int8, SSE ≤ T, antar kelompok): TAR 74,5% (interval 43–100%), TRR 67,6% (59–75%), akurasi seimbang 71,0%. Pembanding: template rata-rata 87,1%, kecerahan rata-rata ROI saja 88,8%. Pada anomali yang berpola saja (34 crop), CAE TRR 46,5% dan AUROC 0,80, template AUROC 0,94.

**Cara menulis di proposal:** posisikan kontribusi utama pada akselerator, paritas int8 yang terbukti, verifikasi bit-exact, dan security-by-design. Tulis kemampuan deteksi sebagai hasil prototipe awal dengan keterbatasan di atas, bukan klaim akurasi.
