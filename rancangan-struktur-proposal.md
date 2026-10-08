# Rancangan Struktur Proposal: PERURI Chip Hackathon 2026

Kategori: IC Chip Design & FPGA Implementation. Topik desain: 03 AI / Edge Accelerator.
Batas pengumpulan proposal: **8 Oktober 2026**.

---

## 0. Aturan yang harus dipenuhi

Ada dua ketentuan yang tampak berbeda, dan saya padukan sebagai berikut.

| Ketentuan | Isi |
|---|---|
| Template dan buku panduan (urutan) | Ringkasan Ide, Latar Belakang & Rumusan Masalah, Proposed Chip Design, Referensi, Lampiran |
| Batas halaman | Maksimal **6 halaman**, tidak termasuk cover, daftar pustaka, dan lampiran teknis |
| 5 bagian wajib | Executive Summary, Problem Statement, Proposed Chip, Technical Design, Security Design |

**Penafsiran yang dipakai:** isi utama (maks. 6 halaman) berisi lima bagian wajib. Bagian "Proposed Chip Design" dari template dipecah menjadi **Proposed Chip** (apa chip-nya) dan **Technical Design** (bagaimana dirancang dan diuji). **Security Design** berdiri sendiri karena buku panduan menyebut bagian itu secara eksplisit. Referensi dan Lampiran berada di luar batas halaman.

Bila ada keraguan apakah Technical Design boleh dipisah dari Proposed Chip, tanyakan ke panitia sebelum 8 Oktober.

---

## 1. Urutan dokumen dan anggaran halaman

| No | Bagian | Halaman | Hitungan |
|---|---|---|---|
| | Cover | di luar batas | tidak dihitung |
| 1 | Executive Summary | 0,50 | dihitung |
| 2 | Problem Statement | 1,00 | dihitung |
| 3 | Proposed Chip | 1,25 | dihitung |
| 4 | Technical Design | 2,00 | dihitung |
| 5 | Security Design | 1,00 | dihitung |
| | Cadangan | 0,25 | dihitung |
| | **Total isi utama** | **6,00** | |
| | Daftar Pustaka | di luar batas | tidak dihitung |
| | Lampiran teknis | di luar batas | tidak dihitung |

Tabel dan gambar ikut dihitung sebagai halaman. Rencanakan **3 gambar** (diagram blok, jadwal dataflow, batas kepercayaan keamanan) dan **4 tabel** (parameter kunci, lebar bit, estimasi resource, ancaman dan kontrol). Sisa detail dipindahkan ke lampiran. Cek jumlah halaman pada PDF akhir, jangan pada dokumen yang masih dapat reflow.

---

## 2. Cover (di luar batas halaman)

- Judul ide desain chip. Usulan:
  1. *LatentGuard: Akselerator Edge-AI One-Class untuk Penyaringan Latent Image Rupiah pada FPGA Cyclone V SoC*
  2. *Akselerator Autoencoder Integer untuk Penyaringan Latent Image "BI" Rupiah dengan Integritas Model Terlindungi*
  3. *Secure One-Class Edge Accelerator untuk Verifikasi Fitur Pengaman Rupiah*
- Nama tim, anggota (2–4 orang, institusi, kontak), dosen pembimbing.
- Kategori dan topik desain (lihat atas).

---

## 3. Isi per bagian

### 3.1 Executive Summary (0,5 halaman)

Lima paragraf pendek, masing-masing 2–3 kalimat, mengikuti urutan template:
1. **Masalah:** pemeriksaan keaslian uang di titik transaksi bergantung pada metode 3D yang subjektif, dan sampel uang palsu untuk melatih AI sangat langka.
2. **Solusi dan chip yang dirancang:** akselerator CAE one-class berbasis integer yang menilai ROI latent image "BI" dari rekonstruksi, dilatih hanya dari uang asli.
3. **Implementasi pada DE10-Nano:** inti di FPGA fabric, HPS sebagai pengirim ROI dan pembaca hasil lewat bridge.
4. **Target pengguna dan dampak:** operator titik transaksi dan perangkat verifikasi uang. Dampaknya berupa penyaring objektif dengan latensi deterministik dan model yang tahan ubah. Jangan klaim angka kinerja yang belum diukur.
5. **Kebaruan dan keunggulan:** kombinasi accelerator khusus latent image, one-class tanpa sampel palsu, desain hardware-aware dengan DSE, dan security-by-design. Jangan menjadikan "dividerless" sebagai kebaruan utama.

Sumber isi: spesifikasi v2.1 bagian 1 dan 3.

### 3.2 Problem Statement (1 halaman)

| Sub-bagian | Isi | Panjang |
|---|---|---|
| Latar belakang | Data BI: 466.535 lembar dimusnahkan (2017–Nov 2025). Rasio temuan 5 ppm (2023), 4 ppm (2024–2025), 1 ppm (April 2026). Kualitas palsu relatif rendah, dikenali lewat 3D | 1 paragraf |
| Tiga masalah | Keterbatasan data, keterbatasan platform edge, keamanan keputusan autentikasi | 3 paragraf pendek |
| Gap | Tabel pendekatan yang ada dibanding pendekatan ini (lihat di bawah) | 1 tabel |
| Rumusan masalah | Satu kalimat tebal: merancang akselerator Edge-AI ringkas, deterministik, dan terlindungi pada Cyclone V SoC untuk ROI latent image | 1 kalimat |
| Batasan dan asumsi | ROI sudah diambil pada protokol capture tetap. Keluaran adalah penyaring. Di luar cakupan: kamera, deteksi objek, alignment | 3–4 butir |

**Tabel gap (baris yang disarankan):**

| Pendekatan | Keterbatasan yang perlu ditulis |
|---|---|
| Pemeriksaan manual 3D | Subjektif, bergantung pemeriksa |
| Klasifikasi supervised | Butuh sampel uang palsu representatif, sulit diperoleh |
| Inferensi software di CPU/MCU | Latensi kurang deterministik, model di penyimpanan yang bisa ditulis ulang |
| Pendekatan ini | One-class, latensi tetap, model dan threshold di ROM |

Draf paragraf "Masalah yang Diangkat" yang sudah kita buat dapat dipakai, dipangkas agar muat 1 halaman. Pasang angka BI dengan sumbernya dan jangan memakai angka yang tidak bisa Anda telusuri.

### 3.3 Proposed Chip (1,25 halaman)

| Sub-bagian | Isi | Visual |
|---|---|---|
| Identitas dan fungsi utama | Nama chip, fungsi: ROI 64×64 masuk, 2 status keluar (AUTHENTIC / NON_AUTHENTIC) | |
| Parameter kunci | Tabel dari spesifikasi v2.1 bagian 1 | Tabel 1 |
| Diagram blok | Register interface, validator, buffer, controller, MAC array, error engine, SHA-256, ROM | **Gambar 1** |
| Input dan output | Daftar sinyal: pixel/start masuk, status/SSE/cycles/integrity_ok keluar | daftar |
| Posisi dalam sistem | Alur HPS → akselerator → HPS | satu kalimat |
| Model ringkas | CAE 6 lapis kernel 4×4, 3.233 parameter, 524.288 MAC, latent 512 nilai | dalam Tabel 1 |

Tabel layer lengkap dipindahkan ke lampiran.

### 3.4 Technical Design (2 halaman)

Ini bagian terpanjang, jadi tiap sub-bagian harus pendek dan memakai tabel.

| Sub-bagian | Isi | Halaman | Item wajib template yang tercakup |
|---|---|---|---|
| 4.1 Arsitektur pemrosesan | MAC array 8 lane, output-stationary, jadwal layer-sequential, pemetaan dasar vs lanjutan, sweep P | 0,35 | arsitektur pemrosesan |
| 4.2 Arsitektur memori | Weight ROM, Input Buffer, RAM A/B ping-pong, ukuran | 0,2 | arsitektur memori |
| 4.3 Datapath dan kuantisasi | Tabel lebar bit (Tabel 2), akumulator 24-bit, requant shift | 0,1 | |
| 4.4 Error engine dan keputusan | SSE streaming, T = N·τ, SSE vs SAD | 0,1 | |
| 4.5 Antarmuka dan komunikasi | Avalon-MM lightweight bridge, register map ringkas, build demo mandiri | 0,2 | antarmuka, komunikasi |
| 4.6 Konsumsi daya | Strategi (on-chip only, int8, clock-enable lane idle) dan metode estimasi (Power Analyzer) | 0,1 | pertimbangan konsumsi daya |
| 4.7 Pendekatan RTL, ISA, IP | SystemVerilog manual. ISA tidak relevan karena fixed-function. IP: TT07 SHA-256 (baseline panitia), Platform Designer untuk bridge HPS | 0,2 | pendekatan RTL, ISA, IP |
| 4.8 Target FPGA/ASIC dan node | DE10-Nano, Cyclone V SE, 28 nm (menurut Intel ARK seri 5CSEA6). ASIC tidak ditargetkan | 0,1 | target FPGA/ASIC, technology node |
| 4.9 Estimasi resource dan tool | Tabel resource (Tabel 3), daftar tool: Quartus Prime Lite, Platform Designer, cocotb/Verilator atau ModelSim, Python | 0,1 | |
| 4.10 Strategi verifikasi dan simulasi | Tiga level (float, integer bit-exact, RTL), vektor terarah, target 0 mismatch | 0,3 | strategi verifikasi, strategi simulasi |
| 4.11 Rencana pengujian dan metrik | Uji RTL, sintesis, demo board (LED/SignalTap), metrik keberhasilan | 0,25 | pengujian, metrik |
| | **Jumlah** | **2,00** | |

**Gambar 2:** jadwal dataflow layer-sequential (E1 → A, E2 → B, ...). Letakkan di 4.1.

**Aturan klaim di proposal:** pada tahap ini tidak ada hasil sintesis maupun akurasi. Tulis angka resource, Fmax, dan AUROC sebagai **target** atau **estimasi**, dan beri tanda yang jelas. Isi angka final di bootcamp.

### 3.5 Security Design (1 halaman)

| Sub-bagian | Isi | Visual |
|---|---|---|
| Prinsip security-by-design | Empat keputusan awal yang membentuk arsitektur: ROM tanpa write port, τ konstanta, status reset = NON_AUTHENTIC, HPS diperlakukan tidak tepercaya | 4 butir |
| Aset dan batas kepercayaan | Bobot, T, keputusan. Fabric tepercaya, HPS dan sumber ROI tidak | **Gambar 3** |
| Ancaman dan kontrol | Tabel ancaman → kontrol → sisa risiko | Tabel 4 |
| Aturan fail-secure | AUTHENTIC hanya jika integritas lulus, input valid, selesai tepat waktu, dan SSE ≤ T, semuanya sekaligus | 1 paragraf |
| Integritas SHA-256 | Payload 3.342 B (53 blok), dicek saat boot dan berkala | 2–3 kalimat |
| Pengujian keamanan | Fault injection per region ROM, input salah ukuran, timeout, perilaku reset | 1 daftar |
| Batas yang jujur | Bitstream di luar cakupan, HPS terkompromi masih bisa memasukkan ROI asli, cetakan palsu bisa lolos, side-channel tidak ditangani | 4 butir |

Poin terpenting: security harus terlihat membentuk keputusan arsitektur sejak awal, bukan sekadar daftar fitur di akhir. Karena itu urutan sub-bagian dimulai dari prinsip desain.

---

## 4. Daftar Pustaka (di luar batas halaman)

Isi dengan sumber yang benar-benar dipakai:
- Siaran pers Bank Indonesia tentang pemusnahan Rupiah palsu.
- Katadata, BI-Bareskrim musnahkan 466 ribu lembar uang palsu.
- Intel, Terasic DE10-Nano Kit. Arrow, Terasic DE10-Nano. Intel ARK, Cyclone V 5CSEA6.
- Baseline panitia: TT07 SHA-256, TT07 Iterative MAC, TinyTPU. Peruri Chip Design datasheet bila dirujuk.
- **Penelitian terdahulu latent image Rupiah** dan **literatur CAE untuk anomaly detection**: isi dengan paper yang sudah Anda baca. Saya tidak menambahkan sitasi yang tidak bisa saya verifikasi.

---

## 5. Lampiran (di luar batas halaman)

| Lampiran | Isi |
|---|---|
| L1 | Rencana bootcamp 3 hari (tabel hari, fokus, target deliverables) |
| L2 | Identitas tim dan pembagian peran (nama, keahlian, tanggung jawab) |
| L3 | Tabel layer CAE lengkap dan varian DSE (32/64, gray/RGB) |
| L4 | Register map lengkap dan daftar sinyal I/O |
| L5 | Rencana dataset dan metrik evaluasi (himpunan G0, G1, A1–A3, split conformal) |
| L6 | Hasil perhitungan: MAC, memori, batas akumulator, sweep P |
| L7 | Asumsi dan hal yang belum diputuskan |

---

## 6. Urutan pengerjaan (sisa waktu sampai 8 Oktober)

1. Tempel bahan yang sudah ada: Problem Statement (draf Masalah), Technical Design (spesifikasi v2.1), Security Design (ringkasan keamanan).
2. Buat 3 gambar: diagram blok, jadwal dataflow, batas kepercayaan.
3. Pangkas ke 6 halaman, lalu cek PDF akhir.
4. Isi cover, lampiran identitas tim, dan daftar pustaka.

---

## 7. Hal yang perlu dikonfirmasi sebelum dikirim

- **Papan FPGA:** template menyebut DE10-Nano, buku panduan menyebut DE1-Nano. Konfirmasi ke panitia.
- **Tabel kapasitas template** (41.910 ALM, 5.570 Kbit M10K) belum terverifikasi, cek di Cyclone V Device Overview.
- **Hubungan dengan baseline panitia:** buku panduan menyatakan panitia menyediakan desain dasar. Jelaskan secara eksplisit bagian mana yang diadopsi atau dirujuk (SHA-256, MAC), dan cek apakah datasheet Peruri Chip Design relevan bagi desain Anda. Saya belum membaca datasheet itu.
- **Format file:** dokumen ini draf Markdown. Template meminta dokumen yang dikirim sebagai proposal, jadi bentuk akhirnya (PDF atau Word) mengikuti ketentuan pengumpulan di situs panitia.
