FORMAT EKSPOR (usulan; RTL harus mengikuti byte yang sama)
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
