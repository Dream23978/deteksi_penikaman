# 📋 ROADMAP & PANDUAN PENGERJAAN PROYEK UAS AI
**Judul Proyek:** Sistem Deteksi Cerdas Tindak Kriminal Penikaman (Stabbing Assault) Berbasis YOLO + Pose Estimation dengan Notifikasi Alarm Darurat ke Smartphone Admin

---

## 💡 Jawaban Penting: Apakah Tetap Perlu Knife Detection?
**YA, TETAP SANGAT PERLU.**
* **Alasannya:** Jika hanya mengandalkan gerakan tangan cepat (pose), sistem akan mudah mengalami *False Alarm* (salah deteksi) saat seseorang hanya melambaikan tangan cepat, menunjuk, atau bergerak refleks.
* **Kombinasi Terbaik:**
  $$\text{Terdeteksi Pisau (Knife)} + \text{Kecepatan Tangan Menusuk (Pose)} + \text{Jarak Dekat ke Target} = \mathbf{ALARM\ BAHAYA}$$
  Kombinasi ini akan membuat sistem Anda sangat solid dan bernilai tinggi di mata dosen penguji.

---

## 🚀 Step-by-Step Pengerjaan Proyek

```
[Tahap 1: Setup Lingkungan]
         │
         ▼
[Tahap 2: Uji Coba Model Pisau & Pose]
         │
         ▼
[Tahap 3: Implementasi Logika Gerakan Menusuk]
         │
         ▼
[Tahap 4: Integrasi Alarm Suara ntfy ke HP Admin]
         │
         ▼
[Tahap 5: Setup Simulasi CCTV (HP 1 = Kamera, HP 2 = Admin)]
         │
         ▼
[Tahap 6: Pengujian & Bahan Presentasi UAS]
```

---

### Tahap 1: Persiapan Environment & Akun
- [ ] Pastikan library utama sudah terpasang di Python:
  ```powershell
  pip install ultralytics opencv-python requests numpy
  ```
- [ ] Install aplikasi **ntfy** di Google Play Store / App Store pada HP penerima (HP Admin).
- [ ] Buat nama topik unik di ntfy (contoh: `uas-ai-keamanan-2026`).

---

### Tahap 2: Penyiapan Model Deteksi Pisau & Pose
Ada dua pilihan untuk model pisau:
1. **Opsi Cepat (Tanpa Training):**
   * Gunakan model bawaan `yolo11n.pt` yang sudah mengenali kelas `knife` (Class ID 43) dan `person` (Class ID 0).
2. **Opsi Nilai Tambah (Training Sendiri via Google Colab):**
   * Download dataset pisau dari Roboflow (format YOLOv8/YOLO11).
   * Latih di Google Colab dengan GPU T4 (50 epoch, ~15 menit).
   * Download file `best.pt` dan masukkan ke folder proyek.

---

### Tahap 3: Pembuatan Logika Gerakan Menikam (Cara A)
Sistem menggabungkan data dari `yolo11n.pt` (atau `best.pt`) dan `yolo11n-pose.pt`:
1. **Deteksi Objek:** Temukan posisi koordinat bounding box pisau dan tubuh manusia.
2. **Titik Kunci Pose:** Ambil titik koordinat pergelangan tangan (*wrist*):
   * Titik 9: Pergelangan tangan kiri.
   * Titik 10: Pergelangan tangan kanan.
3. **Pengecekan Vektor Gerakan:**
   * Simpan posisi $(x, y)$ pergelangan tangan pada frame sebelumnya ($t-1$).
   * Hitung kecepatan tusukan: $v = \sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}$.
   * Jika $v > \text{Threshold}$ (tangan bergerak menusuk dengan akselerasi tinggi) **DAN** pisau terdeteksi di tangan tersebut $\rightarrow$ picu alarm ancaman.

---

### Tahap 4: Integrasi Alarm Darurat ntfy
- [ ] Buat fungsi `kirim_alarm_ntfy(frame_bukti, pesan)`.
- [ ] Pasang konfigurasi prioritas:
  * `Priority: urgent` (membuat HP berdering kencang / bergetar meski dalam mode senyap).
  * `Tags: warning,knife,rotating_light`.
- [ ] Pasang **Cooldown Timer** (misal 8–10 detik) agar HP admin tidak dibanjiri spam notifikasi saat kejadian sedang berlangsung.
- [ ] Sertakan foto frame bukti kejadian ke dalam request POST.

---

### Tahap 5: Simulasi Arsitektur CCTV
Simulasikan seolah-olah sistem berjalan di lingkungan nyata:
1. **HP 1 (Sebagai CCTV):**
   * Install aplikasi **IP Webcam** (Android) atau **IP Camera Lite** (iOS).
   * Sambungkan ke Wi-Fi yang sama dengan laptop.
   * Klik *Start Server*, catat URL RTSP/HTTP (contoh: `http://192.168.1.15:8080/video`).
2. **Laptop (Sebagai Server AI):**
   * Arahkan input kamera di script: `cv2.VideoCapture("http://192.168.1.15:8080/video")`.
3. **HP 2 (Sebagai HP Admin):**
   * Membuka aplikasi ntfy dan menerima peringatan darurat.

---

### Tahap 6: Pengujian & Bahan Laporan/Presentasi UAS
- [ ] **Skenario Uji 1 (Normal):** Orang lewat, berjalan, atau duduk di depan CCTV (Sistem aman, tidak ada alarm).
- [ ] **Skenario Uji 2 (Gerakan Cepat Tanpa Pisau):** Orang melambaikan tangan / memukul angin (Tidak boleh alarm, membuktikan sistem pintar).
- [ ] **Skenario Uji 3 (Penikaman):** Orang memegang pisau dan melakukan gerakan menusuk (Alarm langsung berbunyi di HP admin + foto snapshot masuk).
- [ ] Ambil tangkapan layar (screenshot) deteksi di laptop dan notifikasi di HP untuk dimasukkan ke laporan akhir / slide presentasi.

