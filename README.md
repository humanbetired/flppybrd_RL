# Flappy Bird DQN

Proyek ini mengimplementasikan agen Deep Q-Network (DQN) untuk bermain game Flappy Bird menggunakan pendekatan reinforcement learning. Lingkungan game dibuat dengan Gymnasium dan dirender menggunakan Pygame, dilengkapi grafis menarik seperti sprite burung dengan animasi, pipa, dan latar belakang yang realistis. Proyek ini memisahkan proses pelatihan dan pengujian/visualisasi, sehingga Anda dapat melatih model dan memvisualisasikan performa model terbaik secara terpisah.

## Fitur

- **Lingkungan Flappy Bird Kustom**: Lingkungan yang kompatibel dengan Gymnasium, dengan animasi burung, tanah bergerak, dan grafis pipa yang realistis.
- **Agen DQN**: Model DQN berbasis PyTorch dengan experience replay dan target network untuk pelatihan stabil.
- **Pemisahan Pelatihan dan Pengujian**: Latih model dengan `train.py` dan visualisasikan performa model terbaik dengan `test.py`.
- **Penyimpanan Model**: Model terbaik disimpan otomatis berdasarkan skor atau total reward selama pelatihan.

## Prasyarat

- Python 3.8 atau lebih tinggi
- Disarankan menggunakan virtual environment (misalnya `venv` atau `conda`)

## Instalasi

1. Clone repositori:

   ```bash
   git clone https://github.com/humanbetired/flappy-bird-dqn.git
   cd flappy-bird-dqn
   ```

2. Buat dan aktifkan virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Di Windows: venv\Scripts\activate
   ```

3. Instal dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Siapkan aset grafis:

   - Unduh aset Flappy Bird (misalnya dari FlapPyBird assets).
   - Tempatkan file berikut di folder `assets/`:
     - `bird1.png`, `bird2.png`, `bird3.png` (frame animasi burung)
     - `pipe.png` (sprite pipa)
     - `background.png` (gambar latar belakang)
     - `base.png` (gambar tanah)
   - Jika aset tidak ada, game akan menggunakan persegi panjang berwarna sebagai pengganti.

## Struktur Proyek

```
flappy-bird-dqn/
├── assets/                 # Folder untuk aset grafis (burung, pipa, latar belakang, tanah)
├── flappy_env.py           # Lingkungan Gymnasium untuk Flappy Bird
├── dqn_agent.py            # Logika model dan agen DQN
├── train.py                # Skrip untuk melatih model DQN dan menyimpan model terbaik
├── test.py                 # Skrip untuk memuat dan memvisualisasikan model terbaik
├── best_flappy_model.pth   # Model terbaik yang disimpan (dihasilkan setelah pelatihan)
├── requirements.txt        # Daftar dependensi Python
├── README.md               # Dokumentasi proyek
```

## Cara Penggunaan

### Pelatihan

Untuk melatih model DQN:

```bash
python train.py
```

- Skrip ini melatih model selama 1000 episode (dapat diubah di `train.py`).
- Model terbaik (berdasarkan skor atau total reward) disimpan sebagai `best_flappy_model.pth`.
- Kemajuan pelatihan ditampilkan, termasuk nomor episode, skor, total reward, dan nilai epsilon.
- Game dirender selama pelatihan untuk melihat prosesnya secara visual.

### Pengujian/Visualisasi

Untuk memvisualisasikan model terbaik:

```bash
python test.py
```

- Skrip ini memuat `best_flappy_model.pth` dan menjalankan satu episode dengan model yang telah dilatih.
- Rendering diaktifkan untuk menampilkan burung yang menavigasi pipa dengan sprite animasi.
- Skor akhir dan total reward ditampilkan di akhir visualisasi.

## Konfigurasi

- **Hyperparameter**: Ubah `train.py` untuk menyesuaikan `EPISODES`, `MAX_STEPS`, `UPDATE_TARGET_EVERY`, dll.
- **Pengaturan DQN**: Modifikasi `dqn_agent.py` untuk learning rate, epsilon decay, gamma, dll.
- **Lingkungan**: Sesuaikan `flappy_env.py` untuk mekanisme game (misalnya gravitasi, kecepatan lompat) atau rendering (misalnya FPS, ukuran sprite).

## Catatan

- Pastikan folder `assets/` berisi file sprite yang sesuai untuk visual terbaik. Tanpa aset, game akan menggunakan persegi panjang berwarna.
- Jika performa model kurang baik, coba tingkatkan `EPISODES` atau sesuaikan hyperparameter di `dqn_agent.py` (misalnya `learning_rate=0.0001`, `epsilon_decay=0.999`).
- Untuk mempercepat pelatihan, ubah `render_mode="rgb_array"` di `flappy_env.py` untuk menonaktifkan rendering.

## Peningkatan di Masa Depan

- Menambahkan plot reward untuk memvisualisasikan kemajuan pelatihan.
- Merekam episode visualisasi sebagai video.
- Mengimplementasikan double DQN atau prioritized experience replay untuk performa lebih baik.
- Menambahkan efek suara untuk lompat, skor, dan tabrakan.
- Mendukung beberapa pipa atau tingkat kesulitan dinamis.

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file LICENSE untuk detail.

## Penghargaan

- Terinspirasi dari game Flappy Bird asli.
- Menggunakan aset dari FlapPyBird.
- Dibangun dengan Gymnasium, Pygame, dan PyTorch.

---

*Dibuat oleh HandukBasah*