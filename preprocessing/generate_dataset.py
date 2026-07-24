import pandas as pd
import random

# Tentukan jumlah data (baris) yang ingin dibuat
NUM_ROWS = 500

# Kategori untuk setiap atribut
usia_opts = ['Memenuhi', 'Tidak Memenuhi']
pendidikan_opts = ['SMP', 'SMA', 'D3', 'S1']
pengalaman_opts = ['Ada', 'Tidak Ada']
bahasa_opts = ['Baik', 'Cukup', 'Kurang']
keterampilan_opts = ['Sesuai', 'Tidak Sesuai']
kesehatan_opts = ['Sehat', 'Tidak Sehat']
dokumen_opts = ['Lengkap', 'Tidak Lengkap']

data = []

for _ in range(NUM_ROWS):
    usia = random.choices(usia_opts, weights=[0.8, 0.2])[0]
    pendidikan = random.choices(pendidikan_opts, weights=[0.1, 0.6, 0.15, 0.15])[0]
    pengalaman = random.choices(pengalaman_opts, weights=[0.6, 0.4])[0]
    bahasa = random.choices(bahasa_opts, weights=[0.3, 0.5, 0.2])[0]
    keterampilan = random.choices(keterampilan_opts, weights=[0.7, 0.3])[0]
    kesehatan = random.choices(kesehatan_opts, weights=[0.85, 0.15])[0]
    dokumen = random.choices(dokumen_opts, weights=[0.7, 0.3])[0]
    
    # Logika Penentuan Target (Layak / Tidak Layak)
    # Aturan Wajib (Fatal): Kesehatan dan Dokumen
    if kesehatan == 'Tidak Sehat' or dokumen == 'Tidak Lengkap' or usia == 'Tidak Memenuhi':
        hasil_seleksi = 'Tidak Layak'
    else:
        # Jika lolos syarat fatal, kita hitung skor berdasarkan kualifikasi
        skor = 0
        
        if keterampilan == 'Sesuai':
            skor += 3
        if pengalaman == 'Ada':
            skor += 2
        
        if bahasa == 'Baik':
            skor += 2
        elif bahasa == 'Cukup':
            skor += 1
            
        if pendidikan in ['D3', 'S1']:
            skor += 2
        elif pendidikan == 'SMA':
            skor += 1
            
        # Jika skor mencukupi (misal >= 4), maka Layak.
        # Menambahkan sedikit elemen acak (noise) agar tidak 100% deterministik dan model bisa "belajar" probabilitas
        if skor >= 4:
            hasil_seleksi = random.choices(['Layak', 'Tidak Layak'], weights=[0.95, 0.05])[0]
        else:
            hasil_seleksi = random.choices(['Layak', 'Tidak Layak'], weights=[0.1, 0.9])[0]

    data.append([usia, pendidikan, pengalaman, bahasa, keterampilan, kesehatan, dokumen, hasil_seleksi])

# Simpan ke DataFrame dan Export CSV
df = pd.DataFrame(data, columns=['Usia', 'Pendidikan', 'Pengalaman_Kerja', 'Kemampuan_Bahasa', 'Keterampilan', 'Kesehatan', 'Kelengkapan_Dokumen', 'Hasil_Seleksi'])
df.to_csv('dataset_cpmi.csv', index=False)

print(f"Dataset berhasil dibuat dengan {NUM_ROWS} baris.")
print(df.head())
print("\nDistribusi Hasil Seleksi:")
print(df['Hasil_Seleksi'].value_counts())
