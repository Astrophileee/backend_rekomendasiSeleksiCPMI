import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib

def preprocess():
    print("Membaca dataset_cpmi.csv...")
    file_path = 'dataset_cpmi.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} tidak ditemukan!")
        return

    df = pd.read_csv(file_path)
    print(f"Dataset dimuat dengan {len(df)} baris.")

    encoders = {}

    print("Melakukan encoding pada data kategorikal...")
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            encoders[column] = le
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Mapping untuk {column}: {mapping}")

    # Simpan encoders agar bisa digunakan saat prediksi
    joblib.dump(encoders, 'encoders.joblib')
    print("Encoders disimpan ke encoders.joblib")

    output_file = 'dataset_ready.csv'
    df.to_csv(output_file, index=False)
    print(f"Dataset hasil preprocessing berhasil disimpan ke {output_file}")

if __name__ == "__main__":
    preprocess()

