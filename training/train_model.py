import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():
    print("Membaca dataset_ready.csv...")
    file_path = 'dataset_ready.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} tidak ditemukan! Jalankan preprocessing terlebih dahulu.")
        return

    df = pd.read_csv(file_path)

    # Target: Negara_Penempatan (multi-class)
    X = df.drop(columns=['Negara_Penempatan'])
    y = df['Negara_Penempatan']

    print("Membagi dataset menjadi data latih dan uji (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Melatih model Decision Tree (multi-class)...")
    model = DecisionTreeClassifier(random_state=42, max_depth=8)
    model.fit(X_train, y_train)

    print("Mengevaluasi model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc * 100:.2f}%")
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    model_path = 'decision_tree_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model berhasil disimpan ke {model_path}")

if __name__ == "__main__":
    train()

