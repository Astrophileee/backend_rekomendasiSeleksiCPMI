from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="API Rekomendasi Negara Penempatan CPMI",
    description="API Decision Tree untuk merekomendasikan negara penempatan CPMI"
)

MODEL_PATH = "decision_tree_model.joblib"
ENCODER_PATH = "../preprocessing/encoders.joblib"

model = None
encoders = None

def load_artifacts():
    global model, encoders
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(ENCODER_PATH):
        encoders = joblib.load(ENCODER_PATH)

load_artifacts()

class PredictionRequest(BaseModel):
    usia: int
    pendidikan: int
    pengalaman: int
    bahasa: int
    keterampilan: int
    kesehatan: int
    dokumen: int

class PredictionResponse(BaseModel):
    prediction: str
    is_layak: bool
    prediction_code: int

@app.on_event("startup")
async def startup_event():
    load_artifacts()

@app.get("/")
def root():
    return {"message": "API Rekomendasi Negara Penempatan CPMI Aktif", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global model
    
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503,
                detail="Model belum tersedia. Jalankan training terlebih dahulu."
            )
    
    # Susun input sesuai fitur training
    input_data = pd.DataFrame([{
        'Usia': request.usia,
        'Pendidikan': request.pendidikan,
        'Pengalaman_Kerja': request.pengalaman,
        'Kemampuan_Bahasa': request.bahasa,
        'Keterampilan': request.keterampilan,
        'Kesehatan': request.kesehatan,
        'Kelengkapan_Dokumen': request.dokumen
    }])
    
    try:
        pred_code = int(model.predict(input_data)[0])

        # Decode prediction: gunakan encoder jika tersedia
        if encoders and 'Negara_Penempatan' in encoders:
            le = encoders['Negara_Penempatan']
            pred_str = le.inverse_transform([pred_code])[0]
        else:
            # Fallback manual mapping (alphabetical LabelEncoder order):
            # Arab Saudi=0, Jepang=1, Korea Selatan=2, Malaysia=3, Taiwan=4, Tidak Layak=5
            label_map = {
                0: "Arab Saudi",
                1: "Jepang",
                2: "Korea Selatan",
                3: "Malaysia",
                4: "Taiwan",
                5: "Tidak Layak"
            }
            pred_str = label_map.get(pred_code, "Tidak Layak")

        is_layak = pred_str != "Tidak Layak"

        return {
            "prediction": pred_str,
            "is_layak": is_layak,
            "prediction_code": pred_code
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat prediksi: {str(e)}")
