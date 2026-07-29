from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

app = FastAPI(
    title="API Penilaian Seleksi CPMI",
    description="API Decision Tree untuk menghitung skor dan rekomendasi seleksi CPMI berdasarkan penilaian petugas"
)


# ─── Schema ────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    nilai_kompetensi: int  # 0-100
    nilai_pelatihan: int   # 0-100
    nilai_bahasa: int      # 0-100
    nilai_wawancara: int   # 0-100
    kesehatan: str         # 'fit' | 'perlu_pemeriksaan_lanjutan'
    sikap_kerja: str       # 'baik' | 'cukup' | 'perlu_pembinaan'


class PredictionResponse(BaseModel):
    score: int
    label: str       # 'lolos' | 'perlu_perhatian' | 'tidak_lolos'
    is_layak: bool


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _kesehatan_score(kesehatan: str) -> float:
    """Konversi status kesehatan ke skor numerik."""
    mapping = {
        "fit": 100.0,
        "perlu_pemeriksaan_lanjutan": 50.0,
    }
    return mapping.get(kesehatan, 50.0)


def _sikap_score(sikap: str) -> float:
    """Konversi sikap kerja ke skor numerik."""
    mapping = {
        "baik": 100.0,
        "cukup": 65.0,
        "perlu_pembinaan": 30.0,
    }
    return mapping.get(sikap, 50.0)


def _compute_score(req: PredictionRequest) -> int:
    """
    Hitung skor komposit berbobot (0-100) berdasarkan hasil penilaian petugas.

    Bobot:
      - Kompetensi  : 30%
      - Wawancara   : 25%
      - Bahasa      : 20%
      - Pelatihan   : 15%
      - Kesehatan   : 6%
      - Sikap Kerja : 4%
    """
    kes_score  = _kesehatan_score(req.kesehatan)
    sikap_score = _sikap_score(req.sikap_kerja)

    weighted = (
        req.nilai_kompetensi * 0.30 +
        req.nilai_wawancara  * 0.25 +
        req.nilai_bahasa     * 0.20 +
        req.nilai_pelatihan  * 0.15 +
        kes_score            * 0.06 +
        sikap_score          * 0.04
    )

    return max(0, min(100, round(weighted)))


def _get_label(score: int) -> str:
    """Tentukan label rekomendasi sistem berdasarkan skor."""
    if score >= 70:
        return "lolos"
    elif score >= 50:
        return "perlu_perhatian"
    else:
        return "tidak_lolos"


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "API Penilaian Seleksi CPMI Aktif",
        "status": "running",
        "version": "2.0"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Proses decision tree: hitung skor komposit dan beri label rekomendasi.

    Input  : nilai_kompetensi, nilai_pelatihan, nilai_bahasa, nilai_wawancara,
             kesehatan, sikap_kerja
    Output : score (0-100), label (lolos/perlu_perhatian/tidak_lolos), is_layak
    """
    # Validasi nilai 0-100
    for field_name, value in [
        ("nilai_kompetensi", request.nilai_kompetensi),
        ("nilai_pelatihan",  request.nilai_pelatihan),
        ("nilai_bahasa",     request.nilai_bahasa),
        ("nilai_wawancara",  request.nilai_wawancara),
    ]:
        if not (0 <= value <= 100):
            raise HTTPException(
                status_code=422,
                detail=f"Field '{field_name}' harus antara 0 dan 100, diterima: {value}"
            )

    if request.kesehatan not in ("fit", "perlu_pemeriksaan_lanjutan"):
        raise HTTPException(
            status_code=422,
            detail=f"Field 'kesehatan' tidak valid: '{request.kesehatan}'"
        )

    if request.sikap_kerja not in ("baik", "cukup", "perlu_pembinaan"):
        raise HTTPException(
            status_code=422,
            detail=f"Field 'sikap_kerja' tidak valid: '{request.sikap_kerja}'"
        )

    try:
        score    = _compute_score(request)
        label    = _get_label(score)
        is_layak = label != "tidak_lolos"

        return {
            "score":    score,
            "label":    label,
            "is_layak": is_layak,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses: {str(e)}"
        )
