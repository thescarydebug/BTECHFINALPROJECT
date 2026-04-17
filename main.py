"""
URGMN FastAPI Backend — Fixed Version (DEPLOYMENT READY)
"""

import os, io, warnings, traceback
import numpy as np
import nibabel as nib
import cv2
import torch
import requests

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

warnings.filterwarnings("ignore")

from model import URGMN, FEAT_COLS

# ── Config ──
MODEL_DIR   = "./models"
N_SLICES    = 20
IMG_SIZE    = 224
DEVICE      = torch.device("cpu")
CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_FULL  = [
    "Cognitively Normal",
    "Mild Cognitive Impairment",
    "Alzheimer's Disease"
]

# ── Google Drive Model Links ──
MODEL_URLS = [
    "https://drive.google.com/uc?export=download&id=1fLfqHJcCJkuRcaSUbiET79ZUsGxU0d1a",
    "https://drive.google.com/uc?export=download&id=1EStXqihkn8tsEmI9o7t-mbdkfDBVxWhq",
    "https://drive.google.com/uc?export=download&id=1N-On7IbaNNKu-h73uGlr-DId_Gyu3Y4-",
    "https://drive.google.com/uc?export=download&id=1htnIYQLiJCxD6fGPy9x0oaolV_wfd0jM",
    "https://drive.google.com/uc?export=download&id=1i2cij4WBUNR3ezC9DOjBdW214FYMXG4Y"
]

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for i, url in enumerate(MODEL_URLS, start=1):
        path = os.path.join(MODEL_DIR, f"best_fold{i}.pt")

        if not os.path.exists(path):
            print(f"Downloading model {i}...")

            r = requests.get(url, stream=True)

            if "text/html" in r.headers.get("Content-Type", ""):
                raise Exception("Google Drive blocked download")

            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Model {i} downloaded")

# ── Training statistics ──
FEAT_STATS = {
    "CDR_GLOBAL":  (0.26, 0.25),
    "CDR_SB":      (0.90, 1.02),
    "MMSE_SCORE":  (0.79, 0.14),
    "FAQ_TOTAL":   (0.15, 0.20),
    "APOE_RISK":   (0.30, 0.38),
    "AGE":         (0.55, 0.17),
    "EDUCATION":   (0.62, 0.14),
    "COG_INDEX":   (0.33, 0.23),
    "GENDER_BIN":  (0.45, 0.50),
}

app = FastAPI(title="URGMN API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# ── Load models ──
fold_models = []

@app.on_event("startup")
def load_models():
    download_models()   # 🔥 ONLY ADDITION

    global fold_models
    print("Loading URGMN fold models...")

    for i in range(1, 6):
        path = os.path.join(MODEL_DIR, f"best_fold{i}.pt")

        if not os.path.exists(path):
            alt = os.path.join(MODEL_DIR, "best_model.pt")
            if os.path.exists(alt) and i == 1:
                path = alt
            else:
                print(f"Fold {i}: NOT FOUND")
                continue

        try:
            m = URGMN().to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            fold_models.append(m)
            print(f"Fold {i}: loaded")
        except Exception as e:
            print(f"Fold {i}: ERROR — {e}")

    print(f"Loaded {len(fold_models)} models")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": len(fold_models),
        "device": str(DEVICE),
        "feat_cols": FEAT_COLS,
    }

# ── MRI preprocessing ──
def extract_slices(nifti_bytes: bytes) -> np.ndarray:
    try:
        fh = nib.FileHolder(fileobj=io.BytesIO(nifti_bytes))
        img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    except:
        import gzip, tempfile
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
            tf.write(nifti_bytes)
            tf_path = tf.name
        img = nib.load(tf_path)
        os.unlink(tf_path)

    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[:, :, :, 0]

    z_dim = data.shape[2]
    center = z_dim // 2
    indices = np.linspace(max(0, center - z_dim // 3),
                          min(z_dim - 1, center + z_dim // 3),
                          N_SLICES, dtype=int)

    slices = []
    for idx in indices:
        sl = data[:, :, idx]
        mn, mx = sl.min(), sl.max()
        sl = (sl - mn) / (mx - mn + 1e-6)

        sl = cv2.resize((sl * 255).astype(np.uint8),
                        (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        slices.append(sl)

    return np.stack(slices)

# ── Clinical tensor ──
def build_clinical_tensor(mmse, cdr, faq, age, edu, apoe, gender, cdr_sb):
    cog_idx = (
        (1.0 - mmse / 30.0) * 0.4 +
        (cdr / 3.0) * 0.4 +
        (faq / 30.0) * 0.2
    )

    apoe_risk  = {-1:0,0:0,1:0.5,2:1}.get(int(apoe),0)
    gender_bin = 1.0 if int(gender)==2 else 0.0

    feats = [
        float(cdr), float(cdr_sb),
        float(mmse/30), float(faq/30),
        float(apoe_risk),
        float((age-40)/60),
        float(edu/25),
        float(cog_idx),
        float(gender_bin)
    ]

    return torch.tensor(feats).float().unsqueeze(0)

# ── Prediction ──
@app.post("/predict")
async def predict(
    mri_file: Optional[UploadFile] = File(None),
    mmse: float = Form(24.0),
    cdr: float = Form(0.5),
    faq: float = Form(6.0),
    age: float = Form(72.0),
    edu: float = Form(16.0),
    apoe: int = Form(-1),
    gender: int = Form(1),
    cdr_sb: float = Form(1.0),
):
    if not fold_models:
        return JSONResponse(status_code=503,
                            content={"error": "No models loaded"})

    clin = build_clinical_tensor(
        mmse, cdr, faq, age, edu, apoe, gender, cdr_sb
    ).to(DEVICE)

    has_mri = False
    if mri_file:
        try:
            nifti_bytes = await mri_file.read()
            slices_arr = extract_slices(nifti_bytes)
            has_mri = True
        except:
            has_mri = False

    if not has_mri:
        slices_arr = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    slices_t = torch.from_numpy(slices_arr).unsqueeze(1).repeat(1,3,1,1).to(DEVICE)

    all_probs = []

    with torch.no_grad():
        for m in fold_models:
            slice_probs = []
            for i in range(N_SLICES):
                p,_,_ = m(slices_t[i].unsqueeze(0), clin)
                slice_probs.append(p.cpu().numpy()[0])
            all_probs.append(np.mean(slice_probs, axis=0))

    final_probs = np.mean(all_probs, axis=0)
    pred = int(final_probs.argmax())

    return {
        "diagnosis": CLASS_NAMES[pred],
        "diagnosis_full": CLASS_FULL[pred],
        "probabilities": final_probs.tolist(),
        "has_mri": has_mri
    }