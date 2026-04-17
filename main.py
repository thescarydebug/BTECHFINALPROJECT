"""
URGMN FastAPI Backend — Render Deployment Ready
Uses Hugging Face Hub for model storage (no Google Drive blocking).

SETUP:
1. Upload your .pt files to huggingface.co/YOUR_USERNAME/urgmn-alzheimers
2. Set HF_USERNAME environment variable in Render dashboard
   OR just hardcode your username in HF_USERNAME below.
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

# ─────────────────────────────────────────────
# CONFIG — edit HF_USERNAME to your HuggingFace username
# ─────────────────────────────────────────────
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_HF_USERNAME_HERE")
HF_REPO     = f"{HF_USERNAME}/urgmn-alzheimers"
MODEL_DIR   = "./models"
N_SLICES    = 20
IMG_SIZE    = 224
DEVICE      = torch.device("cpu")
CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_FULL  = [
    "Cognitively Normal",
    "Mild Cognitive Impairment",
    "Alzheimer's Disease",
]


# ─────────────────────────────────────────────
# Download models from Hugging Face at startup
# ─────────────────────────────────────────────
def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for i in range(1, 6):
        path = os.path.join(MODEL_DIR, f"best_fold{i}.pt")
        if os.path.exists(path):
            print(f"  Model {i}: already on disk, skipping download")
            continue

        url = (
            f"https://huggingface.co/{HF_REPO}"
            f"/resolve/main/best_fold{i}.pt"
        )
        print(f"  Downloading model {i} from {url} ...")

        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()

            # Hugging Face returns content-type application/octet-stream
            # or binary/octet-stream for .pt files. If we get HTML,
            # the repo is private or the file doesn't exist.
            ct = r.headers.get("Content-Type", "")
            if "text/html" in ct:
                raise RuntimeError(
                    f"Got HTML instead of a model file. "
                    f"Check that the repo '{HF_REPO}' is PUBLIC and "
                    f"best_fold{i}.pt is uploaded."
                )

            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            size_mb = os.path.getsize(path) / 1e6
            print(f"  Model {i}: downloaded ({size_mb:.1f} MB)")

        except Exception as e:
            # Remove partial file if download failed
            if os.path.exists(path):
                os.remove(path)
            print(f"  Model {i}: DOWNLOAD FAILED — {e}")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────
fold_models = []


@app.on_event("startup")
def load_models():
    global fold_models

    print("\n=== URGMN Model Loading ===")

    if HF_USERNAME == "YOUR_HF_USERNAME_HERE":
        print("WARNING: HF_USERNAME not set. Set it as an environment variable "
              "in Render or edit main.py directly.")
    else:
        download_models()

    for i in range(1, 6):
        path = os.path.join(MODEL_DIR, f"best_fold{i}.pt")

        if not os.path.exists(path):
            # Fallback: if only best_model.pt exists, use it for fold 1
            alt = os.path.join(MODEL_DIR, "best_model.pt")
            if os.path.exists(alt) and i == 1:
                path = alt
                print(f"  Fold {i}: using best_model.pt as fallback")
            else:
                print(f"  Fold {i}: NOT FOUND at {path}")
                continue

        try:
            m = URGMN().to(DEVICE)
            m.load_state_dict(
                torch.load(path, map_location=DEVICE, weights_only=True)
            )
            m.eval()
            fold_models.append(m)
            print(f"  Fold {i}: loaded OK")
        except Exception as e:
            print(f"  Fold {i}: ERROR loading weights — {e}")
            traceback.print_exc()

    print(f"\nLoaded {len(fold_models)} / 5 fold models on {DEVICE}")
    print("===========================\n")


@app.get("/health")
def health():
    return {
        "status":        "ok" if fold_models else "no_models",
        "models_loaded": len(fold_models),
        "device":        str(DEVICE),
        "hf_repo":       HF_REPO,
    }


# ─────────────────────────────────────────────
# MRI preprocessing
# ─────────────────────────────────────────────
def extract_slices(nifti_bytes: bytes) -> np.ndarray:
    """Load NIfTI from bytes, return [N_SLICES, H, W] float32 in [0,1]."""
    try:
        fh  = nib.FileHolder(fileobj=io.BytesIO(nifti_bytes))
        img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    except Exception:
        # Fallback for gzip-compressed files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
            tf.write(nifti_bytes)
            tf_path = tf.name
        img = nib.load(tf_path)
        os.unlink(tf_path)

    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[:, :, :, 0]

    z_dim   = data.shape[2]
    center  = z_dim // 2
    indices = np.linspace(
        max(0, center - z_dim // 3),
        min(z_dim - 1, center + z_dim // 3),
        N_SLICES,
        dtype=int,
    )

    slices = []
    for idx in indices:
        sl       = data[:, :, idx].copy()
        mn, mx   = sl.min(), sl.max()
        sl       = (sl - mn) / (mx - mn + 1e-6)
        sl_uint8 = (sl * 255).astype(np.uint8)
        sl_resized = cv2.resize(sl_uint8, (IMG_SIZE, IMG_SIZE))
        slices.append(sl_resized.astype(np.float32) / 255.0)

    return np.stack(slices, axis=0)   # [N_SLICES, H, W]


# ─────────────────────────────────────────────
# Clinical tensor
# ─────────────────────────────────────────────
def build_clinical_tensor(
    mmse: float, cdr: float, faq: float,
    age: float, edu: float, apoe: int,
    gender: int, cdr_sb: float,
) -> torch.Tensor:
    """
    Build normalised clinical feature tensor matching training preprocessing.
    Order must match FEAT_COLS in model.py exactly.
    """
    cog_idx    = (1.0 - mmse / 30.0) * 0.4 + (cdr / 3.0) * 0.4 + (faq / 30.0) * 0.2
    apoe_risk  = {-1: 0.0, 0: 0.0, 1: 0.5, 2: 1.0}.get(int(apoe), 0.0)
    gender_bin = 1.0 if int(gender) == 2 else 0.0

    # FEAT_COLS order: CDR_GLOBAL, CDR_SB, MMSE_SCORE, FAQ_TOTAL,
    #                  APOE_RISK, AGE, EDUCATION, COG_INDEX, GENDER_BIN
    feats = [
        float(cdr),                              # CDR_GLOBAL
        float(cdr_sb),                           # CDR_SB
        float(mmse / 30.0),                      # MMSE_SCORE (normalised)
        float(faq / 30.0),                       # FAQ_TOTAL  (normalised)
        float(apoe_risk),                        # APOE_RISK
        float(min(1.0, max(0.0, (age - 40.0) / 60.0))),   # AGE
        float(min(1.0, max(0.0, edu / 25.0))),             # EDUCATION
        float(min(1.0, max(0.0, cog_idx))),                # COG_INDEX
        float(gender_bin),                       # GENDER_BIN
    ]
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


# ─────────────────────────────────────────────
# Prediction endpoint
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(
    mri_file: Optional[UploadFile] = File(None),
    mmse:    float = Form(24.0),
    cdr:     float = Form(0.5),
    faq:     float = Form(6.0),
    age:     float = Form(72.0),
    edu:     float = Form(16.0),
    apoe:    int   = Form(-1),
    gender:  int   = Form(1),
    cdr_sb:  float = Form(1.0),
):
    if not fold_models:
        return JSONResponse(
            status_code=503,
            content={"error": "No models loaded. Check Render logs."},
        )

    # ── Clinical features ──
    try:
        clin = build_clinical_tensor(
            mmse, cdr, faq, age, edu, apoe, gender, cdr_sb
        ).to(DEVICE)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Clinical tensor error: {e}"},
        )

    # ── MRI slices ──
    has_mri = False
    if mri_file and mri_file.filename:
        try:
            nifti_bytes = await mri_file.read()
            slices_arr  = extract_slices(nifti_bytes)
            has_mri     = True
            print(f"MRI loaded: {mri_file.filename}  shape={slices_arr.shape}")
        except Exception as e:
            print(f"MRI load failed: {e}")
            traceback.print_exc()
            has_mri = False

    if not has_mri:
        # Blank placeholder — model relies on clinical stream only
        slices_arr = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # [N_SLICES, 3, H, W]
    slices_t = (
        torch.from_numpy(slices_arr)
        .unsqueeze(1)
        .repeat(1, 3, 1, 1)
        .to(DEVICE)
    )

    # ── Ensemble inference ──
    all_fold_probs  = []
    all_fold_r_mri  = []
    all_fold_r_clin = []

    with torch.no_grad():
        for m in fold_models:
            slice_probs  = []
            r_mri_scores = []
            r_clin_scores = []

            for s_idx in range(N_SLICES):
                img_b = slices_t[s_idx].unsqueeze(0)  # [1, 3, H, W]

                # Extract reliability scores
                x = m.stem(img_b)
                x = m.layer1(x)
                x = m.layer2(x); x = m.cbam2(x)
                x = m.layer3(x); x = m.cbam3(x)
                x = m.layer4(x); x = m.cbam4(x)
                mf = m.pool(x).flatten(1)
                cf = m.clin(clin)
                r_mri_scores.append(float(m.r_mri(mf).item()))
                r_clin_scores.append(float(m.r_clin(cf).item()))

                probs, _, _ = m(img_b, clin)
                slice_probs.append(probs.cpu().numpy()[0])

            all_fold_probs.append(np.mean(slice_probs, axis=0))
            all_fold_r_mri.append(np.mean(r_mri_scores))
            all_fold_r_clin.append(np.mean(r_clin_scores))

    final_probs = np.mean(all_fold_probs, axis=0)
    pred_idx    = int(final_probs.argmax())

    # Uncertainty from EDL
    with torch.no_grad():
        mid_slice = slices_t[N_SLICES // 2].unsqueeze(0)
        _, unc_val, _ = fold_models[0](mid_slice, clin)
        if isinstance(unc_val, torch.Tensor):
            uncertainty = float(unc_val.mean().item())
        else:
            uncertainty = float(unc_val)
    uncertainty = max(0.02, min(0.95, uncertainty))

    r_mri_final  = float(np.mean(all_fold_r_mri))
    r_clin_final = float(np.mean(all_fold_r_clin))

    print(
        f"Result: {CLASS_NAMES[pred_idx]} "
        f"({final_probs[pred_idx]*100:.1f}%)  "
        f"unc={uncertainty:.3f}  "
        f"R_mri={r_mri_final:.3f}  R_clin={r_clin_final:.3f}"
    )

    return {
        "diagnosis":      CLASS_NAMES[pred_idx],
        "diagnosis_full": CLASS_FULL[pred_idx],
        "probabilities": {
            "CN":  round(float(final_probs[0]), 4),
            "MCI": round(float(final_probs[1]), 4),
            "AD":  round(float(final_probs[2]), 4),
        },
        "uncertainty":    round(uncertainty, 4),
        "reliability": {
            "mri":      round(r_mri_final, 4),
            "clinical": round(r_clin_final, 4),
        },
        "has_mri":       has_mri,
        "model_version": "URGMN-v2.0",
        "n_models":      len(fold_models),
    }