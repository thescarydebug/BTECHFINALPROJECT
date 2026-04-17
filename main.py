"""
URGMN FastAPI Backend — Fixed Version
Serves the HTML frontend and handles real model inference.
"""

import os, io, warnings, traceback
import numpy as np
import nibabel as nib
import cv2
import torch
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
DEVICE      = torch.device("cpu")   # CPU for deployment
CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_FULL  = [
    "Cognitively Normal",
    "Mild Cognitive Impairment",
    "Alzheimer's Disease"
]

# ── Training statistics for normalisation ──
# These come from the ADNI training data (adni_master_rich.csv descriptive stats)
FEAT_STATS = {
    "CDR_GLOBAL":  (0.26, 0.25),
    "CDR_SB":      (0.90, 1.02),
    "MMSE_SCORE":  (0.79, 0.14),   # already normalised 0-1 in master
    "FAQ_TOTAL":   (0.15, 0.20),   # already normalised 0-1
    "APOE_RISK":   (0.30, 0.38),
    "AGE":         (0.55, 0.17),   # normalised
    "EDUCATION":   (0.62, 0.14),   # normalised
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

# Serve static files (the frontend)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# ── Load models at startup ──
fold_models = []

@app.on_event("startup")
def load_models():
    global fold_models
    print("Loading URGMN fold models...")
    for i in range(1, 6):
        path = os.path.join(MODEL_DIR, f"best_fold{i}.pt")
        if not os.path.exists(path):
            # Also try best_model.pt as fallback
            alt = os.path.join(MODEL_DIR, "best_model.pt")
            if os.path.exists(alt) and i == 1:
                path = alt
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
            print(f"  Fold {i}: loaded ✅  ({path})")
        except Exception as e:
            print(f"  Fold {i}: ERROR — {e}")

    if not fold_models:
        print("WARNING: No models loaded. Check models/ folder.")
    else:
        print(f"\nLoaded {len(fold_models)} model(s) on {DEVICE}")


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": len(fold_models),
        "device":        str(DEVICE),
        "feat_cols":     FEAT_COLS,
    }


# ── MRI preprocessing ──
def extract_slices(nifti_bytes: bytes) -> np.ndarray:
    """
    Load NIfTI from bytes, extract N_SLICES central axial slices.
    Returns array of shape [N_SLICES, IMG_SIZE, IMG_SIZE] dtype float32 in [0,1].
    """
    try:
        fh = nib.FileHolder(fileobj=io.BytesIO(nifti_bytes))
        img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    except Exception:
        # Try as gzip
        import gzip, tempfile
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
        dtype=int
    )

    slices = []
    for idx in indices:
        sl = data[:, :, idx]
        mn, mx = sl.min(), sl.max()
        if mx - mn > 0:
            sl = (sl - mn) / (mx - mn)
        else:
            sl = np.zeros_like(sl)
        sl = cv2.resize(
            (sl * 255).astype(np.uint8),
            (IMG_SIZE, IMG_SIZE)
        ).astype(np.float32) / 255.0
        slices.append(sl)

    return np.stack(slices, axis=0)  # [N, H, W]


def build_clinical_tensor(
    mmse: float, cdr: float, faq: float,
    age: float, edu: float, apoe: int,
    gender: int, cdr_sb: float
) -> torch.Tensor:
    """
    Normalise raw clinical inputs to match training distribution.
    Returns tensor of shape [1, NUM_FEATS].
    """
    # Compute derived features
    cog_idx = (
        (1.0 - mmse / 30.0) * 0.4 +
        (cdr / 3.0) * 0.4 +
        (faq / 30.0) * 0.2
    )
    apoe_risk  = {-1: 0.0, 0: 0.0, 1: 0.5, 2: 1.0}.get(int(apoe), 0.0)
    gender_bin = 1.0 if int(gender) == 2 else 0.0

    # Map MMSE and FAQ to 0-1 range (matching training preprocessing)
    mmse_norm = mmse / 30.0
    faq_norm  = faq / 30.0

    raw = {
        "CDR_GLOBAL":  float(cdr),
        "CDR_SB":      float(cdr_sb),
        "MMSE_SCORE":  float(mmse_norm),
        "FAQ_TOTAL":   float(faq_norm),
        "APOE_RISK":   float(apoe_risk),
        "AGE":         float(min(1.0, max(0.0, (age - 40.0) / 60.0))),
        "EDUCATION":   float(min(1.0, max(0.0, edu / 25.0))),
        "COG_INDEX":   float(min(1.0, max(0.0, cog_idx))),
        "GENDER_BIN":  float(gender_bin),
    }

    feats = [raw[col] for col in FEAT_COLS]
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


# ── Prediction endpoint ──
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
            content={"error": "No models loaded. Check that best_fold1.pt to best_fold5.pt exist in models/"}
        )

    # ── Clinical tensor ──
    try:
        clin = build_clinical_tensor(
            mmse, cdr, faq, age, edu, apoe, gender, cdr_sb
        ).to(DEVICE)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to build clinical tensor: {str(e)}"}
        )

    # ── MRI slices ──
    has_mri = False
    if mri_file is not None and mri_file.filename:
        try:
            nifti_bytes = await mri_file.read()
            slices_arr  = extract_slices(nifti_bytes)
            has_mri     = True
            print(f"MRI loaded: {mri_file.filename}, slices shape: {slices_arr.shape}")
        except Exception as e:
            print(f"MRI load failed: {e}")
            traceback.print_exc()
            has_mri = False

    if not has_mri:
        # Use a blank placeholder — model will rely on clinical stream
        slices_arr = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        print("Using blank MRI placeholder (no file uploaded or failed to load)")

    # Convert slices to tensor [N_SLICES, 3, H, W]
    slices_t = torch.from_numpy(slices_arr).unsqueeze(1).repeat(1, 3, 1, 1).to(DEVICE)

    # ── Ensemble inference over all folds ──
    all_fold_probs  = []
    all_fold_r_mri  = []
    all_fold_r_clin = []

    with torch.no_grad():
        for fold_idx, m in enumerate(fold_models):
            slice_probs = []
            r_mris      = []
            r_clins     = []

            for s_idx in range(slices_t.shape[0]):
                img_b = slices_t[s_idx].unsqueeze(0)  # [1, 3, H, W]

                # Extract reliability scores manually
                x = m.stem(img_b)
                x = m.layer1(x)
                x = m.layer2(x); x = m.cbam2(x)
                x = m.layer3(x); x = m.cbam3(x)
                x = m.layer4(x); x = m.cbam4(x)
                mf = m.pool(x).flatten(1)
                cf = m.clin(clin)

                r_mri_score  = float(m.r_mri(mf).item())
                r_clin_score = float(m.r_clin(cf).item())

                # Full forward pass
                probs, _, _ = m(img_b, clin)
                slice_probs.append(probs.cpu().numpy()[0])
                r_mris.append(r_mri_score)
                r_clins.append(r_clin_score)

            # Average over slices for this fold
            fold_avg_probs = np.mean(slice_probs, axis=0)
            all_fold_probs.append(fold_avg_probs)
            all_fold_r_mri.append(np.mean(r_mris))
            all_fold_r_clin.append(np.mean(r_clins))

    # Average over folds
    final_probs = np.mean(all_fold_probs, axis=0)
    pred_class  = int(final_probs.argmax())

    # Uncertainty from Dirichlet: run one more pass to get alpha
    with torch.no_grad():
        img_b = slices_t[N_SLICES // 2].unsqueeze(0)
        _, unc_val, alpha = fold_models[0](img_b, clin)
        if isinstance(unc_val, torch.Tensor):
            uncertainty = float(unc_val.mean().item())
        else:
            uncertainty = float(unc_val)
    uncertainty = max(0.02, min(0.95, uncertainty))

    r_mri_final  = float(np.mean(all_fold_r_mri))
    r_clin_final = float(np.mean(all_fold_r_clin))

    print(f"Prediction: {CLASS_NAMES[pred_class]} "
          f"({final_probs[pred_class]*100:.1f}%) "
          f"unc={uncertainty:.3f} "
          f"R_mri={r_mri_final:.3f} R_clin={r_clin_final:.3f}")

    return {
        "diagnosis":      CLASS_NAMES[pred_class],
        "diagnosis_full": CLASS_FULL[pred_class],
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