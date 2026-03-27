# ============================================================
# APP: No sidebar • Prototypes computed once • Process 2 images
# ============================================================
import os, random
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet34
import cv2  # preprocessing
import yaml

# =========================
# 1) CONFIGURATION — load from config.yaml
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve(path: str) -> str:
    """Resolve relative path (from config) to absolute path."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(APP_DIR, path))

# Load config.yaml
_cfg_path = os.path.join(APP_DIR, "config.yaml")
if os.path.isfile(_cfg_path):
    with open(_cfg_path, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
else:
    _cfg = {}

# Stage 1: Tumor presence detection
DEFAULT_MODEL_PATH_PRESENCE = _resolve(_cfg.get("model_presence", "../models/best_protonet_class_liver_tumor_resnet34.pth"))
DEFAULT_NO_TUMOR_PATH       = _resolve(_cfg.get("data_no_tumor",  "../data/train_presence/no_tumor"))
DEFAULT_TUMOR_PATH          = _resolve(_cfg.get("data_tumor",     "../data/train_presence/tumor"))

# Stage 2: Benign / Malignant classification
DEFAULT_MODEL_PATH_CLASS    = _resolve(_cfg.get("model_class",    "../models/improved_protonet_liver_tumor_resnet34.pth"))
DEFAULT_BENIGN_PATH         = _resolve(_cfg.get("data_benign",    "../data/train/benign"))
DEFAULT_MALIGNANT_PATH      = _resolve(_cfg.get("data_malignant", "../data/train/malignant"))

# Parameters
IMG_SIZE = 224
MEAN  = [0.485, 0.456, 0.406]
STD   = [0.229, 0.224, 0.225]
K_SHOT = 30
SEED   = 42
ALPHA  = 1.0
EXTS   = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
RECURSIVE = True

# Decision thresholds
TUMOR_THRESHOLD     = 0.50    # p(Tumor) >= threshold -> Tumor detected
MALIGNANT_THRESHOLD = 0.42    # p(Malignant) >= threshold -> Malignant (V3 tuned)

# Grayscale-like filter
GRAY_DIFF_TOL     = 3     # max channel difference (0..255)
GRAY_SAT_Q        = 0.10  # 90th percentile saturation S (0..1)
GRAY_CENTER_RATIO = 0.80  # measure within center region

# Image preview (letterbox both images to same size)
PREVIEW_WH = 420  # square frame 420x420

st.set_page_config(
    page_title="Liver Tumor Classification",
    page_icon="🩺",
    layout="centered"
)

# =========================
# 2) CSS (light background, dark buttons, result cards)
# =========================
st.markdown("""
<style>
h1 { margin-top: 10px !important; }
.block-container { padding-top: 1rem; max-width: 1000px; }

/* Main uploader area (light) */
.block-container [data-testid="stFileUploaderDropzone"]{
  background:#f8fafc !important; border:1px dashed #cbd5e1 !important; border-radius:12px !important;
}
.block-container .stFileUploader label{ color:#334155 !important; }

/* Dark buttons */
.action-area .stButton>button{
  background:#111827 !important; color:#fff !important;
  border:1px solid #111827 !important; border-radius:10px !important; font-weight:700 !important;
}
.action-area .stButton>button:hover{ background:#000 !important; border-color:#000 !important; }
.action-area .stButton>button:disabled{ background:#9ca3af !important; border-color:#9ca3af !important; color:#fff !important; }

/* Result cards */
.big-result{
  padding:14px 16px; border-radius:12px; text-align:center;
  font-size:22px; font-weight:800; color:#fff; margin-top:12px;
  box-shadow:0 1px 2px rgba(15,23,42,.08), 0 1px 1px rgba(15,23,42,.04);
}
.bg-benign{   background:#16a34a; }   /* Benign */
.bg-malig{    background:#dc2626; }   /* Malignant */
.bg-notumor{  background:#2563eb; }   /* No tumor / out of domain */
</style>
""", unsafe_allow_html=True)

st.title("🩺 Liver Tumor Classification: Benign vs Malignant")

# =========================
# 3) UTILITY FUNCTIONS + MODEL
# =========================
def sanitize(p: str) -> str:
    return p.strip().strip('"').strip("'") if isinstance(p, str) else p

def strip_prefix(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):     k2 = k[len("module."):]
        elif k.startswith("backbone."): k2 = k[len("backbone."):]
        else:                           k2 = k
        out[k2] = v
    return out


class ProtoNet(nn.Module):
    """Prototypical Network with a projection head (used by V3 model)."""

    def __init__(self, dropout=0.15):
        super().__init__()
        self.encoder = resnet34(weights=None)
        dim = self.encoder.fc.in_features  # 512
        self.encoder.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.projection(self.encoder(x))


@st.cache_resource
def load_encoder(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = sanitize(ckpt_path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    sd = torch.load(p, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = strip_prefix(sd)

    # Auto-detect architecture: V3 (ProtoNet) vs V2 (plain ResNet34)
    has_projection = any(k.startswith("projection.") for k in sd)
    if has_projection:
        model = ProtoNet()
    else:
        model = resnet34(weights=None)
        model.fc = nn.Identity()

    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    for x in model.parameters():
        x.requires_grad_(False)
    return model, device

# --- Grayscale-like filter ---
def is_grayscale_like(pil_img, diff_tol=GRAY_DIFF_TOL, sat_q=GRAY_SAT_Q, center_crop_ratio=GRAY_CENTER_RATIO, return_diag=False):
    rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    H, W = rgb.shape[:2]
    if 0 < center_crop_ratio < 1.0:
        ch = int(H * center_crop_ratio); cw = int(W * center_crop_ratio)
        y0 = (H - ch) // 2; x0 = (W - cw) // 2
        rgb = rgb[y0:y0+ch, x0:x0+cw]

    r, g, b = rgb[:,:,0].astype(np.int16), rgb[:,:,1].astype(np.int16), rgb[:,:,2].astype(np.int16)
    diff_rg = np.abs(r-g).mean()
    diff_rb = np.abs(r-b).mean()
    diff_gb = np.abs(g-b).mean()
    diff_max = float(max(diff_rg, diff_rb, diff_gb))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = (hsv[:,:,1].astype(np.float32) / 255.0)
    S_p90 = float(np.quantile(S, 0.90))

    ok = (diff_max <= diff_tol) and (S_p90 <= sat_q)
    if return_diag:
        return ok, {"diff_max": diff_max, "S_p90": S_p90}
    return ok

def preprocess_for_model(pil_img: Image.Image, size: int = IMG_SIZE, padding: int = 5):
    ok, _ = is_grayscale_like(pil_img, return_diag=True)
    if not ok:
        return None

    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = np.nan_to_num(gray, nan=0, posinf=255, neginf=0).astype(np.uint8)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        masked = cv2.bitwise_and(rgb, rgb, mask=mask)
        x, y, w0, h0 = cv2.boundingRect(cnt)
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(w, x + w0 + padding), min(h, y + h0 + padding)
        crop = masked[y0:y1, x0:x1]
        if crop.size == 0:
            crop = rgb
    else:
        crop = rgb

    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    mean = np.array(MEAN, dtype=np.float32); std = np.array(STD, dtype=np.float32)
    norm = (resized - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor

def embed(enc, device, pil_img: Image.Image) -> np.ndarray:
    x = preprocess_for_model(pil_img)
    if x is None:
        raise ValueError("Image out of domain (not grayscale-like).")
    x = x.to(device)
    with torch.inference_mode():
        z = enc(x)  # [1, 512]
    return z.squeeze(0).detach().cpu().numpy()

def compute_prototype(enc, device, src: str, k: int = K_SHOT, seed: int = SEED, recursive: bool = RECURSIVE):
    src = sanitize(src)
    if os.path.isfile(src):
        return embed(enc, device, Image.open(src).convert("RGB"))
    files = []
    for r, _, names in os.walk(src):
        for n in names:
            if os.path.splitext(n)[1].lower() in EXTS:
                files.append(os.path.join(r, n))
        if not RECURSIVE: break
    if not files:
        raise RuntimeError(f"No images found in: {src}")
    if len(files) > k:
        files = random.Random(seed).sample(files, k)
    feats = []
    for fp in files:
        try:
            feats.append(embed(enc, device, Image.open(fp).convert("RGB")))
        except Exception:
            pass
    if not feats:
        raise RuntimeError("Could not embed any images (may have been filtered as out-of-domain).")
    return np.mean(np.stack(feats, axis=0), axis=0)

def infer_probs_binary(z: np.ndarray, c_neg: np.ndarray, c_pos: np.ndarray, alpha: float = ALPHA):
    d_neg = np.linalg.norm(z - c_neg)
    d_pos = np.linalg.norm(z - c_pos)
    logits = np.array([-alpha*d_neg, -alpha*d_pos], dtype=np.float32)
    logits -= logits.max()
    p = np.exp(logits); p /= max(p.sum(), 1e-12)
    return float(p[0]), float(p[1])

# --- Resize images to equal size for display (keep aspect ratio, letterbox) ---
def resize_pair_equal(imgs, wh=PREVIEW_WH, fill=(246, 248, 252)):  # fill: #f6f8fc
    """
    Letterbox images to equal size (wh x wh) for display.
    Does not affect inference — the model pipeline resizes to 224x224 separately.
    """
    out = []
    for im in imgs:
        im = im.copy()
        im.thumbnail((wh, wh), Image.Resampling.LANCZOS)  # keep aspect ratio
        bg = Image.new("RGB", (wh, wh), fill)
        x = (wh - im.width) // 2
        y = (wh - im.height) // 2
        bg.paste(im, (x, y))
        out.append(bg)
    return out

# =========================
# 4) LOAD MODELS + COMPUTE PROTOTYPES (first run only)
# =========================
try:
    enc_pre, device_pre = load_encoder(DEFAULT_MODEL_PATH_PRESENCE)
    enc_cls, device_cls = load_encoder(DEFAULT_MODEL_PATH_CLASS)
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

if ("PRES_PROTOS" not in st.session_state) or ("CLS_PROTOS" not in st.session_state):
    with st.spinner("Computing prototypes for the first time..."):
        try:
            pres_no  = compute_prototype(enc_pre, device_pre, DEFAULT_NO_TUMOR_PATH,    k=K_SHOT, seed=SEED, recursive=RECURSIVE)
            pres_yes = compute_prototype(enc_pre, device_pre, DEFAULT_TUMOR_PATH,       k=K_SHOT, seed=SEED, recursive=RECURSIVE)
            cls_b    = compute_prototype(enc_cls, device_cls, DEFAULT_BENIGN_PATH,      k=K_SHOT, seed=SEED, recursive=RECURSIVE)
            cls_m    = compute_prototype(enc_cls, device_cls, DEFAULT_MALIGNANT_PATH,   k=K_SHOT, seed=SEED, recursive=RECURSIVE)
            st.session_state["PRES_PROTOS"] = {"NoTumor": pres_no, "Tumor": pres_yes}
            st.session_state["CLS_PROTOS"]  = {"Benign": cls_b, "Malignant": cls_m}
            st.session_state["PROTOS_READY"] = True
        except Exception as e:
            st.error(f"Prototype computation error: {e}")
            st.stop()

# =========================
# 5) MAIN UI — PROCESS UP TO 2 IMAGES
# =========================
st.subheader("Classification")

# Upload up to 2 images
files = st.file_uploader("Select up to 2 images (jpg/png/bmp/tiff/webp)",
                         type=[e[1:] for e in EXTS],
                         accept_multiple_files=True)

# Store original batch + resized preview
if files:
    imgs = []
    for f in files[:2]:
        try:
            imgs.append(Image.open(f).convert("RGB"))
        except Exception:
            st.error(f"Cannot read image: {getattr(f,'name','(unknown)')}")
    st.session_state["BATCH_ORIG"] = imgs
    st.session_state["BATCH_PREV"] = resize_pair_equal(imgs, wh=PREVIEW_WH)
    if len(files) > 2:
        st.warning("Only the first 2 images will be processed.")

batch_orig = st.session_state.get("BATCH_ORIG", [])
batch_prev = st.session_state.get("BATCH_PREV", [])

# Store per-image results
if "RESULTS" not in st.session_state:
    st.session_state["RESULTS"] = {}  # idx -> {"label": "...", "css": "bg-..."}

protos_ready = st.session_state.get("PROTOS_READY", False)

def classify_one(pil_img):
    """Return dict {'label': str, 'css': str} from 2-stage pipeline."""
    # Stage 1: Tumor presence
    z1  = embed(enc_pre, device_pre, pil_img)
    c_no, c_yes = st.session_state["PRES_PROTOS"]["NoTumor"], st.session_state["PRES_PROTOS"]["Tumor"]
    p_no, p_yes = infer_probs_binary(z1, c_no, c_yes, alpha=ALPHA)
    if p_yes < float(TUMOR_THRESHOLD):
        return {"label": "No tumor detected", "css": "bg-notumor"}

    # Stage 2: Benign / Malignant
    z2 = embed(enc_cls, device_cls, pil_img)
    c_b, c_m = st.session_state["CLS_PROTOS"]["Benign"], st.session_state["CLS_PROTOS"]["Malignant"]
    pb, pm   = infer_probs_binary(z2, c_b, c_m, alpha=ALPHA)
    if pm >= float(MALIGNANT_THRESHOLD):
        return {"label": "Result: Malignant", "css": "bg-malig"}
    else:
        return {"label": "Result: Benign", "css": "bg-benign"}

# Display 0/1/2 images
if len(batch_prev) == 0:
    st.info("Please select up to 2 images to classify.")
else:
    cols = st.columns(2) if len(batch_prev) == 2 else [st.container()]

    # Render images and results
    for i in range(len(batch_prev)):
        with (cols[i] if len(cols) == 2 else cols[0]):
            st.image(batch_prev[i], use_container_width=False)
            if i in st.session_state["RESULTS"]:
                r = st.session_state["RESULTS"][i]
                st.markdown(f'<div class="big-result {r["css"]}">{r["label"]}</div>', unsafe_allow_html=True)

    # Single button: classify all uploaded images
    st.markdown("<div class='action-area'>", unsafe_allow_html=True)
    if st.button("Classify (up to 2 images)", use_container_width=True, disabled=not (protos_ready and len(batch_orig) > 0)):
        st.session_state["RESULTS"].clear()
        for i, img in enumerate(batch_orig):
            try:
                st.session_state["RESULTS"][i] = classify_one(img)
            except Exception as e:
                msg = str(e)
                if "out of domain" in msg:
                    st.session_state["RESULTS"][i] = {"label":"Error: please try another image", "css":"bg-notumor"}
                else:
                    st.session_state["RESULTS"][i] = {"label":f"Inference error (image {i+1}): {e}","css":"bg-notumor"}
        # Results render in the loop above on rerun
    st.markdown("</div>", unsafe_allow_html=True)

    if not protos_ready:
        st.info("Computing prototypes for the first time...")
