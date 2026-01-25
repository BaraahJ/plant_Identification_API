import io
import json
from typing import List
from collections import Counter

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import onnxruntime as ort


# ----------------------------
# Paths
# ----------------------------
CONFIG_PATH = "models/model_config.json"
MODEL_PATH = "models/plant_model_batched.onnx"


# ----------------------------
# Load config
# ----------------------------
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

CLASSES: List[str] = cfg["classes"]
IMG_SIZE: int = int(cfg.get("img_size", 224))

TEMP: float = float(cfg["temperature"])
T_CONF: float = float(cfg["threshold_conf"])        # e.g. 0.75
T_MARGIN: float = float(cfg["threshold_margin"])    # e.g. 0.0

OTHER_NAME: str = cfg.get("other_class", "other")
OTHER_IDX: int = CLASSES.index(OTHER_NAME)


# ----------------------------
# ONNX session
# ----------------------------
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name
OUTPUT_NAME = sess.get_outputs()[0].name
print("✅ ONNX input:", INPUT_NAME, "| output:", OUTPUT_NAME)


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Plant Identifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict later
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Helpers
# ----------------------------
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Return normalized float32 NCHW tensor (1,3,H,W)."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC [0,1]
    arr = arr.transpose(2, 0, 1)  # CHW

    # ImageNet normalization (must match training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    arr = (arr - mean) / std

    return arr[None, ...]  # (1,3,H,W)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def top1_from_probs(probs: np.ndarray):
    """Return (idx1, p1, margin=p1-p2) from probs vector shape (C,)."""
    top2 = probs.argsort()[-2:][::-1]
    idx1, idx2 = int(top2[0]), int(top2[1])
    p1, p2 = float(probs[idx1]), float(probs[idx2])
    return idx1, p1, (p1 - p2)


def predict_one(inp: np.ndarray):
    """Return probs, idx, conf, margin for one image."""
    logits = sess.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]  # (1,C)
    probs = softmax(logits / TEMP)[0]                       # (C,)
    idx1, p1, margin = top1_from_probs(probs)
    return probs, idx1, p1, margin


def build_top3(avg_probs: np.ndarray):
    top3_idx = avg_probs.argsort()[-3:][::-1]
    return [{"label": CLASSES[i], "prob": float(avg_probs[i])} for i in top3_idx]


def is_pass(r):
    """Normal acceptance rule (your tuned one)."""
    return (r["idx"] != OTHER_IDX) and (r["conf"] >= T_CONF) and (r["margin"] >= T_MARGIN)


def is_strong(r):
    """
    Strong rescue rule:
    - Accept a single great photo even if others are blurry.
    You can tune these later if needed.
    """
    return (r["idx"] != OTHER_IDX) and (r["conf"] >= 0.85) and (r["margin"] >= 0.05)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "num_classes": len(CLASSES),
        "temperature": TEMP,
        "threshold_conf": T_CONF,
        "threshold_margin": T_MARGIN,
        "other_class": OTHER_NAME
    }


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    if not (1 <= len(files) <= 3):
        raise HTTPException(status_code=400, detail="Upload 1 to 3 images")

    per_img = []
    for f in files:
        data = await f.read()
        inp = preprocess_image(data)
        probs, idx, conf, margin = predict_one(inp)
        per_img.append({
            "probs": probs,
            "idx": idx,
            "label": CLASSES[idx],
            "conf": conf,
            "margin": margin,
        })

    # Average probs for top3 UI (always useful)
    avg_probs = np.mean(np.stack([r["probs"] for r in per_img], axis=0), axis=0)
    top3 = build_top3(avg_probs)

    # ----------------------------
    # Decision policy (perfect scenarios)
    # ----------------------------

    # ----------------------------
# Decision policy (anti-unfair)
# ----------------------------

    def is_weak(r):
        # صور ضعيفة: يا اما Other/Unknown-like أو ثقتها قليلة
        return (r["idx"] == OTHER_IDX) or (r["conf"] < 0.35)

    def is_strong_any(r):
        # صورة قوية جدا (تقدر تعدل الأرقام)
        return (r["idx"] != OTHER_IDX) and (r["conf"] >= 0.80) and (r["margin"] >= 0.10)

    def is_pass_any(r):
        # Pass عادي (أقل صرامة)
        return (r["idx"] != OTHER_IDX) and (r["conf"] >= T_CONF) and (r["margin"] >= T_MARGIN)

    pass_items = [r for r in per_img if is_pass_any(r)]
    strong_items = [r for r in per_img if is_strong_any(r)]

    # Rule A: 2/3 agreement among PASS
    if len(pass_items) >= 2:
        counts = Counter([r["label"] for r in pass_items])
        label, count = counts.most_common(1)[0]
        if count >= 2:
            best = max([r for r in pass_items if r["label"] == label], key=lambda x: x["conf"])
            return {
                "label": label,
                "confidence": float(best["conf"]),
                "margin": float(best["margin"]),
                "top3": top3,
                "policy": "majority_pass"
            }

    # Rule B: Strong single wins if others are weak
    if len(strong_items) >= 1:
        best_strong = max(strong_items, key=lambda x: x["conf"])

        others = [r for r in per_img if r is not best_strong]

        # إذا الباقي ضعيفين أو Unknown/Other → نقبل القوية
        if all(is_weak(r) for r in others):
            return {
                "label": best_strong["label"],
                "confidence": float(best_strong["conf"]),
                "margin": float(best_strong["margin"]),
                "top3": top3,
                "policy": "strong_single_wins"
            }

    # Rule C: Conflict detection (two strong different labels) => Unknown
    if len(strong_items) >= 2:
        strong_labels = set([r["label"] for r in strong_items])
        if len(strong_labels) >= 2:
            # تضارب حقيقي (غالباً صور لنباتات مختلفة)
            p1_idx = int(avg_probs.argmax())
            p1 = float(avg_probs[p1_idx])
            top2_idx = avg_probs.argsort()[-2:][::-1]
            p2 = float(avg_probs[int(top2_idx[1])]) if len(top2_idx) > 1 else 0.0
            margin = p1 - p2
            return {
                "label": "Unknown",
                "confidence": p1,
                "margin": float(margin),
                "top3": top3,
                "policy": "conflict_two_strong"
            }

    # Rule D: fallback unknown
    p1_idx = int(avg_probs.argmax())
    p1 = float(avg_probs[p1_idx])
    top2_idx = avg_probs.argsort()[-2:][::-1]
    p2 = float(avg_probs[int(top2_idx[1])]) if len(top2_idx) > 1 else 0.0
    margin = p1 - p2

    return {
        "label": "Unknown",
        "confidence": p1,
        "margin": float(margin),
        "top3": top3,
        "policy": "unknown"
    }
