"""
predict_tf_only.py
Predict Actual vs Predicted for all models (customcnn, mobilenetv2, resnet50, vgg16)
using tf.keras only.
"""

import os
import sys
import csv
import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# -------------------- CONFIG --------------------
TEST_DIR = "data/chest_xray/test"     # expects subfolders NORMAL/ and PNEUMONIA/

MODELS = {
    "customcnn": "models/customcnn_model",
    "mobilenetv2": "models/mobilenetv2_model",
    "resnet50": "models/resnet50_model",
    "vgg16": "models/vgg16_model"
}

N_PER_CLASS = 6
GRID_COLS = 4
SEED = 42
# ------------------------------------------------

INPUT_SIZE = {
    "customcnn": (150, 150),
    "mobilenetv2": (224, 224),
    "resnet50": (224, 224),
    "vgg16": (224, 224)
}


# -------------------- HELPERS --------------------
def preprocess_for_model(img_bgr, model_name):
    target = INPUT_SIZE[model_name]
    img = cv2.resize(img_bgr, target)
    img = img.astype("float32") / 255.0
    return img

def load_model_simplified(base_path):
    """
    Tries to load model from either .h5 or .keras using tf.keras only.
    """
    candidates = [base_path + ".h5", base_path + ".keras"]
    for path in candidates:
        if os.path.exists(path):
            try:
                print(f"[INFO] Trying to load {path} with tf.keras...")
                model = load_model(path, compile=False)
                model.compile(optimizer="adam",
                              loss="binary_crossentropy",
                              metrics=["accuracy"])
                print(f"[SUCCESS] Loaded {path}")
                return model
            except Exception as e:
                print(f"[ERROR] Could not load {path}: {e}")
    return None


def list_images_by_class(data_dir):
    classes = ["NORMAL", "PNEUMONIA"]
    files = {}
    for c in classes:
        cdir = Path(data_dir) / c
        files[c] = [str(p) for p in cdir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return files

def sample_images(files_by_class, per_class, seed=42):
    rng = random.Random(seed)
    sampled = []
    for cls, files in files_by_class.items():
        if len(files) == 0:
            continue
        k = min(per_class, len(files))
        sampled_paths = rng.sample(files, k)
        for p in sampled_paths:
            sampled.append((p, cls))
    rng.shuffle(sampled)
    return sampled

def predict_one(model, image_path, model_name):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    x = preprocess_for_model(bgr, model_name)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)
    score = float(pred.ravel()[0])
    label = "PNEUMONIA" if score >= 0.5 else "NORMAL"
    return label, score, bgr

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def annotate_and_save_single(bgr, true_label, pred_label, score, out_path):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(4,4))
    title = f"True: {true_label} | Pred: {pred_label} ({score:.2f})"
    correct = (true_label.upper() == pred_label.upper())
    plt.imshow(rgb)
    plt.title(title, color=("green" if correct else "red"))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def build_panel(rows, cols, items, outfile):
    n = len(items)
    rows = rows if rows is not None else int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for i, it in enumerate(items):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(it["rgb"])
        correct = (it["true_label"].upper() == it["pred_label"].upper())
        ax.set_title(
            f"T:{it['true_label']} | P:{it['pred_label']} ({it['score']:.2f})",
            color=("green" if correct else "red"),
            fontsize=9
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


# -------------------- MAIN --------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    files_by_class = list_images_by_class(TEST_DIR)

    for model_name, base_path in MODELS.items():
        print(f"\n=== Running predictions for {model_name} ===")
        model = load_model_simplified(base_path)
        if model is None:
            print(f"[ERROR] Could not load model {model_name}")
            continue

        OUTPUT_DIR = f"results/actual_vs_pred/{model_name}"
        ensure_dir(OUTPUT_DIR)

        # Collect samples
        samples = sample_images(files_by_class, N_PER_CLASS, seed=SEED)
        if len(samples) == 0:
            print("[ERROR] No test images found!")
            continue

        ts = time.strftime("%Y%m%d-%H%M%S")
        csv_path = os.path.join(OUTPUT_DIR, f"predictions_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "true_label", "pred_label", "score"])

            panel_items = []
            for img_path, true_label in samples:
                try:
                    pred_label, score, bgr = predict_one(model, img_path, model_name)
                except Exception as e:
                    print(f"[WARNING] Failed on {img_path}: {e}")
                    continue

                writer.writerow([img_path, true_label, pred_label, f"{score:.6f}"])

                # Save single annotated image
                single_out = os.path.join(
                    OUTPUT_DIR,
                    f"{Path(img_path).stem}_T-{true_label}_P-{pred_label}_{ts}.png"
                )
                annotate_and_save_single(bgr, true_label, pred_label, score, single_out)

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                panel_items.append({
                    "rgb": rgb,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "score": score,
                    "path": img_path
                })

        # Save panel
        panel_out = os.path.join(OUTPUT_DIR, f"panel_{model_name}_{ts}.png")
        build_panel(None, GRID_COLS, panel_items, panel_out)

        print(f"[INFO] Done {model_name}. Panel -> {panel_out}")


if __name__ == "__main__":
    main()
