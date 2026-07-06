"""
Production Batch Processor – Skip non‑155‑band files
Only processes shapes: (15,15,155) or (155,15,15)
"""

import os
import json
import csv
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model

# ---------- Configuration ----------
MODEL_PATH = r"C:\Users\okyer001\PycharmProjects\pythonProject\models\model.keras"
INPUT_DIR = r"C:\Frank\New folder\Quinoa_dataset-hyper"
OUTPUT_DIR = r"./output_final_skip"
CLASS_NAMES = ['HNHP', 'HNLP', 'LNHP', 'LNLP']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "failed"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "skipped"), exist_ok=True)

print("Loading model...")
model = load_model(MODEL_PATH)
print(f"Model loaded. Input shape: {model.input_shape}")

# ---------- Helpers ----------

def apply_snv(image):
    mean = np.mean(image, axis=2, keepdims=True)
    std = np.std(image, axis=2, keepdims=True) + 1e-8
    return (image - mean) / std

def load_and_fix(filepath):
    """
    Load image and convert to (15,15,155) if possible.
    Returns (fixed_image, error_message) – if error_message not None, skip.
    """
    try:
        if filepath.lower().endswith('.npy'):
            img = np.load(filepath).astype(np.float32)
        else:
            import tifffile as tiff
            img = tiff.imread(filepath).astype(np.float32)
    except Exception as e:
        return None, f"Load error: {e}"

    # Already correct
    if img.shape == (15, 15, 155):
        return img, None

    # Transpose from (155,15,15) to (15,15,155)
    if img.shape == (155, 15, 15):
        return np.moveaxis(img, 0, -1), None

    # Any other shape -> skip
    return None, f"Unsupported shape: {img.shape}"

def quality_check(image, filepath):
    issues = []
    passed = True
    if image.shape != (15, 15, 155):
        issues.append(f"Shape {image.shape} != (15,15,155)")
        passed = False
    nan_count = np.isnan(image).sum()
    if nan_count > 0:
        issues.append(f"NaN count: {nan_count}")
        passed = False
    if np.max(image) > 10000 or np.min(image) < -10:
        issues.append(f"Outliers: min={np.min(image):.2f}, max={np.max(image):.2f}")
        passed = False
    flat = image.reshape(-1, 155)
    noise = np.median(np.abs(flat - np.median(flat, axis=0)), axis=0)
    signal = np.median(flat, axis=0)
    snr = signal / (noise + 1e-8)
    mean_snr = np.mean(snr)
    if mean_snr < 1.0:
        issues.append(f"Low SNR: {mean_snr:.2f}")
        passed = False
    return {"passed": passed, "issues": issues, "snr": float(mean_snr)}

def process_file(filepath):
    result = {
        "file": str(filepath),
        "status": "failed",
        "timestamp": datetime.now().isoformat()
    }

    # Load and fix
    image, error = load_and_fix(filepath)
    if image is None:
        result["error"] = error or "Unsupported shape"
        result["skipped"] = True
        return result

    # QA
    qa = quality_check(image, filepath)
    result["qa"] = qa
    if not qa["passed"]:
        result["error"] = f"QA failed: {qa['issues']}"
        return result

    # Preprocess
    processed = apply_snv(image)
    input_batch = np.expand_dims(processed, axis=0)

    # Predict
    start = time.time()
    probs = model.predict(input_batch, verbose=0)
    latency = (time.time() - start) * 1000

    pred_class = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])

    result.update({
        "status": "success",
        "prediction": CLASS_NAMES[pred_class],
        "class_id": pred_class,
        "confidence": confidence,
        "probabilities": {name: float(probs[0][i]) for i, name in enumerate(CLASS_NAMES)},
        "latency_ms": round(latency, 2)
    })

    # Save JSON
    out_path = os.path.join(OUTPUT_DIR, "predictions", Path(filepath).stem + ".json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result

# ---------- Main ----------
def batch_process():
    # Collect files
    file_list = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(('.npy', '.tiff', '.tif')):
                file_list.append(os.path.join(root, f))

    print(f"Found {len(file_list)} files.")
    if not file_list:
        print("No files found.")
        return

    all_results = []
    total_start = time.time()
    shape_stats = {}

    for idx, filepath in enumerate(file_list, 1):
        print(f"[{idx}/{len(file_list)}] {os.path.basename(filepath)}")
        res = process_file(filepath)
        all_results.append(res)

        # Log if skipped or failed
        if res["status"] == "failed":
            # Save detailed log
            log_path = os.path.join(OUTPUT_DIR, "failed", Path(filepath).stem + ".log")
            with open(log_path, 'w') as f:
                json.dump(res, f, indent=2)
        elif res.get("skipped", False):
            # Also log skipped files
            skip_path = os.path.join(OUTPUT_DIR, "skipped", Path(filepath).stem + ".skip")
            with open(skip_path, 'w') as f:
                json.dump(res, f, indent=2)

    total_time = time.time() - total_start

    # Summary
    success = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed" and not r.get("skipped"))
    skipped = sum(1 for r in all_results if r.get("skipped", False))
    qa_fail = sum(1 for r in all_results if r.get("qa", {}).get("passed") == False)

    summary = {
        "total_files": len(file_list),
        "successful": success,
        "failed": failed,
        "skipped": skipped,
        "qa_failed": qa_fail,
        "total_time_seconds": round(total_time, 2),
        "timestamp": datetime.now().isoformat()
    }

    csv_path = os.path.join(OUTPUT_DIR, "batch_summary_final_skip.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    batch_process()