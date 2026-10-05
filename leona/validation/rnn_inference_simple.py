#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

script_start = time.perf_counter()
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# -------------------------
# Config
# -------------------------
hardware = "rpi4"
location = "Suhopolje"
tflite_model_path = f"../models/RNN_temp_in_C_new.tflite"
lag = 24
n_ahead = 1
inf_type = "e2e_simple"   # label for inference_scripts type
summary_csv = "summary_inference_NEW.csv"

# Input window and ground truth
last_24_raw = [
    5.32, 5.35, 5.48, 5.9, 6.39, 6.41, 6.44, 6.5, 6.5, 6.73, 7.6, 8.8,
    8.38, 7.5, 7.3, 6.5, 6.28, 5.82, 5.32, 4.68, 4.5, 3.99, 4.29, 4.5
]
assert len(last_24_raw) == lag, "Need exactly 24 values for a 24-lag model."
y_true_next_c = 5.3

# Normalization stats (from training!)
t_mean = 13.387054
t_std  = 8.542878

# -------------------------
# Helpers
# -------------------------
def load_latest_tflite(path_pattern: str) -> str:
    if os.path.exists(path_pattern):
        return path_pattern
    cand = sorted(glob.glob(path_pattern))
    if not cand:
        raise FileNotFoundError(f"No TFLite files found for: {path_pattern}")
    return cand[-1]

def quantize_to_int8(x_norm: np.ndarray, in_details) -> np.ndarray:
    q = in_details["quantization_parameters"]
    scale = float(q["scales"][0]) if len(q["scales"]) else 1.0
    zp    = int(q["zero_points"][0]) if len(q["zero_points"]) else 0
    return np.rint(x_norm / scale + zp).astype(np.int8)

def dequantize_from_int8(y_q: np.ndarray, out_details) -> np.ndarray:
    q = out_details["quantization_parameters"]
    scale = float(q["scales"][0]) if len(q["scales"]) else 1.0
    zp    = int(q["zero_points"][0]) if len(q["zero_points"]) else 0
    return (y_q.astype(np.float32) - zp) * scale

# -------------------------
# 1) Load model
# -------------------------
tflite_path = load_latest_tflite(tflite_model_path)
#print(f"Using TFLite model: {tflite_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# -------------------------
# 2) Prepare input
# -------------------------
x_raw = np.array(last_24_raw, dtype=np.float32).reshape(1, lag, 1)

# -------------------------
# 3) Run inference_scripts (E2E timing)
# -------------------------
times_ms = []
preds_real = []

for _ in range(1):  # single sample, but loop for uniformity
    t0 = time.perf_counter()

    # preprocess
    x_norm = (x_raw - t_mean) / t_std
    if in_det["dtype"] == np.float32:
        x_in = x_norm.astype(np.float32)
    elif in_det["dtype"] == np.int8:
        x_in = quantize_to_int8(x_norm.astype(np.float32), in_det)
    else:
        raise TypeError(f"Unsupported input dtype: {in_det['dtype']}")

    # inference_scripts
    interpreter.set_tensor(in_det["index"], x_in)
    interpreter.invoke()
    y_out = interpreter.get_tensor(out_det["index"])

    # postprocess
    if out_det["dtype"] == np.float32:
        y_norm = y_out.astype(np.float32)
    elif out_det["dtype"] == np.int8:
        y_norm = dequantize_from_int8(y_out, out_det)
    else:
        raise TypeError(f"Unsupported output dtype: {out_det['dtype']}")

    y_pred_c = float(y_norm[0, 0] * t_std + t_mean)
    preds_real.append(y_pred_c)

    t1 = time.perf_counter()
    times_ms.append((t1 - t0) * 1000.0)

preds_real = np.array(preds_real, dtype=np.float32)
a = np.array(times_ms)

# -------------------------
# 4) Metrics
# -------------------------
diff = preds_real[0] - y_true_next_c
#print(f"Pred: {preds_real[0]:.4f} °C | True: {y_true_next_c:.4f} °C | Diff: {diff:+.4f} °C")
print(f"E2E time: {a[0]:.4f} ms")

# -------------------------
# 5) Save summary row
# -------------------------

script_end = time.perf_counter()
summary_row = {
    "model_name": os.path.basename(tflite_path),
    "run_timestamp": run_timestamp,
    "inference_type": inf_type,
    "total_samples": len(preds_real),
    "total_wall_ms": round(a.sum(), 4),
    "avg_ms": round(a.mean(), 4) if len(a) else None,
    "min_ms": round(a.min(), 4) if len(a) else None,
    "p50_ms": round(np.percentile(a, 50), 4) if len(a) else None,
    "p90_ms": round(np.percentile(a, 90), 4) if len(a) else None,
    "p95_ms": round(np.percentile(a, 95), 4) if len(a) else None,
    "p99_ms": round(np.percentile(a, 99), 4) if len(a) else None,
    "max_ms": round(a.max(), 4) if len(a) else None,
    "end_time": f"{script_end - script_start:.4f}s",
    "end_timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
    "hw": hardware,
}

if os.path.exists(summary_csv):
    df_summary = pd.read_csv(summary_csv)
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_row])], ignore_index=True)
else:
    df_summary = pd.DataFrame([summary_row])

df_summary.to_csv(summary_csv, index=False, float_format="%.4f")
#print(f"\nSaved summary to: {summary_csv}")


script_end2 = time.perf_counter()

print(f"\n[Script Runtime] {script_end2 - script_start:.4f} seconds total")
