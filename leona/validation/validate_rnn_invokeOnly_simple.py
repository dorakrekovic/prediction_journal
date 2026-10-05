#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# -------------------------
# Config
# -------------------------

tflite_model_path = f"../models/RNN_temp_in_C_new.tflite"  # exact path or glob OK
inf_type = "invoke_simple"
lag = 24
n_ahead = 1

# Training normalization stats (must match the model's training)
t_mean = 13.387054
t_std  = 8.542878


# 24 inputs + true next value
last_24_raw = [
    5.32, 5.35, 5.48, 5.9, 6.39, 6.41, 6.44, 6.5, 6.5, 6.73, 7.6, 8.8,
    8.38, 7.5, 7.3, 6.5, 6.28, 5.82, 5.32, 4.68, 4.5, 3.99, 4.29, 4.5
]
# Input window (24 values) and the true next value to compare against
last_24_raw = np.array([
    5.32, 5.35, 5.48, 5.90, 6.39, 6.41, 6.44, 6.50,
    6.50, 6.73, 7.60, 8.80, 8.38, 7.50, 7.30, 6.50,
    6.28, 5.82, 5.32, 4.68, 4.50, 3.99, 4.29, 4.50
], dtype=np.float32)
y_true_next_c = 5.3

# Summary CSV (one row appended per run)
summary_csv = "summary_inference.csv"

# -------------------------
# Helpers
# -------------------------
def quantize_to_int8(x_norm: np.ndarray, in_details) -> np.ndarray:
    q = in_details["quantization_parameters"]
    if len(q["scales"]) == 0:
        return x_norm.astype(np.int8)
    scale = float(q["scales"][0])
    zp    = int(q["zero_points"][0])
    return np.rint(x_norm / scale + zp).astype(np.int8)

def dequantize_from_int8(y_q: np.ndarray, out_details) -> np.ndarray:
    q = out_details["quantization_parameters"]
    if len(q["scales"]) == 0:
        return y_q.astype(np.float32)
    scale = float(q["scales"][0])
    zp    = int(q["zero_points"][0])
    return (y_q.astype(np.float32) - zp) * scale

# -------------------------
# Sanity checks
# -------------------------
assert last_24_raw.shape[0] == lag, f"Need exactly {lag} values for a {lag}-lag model."

# -------------------------
# Load TFLite model
# -------------------------
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

print("Input spec :", in_det["shape"], in_det["dtype"])
print("Output spec:", out_det["shape"], out_det["dtype"])

# -------------------------
# Build raw input (shape [1, lag, 1])
# -------------------------
x_raw = last_24_raw.reshape(1, lag, 1)

# -------------------------
# E2E inference_scripts timing
# (normalize -> maybe quantize -> set_tensor -> invoke -> get_tensor -> maybe dequantize -> denormalize)
# -------------------------
t0_all = time.perf_counter()

# preprocess
x_norm = (x_raw - t_mean) / t_std
if in_det["dtype"] == np.float32:
    x_in = x_norm.astype(np.float32)
elif in_det["dtype"] == np.int8:
    x_in = quantize_to_int8(x_norm.astype(np.float32), in_det)
else:
    raise TypeError(f"Unsupported input dtype: {in_det['dtype']}")

# io + compute
interpreter.set_tensor(in_det["index"], x_in)
t0_inv = time.perf_counter()
interpreter.invoke()
t1_inv = time.perf_counter()
y_out = interpreter.get_tensor(out_det["index"])

# postprocess
if out_det["dtype"] == np.float32:
    y_norm = y_out.astype(np.float32)
elif out_det["dtype"] == np.int8:
    y_norm = dequantize_from_int8(y_out, out_det)
else:
    raise TypeError(f"Unsupported output dtype: {out_det['dtype']}")

y_pred_c = float(y_norm[0, 0] * t_std + t_mean)
t1_all = time.perf_counter()

invoke_only_ms = (t1_inv - t0_inv) * 1000.0
e2e_ms         = (t1_all - t0_all) * 1000.0

diff = y_pred_c - y_true_next_c

print("\n===== One-Step Prediction =====")
print(f"Predicted:  {y_pred_c:.4f} °C")
print(f"Actual:     {y_true_next_c:.4f} °C")
print(f"Difference: {diff:+.4f} °C")
print(f"Invoke-only: {invoke_only_ms:.4f} ms")
print(f"E2E:         {e2e_ms:.4f} ms")


# -------------------------
# 5) Save ONE summary row to CSV (4 decimals)
# -------------------------
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = os.path.basename(tflite_model_path)

summary_row = {
    "model_name": model_name,
    "run_timestamp": run_timestamp,
    "inference_type": inf_type,
    "total_samples": 1,
    "total_wall_ms": round(e2e_ms, 4),  # E2E for this single sample
    "avg_ms": round(invoke_only_ms, 4),
    "min_ms": round(invoke_only_ms, 4),
    "p50_ms": round(invoke_only_ms, 4),
    "p90_ms": round(invoke_only_ms, 4),
    "p95_ms": round(invoke_only_ms, 4),
    "p99_ms": round(invoke_only_ms, 4),
    "max_ms": round(invoke_only_ms, 4),
}

if os.path.exists(summary_csv):
    df_summary = pd.read_csv(summary_csv)
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_row])], ignore_index=True)
else:
    df_summary = pd.DataFrame([summary_row])

df_summary.to_csv(summary_csv, index=False, float_format="%.4f")
print(f"\nSaved summary to: {summary_csv}")
