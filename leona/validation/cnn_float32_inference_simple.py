#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# ---------------------
# Config
# ---------------------
tflite_model_path = "../models/CNN_float32.tflite"   # path to your float32 model
t_mean = 13.387054  # training mean
t_std  = 8.542878   # training std
true_value = 5.3    # actual next-step temperature
summary_csv = "summary_inference.csv"
inf_type = "e2e_simple"

# ---------------------
# Load TFLite model
# ---------------------
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

print("Input spec :", in_det["shape"], in_det["dtype"])
print("Output spec:", out_det["shape"], out_det["dtype"])

# ---------------------
# Prepare input
# ---------------------
last_24_raw = np.array([
    5.32, 5.35, 5.48, 5.90, 6.39, 6.41, 6.44, 6.50,
    6.50, 6.73, 7.60, 8.80, 8.38, 7.50, 7.30, 6.50,
    6.28, 5.82, 5.32, 4.68, 4.50, 3.99, 4.29, 4.50
], dtype=np.float32)

x_raw = last_24_raw.reshape(1, 24, 1).astype(np.float32)
t0 = time.perf_counter()
x_norm = (x_raw - t_mean) / t_std

interpreter.set_tensor(in_det["index"], x_norm.astype(np.float32))
interpreter.invoke()
y_norm = interpreter.get_tensor(out_det["index"])


#invoke_ms = (t1 - t0) * 1000.0  # ms

# ---------------------
# Postprocess
# ---------------------
y_pred_real = y_norm * t_std + t_mean
t1 = time.perf_counter()
e2e_ms = (t1 - t0) * 1000.0
diff = float(y_pred_real[0, 0] - true_value)

print("\n===== Single-Step Prediction =====")
print(f"Predicted:  {y_pred_real[0,0]:.4f} °C")
print(f"Actual:     {true_value:.4f} °C")
print(f"Difference: {diff:+.4f} °C")
print(f"Inference time (E2E): {e2e_ms:.4f} ms")

# ---------------------
# Save summary to CSV
# ---------------------
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = os.path.basename(tflite_model_path)

summary_row = {
    "model_name": os.path.basename(tflite_model_path),
    "run_timestamp": run_timestamp,
    "inference_type": inf_type,
    "total_samples": 1,
    "total_wall_ms": round(e2e_ms, 4),
    "avg_ms": round(e2e_ms, 4),
    "min_ms": round(e2e_ms, 4),
    "p50_ms": round(e2e_ms, 4),
    "p90_ms": round(e2e_ms, 4),
    "p95_ms": round(e2e_ms, 4),
    "p99_ms": round(e2e_ms, 4),
    "max_ms": round(e2e_ms, 4),
}

if os.path.exists(summary_csv):
    df_summary = pd.read_csv(summary_csv)
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_row])], ignore_index=True)
else:
    df_summary = pd.DataFrame([summary_row])

df_summary.to_csv(summary_csv, index=False, float_format="%.4f")
print(f"\nSaved summary to: {summary_csv}")
