#!/usr/bin/env python3
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import time  # <-- added

# ---- load int8 TFLite model ----
tflite_model_path = "../models/CNN_int8.tflite"
inf_type = "invoke_simple"
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sanity check: shapes & dtypes
print("Input:",  input_details[0]['shape'],  input_details[0]['dtype'])   # should be int8
print("Output:", output_details[0]['shape'], output_details[0]['dtype'])  # should be int8

# ---- training normalization stats ----
t_mean = 13.387054
t_std  = 8.542878

# ---- Prepare raw window ----
last_24_raw = np.array([5.32, 5.35, 5.48, 5.9, 6.39, 6.41, 6.44, 6.5,
                        6.5, 6.73, 7.6, 8.8, 8.38, 7.5, 7.3, 6.5,
                        6.28, 5.82, 5.32, 4.68, 4.5, 3.99, 4.29, 4.5], dtype=np.float32)
x_raw = last_24_raw.reshape(1, 24, 1).astype(np.float32)

# ---- 1) Normalize ----
x_norm = (x_raw - t_mean) / t_std

# ---- 2) Quantize to int8 ----
in_scale      = input_details[0]['quantization_parameters']['scales'][0]
in_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
x_q = np.rint(x_norm / in_scale + in_zero_point).astype(np.int8)

# (Optional) one warm-up invoke to avoid first-run overhead:
# interpreter.set_tensor(input_details[0]['index'], x_q)
# interpreter.invoke()
# _ = interpreter.get_tensor(output_details[0]['index'])

# ---- 3) Invoke + measure model time only ----
interpreter.set_tensor(input_details[0]['index'], x_q)

t0 = time.perf_counter()
interpreter.invoke()
t1 = time.perf_counter()

invoke_ms = (t1 - t0) * 1000.0
print(f"Inference time (invoke only): {invoke_ms:.3f} ms")

y_q = interpreter.get_tensor(output_details[0]['index'])  # int8

# ---- 4) Dequantize output ----
out_scale      = output_details[0]['quantization_parameters']['scales'][0]
out_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
y_norm = (y_q.astype(np.float32) - out_zero_point) * out_scale  # [1, n_ahead]

# ---- 5) Denormalize to °C ----
y_pred_real = y_norm * t_std + t_mean
y_true_real = 5.3
diff = y_pred_real - y_true_real

print(f"Predicted:  {y_pred_real} °C")
print(f"Actual:     {y_true_real:.2f} °C")
print(f"Difference: {diff} °C")

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = os.path.basename(tflite_model_path)
summary_csv = f"summary_inference.csv"

summary_row = {
    "model_name": model_name,
    "run_timestamp": run_timestamp,
    "inference_type": inf_type,
    "total_samples": 1,
    "total_wall_ms": round(invoke_ms, 4),   # same as invoke-only here
    "avg_ms": round(invoke_ms, 4),
    "min_ms": round(invoke_ms, 4),
    "p50_ms": round(invoke_ms, 4),
    "p90_ms": round(invoke_ms, 4),
    "p95_ms": round(invoke_ms, 4),
    "p99_ms": round(invoke_ms, 4),
    "max_ms": round(invoke_ms, 4),
}

if os.path.exists(summary_csv):
    df_summary = pd.read_csv(summary_csv)
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_row])], ignore_index=True)
else:
    df_summary = pd.DataFrame([summary_row])

df_summary.to_csv(summary_csv, index=False, float_format="%.4f")
print(f"Saved summary to: {summary_csv}")
