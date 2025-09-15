#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# ---- config ----
tflite_path = "../models/CNN_int8.tflite"
t_mean = 13.387054
t_std  = 8.542878
true_value = 5.3
summary_csv = "summary_inference.csv"

# ---- model ----
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
print("Input:",  in_det['shape'],  in_det['dtype'])
print("Output:", out_det['shape'], out_det['dtype'])

# ---- input (raw) ----
last_24_raw = np.array([
    5.32, 5.35, 5.48, 5.90, 6.39, 6.41, 6.44, 6.50,
    6.50, 6.73, 7.60, 8.80, 8.38, 7.50, 7.30, 6.50,
    6.28, 5.82, 5.32, 4.68, 4.50, 3.99, 4.29, 4.50
], dtype=np.float32).reshape(1, 24, 1)

# quantization params
in_scale      = float(in_det['quantization_parameters']['scales'][0]) if len(in_det['quantization_parameters']['scales']) else 1.0
in_zero_point = int(in_det['quantization_parameters']['zero_points'][0]) if len(in_det['quantization_parameters']['zero_points']) else 0
out_scale      = float(out_det['quantization_parameters']['scales'][0]) if len(out_det['quantization_parameters']['scales']) else 1.0
out_zero_point = int(out_det['quantization_parameters']['zero_points'][0]) if len(out_det['quantization_parameters']['zero_points']) else 0

# ---------- FULL E2E timing ----------
t0_full = time.perf_counter()

# 1) normalize
x_norm = (last_24_raw - t_mean) / t_std
# 2) quantize to int8 (input is int8 model)
x_q = np.rint(x_norm / in_scale + in_zero_point).astype(np.int8)
# 3) set_tensor + 4) invoke + 5) get_tensor
interpreter.set_tensor(in_det['index'], x_q)
interpreter.invoke()
y_q = interpreter.get_tensor(out_det['index'])
# 6) dequantize + 7) denormalize
y_norm = (y_q.astype(np.float32) - out_zero_point) * out_scale
y_pred_real = y_norm * t_std + t_mean

t1_full = time.perf_counter()
e2e_full_ms = (t1_full - t0_full) * 1000.0

diff = float(y_pred_real[0, 0] - true_value)

print("\n===== INT8 Full E2E =====")
print(f"Pred: {y_pred_real[0,0]:.4f} °C | True: {true_value:.4f} °C | Diff: {diff:+.4f} °C")
print(f"E2E (normalize→quantize→I/O→invoke→dequant→denorm): {e2e_full_ms:.4f} ms")


# ---------- save to CSV (two rows) ----------
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model_name = os.path.basename(tflite_path)

rows = []
for label, ms in [("e2e_simple", e2e_full_ms)]:
    rows.append({
        "model_name": model_name,
        "run_timestamp": run_timestamp,
        "inference_type": label,
        "total_samples": 1,
        "total_wall_ms": round(ms, 4),
        "avg_ms": round(ms, 4),
        "min_ms": round(ms, 4),
        "p50_ms": round(ms, 4),
        "p90_ms": round(ms, 4),
        "p95_ms": round(ms, 4),
        "p99_ms": round(ms, 4),
        "max_ms": round(ms, 4),
    })

if os.path.exists(summary_csv):
    df = pd.read_csv(summary_csv)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
else:
    df = pd.DataFrame(rows)

df.to_csv(summary_csv, index=False, float_format="%.4f")
print(f"\nAppended two rows to: {summary_csv}")
