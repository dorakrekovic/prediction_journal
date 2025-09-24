#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
script_start = time.perf_counter()
run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

# ======================
# Config
# ======================
hardware = "rpi4"
location = "Suhopolje"
excel_path = f"../../{location}2021.xlsx"

tflite_path_pattern = "../models/CNN_float32.tflite"  # exact path or glob pattern


inf_type = "e2e"  # label saved to summary

features_final = ["t2m"]
lag = 24
n_ahead = 1

# Normalization stats source (match training!)
NORMALIZE_ON = "train"     # "train" or "all"
train_stats_share = 0.9

# CSV outputs
save_csv = True
csv_out_detailed = f"{location}_cnn_float32_full_e2e_results.csv"
summary_csv = "summary_inference_NEW.csv"

# Optional benchmarking knobs
WARMUP_RUNS = 5            # a few warmups can stabilize timings
PRINT_PER_SAMPLE = False   # True = print each sample's E2E time

# ======================
# Helpers
# ======================
def load_latest_tflite(pattern: str) -> str:
    if os.path.exists(pattern):
        return pattern
    cand = sorted(glob.glob(pattern))
    if not cand:
        raise FileNotFoundError(f"No TFLite files match: {pattern}")
    return cand[-1]

def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else None

# ======================
# 1) Load RAW data
# ======================
d = pd.read_excel(excel_path)
d["datetime"] = pd.to_datetime(d["datetime"], format="%d/%m/%Y %H:%M")
d.sort_values("datetime", inplace=True)
d = d.groupby("datetime", as_index=False)[features_final].mean()

ts = d[features_final].copy()   # RAW series (°C)
nrows = len(ts)

# ======================
# 2) Compute normalization stats (match training)
# ======================
if NORMALIZE_ON == "train":
    cut_stats = int(nrows * train_stats_share)
    train_for_stats = ts.iloc[:cut_stats].copy()
    train_mean = train_for_stats.mean()
    train_std  = train_for_stats.std(ddof=0).replace(0, 1.0)
elif NORMALIZE_ON == "all":
    train_mean = ts.mean()
    train_std  = ts.std(ddof=0).replace(0, 1.0)
else:
    raise ValueError("NORMALIZE_ON must be 'train' or 'all'.")

print("Normalization stats used:")
for col in train_mean.index:
    print(f"{col}: mean={train_mean[col]:.6f}, std={train_std[col]:.6f}")

# Normalized series only for target alignment (E2E loop uses RAW each time)
ts_norm = (ts - train_mean) / train_std

# ----------------------
# Supervised dimensions
# ----------------------
num_samples = nrows - lag - n_ahead
if num_samples <= 0:
    raise ValueError("Not enough rows for the chosen lag and n_ahead.")

# Targets (normalized), then we’ll denormalize for y_true
Y_all_norm = ts_norm.values[lag : nrows - n_ahead, 0][:, None]  # shape (num_samples, 1)

# Timestamps aligned to samples (drop first lag+n_ahead)
ts_datetimes_all = d["datetime"].iloc[lag + n_ahead :].reset_index(drop=True)

# Sanity checks
assert Y_all_norm.shape[0] == num_samples
assert len(ts_datetimes_all) == num_samples

# ======================
# 3) Load TFLite model
# ======================
tflite_model_path = load_latest_tflite(tflite_path_pattern)
#print(f"Using TFLite model: {tflite_model_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]
#print("Input spec :", in_det["shape"], in_det["dtype"])
#print("Output spec:", out_det["shape"], out_det["dtype"])

# Sanity: CNN float32 model should expect float32 input/output
if in_det["dtype"] != np.float32 or out_det["dtype"] != np.float32:
    print("Warning: Model I/O dtypes aren’t float32. Script still runs, "
          "but it’s intended for CNN float32 models.")

# ======================
# 4) E2E per-sample timing from RAW → prediction
# ======================
preds_real = []
t_prep_ms, t_io_ms, t_post_ms, t_total_ms = [], [], [], []

# Warm-up (optional)
if WARMUP_RUNS and nrows > lag:
    x_raw_w = ts.values[:lag].astype(np.float32).reshape(1, lag, 1)
    x_norm_w = (x_raw_w - float(train_mean["t2m"])) / float(train_std["t2m"])
    interpreter.set_tensor(in_det["index"], x_norm_w.astype(np.float32))
    for _ in range(WARMUP_RUNS):
        interpreter.invoke()
        _ = interpreter.get_tensor(out_det["index"])

t0_loop = time.perf_counter_ns()

for i in range(num_samples):
    # ---------- start E2E ----------
    t0 = time.perf_counter_ns()

    # 1) RAW window (°C)
    x_raw = ts.values[i : i + lag].astype(np.float32).reshape(1, lag, 1)

    # 2) normalize (timed)
    x_norm = (x_raw - float(train_mean["t2m"])) / float(train_std["t2m"])
    t1 = time.perf_counter_ns()

    # 3) set → invoke → get
    t2 = time.perf_counter_ns()
    interpreter.set_tensor(in_det["index"], x_norm.astype(np.float32))
    interpreter.invoke()
    y_out = interpreter.get_tensor(out_det["index"])
    t3 = time.perf_counter_ns()

    # 4) denormalize to °C (post)
    y_real = y_out.astype(np.float32) * float(train_std["t2m"]) + float(train_mean["t2m"])
    preds_real.append(float(y_real[0, 0]))
    t4 = time.perf_counter_ns()

    # times
    t_prep_ms.append((t1 - t0) / 1e6)     # raw->normalize
    t_io_ms.append((t3 - t2) / 1e6)       # set/invoke/get
    t_post_ms.append((t4 - t3) / 1e6)     # denormalize
    t_total_ms.append((t4 - t0) / 1e6)    # full E2E

    if PRINT_PER_SAMPLE:
        print(f"{i:6d}  total_e2e={t_total_ms[-1]:.4f} ms  "
              f"(prep={t_prep_ms[-1]:.4f}, io={t_io_ms[-1]:.4f}, post={t_post_ms[-1]:.4f})")

t1_loop = time.perf_counter_ns()
total_wall_ms = (t1_loop - t0_loop) / 1e6

preds_real = np.array(preds_real, dtype=np.float32)
ytrue_real = (Y_all_norm.astype(np.float32) * float(train_std["t2m"]) + float(train_mean["t2m"])).ravel()

# Sanity checks before DataFrame
assert len(preds_real) == num_samples
assert len(ytrue_real) == num_samples
assert len(ts_datetimes_all) == num_samples

# ======================
# Timing summary
# ======================
def stats(name, arr_ms):
    if not arr_ms:
        print(f"{name}: no samples")
        return
    a = np.array(arr_ms, dtype=np.float64)
    print(f"{name:14s} avg={a.mean():8.4f} ms | min={a.min():8.4f} | "
          f"p50={np.percentile(a,50):8.4f} | p90={np.percentile(a,90):8.4f} | "
          f"p95={np.percentile(a,95):8.4f} | p99={np.percentile(a,99):8.4f} | "
          f"max={a.max():8.4f}")

print(f"\nTotal loop wall time: {total_wall_ms:.4f} ms for {len(t_total_ms)} samples")
if total_wall_ms > 0 and len(t_total_ms) > 0:
    print(f"Throughput (E2E): {len(t_total_ms) / (total_wall_ms/1000.0):.2f} samples/sec")
stats("preprocess", t_prep_ms)
stats("io+compute", t_io_ms)
stats("postprocess", t_post_ms)
stats("total_e2e", t_total_ms)

# ======================
# 5) Save CSVs (rounded to 4 decimals)
# ======================
if save_csv and len(preds_real) > 0:
    # Per-sample detailed CSV
    detailed_df = pd.DataFrame({
        "datetime": ts_datetimes_all,
        "y_true_C": np.round(ytrue_real, 4),
        "y_pred_C": np.round(preds_real, 4),
        "error_C":  np.round(preds_real - ytrue_real, 4),
        "prep_ms":       np.round(t_prep_ms, 4),
        "io_compute_ms": np.round(t_io_ms, 4),
        "post_ms":       np.round(t_post_ms, 4),
        "total_e2e_ms":  np.round(t_total_ms, 4),
    })
    detailed_df.to_csv(csv_out_detailed, index=False, float_format="%.4f")
    #print(f"Saved detailed results to: {csv_out_detailed}")

    # Append one summary row per run
    a_tot = np.array(t_total_ms, dtype=np.float64)
    script_end = time.perf_counter()
    summary_row = {
        "model_name": os.path.basename(tflite_model_path),
        "run_timestamp": run_timestamp,
        "inference_type": inf_type,
        "total_samples": int(len(a_tot)),
        "total_wall_ms": round(total_wall_ms, 4),
        "avg_ms": round(a_tot.mean(), 4) if len(a_tot) else None,
        "min_ms": round(a_tot.min(), 4) if len(a_tot) else None,
        "p50_ms": round(pct(a_tot, 50), 4) if len(a_tot) else None,
        "p90_ms": round(pct(a_tot, 90), 4) if len(a_tot) else None,
        "p95_ms": round(pct(a_tot, 95), 4) if len(a_tot) else None,
        "p99_ms": round(pct(a_tot, 99), 4) if len(a_tot) else None,
        "max_ms": round(a_tot.max(), 4) if len(a_tot) else None,
        "end_time": f"{script_end - script_start:.4f}s",
        "end_timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "hw": hardware,
    }

    if os.path.exists(summary_csv):
        df_sum = pd.read_csv(summary_csv)
        df_sum = pd.concat([df_sum, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df_sum = pd.DataFrame([summary_row])
    df_sum.to_csv(summary_csv, index=False, float_format="%.4f")
    #print(f"Saved summary to: {summary_csv}")

script_end2 = time.perf_counter()

print(f"\n[Script Runtime] {script_end2 - script_start:.4f} seconds total")
