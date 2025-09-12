#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import time  # <-- added

# ---- load int8 TFLite model ----
interpreter = tf.lite.Interpreter(model_path="models/CNN_int8.tflite")
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


# (Optional) End-to-end timing (preprocess -> invoke -> postprocess):
t0_all = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], x_q)
interpreter.invoke()
y_q = interpreter.get_tensor(output_details[0]['index'])

# ---- 4) Dequantize output ----
out_scale      = output_details[0]['quantization_parameters']['scales'][0]
out_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]


y_norm = (y_q.astype(np.float32) - out_zero_point) * out_scale
y_pred_real = y_norm * t_std + t_mean
t1_all = time.perf_counter()
print(f"End-to-end time: {(t1_all - t0_all)*1000:.3f} ms")
