import numpy as np
import tensorflow as tf

# ---- load int8 TFLite model ----
interpreter = tf.lite.Interpreter(model_path="models/CNN_int8.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Sanity check: shapes & dtypes
print("Input:",  input_details[0]['shape'],  input_details[0]['dtype'])   # should be int8
print("Output:", output_details[0]['shape'], output_details[0]['dtype'])  # should be int8

# ---- training normalization stats you saved earlier ----
# Replace with the actual values you used (or load them from disk if you saved them).
t_mean = 13.387054      # scalar
t_std  = 8.542878       # scalar (non-zero)

# ---- Prepare raw window (last 24 timesteps in original units, shape [24]) ----
# Example placeholder; replace with your real values:
last_24_raw = np.array([5.32, 5.35, 5.48, 5.9, 6.39, 6.41, 6.44, 6.5,
                        6.5, 6.73, 7.6, 8.8, 8.38, 7.5, 7.3, 6.5,
                        6.28, 5.82, 5.32, 4.68, 4.5, 3.99, 4.29, 4.5], dtype=np.float32)  # 24 raw °C values

# Reshape to model input [1, lag, n_features]
x_raw = last_24_raw.reshape(1, 24, 1).astype(np.float32)

# ---- 1) Normalize (float32, same as training) ----
x_norm = (x_raw - t_mean) / t_std

# ---- 2) Quantize to int8 using TFLite input scale/zero_point ----
in_scale      = input_details[0]['quantization_parameters']['scales'][0]
in_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]

x_q = np.rint(x_norm / in_scale + in_zero_point).astype(np.int8)
# Optional safety clip (usually not needed after rint):
# x_q = np.clip(x_q, -128, 127).astype(np.int8)

# ---- 3) Invoke ----
interpreter.set_tensor(input_details[0]['index'], x_q)
interpreter.invoke()

y_q = interpreter.get_tensor(output_details[0]['index'])  # int8

# ---- 4) Dequantize output to normalized float ----
out_scale      = output_details[0]['quantization_parameters']['scales'][0]
out_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]

y_norm = (y_q.astype(np.float32) - out_zero_point) * out_scale  # shape [1, n_ahead]

# ---- 5) Denormalize to original units (°C) ----
y_pred_real = y_norm * t_std + t_mean
y_true_real = 5.3

# --- compare ---
diff = y_pred_real - y_true_real

print(f"Predicted: {y_pred_real:} °C")
print(f"Actual:    {y_true_real:.2f} °C")
print(f"Difference: {diff:} °C")



