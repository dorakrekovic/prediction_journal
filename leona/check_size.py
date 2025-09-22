import os

h5_path = "models/WASM_2025-09-22_14-44_RNN.h5"
tflite_path = "models/WASM_2025-09-22_14-44_RNN_float32.tflite"

print("H5 model size:     {:.2f} MB".format(os.path.getsize(h5_path) / (1024 * 1024)))
print("TFLite model size: {:.2f} MB".format(os.path.getsize(tflite_path) / (1024 * 1024)))
