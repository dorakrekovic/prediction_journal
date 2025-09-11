import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="Leona_new.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=== Input Tensor Details ===")
for detail in input_details:
    print(f"Index: {detail['index']}")
    print(f"Name: {detail['name']}")
    print(f"Type: {detail['dtype']}")
    print(f"Scale: {detail['quantization'][0]}")
    print(f"Zero Point: {detail['quantization'][1]}")
    print("-" * 40)

print("\n=== Output Tensor Details ===")
for detail in output_details:
    print(f"Index: {detail['index']}")
    print(f"Name: {detail['name']}")
    print(f"Type: {detail['dtype']}")
    print(f"Scale: {detail['quantization'][0]}")
    print(f"Zero Point: {detail['quantization'][1]}")
    print("-" * 40)
