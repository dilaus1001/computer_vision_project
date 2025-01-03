import tensorflow as tf
import numpy as np

def convert_keras_to_tflite(keras_model_path, output_path):

    print("Loading Keras model...")
    model = tf.keras.models.load_model(keras_model_path)
    
    print("Creating TFLite converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configure the converter for optimal mobile performance
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    print("Converting model to TFLite format...")
    tflite_model = converter.convert()

    print(f"Saving TFLite model to {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    tflite_model_size = len(tflite_model) / 1024 / 1024  # Size in MB
    print(f"Converted model size: {tflite_model_size:.2f} MB")
    
    return tflite_model

def verify_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model information
    print("\nModel Details:")
    print("Input Shape:", input_details[0]['shape'])
    print("Input Type:", input_details[0]['dtype'])
    print("Output Shape:", output_details[0]['shape'])
    print("Output Type:", output_details[0]['dtype'])
    
    # Create a dummy input for testing
    input_shape = input_details[0]['shape']
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    
    # Test inference
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print("\nModel successfully verified!")
    print("Test inference completed without errors.")

if __name__ == "__main__":
    keras_model_path = '/Users/francesco/Repository/computer_vision_project/models/skc_model.keras'
    tflite_output_path = '/Users/francesco/Repository/computer_vision_project/models/skc_model.tflite'
    
    # Convert the model
    converted_model = convert_keras_to_tflite(
        keras_model_path=keras_model_path,
        output_path=tflite_output_path
    )
    
    # Verify the converted model
    verify_tflite_model(tflite_output_path)