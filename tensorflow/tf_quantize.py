import os
import random
import numpy as np
import tensorflow as tf

def convert_model(outfolder, model_path, name):
    # Code taken from students
    model = tf.keras.models.load_model(model_path)

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted model to a .tflite file
    with open(os.path.join(outfolder,name+'_trained_model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    # Load sample training data (adjust shape to match model input)
    def representative_dataset(size=2000):
        #TODO replace with actual data
        for i in range(size):
            sample = np.array([random.randint(0, 100)/100 for _ in range(30720)])
            sample = np.expand_dims(sample, axis=0)  # Add batch dimension
            sample = np.expand_dims(sample, axis=-1)
            yield [sample.astype(np.float32)] # Convert to float32

    # Convert to TFLite with quantization
    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable full integer quantization
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset

    # Specify full int8 quantization
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.int8
    converter_int8.inference_output_type = tf.int8

    # Convert and save
    tflite_model_int8 = converter_int8.convert()

    with open(os.path.join(outfolder,name+"_int8_model.tflite"), "wb") as f:
        f.write(tflite_model_int8)

    converter_int16 = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable optimizations
    converter_int16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int16.representative_dataset = representative_dataset

    # Set supported ops for int16 activations
    converter_int16.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]

    # Force integer only model (optional)
    converter_int16.inference_input_type = tf.int16
    converter_int16.inference_output_type = tf.int16

    tflite_model_int16 = converter_int16.convert()

    with open(os.path.join(outfolder,name+"_int16_model.tflite"), "wb") as f:
        f.write(tflite_model_int16)