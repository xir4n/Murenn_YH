import re
import numpy as np
from ai_edge_quantizer import quantizer, recipe

from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer.utils import tfl_interpreter_utils

def clean_input_layer_name(name: str) -> str:
    name = name.split(":")[0]
    name = re.sub(r"^serving_default_", "", name)
    return name

def model_quantization(quantization_recipe, X_calibrate, tfliteFP32path, tfliteINTpath):
    interpreter = Interpreter(model_path=str(tfliteFP32path))

    input_details = interpreter.get_input_details()

    calibration_samples=[]

    for i, sample in enumerate(X_calibrate):
      calibration_samples.append({
            clean_input_layer_name(input_details[0]['name']): sample
        })

    calibration_data = {
        tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY: calibration_samples,
    }

    qt_static = quantizer.Quantizer(tfliteFP32path)
    qt_static.load_quantization_recipe(quantization_recipe())
    calibration_result = qt_static.calibrate(calibration_data)
    qt_static.quantize(calibration_result).export_model(tfliteINTpath, overwrite=True)

if __name__ == '__main__':
  # Quantization test
  dummy_input = np.random.randn(200, 1, 30720).astype(np.float32)
  prefix="torch2tf/dummy_gabor"
  model_quantization(quantization_recipe=recipe.static_wi8_ai16,
                     X_calibrate=dummy_input, 
                     tfliteFP32path=prefix+".tflite",
                    tfliteINTpath=prefix+"INT16.tflite")
  
  model_quantization(quantization_recipe=recipe.static_wi8_ai8,
                     X_calibrate=dummy_input, 
                     tfliteFP32path=prefix+".tflite",
                    tfliteINTpath=prefix+"INT8.tflite")