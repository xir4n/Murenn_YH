import torch
import sys
import os
import tensorflow as tf
from ai_edge_quantizer import quantizer, recipe
import yaml

from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer.utils import tfl_interpreter_utils
import litert_torch

sys.path.append(os.path.abspath(".."))
from models.network import BCMel
from torch2tf.quantize import model_quantization, clean_input_layer_name
from data.yellowhammer import TrainingDataset


def load_bcresnet_model(ckpt_path, config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print(checkpoint['state_dict'].keys())
    network = BCMel(**cfg['configuration_dict']['model_settings'])

    network_weights = {k.replace("network.", ""): v for k, v in checkpoint['state_dict'].items()}
    network.load_state_dict(network_weights)
    network.eval()

    return network

def prepare_calibration_data(model, data_dir, num_samples=500):
    dataset = TrainingDataset(data_dir)
    idx = torch.randperm(len(dataset))[:num_samples]
    dataset = [dataset[i] for i in idx]
    calibration_samples = []
    for i, sample in enumerate(dataset):
        calibration_samples.append(
            model.fb(sample['waveform']).unsqueeze(1),
        )
    return calibration_samples


if __name__ == '__main__':
    ckpt_path = "/Users/zhang/MuReNN/yellowhammer_outputs/BCResnet/BCMel/last.ckpt"
    config_path = "/Users/zhang/MuReNN/yellowhammer_outputs/BCResnet/BCMel/lightning_logs/version_0/hparams.yaml"
    prefix="torch2tf/bc_backend"

    model = load_bcresnet_model(ckpt_path, config_path)

    dummy_input = torch.randn(1, 30720)
    fb_output = model.fb(dummy_input).unsqueeze(1)
    tflite_model = litert_torch.convert(model.net, (fb_output,))
    tflite_model.export(f"{prefix}.tflite")

    data_dir = '/Users/zhang/MuReNN/YH_data_with_aug/train'
    calibration_samples = prepare_calibration_data(model, data_dir)

    model_quantization(quantization_recipe=recipe.static_wi8_ai16,
                        X_calibrate=calibration_samples,
                        tfliteFP32path=prefix+".tflite",
                        tfliteINTpath=prefix+"INT16.tflite")

    model_quantization(quantization_recipe=recipe.static_wi8_ai8,
                        X_calibrate=calibration_samples,
                        tfliteFP32path=prefix+".tflite",
                        tfliteINTpath=prefix+"INT8.tflite")



