import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import litert_torch


class SongModel(nn.Module):
    def __init__(self, input_shape):
        super(SongModel, self).__init__()

        in_channels = input_shape[0]  # (C, H, W)

        # Conv blocks
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
    

if __name__ == "__main__":

    input_shape=(1, 30720)
    model = SongModel(input_shape=input_shape)
    summary(model, input_size=input_shape)

    dummy_input = torch.randn(1, 1, 30720, dtype=torch.float32).to("cpu")   
    print(dummy_input.shape)

    sample_inputs = (dummy_input,)

    edge_model = litert_torch.convert(model.eval(), sample_inputs)
    edge_model.export("torch2tf/dummy_gabor.tflite")