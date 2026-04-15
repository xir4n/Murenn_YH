import math
import torch
import torch.nn as nn
import murenn
from models.udtcwt import UDTCWTDirect as UDTCWT
from murenn.dtcwt.nn import ModulusStable
from models.gabor import Gabor
from torchaudio.transforms import MelSpectrogram
from models.bcresnet import BCResNets


class BCMel(nn.Module):
    def __init__(self, J, Q, T, J_phi, use_power):
        super().__init__()
        self.fb = MelSpectrogram(
            sample_rate=20480,
            n_fft=512,
            win_length=101,
            hop_length=64,
            f_min=3000,
            f_max=10000,
            n_mels=32,
            mel_scale="htk",
            power=1.0,
        )
        self.net = BCResNets(
            base_c = 16,
            num_classes=6,
        )
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        x = self.fb(x).unsqueeze(1) 
        x = self.net(x)
        return x


class BCGabor(nn.Module):
    def __init__(self, J, Q, T, J_phi, use_power):
        super().__init__()
        self.fb = Gabor(
            n_filters=32,
            win_length=101,
            stride=64,
            fmin=3000,
            fmax=10000,
            n_fft=512,
            scale="mel",
            sample_rate=20480,
            input_shape=30720,
        )
        self.net = BCResNets(
            base_c = 16,
            num_classes=6,
        )
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.fb(x).unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)
        x = self.net(x)
        return x


class BCConv1d(nn.Module):
    def __init__(self, J, Q, T, J_phi, use_power):
        super().__init__()
        self.fb = torch.nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=101,
            stride=64,
            padding=50,
            bias=False,
        )
        self.net = BCResNets(
            base_c = 16,
            num_classes=6,
        )
        self.fc = nn.Linear(64, 6)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.fb(x).unsqueeze(1)
        x = self.net(x)
        return x