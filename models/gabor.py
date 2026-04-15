import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from speechbrain.nnet.CNN import GaborConv1d
from murenn.dtcwt.nn import ModulusStable

class GaborConv(GaborConv1d):
    """Modified GaborConv1d to support linear and mel scale initialization.
    """
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride,
        input_shape=None,
        in_channels=None,
        padding="same",
        padding_mode="constant",
        sample_rate=16000,
        min_freq=50.0,
        max_freq=None,
        n_fft=512,
        normalize_energy=False,
        bias=False,
        sort_filters=False,
        use_legacy_complex=False,
        skip_transpose=False,
        scale="log",
    ):
        super().__init__(
            out_channels,
            kernel_size,
            stride,
            input_shape,
            in_channels,
            padding,
            padding_mode,
            sample_rate,
            min_freq,
            max_freq,
            n_fft,
            normalize_energy,
            bias,
            sort_filters,
            use_legacy_complex,
            skip_transpose,
        )
        print(f"Initialized GaborConv with scale {scale}")
        self.kernel = torch.nn.Parameter(self._initialize_kernel_scales(scale))
    
    def _initialize_kernel_scales(self, scale):
        params_func = self.params_func_log_scale if scale == "log" else self.params_func_mel_scale
        center_frequencies, fwhms = params_func(self.min_freq, self.max_freq, self.filters)
        center_frequencies = center_frequencies * 2* np.pi / self.sample_rate
        inverse_bandwidths = (torch.sqrt(2.0 * torch.log(torch.tensor(2.0)))
                                * self.sample_rate / (np.pi * fwhms))
        output = torch.cat(
            [
                center_frequencies.unsqueeze(1),
                inverse_bandwidths.unsqueeze(1),
            ],
            dim=-1,
        )
        return output

    def params_func_mel_scale(self,f_min, f_max, n_mels, mel_scale='htk'):
        m_min = torchaudio.functional.functional._hz_to_mel(f_min, mel_scale=mel_scale)
        m_max = torchaudio.functional.functional._hz_to_mel(f_max, mel_scale=mel_scale)
        ordinary_mels = torch.linspace(m_min, m_max, n_mels + 2)
        ordinary_freqs = torchaudio.functional.functional._mel_to_hz(ordinary_mels, mel_scale=mel_scale)
        ordinary_Qs = 2 * ordinary_freqs[1:-1] / (ordinary_freqs[2:] - ordinary_freqs[:-2])
        xis = ordinary_freqs[1:-1]
        fwhms = xis / ordinary_Qs
        return xis, fwhms

    def params_func_log_scale(self, f_min, f_max, n_channels):
        freqs = torch.logspace(math.log2(f_min), math.log2(f_max), steps=n_channels + 2, base=2)
        Qs = 2 * freqs[1:-1] / (freqs[2:] - freqs[:-2])
        xis = freqs[1:-1]
        fwhms = xis / Qs
        return xis, fwhms
    
    def forward(self, x):
        """Returns the output of the Gabor convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve.

        Returns
        -------
        x : torch.Tensor
            The output of the Gabor convolution
        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        unsqueeze = x.ndim == 2
        if unsqueeze:
            x = x.unsqueeze(1)

        kernel = self._gabor_constraint(self.kernel)
        if self.sort_filters:
            idxs = torch.argsort(kernel[:, 0])
            kernel = kernel[idxs, :]

        filters = self._gabor_filters(kernel)
        if not self.use_legacy_complex:
            temp = torch.view_as_real(filters)
            real_filters = temp[:, :, 0]
            img_filters = temp[:, :, 1]
        else:
            real_filters = filters[:, :, 0]
            img_filters = filters[:, :, 1]
        stacked_filters = torch.cat(
            [real_filters, img_filters], dim=0
        )
        stacked_filters = stacked_filters.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(x, self.kernel_size)
        elif self.padding == "valid":
            pass
        else:
            raise ValueError(
                "Padding must be 'same' or 'valid'. Got " + self.padding
            )

        output = F.conv1d(
            x, stacked_filters, bias=self.bias, stride=self.stride, padding=0
        )
        if not self.skip_transpose:
            output = output.transpose(1, -1)
        return output

class Gabor(nn.Module):
    def __init__(self, n_filters, win_length, stride, fmin, fmax, n_fft, scale, input_shape, sample_rate):
        super(Gabor, self).__init__()
        self.gaborconv = GaborConv(
            out_channels=n_filters*2,
            kernel_size=win_length,
            stride=stride,
            input_shape=input_shape,
            in_channels=1,
            padding="same",
            padding_mode="constant",
            sample_rate=sample_rate,
            min_freq=fmin,
            max_freq=fmax,
            n_fft=n_fft,
            normalize_energy=False,
            bias=False,
            sort_filters=False,
            use_legacy_complex=True,
            skip_transpose=True,
            scale=scale,
        )
    
    def forward(self, x):
        x = self.gaborconv(x)
        x_real = x[:,:x.shape[1]//2,:]
        x_imag = x[:,x.shape[1]//2:,:]
        x = ModulusStable.apply(x_real, x_imag)
        x = torch.flip(x, dims=(1,))
        return x




if __name__ == "__main__":
    x = torch.randn(4, 1, 160000)
    gabor = Gabor(
        n_filters=64,
        win_length=401,
        stride=1,
        fmin=50,
        n_fft=512,
        sort_filters=False,
        scale="log",
    )
    y = gabor(x)
    print(y.shape)
