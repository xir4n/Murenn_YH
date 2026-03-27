'''
This script defines the MurennLayer class, which is a custom Pytorch Module that implements the hybrid filterbank layer 
named MuReNN.
'''
import math
import torch
import torch.nn as nn
import murenn
from murenn import UDTCWT
from murenn.dtcwt.nn import ModulusStable


class Downsampling(torch.nn.Module):
    """
    Downsample the input signal by a factor of 2**J_phi.
    --------------------
    Args:
        J_phi (int): Number of levels of downsampling.
    """
    def __init__(self, J_phi):
        super().__init__()
        self.J_phi = J_phi
        self.phi = murenn.DTCWT(
            J=1,
            level1="near_sym_b",
            skip_hps=True,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        for j in range(self.J_phi):
            x, _ = self.phi(x)
            x = x[:,:,::2]
        return x


class PowerStable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta):
        x = x.clamp(min=-beta)
        output = (x + beta) ** alpha 
        # output = x ** alpha
        ctx.save_for_backward(x, alpha, beta, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, beta, output = ctx.saved_tensors
        dx, dalpha, dbeta = None, None, None
        if ctx.needs_input_grad[0]:
            dx = grad_output * alpha * ((x+beta) ** (alpha - 1))
            dx.masked_fill_(output == 0, 0)
        if ctx.needs_input_grad[1]:
            dalpha = grad_output * output * torch.log((x+beta))
            dalpha.masked_fill_(output == 0, 0)
        if ctx.needs_input_grad[2]:
            dbeta = grad_output * alpha * ((x+beta) ** (alpha - 1))
            dbeta.masked_fill_(output == 0, 0)
        return dx, dalpha, dbeta

class MuReNNLayer(torch.nn.Module):
    """
    Convolve with psi, then apply modulus, and downsample.
    inputs: (batch, in_channels, time)
    outputs: (batch, in_channels * sum(Q), time / 2**J_phi
    """
    def __init__(self, *, J, Q, T, in_channels, J_phi, use_conv1d=True, use_power=True,):
        super().__init__()
        if isinstance(Q, int):
            self.Q = [Q for j in range(J)]
        elif isinstance(Q, list):
            assert len(Q) == J
            self.Q = Q
        else:
            raise TypeError(f"Q must to be int or list, got {type(Q)}")
        self.T = T
        self.in_channels = in_channels
        self.dtcwt = UDTCWT(J=J)
        if use_conv1d:
            conv1d = []
            for j in range(J):
                conv1d_j = torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.Q[j] * in_channels,
                    kernel_size=self.T,
                    bias=False,
                    padding="same",
                    dilation=2**(j -1) if j > 0 else 1,
                    groups=in_channels,
                )
                conv1d.append(conv1d_j)
            self.conv1d = torch.nn.ParameterList(conv1d)

        self.down = Downsampling(J_phi)
        self.use_power = use_power
        if self.use_power:
            self.root = torch.nn.ParameterList([])
            self.beta = torch.nn.Parameter(torch.zeros(1))
            for j in range(self.dtcwt.J):
                self.root.append(nn.Parameter(torch.ones(1, self.Q[j] * self.in_channels, 1) * 1))
        self.use_conv1d = use_conv1d
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert self.in_channels == x.shape[1]
        lp, bps = self.dtcwt(x)
        u_psi_x = []
        for j in range(self.dtcwt.J):
            xj = bps[j]
            if self.use_conv1d:
                Wx_j_r = self.conv1d[j](xj.real) / math.sqrt(2) ** j
                Wx_j_i = self.conv1d[j](xj.imag) / math.sqrt(2) ** j
            else:
                Wx_j_r = bps[j].real / math.sqrt(2) ** j
                Wx_j_i = bps[j].imag / math.sqrt(2) ** j
            u_psi_x_j = ModulusStable.apply(Wx_j_r, Wx_j_i)

            if self.use_power:
                u_psi_x_j = PowerStable.apply(u_psi_x_j, self.sigmoid(self.root[j]), self.beta)

            u_psi_x_j = self.down(u_psi_x_j)
            u_psi_x.append(u_psi_x_j)
        u_psi_x = torch.cat(u_psi_x, dim=1)
        return u_psi_x
    


if __name__ == "__main__":
    murenn_layer = MuReNNLayer(J=2, Q=16, T=32, in_channels=1, J_phi=6, use_conv1d=True, use_power=True)
    x = torch.randn(1, 1, 2**12)
    y = murenn_layer(x)
    print("------- Layer Architecture -------")
    print(murenn_layer)
    print("------- Input Shape -------")
    print(x.shape)
    print("------- Output Shape -------")
    print(y.shape)