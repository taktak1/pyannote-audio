# The MIT License (MIT)
#
# Copyright (c) 2019 Mirco Ravanelli
# Copyright (c) 2019-2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Hervé Bredin - http://herve.niderb.fr

# Part of this code was taken from https://github.com/mravanelli/SincNet
# (see above license terms).

# Please give proper credit to the authors if you are using SincNet-based
# models  by citing their paper:

# Mirco Ravanelli, Yoshua Bengio.
# "Speaker Recognition from raw waveform with SincNet".
# SLT 2018. https://arxiv.org/abs/1808.00158

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from pyannote.audio.models.asteroid.filterbanks import Encoder, ParamSincFB


class SincConv1d(nn.Module):
    """Sinc-based 1D convolution

    Parameters
    ----------
    in_channels : `int`
        Should be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    stride : `int`, optional
        Defaults to 1.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    min_low_hz: `int`, optional
        Defaults to 50.
    min_band_hz: `int`, optional
        Defaults to 50.

    Usage
    -----
    Same as `torch.nn.Conv1d`

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio. "Speaker Recognition from raw waveform with
    SincNet". SLT 2018. https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
    ):

        super().__init__()

        if in_channels != 1:
            msg = (
                f"SincConv1d only supports one input channel. "
                f"Here, in_channels = {in_channels}."
            )
            raise ValueError(msg)
        self.in_channels = in_channels

        self.out_channels = out_channels

        if kernel_size % 2 == 0:
            msg = (
                f"SincConv1d only support odd kernel size. "
                f"Here, kernel_size = {kernel_size}."
            )
            raise ValueError(msg)
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv1d does not support bias.")
        if groups > 1:
            raise ValueError("SincConv1d does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Half Hamming half window
        n_lin = torch.linspace(
            0, self.kernel_size / 2 - 1, steps=int((self.kernel_size / 2))
        )
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        # Due to symmetry, I only need half of the time axes
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        """Get sinc filters activations

        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Equivalent to Eq.4 of the reference paper
        # I just have expanded the sinc and simplified the terms.
        # This way I avoid several useless computations.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class SincNet(nn.Module):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()

        if sample_rate != 16000:
            raise NotImplementedError("PyanNet only supports 16kHz audio for now.")
            # TODO: add support for other sample rate. it should be enough to multiply
            #  kernel_size by (sample_rate / 16000). but this needs to be double-checked.

        self.wav_norm1d = torch.nn.InstanceNorm1d(1, affine=True)

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        # self.conv1d.append(
        #     Encoder(
        #         ParamSincFB(
        #             80,
        #             251,
        #             stride=1,
        #             sample_rate=sample_rate,
        #             min_low_hz=50,
        #             min_band_hz=50,
        #         )
        #     )
        # )
        self.conv1d.append(
            SincConv1d(
                1,
                80,
                251,
                sample_rate=sample_rate,
                min_low_hz=50,
                min_band_hz=50,
                stride=1,
                padding=0,
                dilation=1,
                bias=False,
                groups=1,
            )
        )
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(80, affine=True))

        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)
        """

        outputs = self.wav_norm1d(waveforms)

        for c, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):

            outputs = conv1d(outputs)

            # https://github.com/mravanelli/SincNet/issues/4
            if c == 0:
                outputs = torch.abs(outputs)

            outputs = F.leaky_relu(norm1d(pool1d(outputs)))

        return outputs
