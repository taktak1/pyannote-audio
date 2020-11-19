# MIT License
#
# Copyright (c) 2020 CNRS
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


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.asteroid.filterbanks import Encoder, ParamSincFB
from pyannote.core.utils.generators import pairwise


class PyanNet(Model):
    """PyanNet segmentation model

    SincFilterbank > Conv1 > Conv1d > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    LSTM_DEFAULTS = {"hidden_size": 128, "num_layers": 2, "bidirectional": True}
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        lstm: dict = None,
        linear: dict = None,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        lstm_hparams = dict(**self.LSTM_DEFAULTS)
        if lstm is not None:
            lstm_hparams.update(**lstm)
        lstm_hparams["batch_first"] = True  # this is not negotiable
        self.hparams.lstm = lstm_hparams

        linear_hparams = dict(**self.LINEAR_DEFAULTS)
        if linear is not None:
            linear_hparams.update(**linear)
        self.hparams.linear = linear_hparams

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        if sample_rate != 16000:
            raise NotImplementedError("PyanNet only supports 16kHz audio for now.")
            # TODO: add support for other sample rate. it should be enough to multiply
            #  kernel_size by (sample_rate / 16000). but this needs to be double-checked.

        self.conv1d.append(
            Encoder(
                ParamSincFB(
                    80,
                    251,
                    stride=1,
                    sample_rate=self.hparams.sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
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

        self.lstm = nn.LSTM(60, **self.hparams.lstm)

        lstm_out_features: int = self.lstm.hidden_size * (
            2 if self.lstm.bidirectional else 1
        )

        if self.hparams.linear["num_layers"] > 0:
            self.linear = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features)
                    for in_features, out_features in pairwise(
                        [
                            lstm_out_features,
                        ]
                        + [self.hparams.linear["hidden_size"]]
                        * self.hparams.linear["num_layers"]
                    )
                ]
            )

    def build(self):

        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.lstm.hidden_size * (2 if self.lstm.bidirectional else 1)

        self.classifier = nn.Linear(
            in_features, len(self.hparams.task_specifications.classes)
        )
        self.activation = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = waveforms
        for conv1d, pool1d, norm1d in zip(self.conv1d, self.pool1d, self.norm1d):
            outputs = norm1d(pool1d(conv1d(outputs)))

        outputs, _ = self.lstm(
            rearrange(outputs, "batch feature frame -> batch frame feature")
        )

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
