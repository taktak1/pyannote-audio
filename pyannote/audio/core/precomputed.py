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

import io
from pathlib import Path
from typing import List, Text, Union

import numpy as np
import yaml

from pyannote.audio.core.io import AudioFile
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.database import get_unique_identifier


class Precomputed:
    """Precomputed inference scores

    Parameters
    ----------
    root_dir : str
        Path to directory where precomputed features are stored.
    window : SlidingWindow, optional
        Sliding window used for feature extraction. This is not used when
        `root_dir` already exists and contains `metadata.yml`.
    dimension : int, optional
        Dimension of feature vectors. This is not used when `root_dir` already
        exists and contains `metadata.yml`.
    classes : iterable, optional
        Human-readable name for each dimension.

    Notes
    -----
    If `root_dir` directory does not exist, one must provide both
    `window` and `dimension` parameters in order to create and
    populate file `root_dir/metadata.yml` when instantiating.
    """

    def __init__(
        self,
        root_dir: Union[Text, Path],
        window: SlidingWindow = None,
        dimension: int = None,
        classes: List[Text] = None,
    ):
        super().__init__()

        self.root_dir = Path(root_dir).expanduser().resolve(strict=False)

        path = self.root_dir / "metadata.yml"
        if path.exists():

            with io.open(path, "r") as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)

            self.dimension = params.pop("dimension")
            self.classes = params.pop("classes", None)
            self.window = SlidingWindow(**params)

            if dimension is not None and self.dimension != dimension:
                msg = 'inconsistent "dimension" (is: {0}, should be: {1})'
                raise ValueError(msg.format(dimension, self.dimension))

            if classes is not None and self.classes != classes:
                msg = 'inconsistent "classes" (is {0}, should be: {1})'
                raise ValueError(msg.format(classes, self.classes))

            if (window is not None) and (
                (window.start != self.window.start)
                or (window.duration != self.window.duration)
                or (window.step != self.window.step)
            ):
                msg = 'inconsistent "windows"'
                raise ValueError(msg)

            return

        if dimension is None:
            if classes is None:
                msg = (
                    "Please provide either `dimension` or `classes` "
                    "parameters (or both) when instantiating "
                    "`Precomputed`."
                )
                raise ValueError(msg)
            dimension = len(classes)

        if window is None or dimension is None:
            msg = (
                f"Either directory {self.root_dir} does not exist or it "
                f"does not contain precomputed features. In case it exists "
                f"and this was done on purpose, please provide both "
                f"`windows` and `dimension` parameters when "
                f"instantianting `Precomputed`."
            )
            raise ValueError(msg)

        self.root_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "start": window.start,
            "duration": window.duration,
            "step": window.step,
            "dimension": dimension,
        }
        if classes is not None:
            params["classes"] = classes

        with io.open(path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)

        self.window = window
        self.dimension = dimension
        self.classes = classes

    def get_path(self, file: AudioFile) -> Path:
        uri = get_unique_identifier(file)
        return self.root_dir / f"{uri}.npy"

    def __call__(self, file: AudioFile) -> SlidingWindowFeature:
        """Load precomputed inference scores

        Parameters
        ----------
        file : AudioFile
            Audio file.

        Returns
        -------
        scores : SlidingWindowFeature
            Precomputed inference scores.
        """

        path = Path(self.get_path(file))

        if not path.exists():
            uri = file["uri"]
            database = file["database"]
            msg = (
                f"Directory {self.root_dir} does not contain "
                f'precomputed features for file "{uri}" of '
                f'"{database}" database.'
            )
            raise ValueError(msg)

        data = np.load(str(path))

        return SlidingWindowFeature(data, self.window)

    def dump(self, file: AudioFile, features: SlidingWindowFeature):
        path = self.get_path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, features.data)
