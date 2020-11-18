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


import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from pyannote.audio.core.model import load_from_checkpoint
from pyannote.audio.core.precomputed import Precomputed
from pyannote.database import FileFinder, get_protocol


@hydra.main(config_path="infer", config_name="config")
def main(cfg: DictConfig) -> None:

    model = load_from_checkpoint(cfg.checkpoint)

    protocol = get_protocol(cfg.protocol.name, preprocessors={"audio": FileFinder()})
    files = getattr(protocol, cfg.protocol.subset)()

    if cfg.inference.device == "auto":
        cfg.inference.device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = instantiate(cfg.inference, model)

    for f, file in enumerate(files):
        scores = inference(file)
        if f == 0:
            #  TODO: handle case where inference.is_multi_task
            precomputed = Precomputed(
                "/tmp/test/infer",
                window=scores.sliding_window,
                dimension=scores.shape[-1],
                classes=inference.task_specifications.classes,
            )
        precomputed.dump(file, scores)


if __name__ == "__main__":
    main()
