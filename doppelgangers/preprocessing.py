# Copyright (c) 2025
# Manuel Cherep <mcherep@mit.edu>
# Nikhil Singh <nsingh1@mit.edu>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import abc
import torch
import torchaudio
from doppelgangers.vggish import VGGishInputProcessor, _SAMPLE_RATE


class BasePreprocessor(abc.ABC):
    def __init__(
        self,
        batch: bool = False,
        sample_rate: int = 16000,
        normalize: bool = False,
        eps: float = 1e-6
    ) -> None:
        self.batch = batch
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.eps = eps

    @abc.abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Assume shape of the data is (n_samples,) or (batch_size, n_samples)
        pass


class SimpleProcessor(BasePreprocessor):
    def __init__(
        self,
        batch: bool = False,
        sample_rate: int = 16000,
        normalize: bool = False,
        eps: float = 1e-6
    ) -> None:
        super().__init__(batch=batch, sample_rate=sample_rate, normalize=normalize, eps=eps)

        self.spec = torch.compile(torchaudio.transforms.MelSpectrogram(self.sample_rate))

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.atleast_2d(data)
        output = self.spec(data)
        output = torch.log(output + 1e-6)

        if not self.batch:
            output = output.squeeze(0)

        return output


class VGGishProcessor(BasePreprocessor):
    def __init__(
        self,
        batch: bool = False,
        sample_rate: int = _SAMPLE_RATE,
        normalize: bool = False,
        eps: float = 1e-6
    ) -> None:
        super().__init__(batch=batch, sample_rate=sample_rate, normalize=normalize, eps=eps)

        if self.batch:
            self.preprocessor = torch.vmap(VGGishInputProcessor())
        else:
            self.preprocessor = VGGishInputProcessor()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(data)
