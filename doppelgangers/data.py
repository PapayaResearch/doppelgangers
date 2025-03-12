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

import os
import math
import abc
import logging
import pandas as pd
import glob
import torch
import torch.nn as nn
import torchaudio
import jax
import jax.flatten_util
import jax.numpy as jnp
import synthax
import synthax.config
import synthax.synth
from doppelgangers.preprocessing import SimpleProcessor, VGGishProcessor


_PREPROCESSORS_ = {
    "simple": SimpleProcessor,
    "vggish": VGGishProcessor
}


class AudioDoppelgangersBaseDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        sample_rate: int,
        transform: nn.Module,
        max_n: int,
        apply_transform: bool,
        preprocessing: str,
        transform_mode: str
    ) -> None:
        self.sample_rate = sample_rate
        self.transform = transform
        self.max_n = max_n
        self.apply_transform = apply_transform
        self.preprocessing = preprocessing
        self.transform_mode = transform_mode


class AudioDoppelgangersDataset(AudioDoppelgangersBaseDataset):
    def __init__(
        self,
        data_dir: str,
        sample_rate: int,
        transform: nn.Module,
        max_n: int,
        apply_transform: bool,
        duration: float,
        preprocessing: str,
        transform_mode: str,
        temporal_jitter: bool,
        n_layers: int
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            transform=transform,
            max_n=max_n,
            apply_transform=apply_transform,
            preprocessing=preprocessing,
            transform_mode=transform_mode
        )

        self.data_dir = data_dir
        self.duration = duration
        self.temporal_jitter = temporal_jitter
        self.n_layers = n_layers

        self.preprocessor = _PREPROCESSORS_[self.preprocessing](
            batch=False,
            sample_rate=self.sample_rate
        )

        # Caching file names to avoid slow disk reads
        cache_file = os.path.join(self.data_dir, "_audio.csv")
        if os.path.isfile(cache_file):
            self.df_audiofiles = pd.read_csv(cache_file)
            self.df_audiofiles["audio_file"] = self.df_audiofiles.audio_file.map(
                lambda f : os.path.join(
                    self.data_dir,
                    os.path.basename(f)
                )
            )
        else:
            self.df_audiofiles = pd.DataFrame()

            self.df_audiofiles["audio_file"] = glob.glob(
                os.path.join(
                    os.path.expanduser(
                        data_dir
                    ),
                    "*.wav"
                )
            )

            self.df_audiofiles.to_csv(cache_file, index=False)

        if (self.max_n != -1) and (len(self.df_audiofiles) > self.max_n):
            # If we have more files than we need, sample randomly
            logging.info("Sampling %d files from %d total." % (self.max_n, len(self.df_audiofiles)))
            self.df_audiofiles = self.df_audiofiles.sample(n=self.max_n, replace=False)

        logging.info("Training with on-disk data %s transformations." % ("with" if apply_transform else "without"))

    def __len__(self):
        return len(self.df_audiofiles)

    def __getitem__(self, index: int) -> torch.Tensor:
        waveform, sr = torchaudio.load(self.df_audiofiles["audio_file"].iloc[index])

        if waveform.shape[0] > 1:
            # If the waveform has multiple channels, average them
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = torchaudio.functional.resample(
            waveform.squeeze(0),
            sr,
            self.sample_rate
        )

        n_frames = int(self.duration * self.sample_rate)
        if waveform.shape[0] < n_frames:
            # If the waveform is too short, pad with zeros
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(n_frames - waveform.shape[0])
                ]
            )

        if self.n_layers > 1:
            waveform = waveform[:waveform.shape[0] - (waveform.shape[0] % n_frames)]
            waveform_seconds = waveform.reshape(
                -1,
                n_frames
            )
            layers_indices = torch.randperm(waveform_seconds.shape[0])[:self.n_layers]
            waveform = waveform_seconds[layers_indices].sum(dim=0)
            waveform = waveform / (waveform.abs().max() + 1e-6)

            waveform_1 = waveform
            waveform_2 = waveform

        if waveform.shape[0] > n_frames and not self.temporal_jitter and not self.n_layers > 1:
            # If the waveform is too long, sample a random segment
            waveform_index = torch.randint(0, waveform.shape[0] - n_frames, (1,)).item()
            waveform = waveform[waveform_index:waveform_index + n_frames]

            waveform_1 = waveform
            waveform_2 = waveform

        if self.temporal_jitter:
            waveform_index1, waveform_index2 = torch.randint(0, waveform.shape[0] - n_frames, (2,)).tolist()
            waveform_1 = waveform[waveform_index1:waveform_index1 + n_frames]
            waveform_2 = waveform[waveform_index2:waveform_index2 + n_frames]


        if self.apply_transform and self.transform_mode == "waveform":
            waveform_1 = self.transform(waveform_1[None, None, :])[0, 0]
            waveform_2 = self.transform(waveform_2[None, None, :])[0, 0]

        x = self.preprocessor(waveform_1)
        y = self.preprocessor(waveform_2)

        if self.apply_transform and self.transform_mode == "spec":
            x = self.transform(x)
            y = self.transform(y)

        x = x.squeeze(0)
        y = y.squeeze(0)

        return x, y, (waveform_1, waveform_2)


class AudioDoppelgangersSyntheticDataset(AudioDoppelgangersBaseDataset):
    def __init__(
        self,
        synth: synthax.synth.BaseSynth,
        batch_size: int,
        delta: float,
        sample_rate: int,
        transform: nn.Module,
        max_n: int,
        apply_transform: bool,
        preprocessing: str,
        transform_mode: str,
        augmentation_chunksize: int = 100,
        eps: float = 1e-6
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            transform=transform,
            max_n=max_n,
            apply_transform=apply_transform,
            preprocessing=preprocessing,
            transform_mode=transform_mode
        )

        self.synth = synth
        self.batch_size = batch_size
        self.delta = delta
        self.augmentation_chunksize = augmentation_chunksize
        self.eps = eps

        self.preprocessor = _PREPROCESSORS_[self.preprocessing](
            batch=True,
            sample_rate=self.sample_rate
        )

        self.generate_params = jax.jit(self.synth.init)
        self.generate_sound = jax.jit(self.synth.apply)
        worker_info = torch.utils.data.get_worker_info()
        self.seed = int(worker_info.seed if worker_info is not None else 0)
        self.key = jax.random.PRNGKey(self.seed)

        init_params = self.generate_params(self.key)
        _, self.unflatten = jax.flatten_util.ravel_pytree(init_params)
        self.sample_noise = jax.jit(jax.random.normal, static_argnums=(1,))

        logging.info("Training with synthetic data %s transformations." % ("with" if apply_transform else "without"))

    def __len__(self):
        return math.ceil(self.max_n / self.batch_size)

    def __getitem__(self, index: int) -> torch.Tensor:
        self.key, subkey = jax.random.split(self.key)
        params = self.generate_params(subkey)

        params_flat, _ = jax.flatten_util.ravel_pytree(params)

        # Apply delta noise to the parameters
        params_noise1 = self.delta * self.sample_noise(subkey, params_flat.shape)
        self.key, subkey = jax.random.split(self.key)
        params_noise2 = self.delta * self.sample_noise(subkey, params_flat.shape)

        # Clip the parameters with an epsilon instead of zero to avoid possible NaNs
        params_aug1 = self.unflatten(jnp.clip(params_flat + params_noise1, self.eps, 1))
        params_aug2 = self.unflatten(jnp.clip(params_flat + params_noise2, self.eps, 1))

        waveform_1 = torch.from_dlpack(jax.dlpack.to_dlpack(self.generate_sound(params_aug1)))
        waveform_2 = torch.from_dlpack(jax.dlpack.to_dlpack(self.generate_sound(params_aug2)))

        if self.apply_transform and self.transform_mode == "waveform":
            # For now, we need to use per_batch processing, and then use mini-batches to conserve GPU memory
            # In practice, this is not ideal (augmentations will not be independent within these mini-batches)
            # However, without this, training will take a very long time
            waveform_1o = []
            waveform_2o = []
            for i in range(0, waveform_1.shape[0], self.augmentation_chunksize):
                waveform_1o.append(self.transform(waveform_1[i:i + self.augmentation_chunksize].unsqueeze(1)).squeeze(1))
                waveform_2o.append(self.transform(waveform_2[i:i + self.augmentation_chunksize].unsqueeze(1)).squeeze(1))
            waveform_1 = torch.cat(waveform_1o)
            waveform_2 = torch.cat(waveform_2o)

        x = self.preprocessor(waveform_1).squeeze(1)
        y = self.preprocessor(waveform_2).squeeze(1)

        if self.apply_transform and self.transform_mode == "spec":
            x = self.transform(x)
            y = self.transform(y)

        return x, y, (waveform_1, waveform_2)
