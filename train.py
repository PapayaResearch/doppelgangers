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
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # Avoid OOM issues with JAX/XLA preallocation
import hydra
import torch
import lightning
from doppelgangers.model import AudioDoppelgangersModel
from doppelgangers.data import AudioDoppelgangersDataset, AudioDoppelgangersSyntheticDataset
from doppelgangers.utils import print_config
from config import Config


config_store = hydra.core.config_store.ConfigStore.instance()
config_store.store(name="base_config", node=Config)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Config) -> None:
    print_config(cfg)

    # Log dir
    log_dir = cfg.trainer.default_root_dir
    os.makedirs(log_dir, exist_ok=True)

    lightning.seed_everything(cfg.system.seed)

    # Load model
    embedding = hydra.utils.instantiate(cfg.embedding.model)
    model = AudioDoppelgangersModel(
        embedding,
        cfg.general.unif_t,
        cfg.general.align_alpha,
        cfg.general.align_w,
        cfg.general.unif_w,
        batch_size=cfg.data.batch_size,
        sample_rate=cfg.data.sample_rate,
        max_audio_log_n=cfg.general.max_audio_log_n,
        optimizer=cfg.general.optimizer,
        n_epochs=cfg.general.epochs,
        scheduler_milestones=cfg.general.scheduler.milestones,
        run_name=cfg.general.run_name,
        apply_transform=cfg.data.apply_transform,
        preprocessing=cfg.embedding.preprocessor
    )


    transform = hydra.utils.instantiate(cfg.transform.transform)
    if cfg.general.synthetic:
        synth = hydra.utils.instantiate(cfg.synth.synth)
        dataset = AudioDoppelgangersSyntheticDataset(
            synth=synth,
            batch_size=cfg.data.batch_size,
            delta=cfg.data.synthetic.delta,
            sample_rate=cfg.data.sample_rate,
            transform=transform,
            max_n=cfg.data.max_n,
            apply_transform=cfg.data.apply_transform,
            preprocessing=cfg.embedding.preprocessor,
            transform_mode=cfg.transform.mode
        )
    else:
        dataset = AudioDoppelgangersDataset(
            cfg.data.path,
            sample_rate=cfg.data.sample_rate,
            transform=transform,
            max_n=cfg.data.max_n,
            apply_transform=cfg.data.apply_transform,
            duration=cfg.data.duration,
            preprocessing=cfg.embedding.preprocessor,
            transform_mode=cfg.transform.mode,
            temporal_jitter=cfg.data.temporal_jitter,
            n_layers=cfg.data.n_layers
        )

    # Data loaders
    train_size = int(cfg.data.train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    dl_batchsize = None if cfg.general.synthetic else cfg.data.batch_size
    num_workers = 0 if cfg.general.synthetic else cfg.data.n_workers
    pin_memory = False if cfg.general.synthetic else True
    persistent_workers = num_workers > 0
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=dl_batchsize,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=dl_batchsize,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
