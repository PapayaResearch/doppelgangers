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

import logging
import torch
import torch.nn as nn
import lightning
from matplotlib import pyplot
from doppelgangers.utils import align_loss, uniform_loss


class AudioDoppelgangersModel(lightning.LightningModule):
    def __init__(
        self,
        embedding_model: nn.Module,
        unif_t: float,
        align_alpha: float,
        align_w: float,
        unif_w: float,
        sample_rate: int,
        max_audio_log_n: int,
        batch_size: int,
        optimizer: str,
        n_epochs: int,
        scheduler_milestones: list,
        **kwargs
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["embedding_model"])

        self.embedding = embedding_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def run_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        emb_x = self.embedding(x)
        emb_y = self.embedding(y)

        emb_x = self.all_gather(emb_x, sync_grads=True).view(-1, emb_x.shape[-1])
        emb_y = self.all_gather(emb_y, sync_grads=True).view(-1, emb_y.shape[-1])

        # Normalize embeddings
        emb_x = emb_x / emb_x.norm(p=2, dim=-1, keepdim=True)
        emb_y = emb_y / emb_y.norm(p=2, dim=-1, keepdim=True)

        align_val = align_loss(emb_x, emb_y, alpha=self.hparams.align_alpha)
        unif_val = (uniform_loss(emb_x, t=self.hparams.unif_t) + uniform_loss(emb_y, t=self.hparams.unif_t)) / 2
        L = align_val * self.hparams.align_w + unif_val * self.hparams.unif_w

        return L, align_val, unif_val

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        L, align_val, unif_val = self.run_step(batch, batch_idx)

        self.log("train/L_align", align_val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/L_unif", unif_val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/L", L, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return L

    def validation_step(self, batch, batch_idx):
        x, y, (w1, w2) = batch
        L, align_val, unif_val = self.run_step((x, y, (w1, w2)), batch_idx)

        self.log("val/L_align", align_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/L_unif", unif_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/L", L, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log audio and spectrograms
        for i, (w1_, w2_) in enumerate(zip(w1, w2)):
            self.logger.experiment.add_audio(
                "val/audio/%d/w1" % i,
                w1_ / w1_.abs().max(),
                self.current_epoch,
                sample_rate=self.hparams.sample_rate
            )

            self.logger.experiment.add_audio(
                "val/audio/%d/w2" % i,
                w2_ / w2_.abs().max(),
                self.current_epoch,
                sample_rate=self.hparams.sample_rate
            )

            try:
                fig, ax = pyplot.subplots(1, 2)
                ax[0].imshow(x[i][0].cpu().numpy())
                ax[0].set_title("s1: %d" % i)
                ax[1].imshow(y[i][0].cpu().numpy())
                ax[1].set_title("s2: %d" % i)
                self.logger.experiment.add_figure("val/spectrogram/%d" % i, fig, self.current_epoch)
                pyplot.close()
            except Exception as error:
                logging.info("Error plotting spectrogram: %s" % error)

            if i == self.hparams.max_audio_log_n:
                break

        return L

    def configure_optimizers(self):
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if self.hparams.optimizer == "adam":
            optim = torch.optim.Adam(
                self.parameters(),
                lr=0.001,
                weight_decay=0.0001
            )
        elif self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(
                self.parameters(),
                lr=0.12 * (self.hparams.batch_size//256) * world_size,
                weight_decay=1e-6
            )

        milestones = [int(self.hparams.n_epochs * m) for m in self.hparams.scheduler_milestones]
        logging.info("Using scheduler milestones: %s." % ",".join(map(str, milestones)))

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            gamma=0.1,
            milestones=milestones
        )

        return [optim], [scheduler]
