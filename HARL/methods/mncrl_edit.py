# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from HARL.losses.byol import byol_loss_func
from HARL.methods.base import BaseMomentumMethod
from HARL.utils.momentum import initialize_momentum_params


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.GA = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def forward(self, X):
        X = self.GA(X)
        X = torch.flatten(X, 1)
        return X


class Indexer(nn.Module):
    def __init__(self):
        super(Indexer, self).__init__()
        self.downsample_f = Downsample()
        self.downsample_b = Downsample()

    def forward(self, X, M_f, M_b):
        """Indicating the foreground and background feature.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            M_f (torch.Tensor) : batch of foreground mask
            M_b (torch.Tensor) : batch of background mask
        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        out['foreground_feature'] = torch.mul(X, M_f)
        # out['foreground_feature'] = self.downsample_f(out['foreground_feature'])
        out['background_feature'] = torch.mul(X, M_b)
        # out['background_feature'] = self.downsample_b(out['background_feature'])

        return out


class MNCRL_edit(BaseMomentumMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements BYOL (https://arxiv.org/abs/2006.07733).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """

        super().__init__(**kwargs)

        if kwargs["mlp_layer"] == 3:
            print("MLP Architecture 3 Layers")
            # projector
            self.projector = nn.Sequential(
                Downsample(),
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )

            # momentum projector
            self.momentum_projector = nn.Sequential(
                Downsample(),
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
            initialize_momentum_params(self.projector, self.momentum_projector)

            # predictor
            self.predictor = nn.Sequential(
                nn.Linear(proj_output_dim, pred_hidden_dim),
                nn.BatchNorm1d(pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, pred_hidden_dim),
                nn.BatchNorm1d(pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, proj_output_dim),
            )
            self.indexer = Indexer()
            self.momentum_indexer = Indexer()

        else:
            print("Implement_Default MLP Architecture")
            # projector
            self.projector = nn.Sequential(
                Downsample(),
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )

            # momentum projector
            self.momentum_projector = nn.Sequential(
                Downsample(),
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
            initialize_momentum_params(self.projector, self.momentum_projector)

            # predictor
            self.predictor = nn.Sequential(
                nn.Linear(proj_output_dim, pred_hidden_dim),
                nn.BatchNorm1d(pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, proj_output_dim),
            )

        self.indexer = Indexer()
        self.momentum_indexer = Indexer()
        self.loss_type = kwargs["loss_type"]
        self.alpha = kwargs["mask_weigted_loss"]
        self.beta = kwargs["weigted_loss"]

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(
            MNCRL_edit, MNCRL_edit).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MNCRL")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)

        # Adding Several Configure for Framework
        parser.add_argument("--mlp_layer", type=int, default=2)
        parser.add_argument("--mask_weigted_loss", type=float, default=0.5)
        parser.add_argument("--weigted_loss", type=float, default=0.5)
        # mask_loss, #sum_basline_mask_loss
        parser.add_argument("--loss_type", type=str, default="mask_loss")

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor, M_f: torch.Tensor, M_b: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        # def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            M_f (torch.Tensor) : batch of foreground mask
            M_b (torch.Tensor) : batch of background mask
        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)

        z_f, z_b = self.indexer(out["feats"], M_f, M_b)
        z_f = self.projector(z_f)
        p_f = self.predictor(z_f)

        z_b = self.projector(z_b)
        p_b = self.predictor(z_b)

        return {**out, "z": z, "p": p, "z_f": z_f, "p_f": p_f, "z_b": z_b, "p_b": p_b}
        # return {**out, "z": z, "p": p,}

    def _shared_step(
        self, feats: List[torch.Tensor], momentum_feats: List[torch.Tensor], M_f: List[torch.Tensor], M_b: List[torch.Tensor]
    ) -> torch.Tensor:

        Z = [self.projector(f) for f in feats]
        P = [self.predictor(z) for z in Z]

        feats_f, feats_b = [], []

        for feat in feats:
            a, b = self.indexer(feat)
            feats_f.append(a)
            feats_b.append(b)

        Z_f = [self.projector(f) for f in feats_f]
        P_f = [self.predictor(z) for z in Z_f]

        Z_b = [self.projector(f) for f in feats_b]
        P_b = [self.predictor(z) for z in Z_b]

        # forward momentum backbone
        with torch.no_grad():
            Z_momentum = [self.momentum_projector(f) for f in momentum_feats]
            feats_f, feats_b = [], []
            for feat in momentum_feats:
                a, b = self.indexer(feat)
                feats_f.append(a)
                feats_b.append(b)
            Z_momentum_f = [self.momentum_projector(f) for f in feats_f]
            Z_momentum_b = [self.momentum_projector(f) for f in feats_b]

        # ------- negative consine similarity loss -------
        if self.loss_type == "sum_baseline_mask_loss":
            print("Updating parameter use Baseline + Mask_loss")
            neg_cos_sim = 0
            for v1 in range(self.num_large_crops):
                for v2 in np.delete(range(self.num_crops), v1):
                    neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

            neg_cos_sim_f = 0
            for v1 in range(self.num_large_crops):
                for v2 in np.delete(range(self.num_crops), v1):
                    neg_cos_sim_f += byol_loss_func(P_f[v2], Z_momentum_f[v1])

            neg_cos_sim_b = 0
            for v1 in range(self.num_large_crops):
                for v2 in np.delete(range(self.num_crops), v1):
                    neg_cos_sim_b += byol_loss_func(P_b[v2], Z_momentum_b[v1])
            total_loss = self.beta*neg_cos_sim + \
                (1-self.beta)*(self.alpha*neg_cos_sim_f + self.alpha*neg_cos_sim_b)

        elif self.loss_type == "mask_loss":
            print("Updating parameter mask loss Only")

            neg_cos_sim_f = 0
            for v1 in range(self.num_large_crops):
                for v2 in np.delete(range(self.num_crops), v1):
                    neg_cos_sim_f += byol_loss_func(P_f[v2], Z_momentum_f[v1])

            neg_cos_sim_b = 0
            for v1 in range(self.num_large_crops):
                for v2 in np.delete(range(self.num_crops), v1):
                    neg_cos_sim_b += byol_loss_func(P_b[v2], Z_momentum_b[v1])

            total_loss = (self.alpha*neg_cos_sim_f + self.alpha*neg_cos_sim_b)
        else:
            raise ValueError(f" Loss type  is not supported.")

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(
                Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        return total_loss, z_std

    def training_step(self, batch: Sequence[Any], M_f: Sequence[Any], M_b: Sequence[Any],  batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        neg_cos_sim, z_std = self._shared_step(
            out["feats"], out["momentum_feats"], M_f, M_b)

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
