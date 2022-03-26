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
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params

class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.GA = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def forward(self, X):
        mask = None
        if type(X) == list:
            mask = X[1]
            X = X[0]

        if mask is None:
            X = self.GA(X)
        else:
            # print(X.shape)
            # print(mask.shape)
            X = X.view(X.shape[0], X.shape[1], -1)
            mask = mask.view(mask.shape[0], mask.shape[1], -1)
            nelements = mask.sum(dim=-1)+1
            X = X.sum(dim=-1) / nelements

        X = torch.flatten(X, 1)
        return X

class Indexer(nn.Module):
    def __init__(self):
        super(Indexer, self).__init__()
    def forward(self, X, M_f, M_b):
        """Indicating the foreground and background feature.
        Args:
            X (torch.Tensor): batch of images in tensor format.
            M_f (torch.Tensor) : batch of foreground mask
            M_b (torch.Tensor) : batch of background mask
        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        # feature_f = torch.mul(X, M_f)
        # # out['foreground_feature'] = self.downsample_f(out['foreground_feature'])
        # feature_b = torch.mul(X, M_b)
        # # out['background_feature'] = self.downsample_b(out['background_feature'])

        feature_f = [torch.mul(f , m_f) for f, m_f in zip(X, M_f)]
        feature_b = [torch.mul(f, m_b) for f, m_b in zip(X, M_b)]

        return feature_f, feature_b

loss_types = ["byol","byol+f_loss+b_loss","byol+f_loss","f_loss+b_loss","f_loss"]

class MNCRL(BaseMomentumMethod):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        loss_type: str,
        **kwargs,
    ):
        """Implements MNCRL.

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
            loss_type (str): which loss need to use ["byol","byol+f_loss+b_loss","byol+f_loss","f_loss+b_loss","f_loss"]
        """

        super().__init__(**kwargs)
        self.alpha = 0.5
        self.beta = 0.5

        assert loss_type in loss_types, "Loss type didn't included"

        self.loss_type = loss_type
        # projector
        self.projector = nn.Sequential(
            Downsample(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # self.projector_f = nn.Sequential(
        #     Downsample(),
        #     nn.Linear(self.features_dim, proj_hidden_dim),
        #     nn.BatchNorm1d(proj_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_hidden_dim, proj_output_dim),
        # )
        #
        # self.projector_b = nn.Sequential(
        #     Downsample(),
        #     nn.Linear(self.features_dim, proj_hidden_dim),
        #     nn.BatchNorm1d(proj_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(proj_hidden_dim, proj_output_dim),
        # )


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

        # self.predictor_f = nn.Sequential(
        #     nn.Linear(proj_output_dim, pred_hidden_dim),
        #     nn.BatchNorm1d(pred_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(pred_hidden_dim, proj_output_dim),
        # )
        #
        # self.predictor_b = nn.Sequential(
        #     nn.Linear(proj_output_dim, pred_hidden_dim),
        #     nn.BatchNorm1d(pred_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(pred_hidden_dim, proj_output_dim),
        # )

        # self.indexer = Indexer()
        # self.momentum_indexer = Indexer()

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MNCRL, MNCRL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("MNCRL")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=256)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # predictor
        parser.add_argument("--pred_hidden_dim", type=int, default=512)
        parser.add_argument("--loss_type", type=str, default="byol+f_loss+b_loss")

        # parameters
        parser.add_argument("--alpha", type=str, default="0.5")
        parser.add_argument("--beta", type=str, default="0.5")


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

    #def forward(self, X: torch.Tensor, M_f: torch.Tensor, M_b: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            M_f (torch.Tensor) : batch of foreground mask
            M_b (torch.Tensor) : batch of background mask
        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        _, X_, M_f, M_b = X
        out = super().forward(X_, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)

        z_f = torch.mul(out["feats"], M_f)
        z_b = torch.mul(out["feats"], M_b)
        # z_b = self.indexer(, M_f, M_b)
        z_f = self.projector(z_f)
        p_f = self.predictor(z_f)

        z_b = self.projector(z_b)
        p_b = self.predictor(z_b)

        return {**out, "z": z, "p": p, "z_f": z_f, "p_f": p_f, "z_b": z_b, "p_b": p_b}
        #return {**out, "z": z, "p": p,}

    def _shared_step(
        self, feats: List[torch.Tensor], momentum_feats: List[torch.Tensor], M_f: List[torch.Tensor], M_b: List[torch.Tensor]
    ) -> torch.Tensor:

        Z = [self.projector(f) for f in feats]
        P = [self.predictor(z) for z in Z]


        feats_f = [torch.mul(f, m_f) for f, m_f in zip(feats, M_f)]
        feats_b = [torch.mul(f, m_b) for f, m_b in zip(feats, M_b)]

        # feats_f, feats_b = [], []
        # for feat in feats:
        #     torch.mul
        #     a , b = self.momentum_indexer(feat,M_f,M_b)
        #     feats_f.append(a)
        #     feats_b.append(b)

        Z_f = [self.projector([f,m]) for f, m in zip(feats_f, M_f)]
        P_f = [self.predictor(z) for z in Z_f]

        Z_b = [self.projector([f,m]) for f, m in zip(feats_b, M_b)]
        P_b = [self.predictor(z) for z in Z_b]


        # forward momentum backbone
        with torch.no_grad():
            Z_momentum = [self.momentum_projector(f) for f in momentum_feats]
            momentum_feats_f = [torch.mul(f, m_f) for f, m_f in zip(momentum_feats, M_f)]
            # self.print(momentum_feats_f)
            momentum_feats_b = [torch.mul(f, m_b) for f, m_b in zip(momentum_feats, M_b)]
            # self.print(momentum_feats_b)
            # feats_f, feats_b = [], []
            # for feat in momentum_feats:
            #     a, b = self.indexer(feat,M_f,M_b)
            #     feats_f.append(a)
            #     feats_b.append(b)
            Z_momentum_f = [self.momentum_projector([f,m]) for f, m in zip(momentum_feats_f, M_f)]
            Z_momentum_b = [self.momentum_projector([f,m]) for f, m in zip(momentum_feats_b, M_b)]


        # ------- negative consine similarity loss -------
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

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        if self.loss_type == "byol+f_loss+b_loss":
            total_loss = self.beta*neg_cos_sim + (1-self.beta)*(self.alpha*neg_cos_sim_f + (1-self.alpha)*neg_cos_sim_b)
        elif self.loss_type == "f_loss+b_loss":
            total_loss = self.alpha*neg_cos_sim_f + (1-self.alpha)*neg_cos_sim_b
        elif self.loss_type == "byol+f_loss":
            total_loss = self.beta*neg_cos_sim + (1-self.beta)*neg_cos_sim_f
        elif self.loss_type == "f_loss":
            total_loss = neg_cos_sim_f
        elif self.loss_type == "byol":
            total_loss = neg_cos_sim
        else:
            total_loss = self.beta*neg_cos_sim + (1-self.beta)*neg_cos_sim_f

        return total_loss , neg_cos_sim, neg_cos_sim_f, neg_cos_sim_b, z_std
        # return neg_cos_sim, z_std

    #def training_step(self, batch: Sequence[Any], M_f: Sequence[Any], M_b: Sequence[Any],  batch_idx: int) -> torch.Tensor:
    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:

        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y, [M_f], [M_b]], where
                [X], [M_f], [M_b] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """
        # M_f = batch[1]
        # M_b = batch[2]
        # batch = batch[0]
        _, batch_, M_f, M_b = batch
        out = super().training_step(batch_, batch_idx)
        class_loss = out["loss"]

        total_loss , neg_cos_sim, neg_cos_sim_f, neg_cos_sim_b, z_std = self._shared_step(out["feats"], out["momentum_feats"], M_f, M_b)
        #neg_cos_sim, z_std = self._shared_step(out["feats"], out["momentum_feats"], M_f, M_b)
        metrics = {
            "total_loss": total_loss,
            "train_neg_cos_sim_b": neg_cos_sim_f,
            "train_neg_cos_sim_f": neg_cos_sim_b,
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_loss_alpha": self.alpha,
            "train_loss_beta": self.beta,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return total_loss + class_loss