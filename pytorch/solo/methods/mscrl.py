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
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from solo.losses.mscrl import mscrl_loss_func
from solo.methods.base import BaseMethod


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

        feature_f = torch.mul(X , M_f)
        feature_b = torch.mul(X, M_b)

        return feature_f, feature_b

loss_types = []

class MSCRL(BaseMethod):
    def __init__(self, proj_output_dim: int, proj_hidden_dim: int, temperature: float, loss_type: str, **kwargs):
        """Implements MSCRL.

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            loss_type (str): which loss need to use ["byol","byol+f_loss+b_loss","byol+f_loss","f_loss+b_loss","f_loss"]
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.alpha = 0.5
        self.beta = 0.5
        assert loss_type in loss_types, "Loss type didn't included"
        self.loss_type = loss_type

        # projector
        self.projector = nn.Sequential(
            Downsample(),
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.indexer = Indexer()

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(MSCRL, MSCRL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simclr")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--alpha", type=str, default="0.5")
        parser.add_argument("--beta", type=str, default="0.5")

        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """
        _, X_, M_f, M_b = X
        out = super().forward(X_, *args, **kwargs)
        z = self.projector(out["feats"])
        z_f , z_b = self.indexer(out["feats"])
        z_f = self.projector(z_f)
        z_b = self.projector(z_b)

        return {**out, "z": z, "z_f": z_f, "z_b": z_b}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes, batch_, M_f, M_b = batch

        out = super().training_step(batch_, batch_idx)
        class_loss = out["loss"]

        feats = out["feats"]

        z = torch.cat([self.projector(f) for f in feats])

        z_f = []
        z_b = []
        for f, m_f, m_b in (feats,M_f,M_b):
            a,b = self.indexer(f,m_f,m_b)
            z_f.append(a)
            z_b.append(b)

        z_f = [self.projector([f, m]) for f, m in zip(z_f, M_f)]
        z_b = [self.projector([f, m]) for f, m in zip(z_b, M_b)]

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = mscrl_loss_func(
            z, z_f ,z_b,
            indexes=indexes,
            temperature=self.temperature,
        )

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss
