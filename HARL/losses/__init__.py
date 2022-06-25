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

from HARL.losses.barlow import barlow_loss_func
from HARL.losses.byol import byol_loss_func
from HARL.losses.deepclusterv2 import deepclusterv2_loss_func
from HARL.losses.dino import DINOLoss
from HARL.losses.moco import moco_loss_func
from HARL.losses.nnclr import nnclr_loss_func
from HARL.losses.ressl import ressl_loss_func
from HARL.losses.simclr import simclr_loss_func
from HARL.losses.simsiam import simsiam_loss_func
from HARL.losses.swav import swav_loss_func
from HARL.losses.vibcreg import vibcreg_loss_func
from HARL.losses.vicreg import vicreg_loss_func
from HARL.losses.wmse import wmse_loss_func

__all__ = [
    "barlow_loss_func",
    "byol_loss_func",
    "deepclusterv2_loss_func",
    "DINOLoss",
    "moco_loss_func",
    "nnclr_loss_func",
    "ressl_loss_func",
    "simclr_loss_func",
    "simsiam_loss_func",
    "swav_loss_func",
    "vibcreg_loss_func",
    "vicreg_loss_func",
    "wmse_loss_func",
]
