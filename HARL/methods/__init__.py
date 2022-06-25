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

from HARL.methods.barlow_twins import BarlowTwins
from HARL.methods.base import BaseMethod
from HARL.methods.byol import BYOL
from HARL.methods.deepclusterv2 import DeepClusterV2
from HARL.methods.dino import DINO
from HARL.methods.linear import LinearModel
from HARL.methods.mocov2plus import MoCoV2Plus
from HARL.methods.nnbyol import NNBYOL
from HARL.methods.nnclr import NNCLR
from HARL.methods.nnsiam import NNSiam
from HARL.methods.ressl import ReSSL
from HARL.methods.simclr import SimCLR
from HARL.methods.simsiam import SimSiam
from HARL.methods.supcon import SupCon
from HARL.methods.swav import SwAV
from HARL.methods.vibcreg import VIbCReg
from HARL.methods.vicreg import VICReg
from HARL.methods.wmse import WMSE
from HARL.methods.mncrl import MNCRL
# from solo.methods.mncrl_edit import MNCRL_edit
from HARL.methods.mscrl import MSCRL
METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "wmse": WMSE,
    "mncrl": MNCRL,
    # "mncrl_edit": MNCRL_edit,
    "mscrl": MSCRL,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "MoCoV2Plus",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
    "mncrl",
    # "mncrl_edit",
    "mscrl",
]

try:
    from HARL.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
