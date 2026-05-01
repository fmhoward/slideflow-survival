# Slideflow-GPL - Add-ons for the deep learning library Slideflow
# Copyright (C) 2024 James Dolezal
#
# This file is part of Slideflow-GPL.
#
# Slideflow-GPL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Slideflow-GPL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Slideflow-GPL. If not, see <https://www.gnu.org/licenses/>.

import numpy as np

from typing import Union, List, Optional, Tuple, Any, TYPE_CHECKING
from slideflow.mil import utils as mil_utils

if TYPE_CHECKING:
    import torch


def run_inference(
    model: "torch.nn.Module",
    bags: Union[np.ndarray, List[str]],
    attention: bool = False,
    device: Optional[Any] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Generate CLAM predictions for a list of bags."""

    import torch
    from .model import CLAM_MB, CLAM_SB

    if isinstance(model, (CLAM_MB, CLAM_SB)):
        clam_kw = dict(return_attention=True, return_instance_loss=False)
    else:
        clam_kw = {}
        attention = False

    y_pred = []
    y_att  = []
    device = mil_utils._detect_device(model, device, verbose=True)
    with torch.inference_mode():
        for bag in bags:
            loaded = mil_utils._load_bag(bag).to(device)
            logits, att = model(loaded, **clam_kw)
            if attention:
                y_att.append(np.squeeze(att.cpu().numpy()))
            y_pred.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())
    yp = np.concatenate(y_pred, axis=0)
    return yp, y_att
