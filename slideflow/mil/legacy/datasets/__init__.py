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
#
# This file incorporates work from CLAM, which is licensed
# under the GNU General Public License, Version 3. The original
# license and source code can be found at https://github.com/mahmoodlab/CLAM.

import os
import torch
from os.path import join
from slideflow.util import path_to_name

from .dataset_generic import Generic_WSI_Classification_Dataset


# -----------------------------------------------------------------------------

class CLAM_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, pt_files, **kwargs):
        super().__init__(**kwargs)
        if isinstance(pt_files, str):
            self.pt_files = {
                path_to_name(filename): join(pt_files, filename)
                for filename in os.listdir(pt_files)
            }
        else:
            self.pt_files = {path_to_name(path): path for path in pt_files}

    def detect_num_features(self):
        features = torch.load(list(self.pt_files.values())[0])
        return features.size()[1]

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide'][idx]
        label = self.slide_data['label'][idx]
        features = torch.load(self.pt_files[slide_id])
        if self.lasthalf:
            features = torch.split(features, 1024, dim = 1)[1]
        return features, label
