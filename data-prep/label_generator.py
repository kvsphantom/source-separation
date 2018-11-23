"""
Created on 11/23/18 11:19 AM    


@author: kvshenoy
"""
import os
import sys
import h5py
import torch

sys.path.append('../')
from settings import *
sys.path.append('../utils/')
from utils import create_folder
import video_utils



def _load_model(self, directory, strict_loading=True, **kwargs):
    self.fscratch = False
    print('Loading pretrained weights')
    if isinstance(directory, dict):
        state_dict = directory
    else:
        state_dict = torch.load(directory, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    self.model.load_state_dict(new_state_dict, strict=strict_loading)

#weights = torch.load(PATH_TO_RETRAINED_RESNET_WEIGHTS)
state_dict = torch.load(PATH_TO_RETRAINED_RESNET_WEIGHTS, map_location=lambda storage, loc: storage)
print ('FKI')