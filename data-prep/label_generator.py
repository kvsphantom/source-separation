"""
Created on 11/23/18 11:19 AM    


@author: kvshenoy
"""
import os
import sys
import h5py
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
import torchvision.models as models

sys.path.append('../')
from settings import *
sys.path.append('../utils/')
from utils import create_folder
import video_utils
from collections import OrderedDict

import os
from PIL import Image
import torch
from torch.utils.data.sampler import RandomSampler
import numpy as np
#import audio as audiolib
import random

from natsort import natsorted, ns
import shutil
from copy import deepcopy

classes={'cello','clarinet','erhu','flute','trumpet','tuba','violin','xylophone'}
duet_class_mapping={'cf':[1,0,0,1,0,0,0,0],
                    'clc':[1,1,0,0,0,0,0,0],
                    'clf':[0,1,0,1,0,0,0,0],
                    'clt':[0,1,0,0,1,0,0,0],
                    'cltu':[0,1,0,0,0,1,0,0],
                    'clv':[0,1,0,0,0,0,1,0],
                    'ct':[1,0,0,0,1,0,0,0],
                    'ec':[1,0,1,0,0,0,0,0],
                    'ef': [0,0,1,1,0,0,0,0],
                    'tf':[0,0,0,1,1,0,0,0],
                    'tut':[0,0,0,0,1,1,0,0],
                    'tuv':[0,0,0,0,0,1,1,0],
                    'vc':[1,0,0,0,0,0,1,0],
                    'vf':[0,0,0,1,0,0,1,0],
                    'vt':[0,0,0,0,1,0,1,0],
                    'xf':[0,0,0,1,0,0,0,1],
                   }

def sync_shuffle(a,b):
    c=deepcopy(a)
    d=deepcopy(b)
    combined = list(zip(c, d))
    random.shuffle(combined)
    c[:], d[:] = zip(*combined)
    return c,d


files_list=[]
labels_list=[]
for category in natsorted(os.listdir(PATH_TO_AV_DUMPS), key=lambda y: y.lower()):
    category_path=os.path.join(PATH_TO_AV_DUMPS,category)
    #store categories - resolve multi-labels and put them in a list
    #Eg. [[1,0,0,1,0,0,0,0] X number of files in the cf.children.children, ...]
    for folder in natsorted(os.listdir(category_path), key=lambda y: y.lower()):
        folder_path=os.path.join(category_path,folder,'visuals')
        for file in natsorted(os.listdir(folder_path), key=lambda y: y.lower()):
            file_path=os.path.join(folder_path,file)
            files_list.append(file_path)
            labels_list.append(duet_class_mapping.get(category))

shuffled_files,shuffled_labels=sync_shuffle(files_list,labels_list)



# def _load_model(self, directory, strict_loading=True, **kwargs):
#     self.fscratch = False
#     print('Loading pretrained weights')
#     if isinstance(directory, dict):
#         state_dict = directory
#     else:
#         state_dict = torch.load(directory, map_location=lambda storage, loc: storage)
#     new_state_dict = OrderedDict()
#
#     for k, v in state_dict.items():
#         if k[:7] == 'module.':
#             name = k[7:]  # remove `module.`
#         else:
#             name = k
#         new_state_dict[name] = v
#     self.model.load_state_dict(new_state_dict, strict=strict_loading)

data_transforms = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
# image_datasets = {x: datasets.ImageFolder(os.path.join(PATH_TO_TRAIN_VAL_DIR, x),
#                                           data_transforms[x])
#                   for x in [phase]}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
#                                               num_workers=4)
#               for x in [phase]}

class createDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, datapoints_list,labels_list,transforms=None):
        self.datapoints_list=datapoints_list
        self.labels_list=labels_list
        self.transforms=transforms

    def __getitem__(self, index):
        file_path=self.datapoints_list[index]
        label=self.labels_list[index]
        image = Image.open(file_path)
        #data =  self.to_tensor(image)
        if self.transforms is not None:
            trans_data = self.transforms(image)
        return (trans_data, label)

    def __len__(self):
        return len(self.datapoints_list)  # of how many examples(images?) you have

#dataset_sizes = {x: len(image_datasets[x]) for x in [phase]}

dataset = createDataset(shuffled_files,shuffled_labels,data_transforms)
dataloader= torch.utils.data.DataLoader(dataset,
                                    batch_size=1, shuffle=False,
                                    num_workers=4)


#weights = torch.load(PATH_TO_RETRAINED_RESNET_WEIGHTS)
state_dict = torch.load(PATH_TO_RETRAINED_RESNET_WEIGHTS, map_location=lambda storage, loc: storage)
new_state_dict = OrderedDict()
# for k, v in state_dict['state_dict'].items():
#     if k[:7] == 'module.':
#         name = k[7:]  # remove `module.`
#     else:
#         name = k
#     new_state_dict[name] = v

model= models.resnet152()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model = torch.nn.DataParallel(model)

model.load_state_dict(state_dict['state_dict'])
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Iterate over data.
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    #labels = labels.to(device)

    outputs = model(inputs)
    scores = torch.nn.functional.sigmoid(outputs)*100

print ('FKI')