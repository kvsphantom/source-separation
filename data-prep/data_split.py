"""
Created on 5:55 PM

@author: kvshenoy
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
from settings import *

def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder

classes=['cello','clarinet','erhu','flute','trumpet','tuba','violin','xylophone']
create_folder(PATH_TO_TRAIN_VAL_DIR)

X=[]
y=[]
for path, subdirs, files in os.walk(PATH_TO_UNCUT_DATASET_FRAMES):
    for name in files:
        #file_list.append(name)
        #input_list.append(os.path.join(path, name))
        id= os.path.join(path, name)
        classname=classes.index(os.path.basename(path))
        X.append(id)
        y.append(classname)

X=np.array(X)
y=np.array(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
#np.save('./X_train',X_train)
#np.save('./y_train',y_train)
#np.save('./X_val',X_val)
#np.save('./y_val',y_val)

#X_train=np.load('X_train.npy')
#X_val=np.load('X_val.npy')
for x in X_train:
    category=os.path.basename(os.path.dirname(x))
    save_to_path=os.path.join(PATH_TO_TRAIN_VAL_DIR, 'train', category, os.path.basename(x))
    create_folder(os.path.dirname(save_to_path))
    copyfile(x,save_to_path)
for x in X_val:
    category = os.path.basename(os.path.dirname(x))
    save_to_path = os.path.join(PATH_TO_TRAIN_VAL_DIR, 'val', category, os.path.basename(x))
    create_folder(os.path.dirname(save_to_path))
    copyfile(x, save_to_path)
