#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
sys.path.append('../')
from settings import *
sys.path.append('../utils/')
from utils import create_folder
import video_utils

times_dict=np.load(TIMESTAMPS)
for i,entry in enumerate(times_dict):
    To = entry['to']
    Tf = entry['tf']
    filename = entry['file']
    filepath = os.path.join(PATH_TO_UNCUT_DATASET, filename + '.mp4')
    category = filename[:-2]
    category_path=os.path.join(PATH_TO_CUT_DATASET, category)
    create_folder(category_path)

    N = len(To)
    namevec = [filename for _ in range(N)]
    dirvec =  [category_path for _ in range(N)]
    lenvec = np.add(np.subtract(Tf, To),np.ones(N, dtype=int))

    outvec = video_utils.svn(dirvec,namevec,To,lenvec)
    inputvec = [filepath for _ in range(N)]
    video_utils.cut_video(inputvec,To,lenvec,outvec)
    print('{0}% Completed'.format(i*100.0/len(times_dict)))