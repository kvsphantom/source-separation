"""
Created on 11/15/18 11:32 AM    


@author: kvshenoy
"""

import os
import sys
import numpy as np
import subprocess
import shlex


sys.path.append('../')
from settings import *
sys.path.append('../utils/')
import utils
import video_utils

classes={'cello','clarinet','erhu','flute','trumpet','tuba','violin','xylophone'}
#         'cf','clc','clf','clt','cltu','tf','tut','tuv'}
duet_class_mapping={'cf':['cello','flute'],
                    'clc':['cello','clarinet'],
                    'clf':['clarinet','flute'],
                    'clt':['clarinet','trumpet'],
                    'cltu':['clarinet','tuba'],
                    'clv':['clarinet','violin'],
                    'ct':['cello','trumpet'],
                    'ec':['erhu','cello'],
                    'ef': ['erhu', 'flute'],
                    'tf':['trumpet','flute'],
                    'tut':['trumpet','tuba'],
                    'tuv':['tuba','violin'],
                    'vc':['cello','violin'],
                    'vf':['flute','violin'],
                    'vt':['trumpet','violin'],
                    'xf':['flute','xylophone']
                   }


utils.create_folder(PATH_TO_AV_DUMPS)

for root,dirs,files in sorted(os.walk(PATH_TO_CUT_DATASET)):
    for file in sorted(files):
        print('[FOLDER] : '+file[:-4])
        #filepath=os.path.join(PATH_TO_ORIGINAL_DATASET,filename+'.mp4')
        category=os.path.basename(root)
        if category in classes: # only choose videos with more than one instruments
            continue
        else:
            ### EXTRACT VISUALS FROM THE VIDEO SEGMENTS
            visuals_dump_path=os.path.join(PATH_TO_AV_DUMPS,category,file[:-4],'visuals')
            utils.create_folder(visuals_dump_path)
            src_path=os.path.join(root,file)
            target_fps=1
            frames=video_utils.get_frames(target_fps,src_path,visuals_dump_path)
            ### EXTRACT WAV FILES FROM THE VIDEO SEGMENTS
            audio_dump_folder = os.path.join(PATH_TO_AV_DUMPS, category, file[:-4], 'audio')
            utils.create_folder(audio_dump_folder)
            audio_dump_path=os.path.join(audio_dump_folder,file[:-4]+'.wav')
            video_utils.get_audio(src_path,audio_dump_path)


