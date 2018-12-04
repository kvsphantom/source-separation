#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:36:25 2018

@author: kvshenoy
"""

import os
import numpy as np
import cv2
import sys
import math

sys.path.append('../')
from settings import *


def create_folder(path):
    if not os.path.exists(path):
        os.umask(0) #To mask the permission restrictions on new files/directories being create
        os.makedirs(path,0o755) # setting permissions for the folder

def extract_frames(src_path,dest_path,req_fps,start_times,stop_times,video_metric_log,empty_frame_log):
    cap = cv2.VideoCapture(src_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    segment = real_fps/float(req_fps)

    with open(video_metric_log, "a") as video_metric_log_file:
        video_metric_log_file.write(os.path.basename(src_path)+","+str(cap.get(cv2.CAP_PROP_FPS))+"\n")

    for index in range(len(start_times)):
        start = start_times[index] * 1000 +500  # in milliseconds
        stop = stop_times[index] * 1000  -500# in milliseconds

        global_frame_number =math.ceil(start*real_fps/float(1000))
        sample_pos = 0
        while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) <= stop:
            selected_sample_number = round((segment * sample_pos) + segment / float(2))  # mid sample from each segment is selected
            sample_pos += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame_number +selected_sample_number - 1)
            ret, image = cap.read()
            if (ret != True and (global_frame_number+selected_sample_number)<total_frames):
                with open(empty_frame_log, "a") as empty_frame_log_file:
                    empty_frame_log_file.write(os.path.basename(src_path) + "," +str(int(global_frame_number+selected_sample_number))+"\n")
                continue
            basename = os.path.join(dest_path, os.path.basename(src_path)[:-4])
            filename = os.path.join(dest_path, basename + "_" + str(int(global_frame_number+selected_sample_number)) + ".png")
            if image is not None:
                resized_image = cv2.resize(image, (224, 224))  # Resize each of the frames to 224X224
                cv2.imwrite(filename, resized_image)
            if (ret != True):
                break
    cap.release()
    #cv2.destroyAllWindows()

fps=0.01 #setting frames per second that we need to sample from each of the video
classes={'cello','clarinet','erhu','flute','trumpet','tuba','violin','xylophone'}
video_metric_log=os.path.join(PATH_TO_LOG_DIR,'metric_log.csv')
create_folder(PATH_TO_LOG_DIR)
empty_frame_log=os.path.join(PATH_TO_LOG_DIR,'empty_frame_log.csv')
with open(video_metric_log, "a") as video_metric_log_file:
    video_metric_log_file.write("ID,FPS\n")
with open(empty_frame_log, "a") as empty_frame_log_file:
    empty_frame_log_file.write("ID,FRAME\n")

create_folder(PATH_TO_UNCUT_DATASET_FRAMES)
for each in classes:
    folder_path=os.path.join(PATH_TO_UNCUT_DATASET_FRAMES,each)
    create_folder(folder_path)

time_dict=np.load(TIMESTAMPS)
for entry in time_dict:
    filename=entry['file']
    filepath=os.path.join(PATH_TO_UNCUT_DATASET,filename+'.mp4')
    category=filename[:-2]
    dest_path = os.path.join(PATH_TO_UNCUT_DATASET_FRAMES,category)
    if category in classes: #selecting only the instrument classes with one instrument
        extract_frames(filepath,dest_path,fps,entry['to'],entry['tf'],video_metric_log,empty_frame_log)
    else:
        continue
