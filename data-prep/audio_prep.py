"""
Created on 11/23/18 7:38 AM    


@author: kvshenoy
"""
import os
import sys
sys.path.append('../')
from settings import *
sys.path.append('../utils/')
from utils import create_folder
import video_utils
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

import librosa
import librosa.display
from scipy.io import wavfile
from natsort import natsorted, ns
import numpy as np

hop_length = 1024
n_fft = 4096
win_length =4096

bases=[]
classes=[i for i in os.listdir(PATH_TO_AV_DUMPS)]
for category in natsorted(classes, key=lambda y: y.lower()):
    for root,dirs,files in os.walk(os.path.join(PATH_TO_AV_DUMPS,category)):
        for dir in natsorted(dirs, key=lambda y: y.lower()):
            audio_file=os.path.join(root,dir,'audio',dir+'.wav')
            data,fs = librosa.load(audio_file,sr=None)
            target_fs = 48000
            resampled_data=librosa.resample(data, fs, target_fs)
            D = librosa.stft(resampled_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            magnitude, phase = librosa.magphase(D)
            plt.figure()
            #librosa.display.specshow(magnitude, y_axis='log')
            #plt.show()
            model = NMF(n_components=10, init='random', random_state=0)
            W = model.fit_transform(magnitude)
            #librosa.display.specshow(W, y_axis='log')
            #plt.show()
            #H = model.components_
            path_to_bases=os.path.join(root,dir,'bases')
            bases.append(path_to_bases+'.npy')
            np.save(path_to_bases,W)

np.save('bases_paths',bases)


