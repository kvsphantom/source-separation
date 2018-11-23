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

hop_length = 1024
n_fft = 4096
win_length =4096

classes={'cf'}
for root,dirs,files in sorted(os.walk(os.path.join(PATH_TO_AV_DUMPS,'cf'))):
    for dir in sorted(dirs):
        audio_file=os.path.join(root,dir,'audio',dir+'.wav')
        data,fs = librosa.load(audio_file,sr=None)
        target_fs = 48000
        resampled_data=librosa.resample(data, fs, target_fs)
        D = librosa.stft(resampled_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        magnitude, phase = librosa.magphase(D)
        plt.figure()
        #librosa.display.specshow(magnitude, y_axis='log')
        #plt.show()
        model = NMF(n_components=9, init='random', random_state=0)
        W = model.fit_transform(magnitude)
        #librosa.display.specshow(W, y_axis='log')
        #plt.show()
        #H = model.components_


