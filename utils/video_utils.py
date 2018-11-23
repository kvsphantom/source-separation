#!/usr/bin/env python2
# -*- coding: utf-8 -*-
__author__ = "Juan Montesinos"
__year__ = "2018"
__version__ = "0.1"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"


import os
import re
import subprocess
import cv2
import sys
import numpy as np
import math
import skimage.io as skio

sys.path.append('../')
from settings import *

def _slice(tensor,dim,element):
    out = ()
    size = tensor.shape
    for i,s in enumerate(size):
        if i==dim:
            out += (element,)
        else:
            out += (slice(s),)
    return out
    
def historia(frames,style,typ=None):
    frames = np.asarray(frames)
    if type(typ) == str:
        if typ=='mean':
            mean = np.mean(frames,axis=0)
        elif typ=='mp':
            mean = np.amax(frames,axis=0)
        else:
            raise Exception('Not recognized type of hist')
    elif (type(typ) == np.ndarray) and (frames.shape[1:] == typ.shape):
        mean = typ
    else:
        raise Exception('Not recognized type of algorithm')
    dif = np.abs(frames-mean)
    if style == 'raw':
        return dif
    elif style == 'RGBc':
        dif /= 255
        return (frames*dif).astype(np.uint8)
    elif style == 'RGB':
        dif = np.amax(dif,axis=np.argmin(dif.shape))/255
        return (frames*mean).astype(np.uint8)
    elif style == 'gray':
        return np.amax(dif,axis=np.argmin(dif.shape)).astype(np.uint8)
        
def time2milisec(time):
    #Converts an input time mm.ss or mm:ss to amount of seconds
    if ':' in time:
        idx = time.find(':')
    elif '.' in time:
        idx = time.find('.')
    if idx == -1:
        print('Wrong input time2K function. Forman should be mm:ss')
        quit()
    minute = int(time[0:idx])
    second = int(time[idx+1:len(time)])
    if second > 60:
        print('Wrong input tim2K function. ss>60')
        quit()
    print(minute)
    print (second)
    milisec = minute*60+second
    return int(milisec)

def ffprobe2ms(time):
    cs = int(time[-2::])
    s  = int(os.path.splitext(time[-5::])[0])
    idx = time.find(':')
    h = int(time[0:idx-1])
    m = int(time[idx+1:idx+3])
    return [h,m,s,cs]

def get_video_metadata(filename,display=None):
    # Get length of video with filename
    time = None
    fps = None
    result = subprocess.Popen(["ffprobe", str(filename)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = [str(x) for x in result.stdout.readlines()]
    info_lines = [x for x in output if "Duration" in x or "Stream" in x]
    duration_line = [x for x in info_lines if "Duration" in x]
    fps_line = [x for x in info_lines if "Stream" in x]
    if duration_line:
        duration_str = duration_line[0].split(",")[0]
        pattern = '\d{2}:\d{2}:\d{2}.\d{2}'
        dt = re.findall(pattern, duration_str)[0]
        time = ffprobe2ms(dt)
    if fps_line:
        pattern = '(\d{2})(.\d{2})* fps'
        fps_elem = re.findall(pattern, fps_line[0])[0]
        fps = float(fps_elem[0]+fps_elem[1])
    if display == 's':
        time = time[0]*3600 + time[1]*60 + time[2] + time[3]/100.0
    elif display == 'ms':
        time = (time[0]*3600 + time[1]*60 + time[2] + time[3]/100.0)*1000
    elif display == 'min':
        time = (time[0]*3600 + time[1]*60 + time[2] + time[3]/100.0)/60
    elif display == 'h':
        time = (time[0]*3600 + time[1]*60 + time[2] + time[3]/100.0)/3600
    return time, fps

def get_video_real_info(filename,display=None):
    result = subprocess.Popen(["ffprobe", filename,'-show_entries','frame=coded_picture_number,pkt_pts,pkt_pts_time',
                               '-select_streams', 'v','-of','compact=p=0','-v','0'],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = [str(x) for x in result.stdout.readlines()]
    if sys.version_info[0] < 3:
        keypts = [[(x.find('=',x.find('pts'))+1,x.find('|',x.find('pts'))),(x.find('=',x.find('time'))+1,x.find('|',x.find('time'))),(x.find('=',x.find('coded'))+1,x.find('\n',x.find('coded')))] for x in output]
    else:
        keypts = [[(x.find('=',x.find('pts'))+1,x.find('|',x.find('pts'))),(x.find('=',x.find('time'))+1,x.find('|',x.find('time'))-1),(x.find('=',x.find('coded'))+1,x.find('\n',x.find('coded'))-2)] for x in output]
    pkt_pts = [int(x[k[0][0]:k[0][1]]) if k[0][0]!=k[0][1] else int(x[k[0][0]]) for k,x in zip(keypts,output) ]
    pkt_time = [float(x[k[1][0]:k[1][1]]) if k[1][0]!=k[1][1] else float(x[k[1][0]]) for k,x in zip(keypts,output) ]
    frame_number = [int(x[k[2][0]:k[2][1]]) if k[2][0]!=k[2][1] else int(x[k[2][0]]) for k,x in zip(keypts,output) ]
    idx = np.where(np.diff(np.sign(pkt_pts)))[0][0]+1
    FPS = (frame_number[-1]-idx)/(pkt_time[-1])
    return FPS,pkt_time[-1],list(zip(pkt_pts,pkt_time,frame_number)),idx

def _cut_video(filename,To,length,outname):
    #os.system('ffmpeg -y -ss '+str(To)+' -i '+filename+' -t '+str(length)+' -c copy '+outname)
    os.system('ffmpeg -y -ss '+str(To)+' -i '+filename+' -t '+str(length)+' -c copy '+outname)
    print(outname)

def cut_video(filename,To,length,outname):
    if type(filename) == list:
        N = len(filename)
        for i in range(N):
            _cut_video(filename[i],To[i],length[i],outname[i])
    else:
        _cut_video(filename,To,length,outname)
        
"""Sequential unique names for auto video cutting"""
def _svn(directory,originalname,To,length): #Structured Video Name
    return os.path.join(directory,os.path.splitext(originalname)[0]+'_'+str(To)+'to'+str(To+length)+'.mp4')

def svn(directory,originalname,To,length):
    if type(originalname) == list:
        N = len(originalname)
        out = []
        for i in range(N):
            out.append(_svn(directory[i],originalname[i],To[i],length[i]))
    else:
        out = _svn(directory,originalname,To,length)
    return out


"""=====================WARNING=============================="""
"""=====================WARNING=============================="""
"""=====================WARNING==============================
This way of loading frames is prepared to deal with videos which has been wronly cut and contains negative timestamps
and desynchronization between audio and video. This happens when, at the time of cutting, the initial time does not 
corresponds to a keyframe. 
As you can see these functions makes use of rm command. Setting a wrong directory may delete your files.
Use it at your own risk
"""


def ffmpeg_frames(video_dir, folder):
    os.system('ffmpeg -i {0} -loglevel error -f image2 {1}/%03d.png'.format(video_dir, folder))


def get_frames(target_fps,video_dir, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(video_dir), str(uuid.uuid4())[:7])
    dangerous_dirs = [f for f in os.listdir('/') if os.path.isdir('/' + f)]
    dangerous_dirs.append('/')
    crazy_pet = [int(out_dir == f) for f in dangerous_dirs]
    if np.max(crazy_pet) > 0:
        raise Exception('You are planning to delete a system folder... Check get_frames directories')
    S, FPS = get_video_metadata(video_dir, 's')
    FPS_real, S_real, timestamps, idx = get_video_real_info(video_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ffmpeg_frames(video_dir, out_dir)
    frame_list = sorted(os.listdir(out_dir))
    frames = [skio.imread(out_dir + '/' + img_dir) for i, img_dir in enumerate(frame_list) if
              (i >= idx and i <= timestamps[-1][2])]
    segment = FPS_real/float(target_fps)
    selected_frames=[int(round((segment * i) + segment / float(2))) for i in range(int(len(frames)/segment))]
    deselected_filenames=['%03d'%id+'.png' for id in range(len(frame_list)+1) if id not in selected_frames]
    deselected_filepaths=[os.path.join(out_dir, file) for file in deselected_filenames]
    [os.system('rm -rf {0}'.format(deselected_filepath)) for deselected_filepath in deselected_filepaths]
    return list(np.array(frames)[selected_frames])


def get_audio(video_dir, out_dir):
    subprocess.Popen(['ffmpeg',
                      '-i', video_dir, '-codec:a',
                      'pcm_s16le', '-ac', '1', out_dir],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def historia(frames, style, typ=None):
    frames = np.asarray(frames)
    if type(typ) == str:
        if typ == 'mean':
            mean = np.mean(frames, axis=0)
        elif typ == 'mp':
            mean = np.amax(frames, axis=0)
        else:
            raise Exception('Not recognized type of hist')
    elif (type(typ) == np.ndarray) and (frames.shape[1:] == typ.shape):
        mean = typ
    else:
        raise Exception('Not recognized type of algorithm')
    dif = np.abs(frames.astype(float) - mean)
    if style == 'raw':
        return dif.astype(np.uint8)
    elif style == 'RGBc':
        dif = dif / 255
        return (frames * dif).astype(np.uint8)
    elif style == 'RGB':
        dif = np.amax(dif, axis=np.argmin(dif.shape)) / np.max(dif)
        dif = np.stack([dif, dif, dif], axis=3)
        return (frames * dif).astype(np.uint8)
    elif style == 'gray':
        return np.amax(dif, axis=np.argmin(dif.shape)).astype(np.uint8)


def flow(video_dir,output_dir,filename):
    S,FPS= get_video_metadata(video_dir,'s')
    FPS_real,S_real,timestamps,idx = get_video_real_info(video_dir)
    expected_frames = int(math.ceil(FPS_real*6+1))
    directory = os.path.join(output_dir,filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    ffmpeg_frames(video_dir,directory)
    frame_list = sorted(os.listdir(directory))
    N_frames = len(frame_list)
    N_real =timestamps[-1][2]-idx

    frames = [skio.imread(directory+'/'+img_dir) for i,img_dir in enumerate(frame_list) if (i>=idx and i<=timestamps[-1][2])]
    N_useful = len(frames)
    if (expected_frames < N_useful):
        frames = frames[0:expected_frames+1]
    elif expected_frames > N_useful:
        print('\r\t'
                       'WARNING: insuficient amount of frames, review video \r\t')
    print('\r\t File : {0} \r\t'
                'FPS metadatos: {1} \r\t'
                'FPS real: {2} \r\t'
                'Length metadata: {3} \r\t'
                'Real length: {4} \r\t'
                'ffmpeg created frames: {5} \r\t'
                'frames enconded in the video: {6} \r\t'
                'Expected frames for 6s at FPS real: {7} \r\t'
                'Useful frames (timestamps>0): {8} \r\t'.format(filename,FPS,FPS_real,S,S_real,N_frames,N_real,expected_frames-1,N_useful))

class videoclass(object):
    def __init__(self,filedir):
        self.video = cv2.VideoCapture(filedir)
        self.nframes = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.FPS  = self.video.get(cv2.CAP_PROP_FPS)
        self.clip_length = float(self.nframes)/self.FPS
        
    def time2K(self,time):
        #Converts an input time mm.ss or mm:ss to amount of seconds
        if ':' in time:
            idx = time.find(':')
        elif '.' in time:
            idx = time.find('.')
        if idx == -1:
            raise Exception('Wrong input time2K function. Forman should be mm:ss')
        minute = int(time[0:idx])
        second = int(time[idx+1:len(time)])
        if second > 60:
            raise Exception('Wrong input tim2K function. ss>60')
        K = self.FPS*(minute*60+second)
        if K > self.nframes:
            raise Exception('Time points out of the clip')
        return K

    def _get_frame(self,time,resize = None ):
        K = self.FPS*time
        self.video.set(1,K)
        ret , frame = self.video.read()
        if ret:
            if resize is not None:
                return cv2.resize(frame,resize)
            else:
                return frame
        else:
            return -1

    def _get_frameK(self,K,resize=None):
        self.video.set(1,K)
        ret , frame = self.video.read()
        if ret:
            if resize is not None:
                return cv2.resize(frame,resize)
            else:
                return frame
        else:
            return -1
        
    def get_frames(self,times,close=True,formato='matplotlib',OF = -1,resize=None):
        frames = []
        offrames = []
        if formato == 'opencv':
            if type(times) == list:
                for time in sorted(times):
                    frames.append(self._get_frame(time,resize=resize))
                    if OF != -1:
                        K = time+OF
                        offrames.append(self._get_frameK(K,resize=resize))
                if close:
                    self.video.release()
                if OF !=-1:
                    return frames,offrames
                else:
                    return frames
            else:
                raise Exception('Required a list of times')
        elif formato == 'matplotlib':
            if type(times) == list:
                for time in sorted(times):
                    frames.append(self._get_frame(time,resize=resize)[:,:,::-1])
                    if OF != -1:
                        K = time+OF
                        offrames.append(self._get_frameK(K,resize=resize)[:,:,::-1])
                if close:
                    self.video.release()
                if OF !=-1:
                    return frames,offrames
                else:
                    return frames                
            else:
                raise Exception('Required a list of times')