#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:41:59 2016

@author: danny
"""
from label_func import label_frames, parse_transcript
from data_functions import list_files, check_files
from create_features import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
from scipy.io.wavfile import read
import numpy

# the processing pipeline can be run in two ways. Either just create features (raw frames
# frequency spectrum, filterbanks or mfcc) or create both features and label them.
def features (params):
    # get list of audio and transcript files 
    audio_files = []
    for line in params[5]:
        audio_files.append(line.split())

    for x in range (0,len(audio_files)):
        print('converting file: \n' + str(x))
        # read audio samples
        input_data = read(audio_files[x][0])
        # sampling frequency
        fs = input_data[0]
        
        data = input_data[1]
        # get window and frameshift size in samples
        window_size = int(fs*params[2])
        frame_shift = int(fs*params[3])
        
        # create features
        if params[4] == 'raw':
            [features, energy] = raw_frames(data, frame_shift, window_size)
        
        elif params[4] == 'freq_spectrum':
            [frames, energy] = raw_frames(data, frame_shift, window_size)
            features = get_freqspectrum(frames, params[0], fs, window_size)
        
        elif params[4] == 'fbanks':
            [frames, energy] = raw_frames(data, frame_shift, window_size)
            freq_spectrum = get_freqspectrum(frames, params[0], fs, window_size)
            features = get_fbanks(freq_spectrum, params[1], fs) 
            
        elif params[4] == 'mfcc':
            [frames, energy] = raw_frames(data, frame_shift, window_size)
            freq_spectrum = get_freqspectrum(frames, params[0], fs, window_size)
            fbanks = get_fbanks(freq_spectrum, params[1], fs)
            features = get_mfcc(fbanks)
            
        # add the frame energy if needed
        if params[7]:
            features = numpy.concatenate([energy[:,None], features],1)
        # add the deltas and double deltas if needed
        if params[6]:
            single_delta= delta (features,2)
            double_delta= delta(single_delta,2)
            features= numpy.concatenate([features,single_delta,double_delta],1)
        
        numpy.save(audio_files[x][1], features)
    return 
