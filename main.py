#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:19:53 2017

@author: danny
"""
# this script functions as a config file where you set the variables for the 
# feature creation and pass it to the function that actually does the work.
import argparse

from process_data import features

parser = argparse.ArgumentParser(description ='create fbank features')
parser.add_argument('speech_list', type = argparse.FileType('r'))
args = parser.parse_args()
# option for the features to return, can be raw for the raw frames 
# freq_spectrum for the fft transformed frames, fbanks for filterbanks or mfcc for mfccs. 
feat = 'fbanks'

file_list= args.speech_list
# some parameters for mfcc creation
params = []
# set alpha for the preemphasis
alpha = 0.97
# set the number of desired filterbanks
nfilters = 40
# windowsize and shift in seconds
t_window = .025
t_shift = .01

# desired sentence length in frames
output_fr = 1024
# option to include delta and double delta features
use_deltas = False
# option to include frame energy
use_energy = False
# put paramaters in a list
params.append(alpha)
params.append(nfilters) 
params.append(t_window)
params.append(t_shift)
params.append(feat)
params.append(file_list)
params.append(use_deltas)
params.append(use_energy)
params.append(output_s)

# call the function that actually does the work
features(params)
