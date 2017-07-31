#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:41:47 2017

@author: danny
"""
from preproc import four,pad,preemph, hamming
from filters import apply_filterbanks,filter_centers, create_filterbanks
from scipy.fftpack import dct
import numpy
import math

# this file contains several feature creation steps. 

def delta (data, n):
# calculate delta features, n is the number of frames to look forward and backward
    dt = []
    
    for j in range (0,data.shape[0]):
        temp = []
        
        for i in range (1, n+1):
            
            if j-i >= 0 and not j+i > (data.shape[0]-1):
                temp.append(n*(data[j+i] - data[j-i]))
                
            elif j-i < 0:
                temp.append(n*(data[j+i]))
            
            else:
                temp.append(n* (0 - data[j-i]))
            
            temp2 = 2 * sum([x*x for x in range (1,n+1)])
        dt.append(sum(temp)/temp2)
        
    return (numpy.array(dt))

def raw_frames(input_data, frame_shift, window_size):
# this function cuts the data into frames and calculates each frames' accuracy

    #determine the number of frames to be extracted
    nframes = math.floor(input_data.size/frame_shift)
    # pad the data
    data = pad(input_data, window_size, frame_shift)
    # slice the frames from the wav file
    # keep a list with the frames and all the values of the samples and 
    # list with the start and end sample# of each frame
    frames = []
    energy = []
    
    for f in range (0, nframes):
        frame = data[f * frame_shift : f * frame_shift + window_size]
        energy.append(numpy.log(numpy.sum(numpy.square(frame), 0)))
        frames.append(frame)
    
    frames = numpy.array(frames)
    energy = numpy.array(energy)  
    # if energy is 0 , the log can not be taken(results in -inf) so we set the 
    # log energy to -50 (log of 2e-22 or approx 0 )
    energy[energy == numpy.log(0)] = -50
    
    return (frames, energy)

def get_freqspectrum(frames, alpha, fs, window_size):
# this function prepares the raw frames for conversion to frequency spectrum
# and applies fft

     # apply preemphasis
    frames = preemph(frames, alpha)
    # apply hamming windowing
    frames = hamming(frames)
    # apply fft 
    freq_spectrum = four(frames,fs,window_size)
    
    return freq_spectrum

def get_fbanks(freq_spectrum, nfilters, fs):
#  this function calculates the filters and creates filterbank features from
#  the fft features
    
    # get the frequencies corresponding to the bins returned by the fft
    xf = numpy.linspace(0.0, fs/2, numpy.shape(freq_spectrum)[1])
    # get the filter frequencies
    fc = filter_centers (nfilters,fs,xf)
    # create filterbanks
    filterbanks = create_filterbanks(nfilters, xf, fc)
    # apply filterbanks
    fbanks = apply_filterbanks(freq_spectrum, filterbanks)    
    
    return fbanks

def get_mfcc(fbanks):
# this function creates mfccs from the fbank features
    
    # apply discrete cosine transform to get mfccs. According to convention, 
    # we discard the first filterbank (which is roughly equal to the method 
    # where we only space filters from 1000hz onwards)
    mfcc = dct(fbanks[:,1:])
    # discard the first coefficient of the mffc as well and take the next 13
    # coefficients.
    mfcc = mfcc[:,1:13]
    
    return mfcc
