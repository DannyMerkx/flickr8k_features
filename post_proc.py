#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:00:26 2017

@author: danny
"""

import argparse
import numpy
import os 

def truncate(input_data, input_len):
    # truncate the data to the number of frames given by input_len
    trunc = len(input_data) - input_len
    if numpy.mod(trunc,2) == 0:
        trunc_begin = int(trunc/2)
        trunc_end = int(trunc/2)
    else:
        trunc_begin = int(numpy.floor(trunc/2))
        trunc_end = int(numpy.ceil(trunc/2))
    assert len(input_data[trunc_begin:-trunc_end, :]) == input_len
    
    return(input_data[trunc_begin:-trunc_end, :])
   
def pad_input(input_data, input_len):
    # pad the data to the number of frames given by input_len
    padsize = input_len -len(input_data)
    if numpy.mod(padsize,2) == 0:
        pad_begin = int(padsize/2)
        pad_end = int(padsize/2)
    else:
        pad_begin = int(numpy.floor(padsize/2))
        pad_end = int(numpy.ceil(padsize/2))
    input_data = numpy.pad(input_data, ((pad_begin, pad_end),(0,0)), 'constant', constant_values =(0,0)) 
    assert len(input_data) == input_len
    
    return(input_data)

parser = argparse.ArgumentParser(description='get the spectral mean')
parser.add_argument('train_list', type=argparse.FileType('r'))
parser.add_argument('val_list', type=argparse.FileType('r'))
parser.add_argument('test_list', type=argparse.FileType('r'))
parser.add_argument('wav2speak', type=argparse.FileType('r'))
args = parser.parse_args()

# get the matched speaker ids/ wav files from the wav2speak.txt
files = []
for line in args.wav2speak:
    files.append(line.split())

# create a dictionary of the wav files and speaker ids
files = dict(files)

# get the number of speakers
n_speak = 0
for keys in files:
    if int(files[keys]) > n_speak:
        n_speak = int(files[keys])

# create arrays to hold the mean, variance and file per speaker
means = [[] for x in range(n_speak)]
variance = [[] for x in range(n_speak)]
count = [[] for x in range(n_speak)]
# desired output length in frames
out_len = 1024

for line in args.train_list:
    # some code to extract locations/filenames from the text files
    paths = line.split()
    base = os.path.basename(paths[0])
    speaker_id = files[base]
    # load the features npy file
    feats = numpy.load(paths[1] + '.npy')
    # sum of the energies over all filterbanks
    eSum = numpy.sum(feats,1)
    # energy of the frame with the highest energy
    eMax = max(eSum)
    # simple voice activation detection, take all frames with at least 5% of the
    # max energy
    VAD = eSum >= (0.05*eMax)
    # take only the non-silence frames for variance and mean calculation
    vadFeats = [feats[x,:] for x in range(len(feats)) if VAD[x]]
    # get the mean, variance and utterence length (for weighting later)
    means[speaker_id-1].append(numpy.mean(vadFeats, 0))
    variance[speaker_id-1].append(numpy.var(vadFeats, 0))
    count[speaker_id-1].append(len(vadFeats))
    
for line in args.val_list:
    paths = line.split()
    base = os.path.basename(paths[0])
    speaker_id = files[base]
    
    feats = numpy.load(paths[1] + '.npy')
    
    eSum = numpy.sum(feats,1)
    eMax = max(eSum)
    
    VAD = eSum >= (0.05*eMax)
    vadFeats = [feats[x,:] for x in range(len(feats)) if VAD[x]]
    
    means[speaker_id-1].append(numpy.mean(vadFeats, 0))
    variance[speaker_id-1].append(numpy.var(vadFeats, 0))
    count[speaker_id-1].append(len(vadFeats))
    
for line in args.test_list:
    paths = line.split()
    base = os.path.basename(paths[0])
    speaker_id = files[base]
    
    feats = numpy.load(paths[1] + '.npy')
    
    eSum = numpy.sum(feats,1)
    eMax = max(eSum)
    
    VAD = eSum >= (0.05*eMax)
    vadFeats = [feats[x,:] for x in range(len(feats)) if VAD[x]]
    
    means[speaker_id-1].append(numpy.mean(vadFeats, 0))
    variance[speaker_id-1].append(numpy.var(vadFeats, 0))
    count[speaker_id-1].append(len(vadFeats))

# now we have the var and mean per file we need to combine them per speaker as a 
# weighted sum
for x in range(n_speak):
    # weight for the means and variances based on caption length
    count[x] = [z/max(count[x]) for z in count[x]]
    for y in range(len(variance[x])):
        # multiply means and variances by weight
        means[x][y] = means[x][y] * count[x][y]
        variance[x][y] = variance[x][y] * count[x][y]
    # take the sum for each speaker and divide by number of files for the speaker
    means[x] = sum(means[x])/len(count[x])
    variance[x] = sum(variance[x])/len(count[x])

for line in args.train_list:
    paths = line.split()
    base = os.path.basename(paths[0])
    speaker_id = files[base]
    # load the features and subtract the mean and normalise the variance
    feats = numpy.load(paths[1] + '.npy')
    feats = feats - means[speaker_id-1]
    feats = feats / variance[speaker_id-1]
    # truncate/pad as needed
    if len(feats) > out_len:
        feats = truncate(feats, out_len)
    if len(feats) < out_len:
        feats = pad_input(feats, out_len)
    # save the new features
    numpy.save(paths[1], feats)
    
for line in args.val_list:
    paths = line.split()
    base = os.path.basename(paths[0])
    speaker_id = files[base]
    
    feats = numpy.load(paths[1] + '.npy')
    feats = feats - means[speaker_id-1]
    feats = feats / variance[speaker_id-1]
    
    if len(feats) > out_len:
        feats = truncate(feats, out_len)
    if len(feats) < out_len:
        feats = pad_input(feats, out_len)
        
    numpy.save(paths[1], feats)
    
for line in args.test_list:
    paths = line.split()
    base = os.path.basename(paths[0])
    speaker_id = files[base]
    
    feats = numpy.load(paths[1] + '.npy')
    feats = feats - means[speaker_id-1]
    feats = feats / variance[speaker_id-1]
    
    if len(feats) > out_len:
        feats = truncate(feats, out_len)
    if len(feats) < out_len:
        feats = pad_input(feats, out_len)
        
    numpy.save(paths[1], feats)