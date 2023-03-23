#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset structure

project/
    -code/
        - config.yml
        - ...
    dataset/
    - metadata.csv
    - wavs/
        - 0001.wav
        - 0002.wav
        - etc
    - mels/
        - 0001.pt
        - 0002.pt
        - etc
    - filelists/
        - audio_text_train_filelist.txt 
        - audio_text_test_filelist.txt 
        - audio_text_val_filelist.txt 
        - mel_text_train_filelist.txt 
        - mel_text_test_filelist.txt 
        - mel_text_val_filelist.txt 
    - recordings/
        - recording_1.wav
        - recording_2.wav
        - etc
    - transcripts/
        - 0001.json
        - 0002.json
        - etc
        
recordings/
The original audio recordings, each file may be several hours in length.

wavs/
The audio clips, 2 to 20 seconds in length, extracted from the recordings.

transcripts/
The output of Amazon Transcribe.

mels/
Mel spectrograms corresponding to each audio clip.

metadata.csv
metadata.csv is a file with a single column, and a row per audio file. It looks like this:

0002|capture one the infinite variety.
0003|one of these creatures are quite likely to be un described by science.
0004|the difficulty will be to find specialists who know enough about the group's concerned to be able to single out the new one.
0005|no one can say just how many species of animals there are in these greenhouse humid, dimly lit jungles.
0006|they contain the richest and the most varied assemblage of animals and plant life to be found anywhere on Earth.
0007|the rate at which this happens can be estimated.

filelist/
The file lists tell the models which data to use for training, testing, and validation. They are tab separated files, the first column is the path relative to the dataset folder, and the second column is the transcript.
Here is an excerpt:

dataset/wavs/0882.wav|although their little known and infrequently seen, they are enormously abundant.
dataset/wavs/0489.wav|they developed gas filled floatation tanks.
dataset/wavs/3502.wav|it will breach in this way again and again.
dataset/wavs/2540.wav|only one reptile today lack such an organ, A strange lizard like creature that lives on a few small islands in New Zealand, the tuatara.

STEPS:
Step 1. Get speech data
Step 2. Split recordings into audio clips
Step 3. Automatically transcribe clips with Amazon Transcribe
Step 4. Make metadata.csv and filelists
Step 5. Download scripts from DeepLearningExamples
Step 6. Get mel spectrograms

"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

path = 'dataset/transcripts/'
files = os.listdir(path)

# Extract the transcript text from the output of Amazon Transcribe
rows = []
for index in range(1, len(files) + 1):  
    name =  '{:04d}'.format(index)
    with open(os.path.join(path, "{}.json".format(name))) as json_file:
        data = json.load(json_file)
        transcript = data["results"]["transcripts"][0]["transcript"]
        rows.append([name, transcript])
   
data = pd.DataFrame(rows, columns = ["name", "transcript"])

# Some of the audio clips may be empty or noise
# Remove  rows with empty transcripts
data = data[data.transcript != ""]

# Add new columns
data["wav_path"] = data["name"].apply("dataset/wavs/{}.wav".format)
data["mel_path"] = data["name"].apply("dataset/mels/{}.pt".format)
data["metadata"] = data["name"] + "|" + data["transcript"]
data["wav_text"] = data["wav_path"] + "|" + data["transcript"]
data["mel_text"] = data["mel_path"] + "|" + data["transcript"]

# Split files intro training, testing, and validation
train, test = train_test_split(data, test_size = 0.2, random_state = 1)
test, val = train_test_split(test, test_size = 0.05, random_state = 1)

metadata = data["metadata"]
audio_text_test_filelist =   test["wav_text"]
audio_text_train_filelist = train["wav_text"]
audio_text_val_filelist =     val["wav_text"]
mel_text_test_filelist =     test["mel_text"]
mel_text_train_filelist =   train["mel_text"]
mel_text_val_filelist =       val["mel_text"]

metadata.to_csv("dataset/metadata.csv", index = False)
os.mkdir("dataset/filelists")
np.savetxt("dataset/filelists/audio_text_test_filelist.txt", audio_text_test_filelist.values, fmt = "%s")
np.savetxt("dataset/filelists/audio_text_train_filelist.txt", audio_text_train_filelist.values, fmt = "%s")
np.savetxt("dataset/filelists/audio_text_val_filelist.txt", audio_text_val_filelist.values, fmt = "%s")
np.savetxt("dataset/filelists/mel_text_test_filelist.txt", mel_text_test_filelist.values, fmt = "%s")
np.savetxt("dataset/filelists/mel_text_train_filelist.txt", mel_text_train_filelist.values, fmt = "%s")
np.savetxt("dataset/filelists/mel_text_val_filelist.txt", mel_text_val_filelist.values, fmt = "%s")
