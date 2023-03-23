#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import exists, join, basename, splitext
from flowtron import Flowtron
from data import Data
import sys
from glow import WaveGlow
from IPython.display import Audio
import matplotlib
import matplotlib.pylab as plt
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from nltk.tokenize import sent_tokenize
sys.path.insert(0, 'tacotron2')
sys.path.insert(0, 'tacotron2/waveglow')
plt.rcParams["axes.grid"] = False

db_dir = "~/tts_data/"
checkpoints_dir = os.path.join(db_dir,'flowtron_checkpoints/')
models_dir = os.path.join(db_dir,'pre_trained_models/')
in_aud_dir = os.path.join(db_dir,'input_audio_files/')
stories_dir = os.path.join(db_dir,'stories/')
out_aud_dir = os.path.join(db_dir,'synthesized_audio_files/')


mod_sp_id = "01_id4051_iter4"

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
# read config
config = json.load(open('./config_files/libritts_train_clean_100_id4051.json')) 
data_config = config["data_config"]
model_config = config["model_config"]
model_config['n_speakers'] = 1


data_config['training_files'] = 'filelists/libritts_train_clean_100_speaker4051_filelist_train.txt' 
data_config['validation_files'] = 'filelists/libritts_train_clean_100_speaker4051_filelist_val.txt'

# load flowtron
model = torch.load(os.path.join(checkpoints_dir, mod_sp_id), map_location='cpu')['model'].cuda() # good

state_dict = model.state_dict()
model.load_state_dict(state_dict)
_ = model.eval()
# load waveglow  
waveglow = torch.load(os.path.join(models_dir,'waveglow_256channels_universal_v5.pt'))['model'].cuda().eval()
waveglow.cuda()
for k in waveglow.convinv:
    k.float()
_ = waveglow.eval()


ignore_keys = ['training_files', 'validation_files']
trainset = Data(data_config['training_files'], **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))


def synthesize(speaker_id, text, sigma=0.8, n_frames=1000):
  speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
  text = trainset.get_text(text).cuda()
  speaker_vecs = speaker_vecs[None]
  text = text[None]
  
  with torch.no_grad():
    residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
    mels, attentions = model.infer(residual, speaker_vecs, text)
    audio = waveglow.infer(mels.float(), sigma).float()
  audio = audio.cpu().numpy()[0]
  # normalize audio for now
  audio = audio / np.abs(audio).max()
  audio = np.int16(audio * 32767)
  return audio 


#%% text normalization     

with open (stories_dir + 'story.txt') as file:
    list_of_sentences = sent_tokenize(file.read())
    for sent_no in range(len(list_of_sentences)):
        print ('sentence {}: {}'.format(sent_no, list_of_sentences[sent_no]) + '\n')

#%% voice synthesis
# generate audio files for each sentence and save it to disk
story_name = "story" 
speaker_id = 4051 #1088
sigma=0.75 
n_frames=1000

for sent_no in range(len(list_of_sentences)):
    syn_audio = synthesize(speaker_id,list_of_sentences[sent_no],sigma,n_frames) 
    # write the numpy array (syn_audio) into a wav file:
    write(out_aud_dir + story_name + "_speakerid_" + mod_sp_id +"_sentence{}_sig{}_nframe{}.wav".format(sent_no,sigma,n_frames), 22050, syn_audio)
  
#%% combining synthesized audio into one single wav file

# pip install pydub
from pydub import AudioSegment
combined_audio = AudioSegment.empty()
for sent_no in range (len(list_of_sentences)):
    combined_audio += AudioSegment.from_wav(out_aud_dir + story_name + "_speakerid_" + mod_sp_id +"_sentence{}_sig{}_nframe{}.wav".format(sent_no,sigma,n_frames))                                                                                  
    combined_audio.export(out_aud_dir + story_name + "_speakerid_" + mod_sp_id +"_sig{}_nframe{}.wav".format(sigma,n_frames), format="wav")
    
#%% audio post-processing

# run this cell to listen to the audio file, if there are many mistakes, you could re-synthesize the whole batch
Audio(filename = out_aud_dir + story_name + "_sig{}_nframe{}.wav".format(sigma,n_frames), rate=22050, autoplay=True)















