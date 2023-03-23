#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% filelist text operations
import os
db_filelist = "~/tts_flowtron/filelists"

# always make a copy of the filelist before this operation
with open (os.path.join(db_filelist,'libritts_train_clean_100_speaker4051_filelist_siddha.txt')) as input_txt:
    text = input_txt.readlines()
    
    # level 1 replacement (speaker id)
with open(os.path.join(db_filelist,'libritts_train_clean_100_speaker4051_filelist_siddha.txt'), 'w') as output_txt:
    
    for line in text:
        if line.startswith('/path_to_libritts/4051/'):
            output_txt.write(line)
        else:
            pass

# level 2 replacement (chapters) e.g.:
    # path_to_libritts/4297/13006/... .wav => input_audio_files/librispeech_tts_train_clean_100_speaker1088_wav/... .wav
    # path_to_libritts/4297/13009/... .wav => input_audio_files/librispeech_tts_train_clean_100_speaker1088_wav/... .wav

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
