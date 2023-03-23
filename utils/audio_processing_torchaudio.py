
# !pip install torchaudio librosa boto3
import os
import sys
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

#%% universal path
dp_git = os.path.join(os.path.dirname(os.path.realpath(__file__)).split("git")[0]+"git")
# sys.path.append(os.path.join(dp_git,'siddha'))
sys.path.append(os.path.join(dp_git,'../flowtron'))

audio_path = os.path.join(dp_git,'../flowtron/data/librispeech_tts_train_clean_100_speaker1088_wav')

#%% investigating the audio files


for index, audio_file in enumerate(os.listdir(audio_path)):
    
    # print metadata e.g. sampling rate, bits_per_sample, encoding, etc.
    if audio_file.endswith('.wav'):
        metadata = torchaudio.info(audio_path + '/' + audio_file)
        print(audio_file, '\n', metadata , '\n')
    
        # resampling audio file    
        waveform, sample_rate = torchaudio.load(audio_path + '/' + audio_file)
        
        F.print_stats(waveform, sample_rate=sample_rate)
        T.plot_waveform(waveform, sample_rate)
        T.plot_specgram(waveform, sample_rate)
        T.play_audio(waveform, sample_rate)


