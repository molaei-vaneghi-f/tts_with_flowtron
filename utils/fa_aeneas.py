import os
import json

fa_db = "~/tts_data/fa_aeneas/output"
json_name = "omr.json"

idx_list = []
dur_list = []
line_list = []

with open(os.path.join(fa_db, json_name), 'r') as f:
  json_file = json.load(f)
  # Pretty Printing JSON string back
  print(json.dumps(json_file, indent = 4, sort_keys=True))


  # read different elements of json components/fragments and organize them in a list to be used for cutting/stitching of the audio (stat+dynamic)
  for fragments in json_file['fragments']:
      
      idx = fragments['id']
      begin = fragments['begin']
      end = fragments['end']
      line = fragments['lines']
      
      print(f'id: {idx} - being at: {begin} - end at: {end} \n line: {line} \n')
      
      idx_list.append(idx)
      dur_list.append([float(begin), float(end)])
      line_list.append(line)

#%% reading in audio files

from scipy.io import wavfile

audio_dir = "~/tts_data/synthesized_audio_files/"

# read in the original story (dynamic + static parts) as an np array
omr_sr, omr_audio = wavfile.read(os.path.join(audio_dir,'omr_story_speakerid_01_id4051_iter4_sig0.75_nframe1000_he.wav'))

# read in the synthesized audio files corresponding to dynamic parts
person_sr, person_audio = wavfile.read(os.path.join(audio_dir,'person.wav'))
location_sr, location_audio = wavfile.read(os.path.join(audio_dir,'location.wav'))
sweet_sr, sweet_audio = wavfile.read(os.path.join(audio_dir,'sweet.wav'))
sweet_loc_sr,  sweet_loc_audio = wavfile.read(os.path.join(audio_dir,'sweet_loc.wav'))

#%%
# static1 = omr_audio []




























