import numpy as np 
import os 
import torchaudio

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')

import os

import numpy as np
from torch.utils import data
import pandas as pd
import torch.nn.functional as F
import json

class PitchClassiDataset(data.Dataset):
    def __init__(self, metadata_dir, split, sample_duration=None, return_audio_path=True):
        # self.cfg = cfg
        self.metadata_dir = os.path.join(metadata_dir, f'nsynth-{split}/examples.json')
        self.metadata = json.load(open(self.metadata_dir,'r'))
        self.metadata = [(k + '.wav', v['pitch']) for k, v in self.metadata.items()]

        self.audio_dir = os.path.join(metadata_dir, f'nsynth-{split}')
        self.return_audio_path = return_audio_path
        self.sample_rate = 16000
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None
    
    def label2class(self, id_list):
        return [ id+9 for id in id_list]
    
    def __getitem__(self, index):
        audio_path = self.metadata[index][0]
        label = self.metadata[index][1] - 9
        
        wav, sr = torchaudio.load(os.path.join(self.audio_dir, "audio", audio_path))
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        audio = wav.squeeze()

        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[0] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[0]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[random_start:random_start+self.sample_duration]
        
        if self.return_audio_path:
            return audio.numpy(), label, audio_path
        return audio.numpy(), label

    def __len__(self):
        return len(self.metadata)
    
    def collate_fn(self, samples):
        return zip(*samples)

