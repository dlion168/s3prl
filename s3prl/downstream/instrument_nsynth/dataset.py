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

class InstrumentDataset(data.Dataset):
    def __init__(self, metadata_dir, split, sample_duration=None, return_audio_path=True):
        # self.cfg = cfg
        self.metadata_dir = os.path.join(metadata_dir, f'nsynth-{split}/examples.json')
        self.metadata = json.load(open(self.metadata_dir,'r'))
        self.metadata = [(k + '.wav', v['instrument_family_str']) for k, v in self.metadata.items()]

        self.audio_dir = os.path.join(metadata_dir, f'nsynth-{split}')
        self.class2id = {
            'bass': 0,
            'brass': 1,
            'flute': 2,
            'guitar': 3,
            'keyboard': 4,
            'mallet': 5,
            'organ': 6,
            'reed': 7,
            'string': 8,
            'synth_lead': 9,
            'vocal': 10
        }
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = 16000
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None
    
    def label2class(self, id_list):
        return [self.id2class[id] for id in id_list]
    
    def __getitem__(self, index):
        audio_path = self.metadata[index][0]
        
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

        # label = self.class2id[audio_path.split('/')[1].split('_')[0]]
        label = self.class2id[audio_path.rsplit('_', 2)[0]]
        if self.return_audio_path:
            return audio.numpy(), label, audio_path
        return audio.numpy(), label

    def __len__(self):
        return len(self.metadata)
    
    def collate_fn(self, samples):
        return zip(*samples)

