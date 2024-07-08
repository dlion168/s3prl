import numpy as np 
import os 
import torchaudio

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')

import os

import numpy as np
from torch.utils import data
import pandas as pd
import torch.nn.functional as F

class SingerDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, sample_duration=None, return_audio_path=True):
        # self.cfg = cfg
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_s.txt'), 
                                    names = ['audio_path'])
        self.audio_dir = audio_dir
        self.class2id = {'f1':0, 'f2':1, 'f3':2, 'f4':3, 'f5':4, 'f6':5, 'f7':6, 'f8':7, 'f9':8, 'm1':9, 'm2':10, 'm3':11, 'm4':12, 'm5':13, 'm6':14, 'm7':15, 'm8':16, 'm9':17, 'm10':18, 'm11':19}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = 16000
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None
    
    def __getitem__(self, index):
        audio_path = self.metadata.iloc[index][0]
        
        wav, sr = torchaudio.load(audio_path)
        wav = F.resample(wav, orig_freq=sr, new_freq=16000)
        audio = wav.squeeze(0)
        
        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[0] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[0]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]

        # # convert
        # audio_features = self.processor(audio, return_tensors="pt", sampling_rate=self.cfg.target_sr, padding=True).input_values[0]
        
        label = self.class2id[audio_path.split('/')[1].split('_')[0]]
        if self.return_audio_path:
            return wav.numpy(), label, audio_path
        return wav.numpy(), label

    def __len__(self):
        return len(self.metadata)
    
    def collate_fn(self, samples):
        return zip(*samples)

