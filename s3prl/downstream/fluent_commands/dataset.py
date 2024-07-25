import os
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
import torch.nn as nn
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000
EXAMPLE_WAV_MAX_SEC = 10


class FluentCommandsDataset(Dataset):
    def __init__(self, df, base_path, Sy_intent, **kwargs):
        self.df = df
        self.base_path = base_path
        self.max_length = SAMPLE_RATE * EXAMPLE_WAV_MAX_SEC
        self.Sy_intent = Sy_intent
        self.upstream_name = kwargs['upstream']
        self.features_path = kwargs['features_path']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.base_path, self.df.loc[idx].path)
        wav, sr = torchaudio.load(wav_path)

        wav = wav.squeeze(0)

        label = []

        for slot in ["action", "object", "location"]:
            value = self.df.loc[idx][slot]
            label.append(self.Sy_intent[slot][value])
        
        if self.features_path:
            feature_path = os.path.join(self.features_path, self.upstream_name, f"{Path(wav_path).stem}.pt")
            if os.path.exists(feature_path):
                feature = torch.load(feature_path)
                return feature, label, True

        return wav.numpy(), np.array(label), Path(wav_path).stem

    def collate_fn(self, samples):
        return zip(*samples)
