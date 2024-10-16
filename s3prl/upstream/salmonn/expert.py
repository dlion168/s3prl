from collections import OrderedDict
from typing import Dict, List, Union

import os
import yaml
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from transformers import WhisperFeatureExtractor
from ..interfaces import UpstreamBase
from .SALMONN.config import Config
from .SALMONN.models.salmonn import SALMONN    
from .SALMONN.utils import prepare_one_sample

class UpstreamExpert(UpstreamBase):
    def __init__(self, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        
        assert (
            os.path.isfile(kwargs['cfg_path'])
        ), "The config file does not exist: {}".format(kwargs['cfg_path'])

        
        self.cfg = OmegaConf.load(kwargs['cfg_path'])

        self.cfg.model.beats_path = kwargs['beats_path']
        self.cfg.model.ckpt = kwargs['ckpt']
        self.cfg.model.prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(kwargs['cfg_path'])),
            self.cfg.model.prompt_path
        )
        self.cfg.model.test_prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(kwargs['cfg_path'])),
            self.cfg.model.test_prompt_path
        )

        self.model = SALMONN.from_config(self.cfg.model).cuda().eval()
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(self.cfg.model.whisper_path)
        # print(self.model)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        features = []
        for wav in wavs:
            samples = prepare_one_sample(np.array([wav.tolist()]), self.wav_processor)
            prompt = [ self.cfg.model.prompt_template.format("<Speech><SpeechHere></Speech> " + "###") ]
            with torch.cuda.amp.autocast(dtype=torch.float16):
                speech_embeds = self.model.generate(samples, self.cfg.generate, prompts=prompt)
                features.append(speech_embeds[0])

        # padded_feats: (batch_size, max_len, hidden_dim)
        padded_feats = pad_sequence(features, batch_first=True).float()
        return {"hidden_states": (padded_feats,), "last_hidden_state": padded_feats}

        # wavs = pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        # # wavs: (batch_size, max_len, 1)

        # hidden = self.model1(wavs)
        # # hidden: (batch_size, max_len, hidden_dim)

        # feature = self.model2(hidden)
        # # feature: (batch_size, max_len, hidden_dim)

        # # The "hidden_states" key will be used as default in many cases
        # # Others keys in this example are presented for SUPERB Challenge
        # return {
        #     "hidden_states": [hidden, feature],
        #     "PR": [hidden, feature],
        #     "ASR": [hidden, feature],
        #     "QbE": [hidden, feature],
        #     "SID": [hidden, feature],
        #     "ASV": [hidden, feature],
        #     "SD": [hidden, feature],
        #     "ER": [hidden, feature],
        #     "SF": [hidden, feature],
        #     "SE": [hidden, feature],
        #     "SS": [hidden, feature],
        #     "secret": [hidden, feature],
        # }
