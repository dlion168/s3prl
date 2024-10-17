# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

import torch
import yaml
from torch import nn 
from .model import load

from ..interfaces import UpstreamBase

class UpstreamExpert(UpstreamBase):
    """
    The MU-LLaMA wrapper
    """

    def __init__(self, ckpt, options_config=None, **kwargs):
        super().__init__(**kwargs)
        # load llama_adapter weights and model_cfg
        print(f'Loading MU-LLaMA from {ckpt}')
        self.model = load(ckpt)
        print('MU-LLaMA initialized.')

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data(wavs, device="cuda:0") }
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO]  # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1)
        
        return {
            "last_hidden_state": inputs_llama,
            "hidden_states": [inputs_llama],
        }
