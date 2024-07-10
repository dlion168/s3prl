# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

from .ImageBind import *
from .ImageBind import data
import torch
import yaml
from torch import nn 

from ..interfaces import UpstreamBase

class UpstreamExpert(UpstreamBase):
    """
    The NextGPT wrapper
    """

    def __init__(self, ckpt, options_config=None, **kwargs):
        super().__init__(**kwargs)
        print(f'Initializing audio encoder from {ckpt} ...')
        self.visual_encoder, self.visual_hidden_size = \
            imagebind_model.imagebind_huge(pretrained=True, store_path=ckpt)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Audio encoder initialized.')

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data(wavs, device="cuda:0") }
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO].unsqueeze(1)  # bsz x 1024
        
        return {
            "last_hidden_state": audio_embeds,
            "hidden_states": [audio_embeds],
        }
