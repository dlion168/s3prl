# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""

import torch
from transformers import LlamaConfig

from omegaconf import OmegaConf
from modelscope.models import Model
from modelscope import snapshot_download
from .src.modeling_blsp import BlspModel
from transformers import WhisperFeatureExtractor
from torch import nn 


from ..interfaces import UpstreamBase

class UpstreamExpert(UpstreamBase):
    """
    The BLSP speech encoder wrapper
    """

    def __init__(self, name, options_config=None, **kwargs):
        super().__init__(**kwargs)
        print(f'Initializing audio encoder from {name} ...')
        self.extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        model_dir = snapshot_download(name)
        self.model = BlspModel.from_pretrained(model_dir)
        print('Audio encoder initialized.')


    def get_downsample_rates(self, key: str) -> int:
        return 160
    

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        speech_inputs = self.extractor(
            wavs,
            sampling_rate=self.extractor.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)
        speech_values = speech_inputs.input_features.cuda()
        speech_attention_mask = speech_inputs.attention_mask.cuda()
        speech_embeds, _ = self.model.get_speech_features(speech_values, speech_attention_mask)
        return {
            "last_hidden_state": speech_embeds,
            "hidden_states": [speech_embeds],
        }
            
