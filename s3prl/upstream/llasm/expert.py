# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""
from transformers import WhisperProcessor, WhisperModel
import torch
import yaml
from torch import nn 

from ..interfaces import UpstreamBase

SAMPLE_RATE = 16000

class UpstreamExpert(UpstreamBase):
    """
    The LLaSM wrapper
    """

    def __init__(self, encoder_ckpt, projector_ckpt, options_config=None, **kwargs):
        super().__init__(**kwargs)
        print(f'Initializing audio encoder from {encoder_ckpt} ...')
        self.extractor = WhisperProcessor.from_pretrained(encoder_ckpt, torch_dtype=torch.float16)
        self.audio_tower = WhisperModel.from_pretrained(encoder_ckpt, torch_dtype=torch.float16)
        self.audio_tower.config.forced_decoder_ids = None
        self.audio_token_len = 64
        print('Audio encoder initialized.')
        print(f'Initializing projector from {projector_ckpt} ...')
        llm_weight = torch.load(projector_ckpt, map_location="cuda:0")
        projector_weight = { k.replace('model.mm_projector.', '') : v for k, v in llm_weight.items() if 'model.mm_projector.' in k}
        self.llama_proj = nn.Linear(
            projector_weight['weight'].shape[1], projector_weight['weight'].shape[0]
        )
        self.llama_proj.load_state_dict(projector_weight)
        self.llama_proj.half()
        for param in self.llama_proj.parameters():
            param.requires_grad = False
        print('Projector initialized.')

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        audio_feat = self.extractor(
            wavs,
            return_tensors = "pt",
            sampling_rate = SAMPLE_RATE,
            device = device
        ).to(device).input_features
        audio_feat = audio_feat.to(dtype=torch.float16)
        decoder_input_ids = torch.ones((audio_feat.shape[0], self.audio_token_len)) * self.audio_tower.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(device).to(torch.long)
        audio_feat = self.audio_tower(audio_feat, decoder_input_ids=decoder_input_ids).last_hidden_state
        dummy_audio_features = self.llama_proj(audio_feat)
        return {
            "last_hidden_state": dummy_audio_features,
            "hidden_states": [dummy_audio_features],
        }
