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
from .codec.MSCodec import MSCodecLM
from torch import nn 

from ..interfaces import UpstreamBase

class LlamaEmbeddings(torch.nn.Module):
    def __init__(self, model_name):
        super(LlamaEmbeddings, self).__init__()
        # Load the configuration
        config = LlamaConfig.from_pretrained(model_name)
        # Initialize the embedding layer with the same dimensions as the original model
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Load the pretrained weights only for the embedding layer
        state_dict = torch.load(model_name + '/pytorch_model.bin', map_location='cpu')
        self.embed_tokens.weight.data.copy_(state_dict['model.embed_tokens.weight'])
    
    def forward(self, input_ids):
        return self.model.embed_tokens(input_ids)

class UpstreamExpert(UpstreamBase):
    """
    The NextGPT wrapper
    """

    def __init__(self, encoder_ckpt, llm_ckpt, feat_mode = "quantized", options_config=None, **kwargs):
        super().__init__(**kwargs)
        print(f'Initializing audio encoder from {encoder_ckpt} ...')
        vq_config_path = './upstream/llm_codec/config.yaml'
        codec_ckpt = encoder_ckpt
        exp_model_config = OmegaConf.load(vq_config_path)
        self.model = MSCodecLM(**exp_model_config.generator.config)  
        parameter_dict = torch.load(codec_ckpt)
        self.model.load_state_dict(parameter_dict['codec_model']) # load model
        self.model.eval()
        print('Audio encoder initialized.')
        self.feat_mode = feat_mode
        if self.feat_mode == 'lm_codebook':
            print(f'Initializing LLM embedding table from {llm_ckpt} ...')
            access_token = "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"
            model_name = "meta-llama/Llama-2-7b-hf"
            model = LlamaEmbeddings(model_name, token=access_token)

            # Extract the embedding table
            self.embedding_table = model.get_input_embeddings()
            print('Embedding Table initialized.')

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        z_q, codes = self.model.encode(wavs)
        if self.feat_mode == "quantized": 
            print(z_q.shape)    
            return {
                "last_hidden_state": z_q,
                "hidden_states": [z_q],
            }
        elif self.feat_mode == 'lm_codebook':
            lm_embeds = self.embedding_table(codes)
            print(lm_embeds.shape)
            return {
                "last_hidden_state": lm_embeds,
                "hidden_states": [lm_embeds],
            }
            
