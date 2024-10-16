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
from torch.nn.utils.rnn import pad_sequence
from .codec.tokenizer import Tokenizer
import numpy as np

from ..interfaces import UpstreamBase
text_tokenizer = Tokenizer(model_path="./upstream/llm_codec/tokenizer.model")

class LlamaEmbeddings(torch.nn.Module):
    def __init__(self, llm_ckpt):
        super(LlamaEmbeddings, self).__init__()
        # Load the configuration
        config = LlamaConfig.from_pretrained(llm_ckpt)
        # Initialize the embedding layer with the same dimensions as the original model
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Load the pretrained weights only for the embedding layer
        state_dict = torch.load('./upstream/llm_codec/embed_llama2.pt', map_location='cpu')

        self.embed_tokens.weight.data.copy_(state_dict['weight'])
    
    def forward(self, input_ids):
        return self.embed_tokens(input_ids)

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
        self.model.load_state_dict(parameter_dict['codec_model'], strict=False) # load model
        self.model.eval()
        self.vq1_texts = np.load("./upstream/llm_codec/layer1.npy", allow_pickle=True)
        print('Audio encoder initialized.')
        self.feat_mode = feat_mode
        if 'lm_codebook' in self.feat_mode:
            print(f'Initializing LLM embedding table from {llm_ckpt} ...')
            #access_token = "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"

            # Extract the embedding table
            self.embedding_table = LlamaEmbeddings(llm_ckpt)#, token=access_token)
            print('Embedding Table initialized.')

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        wavs = pad_sequence(wavs, batch_first=True)
        wavs = wavs.unsqueeze(1)
        z_q, codes = self.model.encode(wavs)
        feature = z_q.transpose(1,2)
        if self.feat_mode == "quantized":    
            return {
                "last_hidden_state": feature,
                "hidden_states": [feature],
            }
        elif self.feat_mode == 'lm_codebook_sem_aco':
            my_code = [ [] for _ in range(codes[0].shape[0]) ]
            # setence = ''
            for kk, code in enumerate(codes):
                for i in range(code.shape[0]):
                    for j in range(code.shape[1]):
                        if kk==0:
                            tmp = code[i,j].item() # index
                            wo = self.vq1_texts[tmp] # get word
                            real_code = text_tokenizer.encode(str(wo), bos=False, eos=False)
                            my_code[i].append(real_code[0])
                            # setence += ' ' + str(wo)
                        else:
                            tmp = code[i,j].item()
                            # wo = text_tokenizer.decode(tmp)
                            # setence += ' ' + str(wo)
                            my_code[i].append(tmp)
            lm_embeds = self.embedding_table(torch.IntTensor(my_code).to(next(self.embedding_table.parameters()).device))
            return {
                "last_hidden_state": lm_embeds,
                "hidden_states": [lm_embeds],
            }
        elif self.feat_mode == 'lm_codebook_sem':
            my_code = [ [] for _ in range(codes[0].shape[0]) ]
            # setence = ''
            for kk, code in enumerate(codes):
                for i in range(code.shape[0]):
                    for j in range(code.shape[1]):
                        if kk==0:
                            tmp = code[i,j].item() # index
                            wo = self.vq1_texts[tmp] # get word
                            real_code = text_tokenizer.encode(str(wo), bos=False, eos=False)
                            my_code[i].append(real_code[0])
            lm_embeds = self.embedding_table(torch.IntTensor(my_code).to(next(self.embedding_table.parameters()).device))
            return {
                "last_hidden_state": lm_embeds,
                "hidden_states": [lm_embeds],
            }
            
