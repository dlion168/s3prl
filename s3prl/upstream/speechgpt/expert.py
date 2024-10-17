# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/speechgpt/expert.py ]
#   Synopsis     [ the SpeechGPT wrapper ]
#   Author       [ Yi-Cheng Lin (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""
from ..interfaces import UpstreamBase

import joblib
import fairseq
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig

import numpy as np

class LlamaEmbeddings(torch.nn.Module):
    def __init__(self):
        super(LlamaEmbeddings, self).__init__()
        # Load the configuration
        config = LlamaConfig.from_pretrained("fnlp/SpeechGPT-7B-cm")
        # Initialize the embedding layer with the same dimensions as the original model
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Load the pretrained weights only for the embedding layer
        state_dict = torch.load('./upstream/speechgpt/embed_speechgpt.pt', map_location='cpu')

        self.embed_tokens.weight.data.copy_(state_dict['weight'])
    
    def forward(self, input_ids):
        return self.embed_tokens(input_ids)

class FeatureReader(nn.Module):
    def __init__(self, ckpt_path, layer=11, max_chunk=1600000, fp16=False, sampling_rate=16000):
        ( 
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        super(FeatureReader, self).__init__()
        self.model = model[0].eval()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()
        
        self.layer_shift = 0
        self.target_sample_hz = sampling_rate

    @torch.no_grad()
    def get_feats(self, waveform):
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half().cuda()
            else:
                x = x.float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                )
        
                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)

class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        self.C = self.C.to(x)
        self.Cnorm = self.Cnorm.to(x)
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.C)
            + self.Cnorm
        )
        min_list = dist.argmin(dim=1).cpu().numpy()
        min_tensor = torch.tensor(min_list)
        feat = self.C.transpose(0,1)[min_tensor]
        return min_list, feat

class UpstreamExpert(UpstreamBase):
    """
    The SpeechGPT wrapper
    """

    def __init__(self, encoder_ckpt, km_ckpt, feat_mode = "encoder", options_config=None, **kwargs):
        super().__init__(**kwargs)

        self.feature_reader = FeatureReader(encoder_ckpt)
        self.apply_kmeans = ApplyKmeans(km_ckpt)
        self.feat_mode = feat_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_table = LlamaEmbeddings()
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        id_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                id_list.append(i)
                count = 1
        return dup_cluster_list, id_list

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        hidden = []
        for wav in wavs:
            feat = self.feature_reader.get_feats(wav)
            if self.feat_mode == "encoder":
                hidden.append(feat)
            else:
                cluster_ids, cluster_feature = self.apply_kmeans(feat)
                dup_cluster_list, id_list = self.merge_duplicates(cluster_ids)
                if self.feat_mode == "kmeans_centroid":
                    hid = cluster_feature.index_select(0, torch.IntTensor(id_list).to(self.device))
                    hidden.append(hid)
                else :
                    lm_embeds = self.embedding_table(torch.IntTensor(id_list).to(self.device))
                    hidden.append(lm_embeds)
                    
        max_length = max(tensor.size(0) for tensor in hidden)
        # Pad each tensor to the maximum length
        padded_tensors = []
        for tensor in hidden:
            current_length = tensor.size(0)
            if current_length < max_length:
                padding_size = max_length - current_length
                # Repeat the last row of the tensor for padding
                last_row = torch.zeros_like(tensor[-1]).unsqueeze(0).repeat(padding_size, 1)
                padded_tensor = torch.cat((tensor, last_row), dim=0)
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)
        
        # Stack the padded tensors
        stacked_tensor = torch.stack(padded_tensors)
        return {"hidden_states": (stacked_tensor), "last_hidden_state": stacked_tensor}