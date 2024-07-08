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
import torch.nn.functional as F

import numpy as np

class FeatureReader(object):
    def __init__(self, ckpt_path, layer=11, max_chunk=1600000, fp16=False, sampling_rate=16000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
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
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            min_list = dist.argmin(dim=1).cpu().numpy()
            min_tensor = torch.tensor(min_list)
            feat = self.C[min_tensor]
            return min_list, feat
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

class UpstreamExpert(UpstreamBase):
    """
    The SpeechGPT wrapper
    """

    def __init__(self, encoder_ckpt, km_ckpt, options_config=None, **kwargs):
        super().__init__(**kwargs)

        self.feature_reader = FeatureReader(encoder_ckpt)
        self.apply_kmeans = ApplyKmeans(km_ckpt)
    
    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i+1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        hidden = []
        for wav in wavs:
            feat = self.feature_reader.get_feats(wav)
            cluster_ids, cluster_feature = self.apply_kmeans(feat).tolist()
            dup_cluster_list, _ = self.merge_duplicates(cluster_ids)
            hid = torch.index_select(cluster_feature, 0, dup_cluster_list)
            hidden.append(hid)
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
