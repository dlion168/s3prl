import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llama import Transformer, ModelArgs, RMSNorm
from .tokenizer import Tokenizer
from util.misc import download
from .utils import sample_top_p

from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torchaudio


class LLaMA_adapter(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, mert_path, legacy_bridge=False):
        super().__init__()

        # 1. mert, mert aggregator and mert projector
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # The model files for MERT can be downloaded here in case of network issues:
        # https://huggingface.co/m-a-p/MERT-v1-330M
        # And set the mert_path argument to directory with the model files
        self.mert_model = AutoModel.from_pretrained(mert_path, trust_remote_code=True).to(self.device)
        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_path, trust_remote_code=True)
        self.mu_mert_agg = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
        self.mu_mert_proj = nn.Linear(1024, 4096)

        if legacy_bridge:
            bridge_norm_layer = nn.LayerNorm
            bridge_bias = True
        else:
            bridge_norm_layer = RMSNorm
            bridge_bias = False

        self.mu_mert_norm_1 = bridge_norm_layer(4096)
        self.mu_mert_f1_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.mu_mert_f2_1 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.mu_mert_f3_1 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.mu_mert_norm_2 = bridge_norm_layer(4096)
        self.mu_mert_f1_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.mu_mert_f2_2 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.mu_mert_f3_2 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        self.mu_mert_norm_3 = bridge_norm_layer(4096)
        self.mu_mert_f1_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)
        self.mu_mert_f2_3 = nn.Linear(4096 * 4, 4096, bias=bridge_bias)
        self.mu_mert_f3_3 = nn.Linear(4096, 4096 * 4, bias=bridge_bias)

        # # 2. tokenizer
        # self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # # 3. llama
        # with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
        #     params = json.loads(f.read())
        # bias_lora = phase == "finetune"
        # model_args: ModelArgs = ModelArgs(
        #     max_seq_len=8192, max_batch_size=1, w_bias=bias_lora, w_lora=bias_lora,
        #     **params)  # max_batch_size only affects inference
        # print(f"model args: {model_args}")
        # model_args.vocab_size = self.tokenizer.n_words
        # if torch.cuda.is_available():
        #     torch.set_default_tensor_type(torch.cuda.HalfTensor)
        # self.llama = Transformer(model_args)
        # torch.set_default_tensor_type(torch.FloatTensor)

        # ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))

        # """
        # Adapted from https://github.com/cedrickchee/llama/blob/main/chattyllama/combined/inference.py
        # """
        # key_to_dim = {
        #     "w1": 0,
        #     "w2": -1,
        #     "w3": 0,
        #     "wo": -1,
        #     "wq": 0,
        #     "wk": 0,
        #     "wv": 0,
        #     "output": 0,
        #     "tok_embeddings": -1,
        #     "ffn_norm": None,
        #     "attention_norm": None,
        #     "norm": None,
        #     "rope": None,
        # }
        # for i, ckpt in enumerate(ckpts):
        #     checkpoint = torch.load(ckpt, map_location="cpu")
        #     for parameter_name, parameter in self.llama.named_parameters():
        #         short_name = parameter_name.split(".")[-2]
        #         if "gate" in parameter_name or "lora" in parameter_name or "bias" in parameter_name:
        #             continue
        #         if key_to_dim[short_name] is None and i == 0:
        #             parameter.data = checkpoint[parameter_name]
        #         elif key_to_dim[short_name] == 0:
        #             size = checkpoint[parameter_name].size(0)
        #             parameter.data[size * i: size * (i + 1), :] = checkpoint[
        #                 parameter_name
        #             ]
        #         elif key_to_dim[short_name] == -1:
        #             size = checkpoint[parameter_name].size(-1)
        #             parameter.data[:, size * i: size * (i + 1)] = checkpoint[
        #                 parameter_name
        #             ]
        #     del checkpoint
        '''
        ckpts_dict = defaultdict(list)
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            for key, val in ckpt.items():
                ckpts_dict[key].append(val)
        
        for key, val in ckpts_dict.items():
            ckpts_dict[key] = torch.cat(val, dim=-1)

        self.llama.load_state_dict(ckpts_dict, strict=False)
        
        print(ckpts)
        for ckpt in ckpts:
            print(ckpt)
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)
        '''
        self.set_default_trainability()

    def set_default_trainability(self):
        for key, value in self.named_parameters():
            value.requires_grad = False

    def load_audio(self, audio_path, target_sr=16000):
        y, sr = torchaudio.load(audio_path)
        resampler = torchaudio.transforms.Resample(sr, target_sr, dtype=y.dtype)
        audio = resampler(y)
        return audio, target_sr

    def encode_audio(self, x):
        xs = []
        for sub_x in x:
            all_inputs = [self.mert_processor(sub_x[ix * self.mert_processor.sampling_rate:min(
                (ix + 60) * self.mert_processor.sampling_rate, len(sub_x))],
                                              sampling_rate=self.mert_processor.sampling_rate,
                                              return_tensors="pt").to(self.device) for ix in
                          range(0, len(sub_x) // (self.mert_processor.sampling_rate * 60) + 1, 60)]
            aggoutputs = torch.zeros(1, 25, 1024).to(self.device)
            for inputs in all_inputs:
                with torch.no_grad():
                    outputs = self.mert_model(**inputs, output_hidden_states=True)
                all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
                sub_x = all_layer_hidden_states.mean(-2).unsqueeze(0)
                aggoutputs += sub_x
            aggoutputs /= len(all_inputs)
            sub_x = self.mu_mert_agg(aggoutputs).squeeze()
            xs.append(sub_x)
        x = torch.stack(xs, dim=0)
        return x

    def forward_audio(self, inputs, cache_size=10, cache_t=20, cache_weight=0.5):
        outputs = []
        outputs_weights = []
        for input_type, (input, input_weight) in inputs.items():
            outputs.append(F.normalize(self.encode_audio(input), dim=-1))
            outputs_weights.append(input_weight)
        outputs_weights = [x / (sum(outputs_weights) + 1e-6) for x in outputs_weights]

        audio_feats = sum([output * output_weight for output, output_weight in zip(outputs, outputs_weights)])
        device = audio_feats.device

        audio_feats = audio_feats.unsqueeze(1)  # B, 1, D
        audio_feats = self.mu_mert_proj(audio_feats)
        audio_feats_norm = self.mu_mert_norm_1(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_1(
            F.silu(self.mu_mert_f1_1(audio_feats_norm)) * self.mu_mert_f3_1(audio_feats_norm))

        audio_feats_norm = self.mu_mert_norm_2(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_2(
            F.silu(self.mu_mert_f1_2(audio_feats_norm)) * self.mu_mert_f3_2(audio_feats_norm))

        audio_feats_norm = self.mu_mert_norm_3(audio_feats)
        audio_feats = audio_feats + self.mu_mert_f2_3(
            F.silu(self.mu_mert_f1_3(audio_feats_norm)) * self.mu_mert_f3_3(audio_feats_norm))
        return audio_feats


def load(model_path, mert_path="m-a-p/MERT-v1-330M", device="cuda" if torch.cuda.is_available() else "cpu"):
    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    adapter_ckpt = torch.load(model_path, map_location='cpu')
    adapter_ckpt = {k:v for k, v in adapter_ckpt['model'].items() if 'mert' in k}
    
    # model_cfg = adapter_ckpt.get('config', {})

    # The model files for MERT can be downloaded here in case of network issues:
    # https://huggingface.co/m-a-p/MERT-v1-330M
    # And set the MERT argument to directory with the model files
    model = LLaMA_adapter(mert_path)

    load_result = model.load_state_dict(adapter_ckpt, strict=False)
    print(load_result)
    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device)