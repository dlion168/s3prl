from collections import OrderedDict
from typing import Dict, List, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import pad_or_trim, log_mel_spectrogram, get_T_after_cnn

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from ..interfaces import UpstreamBase

HIDDEN_DIM = 8

def GetHuggingFaceModel(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

def GetHuggingFaceTokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def process_audio(wavs: List[Tensor]) -> Dict:
    audios, audio_lens, audio_span_tokens = [], [], []
    for audio in wavs:
        L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
        mel_len = L // 160
        audio = pad_or_trim(audio.flatten())
        mel = log_mel_spectrogram(audio)
        audio_len_after_cnn = get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
        audio_len = [audio_len_after_cnn, audio_token_num]
        audios.append(mel)
        audio_lens.append(audio_len)
        audio_span_tokens.append(audio_token_num + 2)  # add audio bos eos
    input_audio_lengths = torch.IntTensor(audio_lens)
    input_audios = torch.stack(audios, dim=0)
    return {"input_audios": input_audios,
            "input_audio_lengths": input_audio_lengths,
            "audio_span_tokens": audio_span_tokens}
    

class UpstreamExpert(UpstreamBase):
    def __init__(self, model_name, **kwargs):
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
        
        self.model = GetHuggingFaceModel(model_name).eval()
        self.tokenizer = GetHuggingFaceTokenizer(model_name)



    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 640

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        audio_info = process_audio(wavs)
        features = self.model.transformer.audio.encode(
                    audio_info["input_audios"],
                    audio_info["input_audio_lengths"],
                    audio_info["audio_span_tokens"]
                )

        # padded_feats: (batch_size, max_len, hidden_dim)
        padded_feats = pad_sequence(features, batch_first=True).float()
        return {"hidden_states": (padded_feats), "last_hidden_state": padded_feats}

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
