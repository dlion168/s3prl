import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import logging
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import WhisperConfig

from .modeling_adapter import Subsampler, CFormer
from .configuration_blsp2 import Blsp2Config
from .configuration_qwen import QWenConfig
from .modeling_utils import length_to_attention_mask
from .modeling_whisper_encoder import WhisperEncoder
import torch.nn.functional as F


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# text_llm_related_losses = {"response_kl", "input_kl"}
# speech_llm_related_losses = {"response_kl", "input_kl", "response_ce", "input_er"}
# lm_related_losses = text_llm_related_losses | speech_llm_related_losses


class Blsp2Model(PreTrainedModel):
    config_class = Blsp2Config
    base_model_prefix = "blsp2"

    def __init__(self, config: Blsp2Config):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.qwen_config = QWenConfig(**config.qwen_config)

        self.whisper_model = WhisperEncoder(self.whisper_config)

        if config.adapter_type == "subsampler":
            self.adapter = Subsampler(self.whisper_config.d_model, config.adapter_inner_dim, self.qwen_config.hidden_size,
                                      config.adapter_hidden_layers, self.whisper_config, config.conv_kernel_sizes)

        elif config.adapter_type == "cformer":
            self.adapter = CFormer(self.whisper_config, self.qwen_config.hidden_size,
                                   self.qwen_config.vocab_size,
                                   num_pre_cif_layers=config.num_pre_cif_layers,
                                   num_post_cif_layers=config.num_post_cif_layers)
        else:
            raise ValueError(f"unsupported adapter type: {config.adapter_type}")
        
        # self.hidden2emotion = nn.Linear(self.qwen_config.hidden_size, self.config.num_emotions, bias=False)

        self.loss_names = [] # must be a list of loss names:  seq_kd, token_kd, or others before training

    def set_loss_names(self, names):
        self.loss_names = names

    def get_speech_features(self, speech_values, speech_attention_mask, num_tokens=None):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        attention_mask = length_to_attention_mask(output.output_lengths)

        speech_embeds, speech_atts, speech_logits, speech_cif_alphas, speech_pred_num_tokens = \
            self.adapter(speech_embeds, attention_mask, num_tokens)

        return speech_embeds, speech_atts