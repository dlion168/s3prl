import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import PreTrainedModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaConfig, WhisperConfig

try:
    from .configuration_blsp import BlspConfig
    from .modeling_whisper_encoder import WhisperEncoder
except:
    from configuration_blsp import BlspConfig
    from modeling_whisper_encoder import WhisperEncoder


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)
    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class Adapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
    ):
        super(Adapter, self).__init__()

        self.fc1 = nn.Linear(in_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, in_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return residual + x


class BlspModel(PreTrainedModel):
    config_class = BlspConfig
    base_model_prefix = "blsp"

    def __init__(self, config: BlspConfig):
        super().__init__(config)
        self.whisper_config = WhisperConfig(**config.whisper_config)
        self.llama_config = LlamaConfig(**config.llama_config)

        self.whisper_model = WhisperEncoder(self.whisper_config)

        in_d = self.whisper_config.d_model
        out_d = self.llama_config.hidden_size
        self.subsampler = Conv1dSubsampler(
            in_d,
            2 * in_d,
            out_d,
            [int(k) for k in config.conv_kernel_sizes.split(",")],
        )
        self.speech_ln = torch.nn.LayerNorm(out_d, 1e-5, True)
        self.adapter = Adapter(out_d, config.adapter_inner_dim)
    

    def get_speech_features(self, speech_values, speech_attention_mask):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.whisper_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        speech_lengths = output.output_lengths

        speech_embeds, speech_lengths = self.subsampler(speech_embeds, speech_lengths)
        speech_embeds = speech_embeds.transpose(0,1) # T x B x C -> B x T x C
        speech_padding_mask = lengths_to_padding_mask(speech_lengths)
        speech_atts = ~speech_padding_mask

        speech_embeds = self.adapter(speech_embeds)
        speech_embeds = self.speech_ln(speech_embeds)

        return speech_embeds, speech_atts

