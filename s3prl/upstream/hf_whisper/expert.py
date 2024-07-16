import logging

import torch
from transformers import AutoFeatureExtractor, WhisperModel

SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        self.extracter = AutoFeatureExtractor.from_pretrained(ckpt)
        self.model = WhisperModel.from_pretrained(ckpt).get_encoder()

    def get_downsample_rates(self, key: str = None) -> int:
        return 160

    def forward(self, wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]
        input = self.extracter(
            wavs,
            return_tensors = "pt",
            padding = "max_length",
            return_attention_mask = True,
            sampling_rate = SAMPLE_RATE,
        ).to(device)
        outputs = self.model(input.input_features, output_hidden_states=True, return_dict=True)
        return outputs
