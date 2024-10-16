# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""
import os 
import torch
import yaml
from torch import nn 

from ..interfaces import UpstreamBase
from .src.factory import create_model_and_transforms

class UpstreamExpert(UpstreamBase):
    """
    The NextGPT wrapper
    """

    def prepare_model(model_config, clap_config, checkpoint_path, device_id=0):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
        model, tokenizer = create_model_and_transforms(
            **model_config,
            clap_config=clap_config,
            use_local_files=False,
            gradient_checkpointing=False,
            freeze_lm_embeddings=False,
        )
        model.eval()
        model = model.to(device_id)

        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        model_state_dict = checkpoint["model_state_dict"]
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict, False)

        return model
    
    def __init__(self, ckpt, name="foundation", **kwargs):
        super().__init__(**kwargs)
        print(f'Initializing {name} audio encoder from {encoder_ckpt} ...')
        config_file = f'configs/{name}.yaml'
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        clap_config = config['clap_config']
        model_config = config['model_config']
        model = prepare_model(
            model_config=model_config, 
            clap_config=clap_config, 
            checkpoint_path=ckpt
        )

        
        self.visual_encoder, self.visual_hidden_size = \
            imagebind_model.imagebind_huge(pretrained=True, store_path=encoder_ckpt)
        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
        print('Audio encoder initialized.')
        print(f'Initializing projector from {projector_ckpt} ...')
        vicuna_weight = torch.load(projector_ckpt, map_location="cuda:0")
        projector_weight = { k.replace('llama_proj.', '') : v for k, v in vicuna_weight.items() if 'llama_proj.' in k}
        self.llama_proj = nn.Linear(
            self.visual_hidden_size, projector_weight['bias'].shape[0]
        )
        self.llama_proj.load_state_dict(projector_weight)
        for param in self.llama_proj.parameters():
            param.requires_grad = False
        print('Projector initialized.')

    def get_downsample_rates(self, key: str) -> int:
        return 160

    def forward(self, wavs):
        inputs = { ModalityType.AUDIO: data.load_and_transform_audio_data(wavs, device="cuda:0") }
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO]  # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1)
        
        return {
            "last_hidden_state": inputs_llama,
            "hidden_states": [inputs_llama],
        }
