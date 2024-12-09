import os
import yaml
import glob
import torch
import random
import argparse
import logging
import torchaudio
import numpy as np
from argparse import Namespace
from torch.distributed import is_initialized, get_world_size
from transformers import AutoModel, AutoConfig
import pdb

from s3prl import hub
from graphs.hubert_graph import HuBERTGraph
from pretrain.multi_distiller.disable_dropout import disable_MERT_encoder_dropout
from s3prl.utility.helper import backup, get_time_tag, hack_isinstance, is_leader_process, override

from huggingface_hub import HfApi, HfFolder

def wrap_mert_weights(mapped_state_dict):
    """Wrap MERT weights with 'model.' prefix to match HuBERT's architecture."""
    wrapped_state_dict = {}
    for key, value in mapped_state_dict.items():
        wrapped_key = f"model.{key}" if not key.startswith("model.") else key
        wrapped_state_dict[wrapped_key] = value
    return wrapped_state_dict


def map_mert_to_hubert(state_dict):
    """Map MERT's keys to match HuBERT's key structure."""
    mapped_state_dict = {}

    for key, value in state_dict.items():
        # Handle convolutional layers
        if key.startswith("feature_extractor.conv_layers"):
            parts = key.split('.')
            layer_idx = parts[2]
            if "conv" in parts:
                new_key = f"feature_extractor.conv_layers.{layer_idx}.0.{'.'.join(parts[4:])}"
            elif "layer_norm" in parts:
                new_key = f"feature_extractor.conv_layers.{layer_idx}.2.{'.'.join(parts[4:])}"
            else:
                new_ley = key  # Skip irrelevant keys like activations
            mapped_state_dict[new_key] = value

        # Handle feature projection
        elif key.startswith("feature_projection"):
            if "projection" in key:
                new_key = key.replace("feature_projection.projection", "post_extract_proj")
            elif "layer_norm" in key:
                new_key = key.replace("feature_projection.layer_norm", "post_extract_proj.layer_norm")
            elif "dropout" in key:
                # Optional: Dropout handling if required
                continue  # Dropout layers might not need to be mapped, as they do not have weights
            mapped_state_dict[new_key] = value

        # Handle encoder layers
        elif key.startswith("encoder.layers"):
            parts = key.split('.')
            layer_idx = parts[2]
            submodule = parts[3]
            rest = parts[4:]
            if submodule == "attention":
                if "q_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.q_proj.{'.'.join(rest[1:])}"
                elif "k_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.k_proj.{'.'.join(rest[1:])}"
                elif "v_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.v_proj.{'.'.join(rest[1:])}"
                elif "out_proj" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.self_attn.out_proj.{'.'.join(rest[1:])}"
            elif submodule == "layer_norm":
                new_key = f"encoder.layers.{layer_idx}.self_attn_layer_norm.{'.'.join(rest)}"
            elif submodule == "final_layer_norm":
                new_key = f"encoder.layers.{layer_idx}.final_layer_norm.{'.'.join(rest)}"
            elif submodule == "feed_forward":
                if "intermediate_dense" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.fc1.{'.'.join(rest[1:])}"
                elif "output_dense" in rest[0]:
                    new_key = f"encoder.layers.{layer_idx}.fc2.{'.'.join(rest[1:])}"
            mapped_state_dict[new_key] = value

        # Handle positional convolutions
        elif key.startswith("encoder.pos_conv_embed.conv"):
            if key.startswith("encoder.pos_conv_embed.conv.bias"):
                new_key = key.replace("encoder.pos_conv_embed.conv.bias", "encoder.pos_conv.0.bias") 
                # mertstate dict: encoder.pos_conv_embed.conv.bias', 'encoder.pos_conv_embed.conv.weight_g', 'encoder.pos_conv_embed.conv.weight_v'
            if key.startswith("encoder.pos_conv_embed.conv.weight_g"):
                new_key = key.replace("encoder.pos_conv_embed.conv.weight_g", "encoder.pos_conv.0.weight_g") 
            
            if key.startswith("encoder.pos_conv_embed.conv.weight_v"):
                new_key = key.replace("encoder.pos_conv_embed.conv.weight_v", "encoder.pos_conv.0.weight_v")

            mapped_state_dict[new_key] = value
        
        elif key == "masked_spec_embed":
            new_key = "model.mask_emb"
            mapped_state_dict[new_key] = value
        
        # # Handle missing layer_norm mapping for MERT
        # elif key == "encoder.layer_norm.weight":
        #     new_key = "model.layer_norm.weight"
        #     mapped_state_dict[new_key] = value
        # elif key == "encoder.layer_norm.bias":
        #     new_key = "model.layer_norm.bias"
        #     mapped_state_dict[new_key] = value

        # Handle other keys (if necessary)
        else:
            print(f"Unrecognized key: {key}")
            mapped_state_dict[key] = value
    
    return mapped_state_dict

def load_hubert_base(model_name="hubert_base"):
    # Load HuBERT base model from s3prl
    #os.environ["TORCH_HOME"] = "/workspace/s3prl/s3prl/cache"
    #base_model = torch.hub.load("s3prl/s3prl", model_name).cuda()
    import s3prl.hub as hub
    merge_type = 'ff+attn'  # Change this to experiment with other types
    model = getattr(hub, 'hubert_base')()
    device = 'cuda'  # or cpu
    model = model.to(device)
    model.model.encoder.layerdrop = 0  # Ensure no dropout in encoder layers
    return model

def get_downstream_args():
    parser = argparse.ArgumentParser()

    # train or test for this experiment
    parser.add_argument('-o', '--override', help='Used to override args and config, this is at the highest priority')

    # distributed training
    parser.add_argument('--backend', default='nccl', help='The backend for distributed training')
    parser.add_argument('--local_rank', type=int,
                        help=f'The GPU id this process should use while distributed training. \
                               None when not launched by torch.distributed.launch')

    # only load the parameters in the checkpoint without overwriting arguments and config, this is for evaluation
    parser.add_argument('-i', '--init_ckpt', metavar='CKPT_PATH', help='Load the checkpoint for evaluation')

    # configuration for the experiment, including runner and downstream
    parser.add_argument('-c', '--config', help='The yaml file for configuring the whole experiment except the upstream model')

    # upstream settings
    parser.add_argument('--hub', default="torch", choices=["torch", "huggingface"],
        help='The model Hub used to retrieve the upstream model.')

    upstreams = [attr for attr in dir(hub) if attr[0] != '_']
    parser.add_argument('-u', '--upstream',  help=""
        'Upstreams with \"_local\" or \"_url\" postfix need local ckpt (-k) or config file (-g). '
        'Other upstreams download two files on-the-fly and cache them, so just -u is enough and -k/-g are not needed. '
        'Please check upstream/README.md for details. '
        f"Available options in S3PRL: {upstreams}. "
    )
    parser.add_argument('-k', '--upstream_ckpt', metavar='{PATH,URL,GOOGLE_DRIVE_ID}', help='Only set when the specified upstream need it')
    parser.add_argument('-g', '--upstream_model_config', help='The config file for constructing the pretrained model')
    parser.add_argument('-r', '--upstream_refresh', action='store_true', help='Re-download cached ckpts for on-the-fly upstream variants')
    parser.add_argument('-f', '--upstream_trainable', action='store_true', help='Fine-tune, set upstream.train(). Default is upstream.eval()')
    parser.add_argument('-s', '--upstream_feature_selection', default='hidden_states', help='Specify the layer to be extracted as the representation')
    parser.add_argument('-l', '--upstream_layer_selection', type=int, help='Select a specific layer for the features selected by -s')
    parser.add_argument('--upstream_feature_normalize', action='store_true', help='Specify whether to normalize hidden features before weighted sum')
    parser.add_argument('--upstream_model_name', default="model.pt", help='The name of the model file in the HuggingFace Hub repo.')
    parser.add_argument('--upstream_revision', help="The commit hash of the specified HuggingFace Repository")
    parser.add_argument('-x', '--fix_feature_len', action='store_true', help="Fix the feature length")
    parser.add_argument('-q', '--ignore_length_dif', action='store_true', help="Fix the feature length")
    parser.add_argument('--json_file', type=str,default="/export/home2/fabian/projects/dataset_distillation/robust-superb/s3prl/results-for-dd-research-f18106ee2c51.json")#i can reuse my same json file.
    parser.add_argument('--sheet_row', type=int, default=1,help='the row number on where this pre-trained model experiment will be in the downstream experiment results.  set to 1 for now.')
    parser.add_argument('--logfile', type=str,default=None)#i can reuse my same json file.

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname',default="random-name", help='Save experiment at result/downstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--cache_dir', help='The cache directory for pretrained model downloading')
    parser.add_argument('--verbose', action='store_true', help='Print model infomation')
    parser.add_argument('--disable_cudnn', action='store_true', help='Disable CUDNN')

    args = parser.parse_args()
    backup_files = []

    print(f"args: {args}")
    if args.expdir is None:
        args.expdir = f'result/pretrain/permute-merging/{args.expname}'
    
    
    print('[Runner] - Start a new experiment')
    os.makedirs(args.expdir, exist_ok=True)


    # we do not need this for now.
    # if args.upstream_model_config is not None and os.path.isfile(args.upstream_model_config):
    #     backup_files.append(args.upstream_model_config)

    # if args.override is not None and args.override.lower() != "none":
    #     override(args.override, args, config)
    #     os.makedirs(args.expdir, exist_ok=True)
        
    return args


def main():
    logging.basicConfig(level=logging.INFO)

    torch.multiprocessing.set_sharing_strategy('file_system')
    torchaudio.set_audio_backend('soundfile')
    hack_isinstance()

    # get config and arguments
    args  = get_downstream_args()
    if args.cache_dir is not None:
        torch.hub.set_dir(args.cache_dir)

    
    #model_paths = [
    #    'result/pretrain/distill_mert_init_mert_music4all_avgpool/states-epoch-65.ckpt',  # Theta M
    #    'result/pretrain/distilhubert-ls960-own/states-epoch-25.ckpt',  # Theta H
    #]
    model1 = load_hubert_base(model_name="hubert_base")
    temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
    temp_config.output_hidden_states = True  # Enable hidden states in the output
    model2 = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).to("cuda")
    disable_MERT_encoder_dropout(model2)
    new_state_dict = map_mert_to_hubert(model2.state_dict())
    wrapped_weights = wrap_mert_weights(new_state_dict)
    hubert_model = load_hubert_base(model_name="hubert_base")

    # Step 5: Load the wrapped weights into the HuBERT model
    missing_keys, unexpected_keys = hubert_model.load_state_dict(wrapped_weights, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    merge_type = 'ff+attn'  # Change this to experiment with other types


    graph1 = HuBERTGraph(model1, merge_type=merge_type).graphify()
    pdb.set_trace()
    graph2 = HuBERTGraph(hubert_model, merge_type=merge_type, model_type="mert").graphify()

    # Fix seed and make backends deterministic
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    with torch.no_grad():
        node_results = run_auxiliary_experiment(  ### this I need to create it later, now I am just trying to see if I can have MERT as HUBERT so then I can create a graph for both! pleas ehelp check this.
            merging_fn=args.merging_fn, #match_tensors_permute
            merge_type=args.merge_type, # ff+attn
            experiment_config=raw_config, 
            pairs=run_pairs, 
            device=device, 
            args=args # here there is the permute_heads as True!. ignore_heads as False 
        )

if __name__ == '__main__':
    main()
