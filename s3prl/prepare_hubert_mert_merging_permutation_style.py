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
from metric_calculators import get_metric_fns
from merging_utils.dataset import get_dataloader
from copy import deepcopy
import json
import pdb

from s3prl import hub
from graphs.hubert_graph import HuBERTGraph
from pretrain.multi_distiller.disable_dropout import disable_MERT_encoder_dropout
from s3prl.utility.helper import backup, get_time_tag, hack_isinstance, is_leader_process, override

from huggingface_hub import HfApi, HfFolder
from inspect import getmembers, isfunction

from merging_utils.model_merger import ModelMerge

def clear_hooks_and_prepare(model):
    for module in model.modules():
        module._backward_hooks = {}
        module._forward_hooks = {}
        module._forward_pre_hooks = {}

def assemble_new_checkpoint(averaged_state_dict):
    new_checkpoint = {}
    new_checkpoint['model_cfg'] = {'_name': 'hubert', 'label_rate': 50.0, 'extractor_mode': 'default', 'encoder_layers': 12, 'encoder_embed_dim': 768, 'encoder_ffn_embed_dim': 3072, 'encoder_attention_heads': 12, 'activation_fn': 'gelu', 'layer_type': 'transformer', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'encoder_layerdrop': 0.05, 'dropout_input': 0.1, 'dropout_features': 0.1, 'final_dim': 256, 'untie_final_proj': False, 'layer_norm_first': False, 'conv_feature_layers': '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2', 'conv_bias': False, 'logit_temp': 0.1, 'target_glu': False, 'feature_grad_mult': 0.1, 'mask_length': 10, 'mask_prob': 0.8, 'mask_selection': 'static', 'mask_other': 0.0, 'no_mask_overlap': False, 'mask_min_space': 1, 'mask_channel_length': 10, 'mask_channel_prob': 0.0, 'mask_channel_selection': 'static', 'mask_channel_other': 0.0, 'no_mask_channel_overlap': False, 'mask_channel_min_space': 1, 'conv_pos': 128, 'conv_pos_groups': 16, 'latent_temp': [2.0, 0.5, 0.999995], 'skip_masked': False, 'skip_nomask': False, 'checkpoint_activations': False, 'required_seq_len_multiple': 2, 'depthwise_conv_kernel_size': 31, 'attn_type': '', 'pos_enc_type': 'abs', 'fp16': True}
    new_checkpoint["dictionaries_symbols"] = [['<s>', '<pad>', '</s>', '<unk>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499']]
    new_checkpoint['model_weight'] = averaged_state_dict  # Set to None or omit if not needed
    new_checkpoint["task_cfg"] = {'_name': 'hubert_pretraining', 'data': '/checkpoint/wnhsu/data/librispeech/960h/iter/250K_50hz_km100_mp0_65_v2', 'fine_tuning': False, 'labels': ['layer6.km500'], 'label_dir': None, 'label_rate': 50.0, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_keep_size': None, 'max_sample_size': 250000, 'min_sample_size': 32000, 'single_target': False, 'random_crop': True, 'pad_audio': False}
    return new_checkpoint


def remove_model_prefix(state_dict):
    """Removes the 'model.' prefix from state_dict keys."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_key = key[len("model."):]  # Remove the 'model.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def contains_name(layer_name, node_list):
    for node in node_list:
        if node in layer_name:
            return True
    return False

def get_merging_fn(name):
    """ Get alignment function from name. """
    import matching_functions
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    return matching_fns[name]

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

def load_randomized_hubert():
    """
    Load a HuBERT model and randomize its parameters.
    """
    # Load the pretrained HuBERT base model
    hubert_model = getattr(hub, 'hubert_base')()

    # Reinitialize the parameters with random values
    for param in hubert_model.parameters():
        if param.requires_grad:
            torch.nn.init.normal_(param, mean=0, std=0.02)  # Random Gaussian initialization

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hubert_model = hubert_model.to(device)

    return hubert_model

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
    parser.add_argument("--interp_weights", type=float, nargs=2, default=[0.5, 0.5],
                    help="Weights for interpolating the merging of two models.")
    parser.add_argument('--json_file', type=str,default="/export/home2/fabian/projects/dataset_distillation/robust-superb/s3prl/results-for-dd-research-f18106ee2c51.json")#i can reuse my same json file.
    parser.add_argument('--sheet_row', type=int, default=1,help='the row number on where this pre-trained model experiment will be in the downstream experiment results.  set to 1 for now.')
    parser.add_argument('--logfile', type=str,default=None)#i can reuse my same json file.

    # experiment directory, choose one to specify
    # expname uses the default root directory: result/downstream
    parser.add_argument('-n', '--expname',default="baseline", help='Save experiment at result/merged_upstream/expname')
    parser.add_argument('-p', '--expdir', help='Save experiment at expdir')

    # options
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--device', default='cuda', help='model.to(device)')
    parser.add_argument('--cache_dir', help='The cache directory for pretrained model downloading')
    parser.add_argument('--verbose', action='store_true', help='Print model infomation')
    parser.add_argument('--disable_cudnn', action='store_true', help='Disable CUDNN')

    args = parser.parse_args()
    backup_files = []

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

    print(f"args: {args}")
    if args.expdir is None:
        args.expdir = f'result/merged_pretrain_upstream/permutation-covariance/{args.expname}'
    
    
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

    ### Here I am missing to get a dataloader ######
    # Load YAML configuration
    with open("merging_utils/data_config.yaml", "r") as file:
        data_config = yaml.load(file, Loader=yaml.FullLoader)
    
    
    ####### get the dataset ########

    # Use the get_dataloader function
    dataloader = get_dataloader(data_config, split="train") ### this is loading 5k samples from librispeech100.     ## Do not use the whole librispeech 100.

    ## from the dataloader I just need the wav and then I need to compute
    ## its length and zero pad on the batches, this may be already implemented in the s3prl way

    graph1 = HuBERTGraph(model1, merge_type=merge_type).graphify()   ###### IN THE ORIGINAL CODE THEY DO DEEPCOPY, BE SURE THAT THIS WILL NOT AFFECT!.
    graph2 = HuBERTGraph(hubert_model, merge_type=merge_type).graphify()
    graphs = []
    graphs.append(  [graph1, graph2] )

    #graphs = [Grapher(deepcopy(base_model), merge_type=merge_type, qk=args.qk, classifier=True).graphify() for base_model in base_models]

    model_to_merge = load_randomized_hubert()
    merging_function = get_merging_fn("match_tensors_permute")

    merging_metric = get_metric_fns(["covariance"])



    #### I need a new model where i will end up doing the merging, How to have a HuBERT randomly initalized? check this.
    for node in graph1.G.nodes:
        info = graph1.get_node_info(node)
        print(f"Node {node}: {info}")
        print(f"Predecessors: {graph1.preds(node)}")
        print(f"Successors: {graph1.succs(node)}")


    ##### initialize the merger#####
    Merge = ModelMerge(graph1, graph2, device="cuda")
    
    ## the transform function later may be modifying the model1 and model2 so you may need to reload a fresh version of them if needed.
    unmerge, cost_dict = Merge.transform(
            model_to_merge, 
            dataloader, 
            sentence_level=None,#None
            special_toks=False,#True
            transform_fn=merging_function, #merging_fn is match_tensors_permute and get_merging_fn gets the implementation from the matching_functions.py script -> see line 28.
            metric_classes=merging_metric,#{'covariance': <class 'metric_calculators.CovarianceMetric'>} 
            permute_heads=True,#True
            ignore_heads=False,#False
            save_both=False,#False
            merge_cls=False,#False, we do not need this for hubert and mert.
            no_absval=True,#True -> not sure what it does.
            saved_features=None,#None
            res_type="first",#"first" -> not sure what this means.
        )
    
    
    param_tail=f'model_average_both_models_equal_weight' # change this later... not now....

    save_dir =  args.expdir
    model1 = "hubert_base"
    model2 = "mert_base"
    merging_fn = "match_tensors_permute"
    merge_type = "ff+attn"
    args.save_both = False
    if os.path.exists(os.path.join(save_dir, 'individual_models')) == False:
        os.makedirs(os.path.join(save_dir, 'individual_models'))
        
    with open(f'{save_dir}/test_{merging_fn}_{merge_type}_hubert_base_mert_base_{param_tail}.args', 'w+') as f_args:
            f_args.write(str(vars(args)) + '\n')
            f_args.write(str(data_config))

    
    # Clear hooks from the merged model
    clear_hooks_and_prepare(Merge.merged_model)
    # Assemble a new checkpoint
    averaged_state_dict = Merge.merged_model.state_dict()  # Adjust if further processing is needed
    cleaned_state_dict = remove_model_prefix(averaged_state_dict)

    new_checkpoint = assemble_new_checkpoint(cleaned_state_dict)


    if args.save_both:
        torch.save(Merge.merged_model1.state_dict(), 
                f'{save_dir}/test_{merging_fn}_{merge_type}_{model1}_{model2}_0_{param_tail}.ckpt')
        torch.save(Merge.merged_model2.state_dict(), 
                f'{save_dir}/test_{merging_fn}_{merge_type}_{model1}_{model2}_1_{param_tail}.ckpt')
    else:
        torch.save(new_checkpoint, f'{save_dir}/test_{merging_fn}_{merge_type}_{model1}_{model2}_{param_tail}.ckpt')
    torch.save(Merge.graphs[0].model.state_dict(), f'{save_dir}/individual_models/match_tensors_permute_{merge_type}_0_{model1}_{param_tail}.ckpt')
    torch.save(Merge.graphs[1].model.state_dict(), f'{save_dir}/individual_models/match_tensors_permute_{merge_type}_1_{model2}_{param_tail}.ckpt')
    ### pending something like : merged_state_dict = Merge.get_merged_state_dict(interp_w=args.interp_weights, save_both=False)  -> inside model_merger.py

    print(f"cost in each node for the merging is:")
    print(cost_dict)


    if unmerge != None:
        if os.path.exists(os.path.join(save_dir, 'unmerge')) == False:
            os.makedirs(os.path.join(save_dir, 'unmerge'))
        torch.save(unmerge, f'{save_dir}/unmerge/unmerge_mat_{args.seed0}_{args.seed1}.ckpt')
        
    if os.path.exists(os.path.join(save_dir, 'costs')) == False:
        os.makedirs(os.path.join(save_dir, 'costs'))
    with open(f'{save_dir}/costs/costs_hubert_base_mert_base.ckpt', 'w+') as costs_out:
        json.dump(cost_dict, costs_out)

    

if __name__ == '__main__':
    main()
