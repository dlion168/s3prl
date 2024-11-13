import torch
import os
from transformers import AutoModel, AutoConfig
from pretrain.multi_distiller.disable_dropout import disable_MERT_encoder_dropout, disable_AST_encoder_dropout, disable_SSAST_encoder_dropout


def convert_mert_attn_state_dict(state_dict):
    new_encoder_layer_dict = {}
    ## mert transformer encoder layers.
    for l in range(12):
        mert_encoder_layer = state_dict.encoder.layers[l].state_dict()
        for key, value in mert_encoder_layer.items():
            # Rename attention block
            if 'attention.' in key:
                new_key = key.replace('attention.', 'self_attn.')
            # Rename layer_norm to self_attn_layer_norm
            elif 'layer_norm' in key and 'final_layer_norm' not in key:
                new_key = key.replace('layer_norm', 'self_attn_layer_norm')
            elif 'final_layer_norm' in key:
                new_key = key  # No changes for final_layer_norm
            # Rename feed forward layers
            elif 'feed_forward.intermediate_dense' in key:
                new_key = key.replace('feed_forward.intermediate_dense', 'fc1')
            elif 'feed_forward.output_dense' in key:
                new_key = key.replace('feed_forward.output_dense', 'fc2')
            else:
                new_key = key  # If no changes are needed, keep the key the same
            # Add the mapped key and value to the new dict
            new_encoder_layer_dict[new_key] = value # this may have a mistake because I should modify the dict in place most likely or per layer outside this function and make a loop outside...
    return new_encoder_layer_dict



def convert_mert_conv_state_dict(state_dict):
    new_state_dict_mert = {}
    for key, value in state_dict.items():
        # Convert "conv_layers.0.conv.weight" to "conv_layers.0.0.weight"
        # Convert "conv_layers.0.layer_norm.weight" to "conv_layers.0.2.weight" (assuming layer_norm is at index 2)
        if "conv_layers" in key:
            # Handle the convolution layers
            if "conv.weight" in key:
                new_key = key.replace("conv.weight", "0.weight")
            # Handle the normalization layers
            elif "layer_norm" in key:
                new_key = key.replace("layer_norm.weight", "2.weight").replace("layer_norm.bias", "2.bias")
            # Handle activation layers if needed (you can add this if distilHuBERT expects it)
            else:
                new_key = key
            new_state_dict_mert[new_key] = value
        else:
            new_state_dict_mert[key] = value
    return new_state_dict_mert

def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    return checkpoint

def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def average_weights(mapped_state_dicts):
    """Averages the weights from multiple state_dicts."""
    avg_dict = collections.OrderedDict()

    keys = mapped_state_dicts[0].keys()
    for key in keys:
        weights = [sd[key] for sd in mapped_state_dicts]
        avg_dict[key] = torch.mean(torch.stack(weights), dim=0)

    return avg_dict

def modify_config(config):
    # Modify 'teacher_names' to ['hubert_base']
    if 'multi_distiller' in config and 'teacher_names' in config['multi_distiller']:
        config['multi_distiller']['teacher_names'] = ['hubert_base']
    else:
        print("Warning: 'teacher_names' not found in 'multi_distiller' config.")

    # Modify 'teacher': 'models' to ['hubert_base']
    if 'teacher' in config and 'models' in config['teacher']:
        config['teacher']['models'] = ['hubert_base']
    else:
        print("Warning: 'models' not found in 'teacher' config.")

    return config

def assemble_new_checkpoint(averaged_state_dict, config, args=None):
    new_checkpoint = {}
    new_checkpoint['Distiller'] = averaged_state_dict
    new_checkpoint['Config'] = config
    new_checkpoint['Optimizer'] = None  # Set to None or omit if not needed
    new_checkpoint['Step'] = 0  # Reset step to 0
    if args is not None:
        new_checkpoint['Args'] = args
    return new_checkpoint

def average_models(model_paths, save_path):
    teacher_1 = torch.hub.load("s3prl/s3prl","hubert_base").cuda()
    teacher_1.model.encoder.layerdrop = 0

    temp_config = AutoConfig.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
    temp_config.output_hidden_states = True  # Enable hidden states in the output
    teacher_2 = AutoModel.from_pretrained("m-a-p/MERT-v0-public", config=temp_config, trust_remote_code=True).cuda()
    disable_MERT_encoder_dropout(teacher_2)
    freeze_model(teacher_1)
    freeze_model(teacher_2)
    
    hubert_state_dict = teacher_1.model.feature_extractor.state_dict()
    mert_state_dict = teacher_2.feature_extractor.state_dict()
    mert_state_dict = convert_mert_conv_state_dict(mert_state_dict)
    averaged_conv_layers = average_weights([hubert_state_dict, mert_state_dict])
    averaged_post_extract_proj = average_weights([teacher_1.model.post_extract_proj.state_dict(), teacher_2.feature_projection.projection.state_dict()])
    
    
    convert_mert_attn_state_dict(#### how to pass mert here, what is better??)

    averaged_encoder = average_weights([new_encoder_layer_dict,hubert_encoder_layer])

    ##### this coming part needs to be fixed as well, this is wrong....
    # Use the 'Config' from the first checkpoint and modify it
    config = checkpoints[0]['Config']
    config = modify_config(config)

    # Optionally, collect 'Args' from the first checkpoint
    args = checkpoints[0].get('Args', None)

    # Assemble the new checkpoint
    new_checkpoint = assemble_new_checkpoint(averaged_state_dict, config, args=args)

    # Save the new checkpoint
    torch.save(new_checkpoint, save_path)
    print(f"Averaged model checkpoint saved to {save_path}")


if __name__ == "__main__":
    # Paths to your three distilled models

    # Path to save the averaged model
    save_path = 'result/pretrain/hubert-mert-teachers-average/averaged_model.ckpt'



    average_models(save_path)
