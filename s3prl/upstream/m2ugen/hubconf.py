# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/nextgpt/hubconf.py ]
#   Synopsis     [ the NextGPT torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

def llm_codec_local(encoder_ckpt, llm_ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(encoder_ckpt)
    return _UpstreamExpert(encoder_ckpt, llm_ckpt, *args, **kwargs)


def llm_codec_url(encoder_ckpt, llm_ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return llm_codec_local(_urls_to_filepaths(encoder_ckpt, refresh=refresh), llm_ckpt, *args, **kwargs)

def llm_codec(refresh=False, *args, **kwargs):
    kwargs["encoder_ckpt"] = "https://huggingface.co/Dongchao/2024/resolve/main/semantic_acoustic.pth"
    kwargs["llm_ckpt"] = "meta-llama/Llama-2-7b-hf"
    return llm_codec_url(refresh=refresh, *args, **kwargs)

def llm_codec_quant(*args, **kwargs):
    return llm_codec(feat_mode = "quantized", *args, **kwargs)

def llm_codec_lm_code_sem_aco(*args, **kwargs):
    return llm_codec(feat_mode = "lm_codebook_sem_aco", *args, **kwargs)

def llm_codec_lm_code_sem(*args, **kwargs):
    return llm_codec(feat_mode = "lm_codebook_sem", *args, **kwargs)