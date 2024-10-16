# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/llasm/hubconf.py ]
#   Synopsis     [ the llasm torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

def llasm_local(encoder_ckpt, projector_ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(projector_ckpt)
    return _UpstreamExpert(encoder_ckpt, projector_ckpt, *args, **kwargs)


def llasm_url(encoder_ckpt, projector_ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return llasm_local(encoder_ckpt, _urls_to_filepaths(projector_ckpt, refresh=refresh), *args, **kwargs)

def llasm_llama2(refresh=False, *args, **kwargs):
    kwargs["encoder_ckpt"] = "openai/whisper-large-v2"
    kwargs["projector_ckpt"] = "https://huggingface.co/LinkSoul/LLaSM-Cllama2/resolve/main/pytorch_model-00003-of-00003.bin"
    return llasm_url(refresh=refresh, *args, **kwargs)

def llasm_baichuan(refresh=False, *args, **kwargs):
    kwargs["encoder_ckpt"] = "openai/whisper-large-v2"
    kwargs["projector_ckpt"] = "https://huggingface.co/LinkSoul/LLaSM-Baichuan/resolve/main/pytorch_model-00003-of-00003.bin"
    return llasm_url(refresh=refresh, *args, **kwargs)