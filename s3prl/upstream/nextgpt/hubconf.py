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

def nextgpt_local(encoder_ckpt, projector_ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(encoder_ckpt)
    assert os.path.isfile(projector_ckpt)
    return _UpstreamExpert(encoder_ckpt, projector_ckpt, *args, **kwargs)


def nextgpt_url(encoder_ckpt, projector_ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return nextgpt_local(_urls_to_filepaths(encoder_ckpt, refresh=refresh), _urls_to_filepaths(projector_ckpt, refresh=refresh), *args, **kwargs)

def nextgpt(refresh=False, *args, **kwargs):
    kwargs["encoder_ckpt"] = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
    kwargs["projector_ckpt"] = "https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0/resolve/main/pytorch_model.pt"
    return nextgpt_url(refresh=refresh, *args, **kwargs)