# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mullama/hubconf.py ]
#   Synopsis     [ the mullama torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

def mullama_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def mullama_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return mullama_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def mullama(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "https://huggingface.co/ChocoWu/mullama_7b_tiva_v0/resolve/main/pytorch_model.pt"
    return mullama_url(refresh=refresh, *args, **kwargs)