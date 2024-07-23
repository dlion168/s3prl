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

def blsp_local(name, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    return _UpstreamExpert(name, *args, **kwargs)

def blsp(refresh=False, *args, **kwargs):
    kwargs["name"] = "iic/blsp_lslm_7b"
    return blsp_local(refresh=refresh, *args, **kwargs)