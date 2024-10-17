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

def nextgpt_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def nextgpt_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return nextgpt_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def imagebind(refresh=False, *args, **kwargs):
    kwargs["ckpt"] = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
    return nextgpt_url(refresh=refresh, *args, **kwargs)