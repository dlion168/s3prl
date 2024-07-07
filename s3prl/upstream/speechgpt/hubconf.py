# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/npc/hubconf.py ]
#   Synopsis     [ the npc torch hubconf ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

def speechgpt_local(encoder_ckpt, km_ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(encoder_ckpt)
    assert os.path.isfile(km_ckpt)
    return _UpstreamExpert(encoder_ckpt, km_ckpt, *args, **kwargs)


def speechgpt_url(encoder_ckpt, km_ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return speechgpt_local(_urls_to_filepaths(encoder_ckpt, refresh=refresh), _urls_to_filepaths(km_ckpt, refresh=refresh), *args, **kwargs)


def speechgpt(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["encoder_ckpt"] = "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt"
    kwargs["km_ckpt"] = "https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"
    return speechgpt_url(refresh=refresh, *args, **kwargs)

