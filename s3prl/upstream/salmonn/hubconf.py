from .expert import UpstreamExpert as _UpstreamExpert
from s3prl.util.download import _urls_to_filepaths


def SalmonnCustom(refresh=False, *args, **kwargs):
    """
    To enable your customized pretrained model, you only need to implement
    upstream/example/expert.py and leave this file as is. This file is
    used to register the UpstreamExpert in upstream/example/expert.py
    The following is a brief introduction of the registration mechanism.

    The s3prl/hub.py will collect all the entries registered in this file
    (callable variables without the underscore prefix) as a centralized
    upstream factory. One can pick up this upstream from the factory via

    1.
    from s3prl.hub import customized_upstream
    model = customized_upstream(ckpt, model_config)

    2.
    model = torch.hub.load(
        'your_s3prl_path',
        'customized_upstream',
        ckpt,
        model_config,
        source='local',
    )

    Our run_downstream.py and downstream/runner.py follows the first usage
    """
    kwargs['beats_path'] = _urls_to_filepaths(kwargs['url_to_BEATs'], refresh=refresh)
    kwargs['ckpt'] = _urls_to_filepaths(kwargs['url_to_SALMONN'], refresh=refresh)
    return _UpstreamExpert(**kwargs)

def Salmonn13B(*args, **kwargs):
    kwargs['cfg_path'] = './upstream/salmonn/SALMONN/configs/decode_config_13B.yaml'
    kwargs['url_to_BEATs'] = 'https://huggingface.co/spaces/fffiloni/SALMONN-7B-gradio/resolve/677c0125de736ab92751385e1e8664cd03c2ce0d/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    kwargs['url_to_SALMONN'] = 'https://huggingface.co/tsinghua-ee/SALMONN/resolve/main/salmonn_v1.pth'
    return SalmonnCustom(*args, **kwargs)


def Salmonn7B(*args, **kwargs):
    kwargs['cfg_path'] = './upstream/salmonn/SALMONN/configs/decode_config_7B.yaml'
    kwargs['url_to_BEATs'] = 'https://huggingface.co/spaces/fffiloni/SALMONN-7B-gradio/resolve/677c0125de736ab92751385e1e8664cd03c2ce0d/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    kwargs['url_to_SALMONN'] = 'https://huggingface.co/tsinghua-ee/SALMONN-7B/resolve/main/salmonn_7b_v0.pth'
    return SalmonnCustom(*args, **kwargs)
