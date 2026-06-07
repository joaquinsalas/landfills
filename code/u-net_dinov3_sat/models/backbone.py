import torch
import timm
import torch.nn as nn

def build_backbone(name: str, path: str = None):
    if 'dinov3' in name.lower():
        base_name = name.split('_pretrain')[0]
        model = torch.hub.load('facebookresearch/dinov3', base_name, pretrained=(path is None))
        if hasattr(model, 'head'): model.head = nn.Identity()
    else:
        model = timm.create_model(name, pretrained=(path is None), num_classes=0)

    if path:
        state = torch.load(path, map_location='cpu', weights_only=False)
        state = state.get('model', state.get('state_dict', state.get('teacher', state)))
        model.load_state_dict({k.replace('backbone.', '').replace('model.', ''): v for k, v in state.items()}, strict=False)

    return model
