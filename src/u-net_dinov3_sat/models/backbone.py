import torch
import timm
import torch.nn as nn

def build_backbone(name: str, path: str = None, freeze: bool = False):
    if 'dinov3' in name.lower():
        model_name = name
        if 'sat' not in model_name.lower():
            model_name = f'{model_name}_pretrain_sat493m'

        model = torch.hub.load(
            'facebookresearch/dinov3',
            model_name,
            pretrained=(path is None),
        )
        if hasattr(model, 'head'):
            model.head = nn.Identity()
    else:
        model = timm.create_model(name, pretrained=(path is None), num_classes=0)

    if path:
        state = torch.load(path, map_location='cpu', weights_only=False)
        state = state.get('model', state.get('state_dict', state.get('teacher', state)))
        cleaned = {}
        for k, v in state.items():
            for prefix in ('backbone.', 'encoder.', 'module.', 'model.'):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[backbone] llaves faltantes: {len(missing)} -> {missing[:3]}...")
        if unexpected:
            print(f"[backbone] llaves inesperadas: {len(unexpected)} -> {unexpected[:3]}...")

    if freeze:
        print("[backbone] Congelando pesos del backbone.")
        for param in model.parameters():
            param.requires_grad = False

    return model
