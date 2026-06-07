import timm
import torch
import torch.nn as nn

def build_backbone(backbone_name: str, pretrained_path: str = None, freeze: bool = False):
    if 'dinov3' in backbone_name.lower() and 'sat' in backbone_name.lower():
        print(f"[backbone] Detectado modelo nativo de DINOv3 Satellite. Cargando via torch.hub...")
        model = torch.hub.load(
            'facebookresearch/dinov3', 
            'dinov3_vitl16_pretrain_sat493m', 
            pretrained=False
        )
        
        if hasattr(model, 'head'):
            model.head = nn.Identity()
            
    else:
        print(f"[backbone] Creando modelo estándar desde timm: {backbone_name}")
        model = timm.create_model(
            backbone_name,
            pretrained=pretrained_path is None,
            num_classes=0,
        )

    if pretrained_path:
        print(f"[backbone] Cargando pesos locales desde: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        state = ckpt.get('model') or ckpt.get('state_dict') or ckpt.get('teacher') or ckpt
        
        cleaned = {}
        for k, v in state.items():
            for prefix in ('backbone.', 'encoder.', 'module.', 'model.'):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v
            
        is_strict = not ('dinov3' in backbone_name.lower())
        
        missing, unexpected = model.load_state_dict(cleaned, strict=is_strict)
        
        if missing:
            print(f"[backbone] llaves faltantes (missing): {len(missing)} -> {missing[:3]}...")
        if unexpected:
            print(f"[backbone] llaves inesperadas (unexpected): {len(unexpected)} -> {unexpected[:3]}...")

    if freeze:
        print("[backbone] Congelando pesos del backbone.")
        for p in model.parameters():
            p.requires_grad = False

    return model
