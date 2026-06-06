import timm
import torch
import torch.nn as nn

def build_backbone(backbone_name: str, pretrained_path: str = None, freeze: bool = False):
    """
    Construye el backbone para la UNet. Soporta modelos estándar de timm y 
    arquitecturas nativas de DINOv3 (usando los nombres oficiales del Hub).
    """
    
    # 1. Detectar si estamos usando DINOv3
    if 'dinov3' in backbone_name.lower():
        # Mapeamos tu configuración al callable oficial de Meta
        if 'vitl16' in backbone_name.lower() or 'large' in backbone_name.lower():
            model_callable = 'dinov3_vitl16'
        elif 'vitb16' in backbone_name.lower() or 'base' in backbone_name.lower():
            model_callable = 'dinov3_vitb16'
        elif 'vits16' in backbone_name.lower() or 'small' in backbone_name.lower():
            model_callable = 'dinov3_vits16'
        else:
            model_callable = 'dinov3_vitl16' # Por defecto Large si no se especifica
            
        print(f"[backbone] Cargando arquitectura nativa '{model_callable}' de DINOv3 via torch.hub...")
        
        # Instanciamos el esqueleto vacío de DINOv3
        model = torch.hub.load(
            'facebookresearch/dinov3', 
            model_callable, 
            pretrained=False
        )
        
        # Eliminar la cabeza de clasificación si existe
        if hasattr(model, 'head'):
            model.head = nn.Identity()
            
    else:
        # Inicialización normal con timm para otros modelos estándar
        print(f"[backbone] Creando modelo estándar desde timm: {backbone_name}")
        model = timm.create_model(
            backbone_name,
            pretrained=pretrained_path is None,
            num_classes=0,
        )

    # 2. Carga de pesos locales (.pth)
    if pretrained_path:
        print(f"[backbone] Cargando pesos locales desde: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # Extraer diccionario de pesos limpio
        state = ckpt.get('model') or ckpt.get('state_dict') or ckpt.get('teacher') or ckpt
        
        cleaned = {}
        for k, v in state.items():
            for prefix in ('backbone.', 'encoder.', 'module.', 'model.'):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            cleaned[k] = v
            
        # Al usar la arquitectura oficial de DINOv3, forzamos strict=True 
        # para asegurar que RoPE y todo encaje a la perfección.
        is_strict = 'dinov3' in backbone_name.lower()
        
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        
        if missing:
            print(f"[backbone] llaves faltantes (missing): {len(missing)} -> {missing[:3]}...")
        if unexpected:
            print(f"[backbone] llaves inesperadas (unexpected): {len(unexpected)} -> {unexpected[:3]}...")

    # 3. Congelar gradientes si es necesario
    if freeze:
        print("[backbone] Congelando pesos del backbone.")
        for p in model.parameters():
            p.requires_grad = False

    return model
