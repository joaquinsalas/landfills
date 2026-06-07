"""
inspect_ckpt.py — Diagnóstico del checkpoint ANTES de instanciar el modelo.
Usa poca memoria porque nunca construye el ViT.

Uso:
    python inspect_ckpt.py dinov3_vit7b16_pretrain_sat493m-a6675841.pth
"""

import sys
import torch

def human_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"

def inspect(path: str):
    print(f"\n{'='*60}")
    print(f"  Checkpoint: {path}")
    print(f"{'='*60}\n")

    # Carga segura en CPU
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        print("✓ Cargado en modo seguro (weights_only=True)")
    except Exception as e:
        print(f"⚠ Modo seguro falló ({e}), usando legacy ...")
        raw = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\nTipo del objeto raíz: {type(raw).__name__}")

    if not isinstance(raw, dict):
        print(f"⚠ No es un dict. Es {type(raw)}.")
        print("  Si es un nn.Module, usa torch.save(model.state_dict(), ...) para exportarlo.")
        return

    print(f"\nKeys de nivel superior ({len(raw)}):")
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            print(f"  [{k}]  Tensor {tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, dict):
            n = len(v)
            n_params = sum(x.numel() for x in v.values() if isinstance(x, torch.Tensor))
            print(f"  [{k}]  dict con {n} entradas, ~{n_params/1e6:.1f}M parámetros")
        else:
            print(f"  [{k}]  {type(v).__name__} = {str(v)[:80]}")

    # Detectar cuál key contiene el state_dict real
    state = raw
    for wrap_key in ("model", "state_dict", "teacher", "student", "module"):
        if wrap_key in raw and isinstance(raw[wrap_key], dict):
            state = raw[wrap_key]
            print(f"\n→ State dict encontrado bajo key '{wrap_key}'")
            break

    # Análisis del state dict
    keys = list(state.keys())
    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    total_params = sum(v.numel() for v in tensors.values())
    total_bytes  = sum(v.numel() * v.element_size() for v in tensors.values())

    print(f"\nEstadísticas del state dict:")
    print(f"  Número de tensores : {len(tensors)}")
    print(f"  Total parámetros   : {total_params/1e6:.2f}M")
    print(f"  Tamaño en memoria  : {human_size(total_bytes)}")

    print(f"\nPrimeras 10 keys:")
    for k in keys[:10]:
        v = state[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k:50s}  {str(tuple(v.shape)):25s}  {v.dtype}")
        else:
            print(f"  {k:50s}  {type(v).__name__}")

    # Heurística: ¿cuántos bloques transformer hay?
    block_indices = set()
    for k in keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p in ("blocks", "layers", "encoder") and i+1 < len(parts):
                try:
                    block_indices.add(int(parts[i+1]))
                except ValueError:
                    pass
    if block_indices:
        print(f"\n→ Bloques transformer detectados: {sorted(block_indices)}")
        print(f"  Total bloques: {max(block_indices)+1}")
        if max(block_indices) == 11:
            print("  ✓ Parece ViT-B (12 bloques) — configuración correcta para model.py")
        elif max(block_indices) == 23:
            print("  → Parece ViT-L (24 bloques) — cambia INTERMEDIATE_LAYERS en model.py")

    # ¿Tiene prefijo 'module.'?
    if any(k.startswith("module.") for k in keys):
        print("\n⚠ Las keys tienen prefijo 'module.' (DataParallel) — model.py lo quita automáticamente")

    print(f"\n{'='*60}")
    print("Diagnóstico completo. Si ves ViT-B y 12 bloques, el modelo.py funciona sin cambios.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
    inspect(path)
