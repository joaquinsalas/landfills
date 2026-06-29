import torch
import os
import config as ini

OUTPUT_ALPHA_DIR = os.path.join(ini.OUTPUT_DIR, "patches")

archivos = sorted([f for f in os.listdir(OUTPUT_ALPHA_DIR) if f.endswith('.pt')])
print(f"Total de tensores encontrados: {len(archivos)}\n")

for fname in archivos:
    path = os.path.join(OUTPUT_ALPHA_DIR, fname)
    tensor = torch.load(path)
    
    ok = tensor.shape == torch.Size([64, 512, 512])
    status = "yes" if ok else "bad"
    
    print(f"{status} {fname}")
    print(f"   Forma:  {tensor.shape}  {'(correcto)' if ok else '(INCORRECTO, esperado [64,512,512])'}")
    print(f"   Dtype:  {tensor.dtype}")
    print(f"   Min:    {tensor.min().item():.4f}")
    print(f"   Max:    {tensor.max().item():.4f}\n")