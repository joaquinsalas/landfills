"""
diagnostico.py — Verifica el entorno paso a paso.
Corre esto ANTES que cualquier otra cosa.

Uso:
    python diagnostico.py
"""

import sys
print(f"Python: {sys.version}")
print(f"Ejecutable: {sys.executable}")
print()

# ── 1. PyTorch básico ────────────────────────────────────────────────────────
print("=" * 50)
print("1. PyTorch")
print("=" * 50)
try:
    import torch
    print(f"  Versión: {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"  VRAM total: {vram_gb:.1f} GB")
        # Test tensor pequeño en GPU
        t = torch.tensor([1.0]).cuda()
        print(f"  Tensor en GPU: OK ({t.device})")
        del t
        torch.cuda.empty_cache()
    else:
        print("  ⚠ CUDA no disponible — se usará CPU")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

print()

# ── 2. timm ──────────────────────────────────────────────────────────────────
print("=" * 50)
print("2. timm")
print("=" * 50)
try:
    import timm
    print(f"  Versión: {timm.__version__}")
    # Instanciar un modelo TINY para probar que timm funciona
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, global_pool="")
    print(f"  ViT-Tiny instanciado OK ({sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")
    del m
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print()

# ── 3. RAM disponible ─────────────────────────────────────────────────────────
print("=" * 50)
print("3. Memoria del sistema")
print("=" * 50)
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"  RAM total    : {mem.total/1024**3:.1f} GB")
    print(f"  RAM disponible: {mem.available/1024**3:.1f} GB")
    print(f"  RAM usada    : {mem.percent:.0f}%")
except ImportError:
    # Sin psutil, leer /proc/meminfo en Linux
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {l.split(":")[0]: int(l.split()[1]) for l in lines if ":" in l}
        total = info.get("MemTotal", 0) / 1024**2
        avail = info.get("MemAvailable", 0) / 1024**2
        print(f"  RAM total    : {total:.1f} GB")
        print(f"  RAM disponible: {avail:.1f} GB")
    except Exception as e:
        print(f"  No se pudo leer memoria: {e}")

print()

# ── 4. Checkpoint ─────────────────────────────────────────────────────────────
print("=" * 50)
print("4. Checkpoint")
print("=" * 50)
import os, sys

ckpt_name = "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
candidates = [
    ckpt_name,
    os.path.join(os.path.dirname(__file__), ckpt_name),
]
# También buscar en cwd y directorios padres
for root, dirs, files in os.walk("."):
    for f in files:
        if f.endswith(".pth"):
            candidates.append(os.path.join(root, f))
    break  # solo nivel actual

found = None
for c in candidates:
    if os.path.isfile(c):
        found = c
        break

if found:
    size_gb = os.path.getsize(found) / 1024**3
    print(f"  Archivo encontrado: {found}")
    print(f"  Tamaño en disco: {size_gb:.2f} GB")

    # Leer solo los primeros bytes para ver magic number
    with open(found, "rb") as f:
        magic = f.read(10)
    print(f"  Magic bytes: {magic.hex()}")

    # ¿Es un zip (formato PyTorch moderno)?
    if magic[:2] == b'PK':
        print("  Formato: ZIP (PyTorch moderno ✓)")
    elif magic[:6] == b'\x80\x02}q':
        print("  Formato: Pickle legacy")
    else:
        print(f"  Formato: desconocido ({magic[:4].hex()})")

    # Intentar cargar SOLO el índice (sin deserializar tensores)
    try:
        import zipfile
        if zipfile.is_zipfile(found):
            with zipfile.ZipFile(found) as zf:
                names = zf.namelist()
                print(f"  Entradas en el ZIP: {len(names)}")
                print(f"  Primeras 5: {names[:5]}")
    except Exception as e:
        print(f"  No se pudo inspeccionar como ZIP: {e}")
else:
    print(f"  ✗ No se encontró {ckpt_name} en el directorio actual")
    print(f"    Corre este script desde la misma carpeta que el .pth")
    print(f"    O pasa la ruta: python diagnostico.py /ruta/al/checkpoint.pth")

    # Listar .pth en cwd
    pths = [f for f in os.listdir(".") if f.endswith(".pth")]
    if pths:
        print(f"  .pth encontrados en cwd: {pths}")

print()
print("=" * 50)
print("Diagnóstico completo. Comparte este output completo.")
print("=" * 50)
