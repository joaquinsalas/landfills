from PIL import Image
import rasterio
import numpy as np
from pathlib import Path

carpeta = Path("./outputs/")

palette = [
    139, 69, 19,   # Brown
    255, 165, 0,   # Orange
    255, 255, 0,   # Yellow
    0, 255, 0,     # Green
    0, 255, 255,   # Cyan
    0, 0, 255      # Blue
]

full_palette = []
for i in range(256):
    ratio = i / 255.0
    idx = int(ratio * (len(palette) // 3 - 1))
    idx = min(idx, len(palette) // 3 - 2)
    r1, g1, b1 = palette[idx*3 : (idx+1)*3]
    r2, g2, b2 = palette[(idx+1)*3 : (idx+2)*3]
    local_ratio = (ratio * (len(palette) // 3 - 1)) - idx
    r = int(r1 + local_ratio * (r2 - r1))
    g = int(g1 + local_ratio * (g2 - g1))
    b = int(b1 + local_ratio * (b2 - b1))
    full_palette.extend([r, g, b])

for tif in carpeta.glob("*.tif"):
    with rasterio.open(tif) as src:
        data = src.read()  
        nodata = src.nodata  

    if data.shape[0] == 3:  
        array = data.transpose(1, 2, 0)  
        array = np.clip(array, 0, 255).astype(np.uint8)
        img = Image.fromarray(array)
    else:  
        array = data[0]  

        if nodata is not None:
            mask = array == nodata
        else:
            mask = np.isnan(array) | np.isinf(array)

        valid_data = array[~mask]
        if valid_data.size > 0:
            vmin, vmax = valid_data.min(), valid_data.max()
            vmin = max(vmin, 0)    
            vmax = min(vmax, 1)
        else:
            vmin, vmax = 0, 1

        normalized = (array - vmin) / (vmax - vmin + 1e-8)
        normalized = np.clip(normalized, 0, 1)

        scaled = (normalized * 255).astype(np.uint8)

        scaled[mask] = 0

        img = Image.fromarray(scaled, mode='L')  
        img.putpalette(full_palette)

        img = img.convert('RGB')

    png_path = tif.with_suffix('.png')
    img.save(png_path)
    print(f"Saved: {png_path.name}")
