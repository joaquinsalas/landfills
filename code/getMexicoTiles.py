import os
import math
import requests
from tqdm import tqdm

# ==============================
# CONFIGURA ESTO
# ==============================

API_KEY = "vGmiSsqHshsF573JvDce"
ZOOM = 14

# Bounding Box México ejemplo (puedes cambiarlo)
# (lat_min, lon_min, lat_max, lon_max)
BBOX = (14.5, -118.5, 32.8, -86.7)
BBOXCDMX = (19.40, -99.20, 19.45, -99.10)

OUTPUT_DIR = "tiles"

# ==============================
# FUNCIONES GEOGRÁFICAS
# ==============================

def deg2num(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0
        * n
    )
    return xtile, ytile


def get_tile_range(bbox, zoom):
    lat_min, lon_min, lat_max, lon_max = bbox

    x_min, y_max = deg2num(lat_min, lon_min, zoom)
    x_max, y_min = deg2num(lat_max, lon_max, zoom)

    return x_min, x_max, y_min, y_max


# ==============================
# DESCARGA DE TILE
# ==============================

def download_tile(z, x, y):
    url = f"https://api.maptiler.com/maps/satellite/{z}/{x}/{y}.jpg?key={API_KEY}"
    path = os.path.join(OUTPUT_DIR, str(z), str(x))
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{y}.jpg")

    # Cache: si ya existe, no descarga
    if os.path.exists(filename):
        return

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                f.write(r.content)
    except Exception as e:
        print(f"Error en tile {z}/{x}/{y}: {e}")


# ==============================
# MAIN
# ==============================

def main():
    x_min, x_max, y_min, y_max = get_tile_range(BBOX, ZOOM)

    total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    print(f"Tiles a descargar: {total_tiles}")

    for x in tqdm(range(x_min, x_max + 1)):
        for y in range(y_min, y_max + 1):
            download_tile(ZOOM, x, y)


if __name__ == "__main__":
    main()
