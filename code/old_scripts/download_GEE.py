import ee
import geemap
import os

project_id = 'rellenos-sanitarios-mexico'

try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# Define parameters
output_dir = './outputs/'
filename = 'output.tif'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load Mexico boundary from FAO GAUL
mexico = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
    ee.Filter.eq('ADM0_NAME', 'Mexico')
)

# Load and process SMAP dataset
dataset = ee.ImageCollection('NASA/SMAP/SPL3SMP_E/005') \
    .filterBounds(mexico)

soil_moisture = dataset.select('soil_moisture_am') \
    .mean() \
    .clip(mexico)

# Set export parameters
region = mexico.geometry()
scale = 10000  # SMAP native resolution is ~9 km

# Download the image
geemap.ee_export_image(
    soil_moisture,
    filename=os.path.join(output_dir, filename),
    region=region,
    scale=scale,
    file_per_band=False
)

print(f"Image saved to {os.path.join(output_dir, filename)}")
