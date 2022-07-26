import geopandas as gpd
import os
from glob import glob


def get_bounds(region):
    file_path = glob(os.path.join("AOI_files", region, "*.shp"))[0]
    return gpd.read_file(file_path).to_crs(epsg=4326).geometry.values[0].bounds
