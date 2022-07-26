import geopandas as gpd
import os
import pandas as pd
from glob import glob


def get_bounds(region):
    file_path = glob(os.path.join("AOI_files", region, "*.shp"))[0]
    return gpd.read_file(file_path).to_crs(epsg=4326).geometry.values[0].bounds


def lin_interp(data):
    """
    data: array (Last axis MUST be time!)
    """
    ndim = data.ndim
    in_shape = data.shape
    if ndim > 2:
        data = data.reshape(-1, data.shape[-1])
    data = pd.DataFrame(data.T).interpolate(method="linear", limit_direction="forward").interpolate(method="linear", limit_direction="backward").values.T
    return data.reshape(in_shape)

