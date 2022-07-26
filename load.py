from glob import glob
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.cluster import KMeans

from download import DATA_DIR
import utils


def parse_filename(filename):
    yyyymmdd = filename.split("P_E_")[-1].split("_R")[0]
    year, month, day = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:])
    return year, month, day


def to_datetime(year_month_day_tuple):
    return datetime.date(*year_month_day_tuple)


def get_files():
    return glob(os.path.join(DATA_DIR, "*.h5"))


def open_file(file, aoi, quantity="soil_moisture"):
    if aoi is None:
        aoi = -180, -90, 180, 90
    with h5py.File(file, "r") as f:
        soil_moisture = np.stack([
            crop(f["Soil_Moisture_Retrieval_Data_AM"][quantity][...][...],
                 nan_fill(f["Soil_Moisture_Retrieval_Data_AM"]["latitude"][...]),
                 nan_fill(f["Soil_Moisture_Retrieval_Data_AM"]["longitude"][...]),
                 aoi
                 ),
            crop(f["Soil_Moisture_Retrieval_Data_PM"][quantity + "_pm"][...],
                 nan_fill(f["Soil_Moisture_Retrieval_Data_PM"]["latitude_pm"][...]),
                 nan_fill(f["Soil_Moisture_Retrieval_Data_PM"]["longitude_pm"][...]),
                 aoi
                 ),
        ], axis=-1)
    return soil_moisture


def open_data(quantity="soil_moisture", aoi=None, with_nans=True):
    files = get_files()
    dates = [to_datetime(parse_filename(f)) for f in files]

    files = [file for _, file in sorted(zip(dates, files))]
    dates = sorted(dates)

    data = np.concatenate(
        [open_file(file, aoi, quantity) for file in files],
        axis=-1
    )
    if with_nans:
        return nan_fill(data)
    else:
        return data


def nan_fill(data, no_data_value=-9999):
    data[data == no_data_value] = np.nan
    return data


def crop(data, latitudes, longitudes, bounds):
    minx, miny, maxx, maxy = bounds

    left = np.min(np.argwhere(longitudes >= minx)[:, 1])
    right = np.max(np.argwhere(longitudes <= maxx)[:, 1])
    top = np.min(np.argwhere(latitudes <= maxy)[:, 0])
    bottom = np.max(np.argwhere(latitudes >= miny)[:, 0])

    return data[top: bottom, left: right]


def kmeans_classify(data):
    folded = data[...].reshape(data.shape[:2] + (-1, 6))  # 3 day average
    data = np.nanmean(folded, axis=-1).reshape(data.shape[:2] + (-1,))

    never_seen = np.all(np.isnan(data), axis=-1)  # bool

    # flatten
    data_seen = data[~never_seen]
    # linearly interpolate to replace any nans still present
    data_seen = utils.lin_interp(data_seen)

    labels = KMeans(n_clusters=6).fit_predict(data_seen)

    labels_real = np.zeros(never_seen.shape) * np.nan
    labels_real[~never_seen] = labels
    return labels_real


def main():
    if not os.path.isfile("California_data.npy"):
        calif = open_data(aoi=utils.get_bounds("California"))
        np.save("California_data.npy", calif)
    else:
        calif = np.load("California_data.npy")

    if not os.path.isfile("Italy_data.npy"):
        italy = open_data(aoi=utils.get_bounds("Italy"))
        np.save("Italy_data.npy", italy)
    else:
        italy = np.load("Italy_data.npy")

    if not os.path.isfile("SouthAfrica_data.npy"):
        sa = open_data(aoi=utils.get_bounds("SouthAfrica"))
        np.save("SouthAfrica_data.npy", sa)
    else:
        sa = np.load("SouthAfrica_data.npy")

    if not os.path.isfile("labels/California.npy"):
        calif_labels = kmeans_classify(calif)
        np.save("labels/California.npy", calif_labels)
    else:
        calif_labels = np.load("labels/California.npy")
    if not os.path.isfile("labels/Italy.npy"):
        italy_labels = kmeans_classify(italy)
        np.save("labels/Italy.npy", italy_labels)
    else:
        italy_labels = np.load("labels/Italy.npy")
    if not os.path.isfile("labels/SouthAfrica.npy"):
        sa_labels = kmeans_classify(sa)
        np.save("labels/SouthAfrica.npy", sa_labels)
    else:
        sa_labels = np.load("labels/SouthAfrica.npy")

    fig, ax = plt.subplots()
    ax.matshow(calif_labels, cmap=plt.cm.Accent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("California Soil Moisture\n(k-means classified)")
    fig.savefig("figures/California_labels.png", bbox_inches="tight", dpi=400)

    fig, ax = plt.subplots()
    ax.matshow(italy_labels, cmap=plt.cm.Accent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Italy Soil Moisture\n(k-means classified)")
    fig.savefig("figures/Italy_labels.png", bbox_inches="tight", dpi=400)

    fig, ax = plt.subplots()
    ax.matshow(sa_labels, cmap=plt.cm.Accent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("South Africa Soil Moisture\n(k-means classified)")
    fig.savefig("figures/SouthAfrica_labels.png", bbox_inches="tight", dpi=400)


if __name__ == '__main__':
    main()
