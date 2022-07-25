from glob import glob
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from download import DATA_DIR


def parse_filename(filename):
    yyyymmdd = filename.split("P_E_")[-1].split("_R")[0]
    year, month, day = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:])
    return year, month, day


def to_datetime(year_month_day_tuple):
    return datetime.date(*year_month_day_tuple)


def get_files():
    return glob(os.path.join(DATA_DIR, "*.h5"))


def open_file(file):
    with h5py.File(file, "r") as f:
        soil_moisture = np.stack([
            f["Soil_Moisture_Retrieval_Data_AM"]["soil_moisture"][...],
            f["Soil_Moisture_Retrieval_Data_PM"]["soil_moisture_pm"][...]
        ], axis=-1)
    return soil_moisture


def open_data():
    files = get_files()
    dates = [to_datetime(parse_filename(f)) for f in files]

    files = [file for _, file in sorted(zip(dates, files))]
    dates = sorted(dates)

    return np.concatenate(
        [open_file(file) for file in files],
        axis=-1
    )


def nan_fill(data, no_data_value=-9999):
    data[data == no_data_value] = np.nan
    return data


def main():
    data = open_data()
    data = nan_fill(data)

    fig, ax = plt.subplots(1, 6)
    for i in range(6):
        ax[i].matshow(data[..., i], cmap="terrain_r")
    plt.show()


if __name__ == '__main__':
    main()
