import numpy as np
import os



def make_splits(numb_nodes, seed, train_ratio, val_ratio):
    """
    numb_nodes: int (number of nodes in graph)
    seed: int (use this to set numpy random seed to make random splits reproduceable)
    train_ratio: float (between 0 and 1)
    val_ration: float (between 0 and 1)
    
    Write a function which will return a numpy array of shape (numb_nodes,)
    The array should contain strings (either "train", "val" or "test") such that
    the proportion of total entries which are "train" is given by `train_ratio`,
    the proportion of total entries which are "val" is given by `val_ratio`,
    and the rest are "test"
    """
    
    np.random.seed(seed)
    idxs = np.random.choice(np.arange(numb_nodes), numb_nodes, replace=False)

    splits = np.array(["empty"] * numb_nodes)
    splits[idxs[:int(numb_nodes * train_ratio)]] = "train"
    splits[idxs[int(numb_nodes * train_ratio): int(numb_nodes * (train_ratio + val_ratio))]] = "val"
    splits[idxs[int(numb_nodes * (train_ratio + val_ratio)):]] = "test"

    return splits


def load_calif():
    return np.load("California_data.npy"), np.load(os.path.join("labels", "California.npy"))


def load_southafrica():
    return np.load("SouthAfrica_data.npy"), np.load(os.path.join("labels", "SouthAfrica.npy"))


def normalize(data_values):
    maximum = np.nanmax(data_values, axis=0, keepdims=True)
    minimum = np.nanmin(data_values, axis=0, keepdims=True)
    return (data_values - minimum) / (maximum - minimum)


def flatten(data_values):
    """
    data_values: numpy array (ndim=3)
    
    return:
    data_values: numpy array (ndim=2)
    """
    return data_values.reshape(-1, data_values.shape[-1])


def get_vacuity(alphas):
    """
    alphas: numpy array (2-dimensinal) of shape (#nodes, #classes)
    """
    return alphas.shape[1] / np.sum(alphas, axis=1)


def expected_probability(alphas):
    """
    alphas: numpy array (2-dimensional) of shape (#nodes, #classes)
    
    return a numpy array (2-dimensional) of shape (#nodes, #classes)
    """
    return alphas / alphas.sum(axis=1, keepdims=True)
