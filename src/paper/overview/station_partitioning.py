import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from sklearn.neighbors import KDTree, BallTree

from src.model.geometry import Coordinates

data = pd.read_csv('C:\\Users\\kevin\\data\\ECMWF\\Vis\\station_locations_nearest.csv', index_col=0)
data = data.set_index('stnid').sort_index()

coords = Coordinates.from_dataframe(data).as_lat_lon().values

def plot_kdtree():
    tree = KDTree(coords, leaf_size=64, metric='euclidean')

    arrays = tree.get_arrays()

    node_data = np.array([tuple(item) for item in arrays[2]])
    idx_start, idx_end, is_leaf, radius = list(node_data.T)

    bounds = arrays[3]

    lower_left_corner = bounds[0]
    upper_right_corner = bounds[1]
    wh = upper_right_corner - lower_left_corner
    dlon = wh[:, 0]
    dlat = wh[:, 1]

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(20, 10))

    for i, (xy, w, h) in enumerate(zip(lower_left_corner, dlon, dlat)):
        if is_leaf[i]:
            rectangle = Rectangle(xy, w, h, edgecolor='k', facecolor='r')
            ax.add_patch(rectangle)

    ax.scatter(coords[:, 0], coords[:, 1], zorder=40, alpha=0.1, s=2)
    ax.set(xlim=(-15, 35), ylim=(30, 80))
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_balltree():
    tree = BallTree(coords, leaf_size=128, metric='euclidean')

    arrays = tree.get_arrays()

    node_data = np.array([tuple(item) for item in arrays[2]])
    idx_start, idx_end, is_leaf, radius = list(node_data.T)

    bounds = arrays[3]

    center = bounds[0]

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(20, 10))

    for i, (c, r) in enumerate(zip(center, radius)):
        if is_leaf[i]:
            rectangle = Circle(c, r, edgecolor='k', facecolor='r', alpha=0.1)
            ax.add_patch(rectangle)

    ax.scatter(coords[:, 0], coords[:, 1], zorder=40, alpha=0.1, s=2)
    ax.set(xlim=(-15, 35), ylim=(30, 80))
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_kdtree()