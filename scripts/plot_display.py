from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis
from matplotlib.colors import Colormap, Normalize


class VolumeDisplay(object):

    def __init__(
            self,
            t: np.ndarray, z: np.ndarray,
            origin: float = 0., width: float = 1.
    ):
        self.t = t
        self.z = z
        self.origin = origin
        self.width = width

    def draw(self, ax: Axis, cmap: Colormap = None, norm: Normalize = None) -> None:
        x = self.origin * np.arange(2)[None, :] * self.width
        y = self.z.ravel()[:, None]
        t = self.t.ravel()[:, None]
        plot = ax.pcolor(x, y, t, cmap=cmap, norm=norm)
        return plot


def _test():
    z_vol = np.array([1092.73, 987.15, 889.29, 798.72, 715.02, 637.76, 566.54, 500.95, 440.61, 385.16, 334.24, 287.52, 244.69, 205.44, 169.51, 136.62, 106.54, 79.04, 53.92, 30.96, 10.])
    t_vol = np.array([281.05, 281.73, 282.37, 282.96, 283.5, 284, 284.47, 284.89, 285.29, 285.65, 285.98, 286.28, 286.56, 286.81, 287.05, 287.26, 287.46, 287.64, 287.8, 287.95, 288.09])
    t_surf = 273.15 + 15.
    z_surf = 0.
    z_ext = np.linspace(-500., 500., 32)
    t_ext = t_surf - 0.00065 * z_ext

    fig, ax = plt.subplots(1, 1)



if __name__ == "__main__":
    _test()