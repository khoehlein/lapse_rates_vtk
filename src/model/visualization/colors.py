import numpy as np
from PyQt5.QtCore import QObject
from PyQt5.QtGui import QColor
from matplotlib import pyplot as plt


class ColormapModel(QObject):
    
    def __init__(self, is_uniform: bool, parent=None):
        super().__init__(parent)
        self.is_uniform = is_uniform


class UniformColormap(ColormapModel):
    
    def __init__(self, color: QColor, parent=None):
        super().__init__(True, parent)
        self.color = color


class SequentialColormap(ColormapModel):

    def __init__(
            self,
            name: str,
            vmin: float, vmax: float,
            num_steps: int = 256,
            color_below_range: QColor = None, color_above_range: QColor = None,
            scalar_name: str=None,
            parent=None
    ):
        super().__init__(False, parent)
        self.name = name
        assert vmax > vmin
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        cmap = plt.get_cmap(self.name)
        if color_below_range is None:
            color_below_range = numpy_to_qcolor(cmap.get_under())
        self.color_below_range = color_below_range
        if color_above_range is None:
            color_above_range = numpy_to_qcolor(cmap.get_over())
        self.color_above_range = color_above_range
        num_steps = int(num_steps)
        assert num_steps > 0
        self.num_steps = num_steps
        self.scalar_name = str(scalar_name)

    def get_cmap(self):
        return plt.get_cmap(self.name)


class DivergingColormap(SequentialColormap):

    def __init__(
            self,
            name: str,
            vmin: float, vmax: float,
            center: float = None, num_steps: int = 256,
            color_below_range: QColor = None, color_above_range: QColor = None,
            scalar_name: str=None,
            parent=None
    ):
        super().__init__(name, vmin, vmax, num_steps, color_below_range, color_above_range, scalar_name, parent)
        if center is None:
            center = (self.vmax - self.vmin) / 2.
        center = float(center)
        assert center > vmin
        assert vmax > center
        self.center = center


def numpy_to_qcolor(numpy_color: np.ndarray) -> QColor:
    return QColor(*np.floor(256 * numpy_color).astype(int).clip(max=255))