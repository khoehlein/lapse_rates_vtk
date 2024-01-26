from PyQt5.QtCore import QObject
from PyQt5.QtGui import QColor


class ColormapModel(QObject):
    
    def __init__(self, parent=None):
        super().__init__(parent)


class UniformColor(ColormapModel):
    
    def __init__(self, color: QColor, opacity: float = None, parent=None):
        super().__init__(parent)
        self.color = color
        self.opacity = opacity
