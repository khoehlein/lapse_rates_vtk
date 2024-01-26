from PyQt5.QtCore import QObject


class LightingModel(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)
