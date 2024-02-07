from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QColorDialog, QPushButton


class SelectColorButton(QPushButton):

    color_changed = pyqtSignal(QColor)

    def __init__(self, color: QColor = None, parent=None):
        super().__init__(parent)
        if color is None:
            color = QColor(0, 0, 0)
        self.current_color = color
        self._update_button_icon()
        self.clicked.connect(self._select_color)

    def set_current_color(self, color: QColor):
        self.current_color = color
        self._update_button_icon()
        self.color_changed.emit(color)
        return self

    def resizeEvent(self, a0=None):
        out = super().resizeEvent(a0)
        self._update_button_icon()
        return out

    def _update_button_icon(self):
        pixmap = QPixmap(12, 12)
        pixmap.fill(self.current_color)
        self.setIcon(QIcon(pixmap))

    def _select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color
            self._update_button_icon()
            self.color_changed.emit(self.current_color)
