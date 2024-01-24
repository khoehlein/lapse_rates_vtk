from PyQt5.QtCore import pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QToolButton, QColorDialog, QVBoxLayout, QPushButton


class SelectColorButton(QPushButton):

    color_changed = pyqtSignal(QColor)
    size_changed = pyqtSignal(QSize)

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self.current_color = color
        # self.display_button = QToolButton(self)
        self._update_button_icon()
        self.clicked.connect(self._select_color)

    def resizeEvent(self, a0=None):
        out = super().resizeEvent(a0)
        self._update_button_icon()
        return out

    def _update_button_icon(self):
        pixmap = QPixmap(16, 16)
        pixmap.fill(self.current_color)
        self.setIcon(QIcon(pixmap))

    def _select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color
            self._update_button_icon()
            self.color_changed.emit(self.current_color)