from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QToolButton, QColorDialog, QVBoxLayout


class SelectColorButton(QToolButton):

    color_changed = pyqtSignal(QColor)

    def __init__(self, default_color: QColor, parent=None):
        super().__init__(parent)
        self.default_color = default_color
        self.display_button = QToolButton(self)
        self._update_button_icon(default_color)
        self.display_button.clicked.connect(self._select_color)

    def _update_button_icon(self, color: QColor):
        pixmap = QPixmap(self.width(), self.height())
        pixmap.fill(color)
        self.display_button.setIcon(QIcon(pixmap))

    def _select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color
            self._update_button_icon(color)
            self.color_changed.emit(self.current_color)