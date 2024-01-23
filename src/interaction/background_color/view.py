from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QToolButton, QColorDialog, QVBoxLayout


class SelectColorMenu(QWidget):

    def __init__(self, default_color: QColor, parent=None):
        super().__init__(parent)
        self.display_button = QToolButton(self)
        self.current_color = default_color
        layout = QVBoxLayout()
        layout.addWidget(self.display_button)
        self.setLayout(layout)
        self._update_button_icon(default_color)
        self.display_button.clicked.connect(self._select_color)

    def _update_button_icon(self, color: QColor):
        pixmap = QPixmap(16 ,16)
        pixmap.fill(color)
        self.display_button.setIcon(QIcon(pixmap))

    def _select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.current_color = color
            self._update_button_icon(color)