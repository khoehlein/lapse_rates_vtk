from PyQt5.QtCore import QObject

from src.interaction.domain_selection.view import DomainSelectionView


class DomainSelectionControl(QObject):

    def __init__(self, selection_view: DomainSelectionView, parent=None)
        super().__init__(parent)