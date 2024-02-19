from typing import Dict, Any, Union

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel, QComboBox, QStackedLayout, \
    QPushButton, QVBoxLayout

from src.interaction.interface import PropertyModelView, PropertyModelController
from src.model.data.data_store import GlobalData
from src.model.downscaling.neighborhood import NeighborhoodModel, NeighborhoodType, LookupType, \
    DEFAULT_NEIGHBORHOOD_RADIAL, DEFAULT_NEIGHBORHOOD_NEAREST_NEIGHBORS


class _NeighborhoodMethodHandles(PropertyModelView):

    def __init__(self, neighborhood_type: NeighborhoodType, parent=None):
        super().__init__(parent)
        self.neighborhood_type = neighborhood_type
        self.spinner_neighborhood_size = QDoubleSpinBox(self)
        self.spinner_threshold = QDoubleSpinBox(self)
        self.spinner_threshold.setMinimum(0.)
        self.spinner_threshold.setMaximum(1.)
        self.spinner_threshold.setSingleStep(0.05)
        self.combo_lookup_type = QComboBox(self)
        self.combo_lookup_type.addItem('auto', LookupType.AUTO)
        self.combo_lookup_type.addItem('Ball Tree', LookupType.BALL_TREE)
        self.combo_lookup_type.addItem('KD-Tree', LookupType.KD_TREE)
        self.combo_lookup_type.addItem('Brute', LookupType.BRUTE)
        self._lookup_opts: Dict[LookupType, int] = {
            LookupType.AUTO: 0,
            LookupType.BALL_TREE: 1,
            LookupType.KD_TREE: 2,
            LookupType.BRUTE: 3
        }
        self.combo_num_processes = QComboBox(self)
        self.combo_num_processes.addItem('all', -1)
        self.combo_num_processes.addItem('1', 1)
        self.combo_num_processes.addItem('2', 2)
        self.combo_num_processes.addItem('4', 4)
        self.combo_num_processes.addItem('8', 8)
        self._process_opts: Dict[int, int] = {
            -1: 0, 1: 1, 2: 2, 4: 3, 8: 4,
        }
        self._set_layout()

    def _set_layout(self):
        layout = self.build_layout()
        self.setLayout(layout)

    def build_layout(self):
        raise NotImplementedError()

    def update_settings(self, settings: NeighborhoodModel.Properties):
        assert settings.neighborhood_type == self.neighborhood_type
        self.spinner_neighborhood_size.setValue(settings.neighborhood_size)
        self.spinner_threshold.setValue(settings.lsm_threshold)
        self.combo_lookup_type.setCurrentIndex(self._lookup_opts.get(settings.tree_type))
        self.combo_num_processes.setCurrentIndex(self._process_opts.get(settings.num_jobs))
        return self

    def get_settings(self):
        return NeighborhoodModel.Properties(
            self.neighborhood_type,
            self._get_neighborhood_size(),
            self.combo_lookup_type.currentData(),
            self.combo_num_processes.currentData(),
            self.spinner_threshold.value(),
        )

    def _get_neighborhood_size(self) -> Union[int, float]:
        raise NotImplementedError()


class RadialNeighborhoodHandles(_NeighborhoodMethodHandles):

    def __init__(self, parent=None):
        super(RadialNeighborhoodHandles, self).__init__(NeighborhoodType.RADIAL, parent)
        self.spinner_neighborhood_size.setSuffix(' km')
        self.spinner_neighborhood_size.setMinimum(16.)
        self.spinner_neighborhood_size.setMaximum(256.)

    def build_layout(self):
        layout = QFormLayout(self)
        layout.addRow(QLabel('Radius:'), self.spinner_neighborhood_size)
        layout.addRow(QLabel('Land/sea threshold:'), self.spinner_threshold)
        layout.addRow(QLabel('Lookup method:'), self.combo_lookup_type)
        layout.addRow(QLabel('Lookup processes:'), self.combo_num_processes)
        return layout

    def _get_neighborhood_size(self) -> Union[int, float]:
        return float(self.spinner_neighborhood_size.value())


class KNNNeighborhoodHandles(_NeighborhoodMethodHandles):

    def __init__(self, parent=None):
        super(KNNNeighborhoodHandles, self).__init__(NeighborhoodType.RADIAL, parent)
        self.spinner_neighborhood_size.setSingleStep(1)
        self.spinner_neighborhood_size.setMinimum(16)
        self.spinner_neighborhood_size.setMaximum(256)

    def build_layout(self):
        layout = QFormLayout(self)
        layout.addRow(QLabel('Neighborhood size:'), self.spinner_neighborhood_size)
        layout.addRow(QLabel('Land/sea threshold:'), self.spinner_threshold)
        layout.addRow(QLabel('Lookup method:'), self.combo_lookup_type)
        layout.addRow(QLabel('Lookup processes:'), self.combo_num_processes)
        return layout

    def _get_neighborhood_size(self) -> Union[int, float]:
        return int(self.spinner_neighborhood_size.value())


class NeighborhoodModelView(PropertyModelView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.combo_lookup_type = QComboBox(self)
        self.interface_stack = QStackedLayout()
        self._handle_widgets = {}
        self.combo_lookup_type.addItem('Radius', NeighborhoodType.RADIAL)
        radial_interface = RadialNeighborhoodHandles(self)
        self.interface_stack.addWidget(radial_interface)
        self._handle_widgets[NeighborhoodType.RADIAL] = radial_interface

        self.combo_lookup_type.addItem('Nearest neighbors', NeighborhoodType.NEAREST_NEIGHBORS)
        knn_interface = KNNNeighborhoodHandles(self)
        self.interface_stack.addWidget(knn_interface)
        self._handle_widgets[NeighborhoodType.NEAREST_NEIGHBORS] = knn_interface

        self.combo_lookup_type.currentIndexChanged.connect(self.interface_stack.setCurrentIndex)

        self.button_apply = QPushButton('Apply')
        self.button_apply.clicked.connect(self.settings_changed.emit)

        self._set_layout()

    def _set_layout(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Neighborhood method:'))
        layout.addWidget(self.combo_lookup_type)
        layout.addLayout(self.interface_stack)
        layout.addStretch()
        layout.addWidget(self.button_apply)
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def get_settings(self):
        return self.interface_stack.currentWidget().get_settings()

    def update_settings(self, settings: NeighborhoodModel.Properties):
        neighborhood_type = settings.neighborhood_type
        self._handle_widgets[neighborhood_type].update_settings(settings)
        idx = {NeighborhoodType.RADIAL: 0, NeighborhoodType.NEAREST_NEIGHBORS: 1}.get(neighborhood_type)
        self.combo_lookup_type.setCurrentIndex(idx)
        return self


class NeighborhoodModelController(PropertyModelController):

    @classmethod
    def from_view(cls, view: NeighborhoodModelView, data_store: GlobalData) -> 'NeighborhoodModelController':
        model = NeighborhoodModel(data_store)
        return cls(view, model)

    def set_defaults(self):
        self.view.blockSignals(True)
        self.view.update_settings(DEFAULT_NEIGHBORHOOD_NEAREST_NEIGHBORS)
        self.view.blockSignals(False)
        self.view.update_settings(DEFAULT_NEIGHBORHOOD_RADIAL)

    def default_settings(self):
        return NeighborhoodModel.Properties(
            neighborhood_type=NeighborhoodType.RADIAL,
            neighborhood_size=60.,
            tree_type=LookupType.AUTO,
            num_jobs=-1,
            lsm_threshold=0.5
        )
