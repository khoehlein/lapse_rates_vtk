from typing import Any, List, Type
from PyQt5.QtGui import QColor


class Parameter(object):

    def __init__(self, name: str, dtype: Type = None, value: Any = None, parent: 'ParameterGroup' = None):
        self.name = name
        self.dtype = dtype
        self.value = value
        self.parent = parent

    def is_set(self):
        return self.value is not None

    def __set__(self, instance, value):
        self.value = value

    def __get__(self, instance, owner):
        return self.value


class Constant(Parameter):

    def __init__(self, name: str, value: Any):
        super().__init__(name=name, value=value)

    def __set__(self, instance, value):
        raise RuntimeError(f'[ERROR] Parameter {self.name} is constant and cannot be set.')


class ChoiceParameter(Parameter):

    def __init__(self, name: str, dtype: Type = None, value: Any = None, choices: List[Any] = None):
        super().__init__(name, dtype, value)
        if choices is not None:
            self.choices = set(choices)
        self.choices = choices


class Boolean(Parameter):

    def __init__(self, name: str, value: bool = None, parent: 'ParameterGroup' = None):
        super().__init__(name, bool, value, parent)


class _RangeParameter(Parameter):

    def __init__(self, name: str, dtype: Type, value: Any, min_value: Any = None, max_value: Any = None):
        if min_value is not None:
            min_value = dtype(min_value)
        self.min_value = min_value
        if max_value is not None:
            max_value = dtype(max_value)
        self.max_value = max_value
        super().__init__(name, dtype, value)


class FloatParameter(_RangeParameter):

    def __init__(self, name: str, value: Any = None, min_value: Any = None, max_value: Any = None):
        super().__init__(name, float, value, min_value, max_value)


class IntParameter(_RangeParameter):

    def __init__(self, name: str, value: Any = None, min_value: Any = None, max_value: Any = None):
        super().__init__(name, int, value, min_value, max_value)


class ParameterGroup(object):

    def __init__(self, name: str, parent: 'ParameterGroup' = None):
        self.name = str(name)
        self.parameters = {}
        self.children = {}
        self.parent = parent

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            if name in self.parameters:
                raise RuntimeError('[ERROR] {} already exists'.format(name))
            self.parameters[name] = value
            value.parent = self
        if isinstance(value, ParameterGroup):
            if name in self.children:
                raise RuntimeError('[ERROR] {} already exists'.format(name))
            self.children[name] = value
            value.parent = self
        return super().__setattr__(name, value)

    def values(self):
        return {
            **{child.name: child.values() for child in self.children.values()},
            **{param.name: param.value for param in self.parameters.values()}
        }


class ColormapModel(ParameterGroup):

    def __init__(self):
        super().__init__('ColormapModel')
        self.colormap_name = Parameter('ColormapName', str)
        self.min_value = Parameter('ValueMin', float)
        self.max_value = Parameter('ValueMax', float)
        self.opacity = FloatParameter('Opacity', min_value=0., max_value=1.)
        self.below_range_color = Parameter('BelowRangeColor', QColor)
        self.above_range_color = Parameter('AboveRangeColor', QColor)


class MeshProperties(ParameterGroup):

    def __init__(self, name: str, style: str):
        super().__init__(name)
        self.style = Constant('Style', style)


class SurfaceProperties(MeshProperties):

    def __init__(self):
        super().__init__('SurfaceProperties', 'surface')
        self.edge_color = Parameter('EdgeColor', QColor)
        self.edge_opacity = FloatParameter('EdgeOpacity', min_value=0., max_value=1.)
        self.show_edges = Boolean('ShowEdges')


class WireframeProperties(MeshProperties):

    def __init__(self):
        super().__init__('WireframeProperties', 'wireframe')
        self.line_width = FloatParameter('LineWidth', min_value=0.)
        self.lines_as_tubes = Boolean('LineAsTubes')


class PointsProperties(MeshProperties):

    def __init__(self):
        super().__init__('PointsProperties', 'points')
        self.point_size = FloatParameter('PointSize', min_value=0.)
        self.points_as_spheres = Boolean('PointsAsSpheres')


class MeshGeometryModel(ParameterGroup):

    def __init__(self):
        super().__init__('MeshGeometryModel')
        self.color = Parameter('Colors', QColor)
        self.opacity = FloatParameter('Opacity', min_value=0., max_value=1.)
        self.visibility = FloatParameter('Visibility')
        self.mesh_properties: MeshProperties = None


