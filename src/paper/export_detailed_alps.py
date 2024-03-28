import os

import numpy as np
import pandas as pd
import xarray as xr

from src.model.geometry import OctahedralGrid, DomainLimits, DomainBoundingBox, LocationBatch, Coordinates, TriangleMesh
from src.model.level_heights import compute_standard_surface_pressure, compute_full_level_pressure, \
    compute_approximate_level_height
import networkx as nx


DOMAIN_NAME = 'central_europe'
DEFAULT_DOMAIN = DomainLimits(43., 47., 6., 12.)
