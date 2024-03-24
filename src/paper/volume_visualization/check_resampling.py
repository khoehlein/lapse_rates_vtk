import pyvista as pv
from pyvista import examples
import numpy as np

data_to_probe = examples.load_uniform()
points = np.array([[1.5, 5.0, 6.2], [6.7, 4.2, 8.0]])
mesh = pv.PolyData(points)
result = mesh.sample(data_to_probe)
output = result["Spatial Point Data"]
npout = np.asarray(output)
print('Done')