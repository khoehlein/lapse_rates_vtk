import pyvista as pv
import numpy as np


mesh = pv.Sphere()

origins = np.zeros((100, 3), dtype=float)
origins[:, 0] = - 10
origins[:, 1] = np.linspace(-10, 10, 100)
directions = np.zeros((100, 3), dtype=float)
directions[:, 0] = 1.

points, rays, cells = mesh.multi_ray_trace(origins, directions, first_point=True)

print('Done')