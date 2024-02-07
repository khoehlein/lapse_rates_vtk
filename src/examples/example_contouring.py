import pyvista as pv
import numpy as np


mesh = pv.Sphere()
mesh['my_scalar_z'] = mesh.points[:, -1] ** 2.
mesh['my_scalar_x'] = mesh.points[:, -1] ** 2.

contours_z = mesh.contour()

print(mesh.active_scalars_name)
print('Done')