import pyvista as pv
import tetgen
import numpy as np
pv.set_plot_theme('document')

sphere = pv.Sphere()
tet = tetgen.TetGen(sphere)
tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
grid = tet.grid
print(type(grid))
grid.plot(show_edges=True)
