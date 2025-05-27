#!/usr/bin/env python
# pip install -U "pyvista>=0.44" numpy
import numpy as np, pyvista as pv

# ---------- cube -------------------------------------------------------
verts = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)

faces = np.array([[0,1,2,3],[4,5,6,7],[0,1,5,4],
                  [1,2,6,5],[2,3,7,6],[3,0,4,7]], int)

def pv_faces(c):               # prepend size column, then flatten
    return np.hstack([np.full((c.shape[0],1),c.shape[1]), c]).ravel()

mesh = pv.PolyData(verts, pv_faces(faces))

# ---------- viewer -----------------------------------------------------
pl = pv.Plotter(window_size=(900,600))
pl.add_mesh(mesh, show_edges=True, color="lightsteelblue", opacity=0.8)
pl.add_axes(); pl.show_grid(color="silver")
pl.camera.zoom(1.25)           # a nicer initial view

# ---------- one sphere widget per vertex -------------------------------
def make_cb(i):
    def _cb(point):
        mesh.points[i] = point
        pl.update_coordinates(mesh.points, render=True)
    return _cb

for i, p in enumerate(mesh.points):
    pl.add_sphere_widget(
        callback     = make_cb(i),
        center       = p,
        radius       = 0.05,
        color        = "orange",
        selected_color = "yellow",
        style        = "surface"
    )

pl.show()
