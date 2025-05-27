#!/usr/bin/env python3
# monopole_vtk.py ------------------------------------------------------------
"""
Simple VTK demo that draws the sphere parameterised by Thevenaz'
"monopole" mapping (Eq. 3.5 of the notes).  No textures, no morphing,
just geometry + shading.
"""

import vtk
import numpy as np
import math

# ---------------------------------------------------------------------------#
# Parameters you may tweak
# ---------------------------------------------------------------------------#
M, N = 128, 128           # grid resolution in u (horizontal) and v (vertical)
eps       = 1.0e-4        # margin to stay clear of the single pole
background = (0.05, 0.1, 0.15)  # renderer background colour (RGB)

# ---------------------------------------------------------------------------#
# Mapping Γ_U(u,v)  ->  (x,y,z)
# ---------------------------------------------------------------------------#
def monopole(u: float, v: float):
    """
    Evaluate Γ_U(u,v) as defined in (1).

    Parameters
    ----------
    u, v : floats in (0,1)

    Returns
    -------
    tuple (x, y, z)
    """
    su  = math.sin(math.pi * u)
    sv  = math.sin(math.pi * v)
    s2u = su * su                     # sin²(π u)
    x = -math.sin(2.0 * math.pi * u) * sv       # -sin(2πu) * sin(πv)
    y = -s2u * math.sin(2.0 * math.pi * v)      # -sin²(πu) * sin(2πv)
    z = -(1.0 - 2.0 * s2u * sv * sv)            # -[1 - 2 sin²(πu) sin²(πv)]
    return x, y, z

# ---------------------------------------------------------------------------#
# Build a vtkPolyData from the parameter grid
# ---------------------------------------------------------------------------#
points   = vtk.vtkPoints()
polys    = vtk.vtkCellArray()
point_id = np.empty((M + 1, N + 1), dtype=int)

for i in range(M + 1):
    u = eps + (1.0 - 2 * eps) * i / M          # avoid 0 & 1 exactly
    for j in range(N + 1):
        v = eps + (1.0 - 2 * eps) * j / N
        pid = points.InsertNextPoint(*monopole(u, v))
        point_id[i, j] = pid

# two triangles per grid quad
for i in range(M):
    for j in range(N):
        p0 = point_id[i,     j]
        p1 = point_id[i + 1, j]
        p2 = point_id[i + 1, j + 1]
        p3 = point_id[i,     j + 1]

        polys.InsertNextCell(3)
        polys.InsertCellPoint(p0)
        polys.InsertCellPoint(p1)
        polys.InsertCellPoint(p2)

        polys.InsertNextCell(3)
        polys.InsertCellPoint(p0)
        polys.InsertCellPoint(p2)
        polys.InsertCellPoint(p3)

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetPolys(polys)

# generate vertex normals so lighting looks good
normals = vtk.vtkPolyDataNormals()
normals.SetInputData(polydata)
normals.AutoOrientNormalsOn()
normals.Update()

# ---------------------------------------------------------------------------#
# VTK pipeline & window
# ---------------------------------------------------------------------------#
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(normals.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(*background)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(800, 800)
renWin.SetWindowName("Monopole Sphere – VTK")

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renWin)

# Some sensible camera settings
renderer.ResetCamera()
cam = renderer.GetActiveCamera()
cam.Elevation(-20)
cam.Azimuth(30)
cam.Zoom(1.5)

renWin.Render()
interactor.Start()
