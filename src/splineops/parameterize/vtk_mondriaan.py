"""
VTK demo window for the 3-D Mondriaan surface.
"""
import time
import numpy as np
import vtk
from vtk.util import numpy_support as vtknp

from constants import *
from brownian import BrownianVector3, BrownianRotation4
from morph import Morph
from mondriaan_layers import MondriaanLayers

# --------------------------------------------------------------------------
class MondriaanVTK:
    def __init__(self):
        self.morph = Morph()
        self.layers = MondriaanLayers()
        self.pos_b = BrownianVector3()
        self.rot_b = BrownianRotation4()

        self._build_geometry()
        self._build_scene()
        self.start_time = time.perf_counter()
        self._timer = self.iren.AddObserver('TimerEvent', self._on_timer)
        self.iren.CreateRepeatingTimer(10)  # ~100 fps

    # ----------------------------------------------------------------------
    def _build_geometry(self):
        # parametric grid of (s,t) in [0,1]²
        s = np.linspace(0, 1, MESH_W, dtype='f4')
        t = np.linspace(0, 1, MESH_H, dtype='f4')
        self.st_grid = np.dstack(np.meshgrid(s, t, indexing='xy')).reshape(-1, 2)

        # VTK points & normals
        self.vtk_points = vtk.vtkPoints()
        self.vtk_points.SetData(vtknp.numpy_to_vtk(
            np.zeros((self.st_grid.shape[0], 3), 'f4')))
        self.vtk_normals = vtk.vtkFloatArray()
        self.vtk_normals.SetNumberOfComponents(3)
        self.vtk_normals.SetNumberOfTuples(self.st_grid.shape[0])

        # cell connectivity (triangles)
        cells = vtk.vtkCellArray()
        for j in range(MESH_H - 1):
            for i in range(MESH_W - 1):
                a = j * MESH_W + i
                b = a + 1
                c = a + MESH_W
                d = c + 1
                for tri in ((a, c, b), (b, c, d)):
                    triangle = vtk.vtkTriangle()
                    for n, pid in enumerate(tri):
                        triangle.GetPointIds().SetId(n, pid)
                    cells.InsertNextCell(triangle)

        # polydata
        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.vtk_points)
        self.poly.GetPointData().SetNormals(self.vtk_normals)
        self.poly.SetPolys(cells)

        self.normals_filter = vtk.vtkPolyDataNormals()
        self.normals_filter.SetInputData(self.poly)
        self.normals_filter.SplittingOff()

    # ----------------------------------------------------------------------
    def _build_scene(self):
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.normals_filter.GetOutputPort())
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.texture = vtk.vtkTexture()
        self.texture.InterpolateOn()
        self.texture.RepeatOn()
        self.actor.SetTexture(self.texture)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.05, 0.1, 0.15)
        self.ren.AddActor(self.actor)

        self.cam = self.ren.GetActiveCamera()
        self.cam.SetViewAngle(ZOOM_ANGLE)
        self.cam.SetClippingRange(0.5, 20)

        self.win = vtk.vtkRenderWindow()
        self.win.AddRenderer(self.ren)
        self.win.SetSize(1200, 900)

        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.win)

    # ----------------------------------------------------------------------
    def _on_timer(self, _obj, _evt):
        t = time.perf_counter() - self.start_time

        # update geometry ---------------------------------------------------
        verts = np.empty((self.st_grid.shape[0], 3), 'f4')
        nrms  = np.empty_like(verts)
        for p, st in enumerate(self.st_grid):
            verts[p], nrms[p] = self.morph.evaluate(st)
        self.vtk_points.SetData(vtknp.numpy_to_vtk(verts))
        self.vtk_normals.SetArray(nrms.ravel(), nrms.size, 1)
        self.vtk_points.Modified(); self.vtk_normals.Modified()
        self.normals_filter.Update()

        # update camera -----------------------------------------------------
        eye = self.pos_b.update(t) + np.array([0, 0, DISTANCE], 'f4')
        rot = self.rot_b.update(t)[:3,:3]
        up  = rot @ np.array([0, 1, 0])
        self.cam.SetPosition(*eye)
        self.cam.SetFocalPoint(0, 0, 0)
        self.cam.SetViewUp(*up)

        # update texture ----------------------------------------------------
        rgb = self.layers.update(t)                             # (H,W,3)
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(TEX_W, TEX_H, 1)
        vtk_img.AllocateScalars(vtk.VTK_FLOAT, 3)
        vtk_img.GetPointData().SetScalars(
            vtknp.numpy_to_vtk(rgb[::-1].reshape(-1, 3)))
        self.texture.SetInputData(vtk_img)

        # render frame ------------------------------------------------------
        self.win.Render()

    def start(self):
        self.iren.Initialize()
        self.iren.Start()


# --------------------------------------------------------------------------
if __name__ == '__main__':
    MondriaanVTK().start()
