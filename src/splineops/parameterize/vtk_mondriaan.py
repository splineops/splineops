import time, numpy as np, vtk
from vtk.util import numpy_support as vtknp

from constants          import *
from brownian           import BrownianVector3, BrownianRotation4
from morph              import Morph
from mondriaan_layers   import MondriaanLayers

class MondriaanVTK:
    def __init__(self):
        self.morph  = Morph()
        self.layers = MondriaanLayers()
        self.pos_b  = BrownianVector3()
        self.rot_b  = BrownianRotation4()

        self._init_polydata()
        self._init_isolines()       # ← new call
        self._init_vtk_scene()

        self.t_prev_print = -1      # debug counter
        self.t0 = time.perf_counter()

    # ------------ geometry ------------------------------------------------
    def _init_polydata(self):
        s = np.linspace(0, 1, MESH_W, dtype='f4')
        t = np.linspace(0, 1, MESH_H, dtype='f4')
        self.st_grid = np.dstack(np.meshgrid(s, t, indexing='xy')).reshape(-1, 2)

        self.vtk_pts = vtk.vtkPoints()
        self.vtk_pts.SetData(vtknp.numpy_to_vtk(
            np.zeros((self.st_grid.shape[0], 3), 'f4')))
        self.vtk_nrm = vtk.vtkFloatArray()
        self.vtk_nrm.SetNumberOfComponents(3)
        self.vtk_nrm.SetNumberOfTuples(self.st_grid.shape[0])

        cells = vtk.vtkCellArray()
        for j in range(MESH_H - 1):
            for i in range(MESH_W - 1):
                a = j * MESH_W + i
                b = a + 1
                c = a + MESH_W
                d = c + 1
                for tri in ((a, c, b), (b, c, d)):
                    tri_cell = vtk.vtkTriangle()
                    for n, pid in enumerate(tri):
                        tri_cell.GetPointIds().SetId(n, pid)
                    cells.InsertNextCell(tri_cell)

        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.vtk_pts)
        self.poly.GetPointData().SetNormals(self.vtk_nrm)
        self.poly.SetPolys(cells)

        self.normals_flt = vtk.vtkPolyDataNormals()
        self.normals_flt.SetInputData(self.poly)
        self.normals_flt.SplittingOff()

    # ------------------------------------------------------------------+
    def _init_isolines(self, iso_step: int = 16):
        """
        Build a vtkPolyData containing parametric grid lines every
        `iso_step` vertices.  Lines share *their own* point array so we
        can update them independently of the main mesh.
        """
        self.iso_map = []                       # maps isoline-point → mesh index

        # collect indices along constant-v (rows) and constant-u (cols)
        for row in range(0, MESH_H, iso_step):
            for col in range(MESH_W):
                self.iso_map.append(row * MESH_W + col)
        row_break = len(self.iso_map)           # delimiter
        for col in range(0, MESH_W, iso_step):
            for row in range(MESH_H):
                self.iso_map.append(row * MESH_W + col)

        # vtkPoints placeholder
        self.iso_pts = vtk.vtkPoints()
        self.iso_pts.SetData(vtknp.numpy_to_vtk(
            np.zeros((len(self.iso_map), 3), 'f4')))

        # connectivity as polylines ---------------------------------------
        lines = vtk.vtkCellArray()

        # first bundle: horizontal lines (constant v)
        offset = 0
        for _ in range(0, MESH_H, iso_step):
            poly = vtk.vtkPolyLine()
            poly.GetPointIds().SetNumberOfIds(MESH_W)
            for i in range(MESH_W):
                poly.GetPointIds().SetId(i, offset + i)
            lines.InsertNextCell(poly)
            offset += MESH_W

        # second bundle: vertical lines (constant u) ----------------------
        offset = row_break                      # first vertical point index
        for _ in range(0, MESH_W, iso_step):
            poly = vtk.vtkPolyLine()
            poly.GetPointIds().SetNumberOfIds(MESH_H)
            for j in range(MESH_H):
                poly.GetPointIds().SetId(j, offset + j)
            lines.InsertNextCell(poly)
            offset += MESH_H                    # jump to next column bloc

        self.iso_poly = vtk.vtkPolyData()
        self.iso_poly.SetPoints(self.iso_pts)
        self.iso_poly.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.iso_poly)

        self.iso_actor = vtk.vtkActor()
        self.iso_actor.SetMapper(mapper)
        self.iso_actor.GetProperty().SetColor(0, 0, 0)
        self.iso_actor.GetProperty().SetLineWidth(1.0)
        self.iso_actor.GetProperty().LightingOff()


    # ------------ VTK scene ----------------------------------------------
    def _init_vtk_scene(self):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.normals_flt.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.tex = vtk.vtkTexture()
        self.tex.InterpolateOn()
        self.tex.RepeatOn()
        self.actor.SetTexture(self.tex)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.05, 0.1, 0.15)
        self.ren.AddActor(self.actor)
        self.ren.AddActor(self.iso_actor)     # ← draw grid lines


        self.cam = self.ren.GetActiveCamera()
        self.cam.SetViewAngle(ZOOM_ANGLE)
        self.cam.SetClippingRange(0.5, 20)

        self.win = vtk.vtkRenderWindow()
        self.win.AddRenderer(self.ren)
        self.win.SetSize(1200, 900)

        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.win)

    # ------------ timer callback -----------------------------------------
    def _on_timer(self, *_):
        t = time.perf_counter() - self.t0

        # update geometry
        verts = np.empty((self.st_grid.shape[0], 3), 'f4')
        nrms  = np.empty_like(verts)
        for p, st in enumerate(self.st_grid):
            verts[p], nrms[p] = self.morph.evaluate(st)

        # update isoline points (share main verts via mapping)
        iso_xyz = verts[self.iso_map]
        self.iso_pts.SetData(vtknp.numpy_to_vtk(iso_xyz))
        self.iso_pts.Modified()
        self.vtk_pts.SetData(vtknp.numpy_to_vtk(verts))
        self.vtk_nrm.SetArray(nrms.ravel(), nrms.size, 1)
        self.vtk_pts.Modified(); self.vtk_nrm.Modified()
        self.normals_flt.Update()

        # update camera
        eye = self.pos_b.update(t) + np.array([0, 0, DISTANCE], 'f4')
        up  = self.rot_b.update(t)[:3, :3] @ np.array([0, 1, 0])
        self.cam.SetPosition(*eye)
        self.cam.SetFocalPoint(0, 0, 0)
        self.cam.SetViewUp(*up)

        # update texture
        rgb = self.layers.update(t)
        vtk_img = vtk.vtkImageData()
        vtk_img.SetDimensions(TEX_W, TEX_H, 1)
        vtk_img.AllocateScalars(vtk.VTK_FLOAT, 3)
        vtk_img.GetPointData().SetScalars(
            vtknp.numpy_to_vtk(rgb[::-1].reshape(-1, 3)))
        self.tex.SetInputData(vtk_img)

        # debug print once a second
        ti = int(t)
        if ti != self.t_prev_print:
            self.t_prev_print = ti
            zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
            g = rgb.mean(axis=2); gmin, gmax = g.min(), g.max()
            print(f"[t={ti}]  z-range {zmin:+.3f} … {zmax:+.3f}   "
                  f"gray {gmin:.3f} … {gmax:.3f}", flush=True)

        self.win.Render()

    # ------------ entry point --------------------------------------------
    def start(self):
        self.iren.Initialize()            # must come first!
        self.iren.AddObserver('TimerEvent', self._on_timer)
        self.iren.CreateRepeatingTimer(10)
        self.iren.Start()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    MondriaanVTK().start()
