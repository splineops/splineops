"""
mondriaan_animation.py  –  real-time “breathing Mondriaan” demo with VTK
========================================================================

 • Physics step : 40 ms   (≈25 Hz)
 • Sub-frames    : 15     (so 16 drawings / physics step → ≈400 FPS visual)
 • Breathing     : 0.04 Hz period (≈25 s), amplitude 0.12
 • Camera drift  : Brownian walk scaled by 0.5
 • Extras        : Mondriaan colour texture + u,v-isolines

Mouse controls
──────────────
 Trackball-camera style (VTK default) — left-drag rotate, middle-drag pan,
 wheel zoom.  MotionFactor is low, so the model stops when you stop
 dragging.
"""

import time
import numpy as np
import vtk
from  vtk.util import numpy_support as vtknp

from constants        import *
from brownian         import BrownianVector3, BrownianRotation4
from morph            import Morph
from mondriaan_layers import MondriaanLayers

# ───────────── pacing knobs — adjust to taste ────────────────────────────
PHYSICS_DT_MS = 40          # one physics tick every 40 ms → 25 Hz
SUB_FRAMES    = 15          # software in-betweens per physics tick
CAMERA_SCALE  = 0.5         # Brownian drift amplitude
BREATH_FREQ   = 0.04        # Hz  (≈25 s cycle)
BREATH_AMP    = 0.12        # radial bulge amplitude
# ------------------------------------------------------------------------

class MondriaanAnimation:
    """VTK window running the full animated pipeline."""

    # ───────── initialisation ────────────────────────────────────────────
    def __init__(self):
        self.morph   = Morph(freq_hz=BREATH_FREQ, amp=BREATH_AMP)
        self.layers  = MondriaanLayers()
        self.pos_b   = BrownianVector3()
        self.rot_b   = BrownianRotation4()

        self._build_mesh()
        self._build_isolines()
        self._build_scene()

        self.t0        = time.perf_counter()
        self.prev_t    = self.t0
        self.prev_xyz  = None        # forces first frame path

    # ───────── analytic mesh (monopole) ──────────────────────────────────
    def _build_mesh(self):
        s = np.linspace(0, 1, MESH_W, dtype='f4')
        t = np.linspace(0, 1, MESH_H, dtype='f4')
        self.st_grid = np.dstack(np.meshgrid(s, t, indexing='xy')).reshape(-1, 2)

        self.vtk_pts = vtk.vtkPoints()
        self.vtk_pts.SetData(vtknp.numpy_to_vtk(np.zeros((self.st_grid.shape[0], 3), 'f4')))
        self.vtk_nrm = vtk.vtkFloatArray(); self.vtk_nrm.SetNumberOfComponents(3)
        self.vtk_nrm.SetNumberOfTuples(self.st_grid.shape[0])

        # connectivity (two triangles per quad)
        cells = vtk.vtkCellArray()
        for j in range(MESH_H - 1):
            for i in range(MESH_W - 1):
                a = j * MESH_W + i; b = a + 1; c = a + MESH_W; d = c + 1
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

    # ───────── isoline overlay ───────────────────────────────────────────
    def _build_isolines(self, iso_step: int = 16):
        self.iso_idx = []
        for row in range(0, MESH_H, iso_step):
            self.iso_idx.extend(row * MESH_W + np.arange(MESH_W))
        row_break = len(self.iso_idx)
        for col in range(0, MESH_W, iso_step):
            self.iso_idx.extend(col + MESH_W * np.arange(MESH_H))

        self.iso_pts = vtk.vtkPoints()
        self.iso_pts.SetData(vtknp.numpy_to_vtk(np.zeros((len(self.iso_idx), 3), 'f4')))

        lines = vtk.vtkCellArray(); off = 0
        for _ in range(0, MESH_H, iso_step):            # horizontal strips
            pl = vtk.vtkPolyLine(); pl.GetPointIds().SetNumberOfIds(MESH_W)
            for i in range(MESH_W):
                pl.GetPointIds().SetId(i, off + i)
            lines.InsertNextCell(pl); off += MESH_W
        off = row_break                                  # vertical strips
        for _ in range(0, MESH_W, iso_step):
            pl = vtk.vtkPolyLine(); pl.GetPointIds().SetNumberOfIds(MESH_H)
            for j in range(MESH_H):
                pl.GetPointIds().SetId(j, off + j)
            lines.InsertNextCell(pl); off += MESH_H

        iso_poly = vtk.vtkPolyData(); iso_poly.SetPoints(self.iso_pts); iso_poly.SetLines(lines)
        mapper   = vtk.vtkPolyDataMapper(); mapper.SetInputData(iso_poly)

        self.iso_actor = vtk.vtkActor(); self.iso_actor.SetMapper(mapper)
        prop = self.iso_actor.GetProperty()
        prop.SetColor(0,0,0); prop.SetLineWidth(1.0); prop.LightingOff()

    # ───────── VTK scene + window ────────────────────────────────────────
    def _build_scene(self):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.normals_flt.GetOutputPort())

        self.surface_actor = vtk.vtkActor(); self.surface_actor.SetMapper(mapper)
        self.texture       = vtk.vtkTexture(); self.texture.RepeatOn(); self.texture.InterpolateOn()
        self.surface_actor.SetTexture(self.texture)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.1, 0.15)
        self.renderer.AddActor(self.surface_actor)
        self.renderer.AddActor(self.iso_actor)

        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetViewAngle(ZOOM_ANGLE); self.camera.SetClippingRange(0.5, 20)

        self.window  = vtk.vtkRenderWindow(); self.window.AddRenderer(self.renderer)
        self.window.SetSize(1200, 900)

        self.interactor = vtk.vtkRenderWindowInteractor(); self.interactor.SetRenderWindow(self.window)
        style = vtk.vtkInteractorStyleTrackballCamera(); style.SetMotionFactor(2.0)
        self.interactor.SetInteractorStyle(style)

    # ───────── main timer callback ───────────────────────────────────────
    def _on_timer(self, *_):
        t_now = time.perf_counter() - self.t0
        dt    = t_now - self.prev_t
        if dt <= 0 and self.prev_xyz is not None:
            return

        # ---------- physics update at t_now ------------------------------
        self.morph.update(t_now)
        xyz_now = np.empty((self.st_grid.shape[0], 3), 'f4')
        nrm_now = np.empty_like(xyz_now)
        for p, st in enumerate(self.st_grid):
            xyz_now[p], nrm_now[p] = self.morph.evaluate(st)

        # first frame -----------------------------------------------------
        if self.prev_xyz is None:
            self._draw_frame(xyz_now, nrm_now, t_now, update_texture=True)
            self.prev_xyz, self.prev_t = xyz_now, t_now
            return

        # in-between interpolated frames ---------------------------------
        for i in range(1, SUB_FRAMES + 1):
            a = i / (SUB_FRAMES + 1)
            xyz_mid = self.prev_xyz + a * (xyz_now - self.prev_xyz)
            nrm_mid = nrm_now                  # normals good enough
            self._draw_frame(xyz_mid, nrm_mid,
                             self.prev_t + a * dt, update_texture=False)

        # final physics frame --------------------------------------------
        self._draw_frame(xyz_now, nrm_now, t_now, update_texture=True)
        self.prev_xyz, self.prev_t = xyz_now, t_now

    # ───────── push data to VTK and render ───────────────────────────────
    def _draw_frame(self, xyz, nrm, t, *, update_texture: bool):
        self.vtk_pts.SetData(vtknp.numpy_to_vtk(xyz))
        self.vtk_nrm.SetArray(nrm.ravel(), nrm.size, 1)
        self.vtk_pts.Modified(); self.vtk_nrm.Modified(); self.normals_flt.Update()

        self.iso_pts.SetData(vtknp.numpy_to_vtk(xyz[self.iso_idx]))
        self.iso_pts.Modified()

        eye_off = CAMERA_SCALE * self.pos_b.update(t)
        eye     = eye_off + np.array([0,0,DISTANCE], 'f4')
        up_vec  = self.rot_b.update(t)[:3,:3] @ np.array([0,1,0], 'f4')
        self.camera.SetPosition(*eye); self.camera.SetFocalPoint(0,0,0); self.camera.SetViewUp(*up_vec)

        if update_texture:
            rgb = self.layers.update(t)
            img = vtk.vtkImageData(); img.SetDimensions(TEX_W, TEX_H, 1)
            img.AllocateScalars(vtk.VTK_FLOAT, 3)
            img.GetPointData().SetScalars(vtknp.numpy_to_vtk(rgb[::-1].reshape(-1,3)))
            self.texture.SetInputData(img)

        self.window.Render()

    # ───────── entry point ───────────────────────────────────────────────
    def run(self):
        self.interactor.Initialize()
        self.interactor.AddObserver('TimerEvent', self._on_timer)
        self.interactor.CreateRepeatingTimer(PHYSICS_DT_MS)
        self._on_timer()          # immediate first frame
        self.interactor.Start()


# ------------------------------------------------------------------------
if __name__ == "__main__":
    MondriaanAnimation().run()
