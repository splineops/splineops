"""
mondriaan_navigation.py  –  explore one frozen Mondriaan ‘breathing’ frame
===========================================================================

• SNAP_T picks the animation time-stamp (seconds).
• Builds the analytic monopole, applies breathing bulge once, colours it,
  overlays isolines.
• Camera is positioned at (0,0,DISTANCE) then rotated: you start outside.
• Trackball MotionFactor = 2 gives tight, inertia-free controls.
"""

import numpy as np, vtk
from  vtk.util import numpy_support as vtknp

from constants        import *               # DISTANCE, mesh sizes …
from morph            import Morph
from mondriaan_layers import MondriaanLayers

# ───────────── user-tweakable constants ──────────────────────────────────
SNAP_T        =  7.0     # seconds into the timeline
ISO_STEP      = 16       # isolines every N mesh samples

BREATH_FREQ   = 0.04     # Hz (matches animated script)
BREATH_AMP    = 0.12

CAMERA_AZIM   = 25       # deg
CAMERA_ELEV   = 12       # deg
# ------------------------------------------------------------------------

# ──────────────────────── geometry helpers ───────────────────────────────
def build_mesh(morph: Morph):
    s = np.linspace(0, 1, MESH_W, 'f4')
    t = np.linspace(0, 1, MESH_H, 'f4')
    st_grid = np.dstack(np.meshgrid(s, t, indexing='xy')).reshape(-1, 2)

    xyz  = np.empty((st_grid.shape[0], 3), 'f4')
    nrm  = np.empty_like(xyz)
    for p, st in enumerate(st_grid):
        xyz[p], nrm[p] = morph.evaluate(st)

    pts = vtk.vtkPoints(); pts.SetData(vtknp.numpy_to_vtk(xyz))
    vtk_nrm = vtknp.numpy_to_vtk(nrm.ravel(), deep=1); vtk_nrm.SetNumberOfComponents(3)

    cells = vtk.vtkCellArray()
    for j in range(MESH_H - 1):
        for i in range(MESH_W - 1):
            a = j * MESH_W + i; b = a + 1; c = a + MESH_W; d = c + 1
            for tri in ((a, c, b), (b, c, d)):
                tc = vtk.vtkTriangle()
                for k, pid in enumerate(tri):
                    tc.GetPointIds().SetId(k, pid)
                cells.InsertNextCell(tc)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().SetNormals(vtk_nrm)
    poly.SetPolys(cells)
    return poly, xyz


def build_isolines(xyz: np.ndarray):
    idx = []
    for row in range(0, MESH_H, ISO_STEP):
        idx.extend(row * MESH_W + np.arange(MESH_W))
    row_break = len(idx)
    for col in range(0, MESH_W, ISO_STEP):
        idx.extend(col + MESH_W * np.arange(MESH_H))

    iso_pts = vtk.vtkPoints(); iso_pts.SetData(vtknp.numpy_to_vtk(xyz[idx]))

    lines = vtk.vtkCellArray(); off = 0
    for _ in range(0, MESH_H, ISO_STEP):        # horizontal
        pl = vtk.vtkPolyLine(); pl.GetPointIds().SetNumberOfIds(MESH_W)
        for i in range(MESH_W):
            pl.GetPointIds().SetId(i, off + i)
        lines.InsertNextCell(pl); off += MESH_W
    off = row_break                              # vertical
    for _ in range(0, MESH_W, ISO_STEP):
        pl = vtk.vtkPolyLine(); pl.GetPointIds().SetNumberOfIds(MESH_H)
        for j in range(MESH_H):
            pl.GetPointIds().SetId(j, off + j)
        lines.InsertNextCell(pl); off += MESH_H

    iso_poly = vtk.vtkPolyData(); iso_poly.SetPoints(iso_pts); iso_poly.SetLines(lines)
    mapper   = vtk.vtkPolyDataMapper(); mapper.SetInputData(iso_poly)
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    prop = actor.GetProperty(); prop.SetColor(0,0,0); prop.SetLineWidth(1.0); prop.LightingOff()
    return actor


def build_texture(layers: MondriaanLayers, t: float):
    rgb = layers.update(t)
    img = vtk.vtkImageData(); img.SetDimensions(TEX_W, TEX_H, 1)
    img.AllocateScalars(vtk.VTK_FLOAT, 3)
    img.GetPointData().SetScalars(vtknp.numpy_to_vtk(rgb[::-1].reshape(-1,3)))
    tex = vtk.vtkTexture(); tex.RepeatOn(); tex.InterpolateOn(); tex.SetInputData(img)
    return tex


# ────────────────────────── main routine ────────────────────────────────
def main():
    morph = Morph(freq_hz=BREATH_FREQ, amp=BREATH_AMP)
    morph.update(SNAP_T)
    poly, xyz = build_mesh(morph)

    tex  = build_texture(MondriaanLayers(), SNAP_T)
    surf = vtk.vtkActor(); surf.SetMapper(vtk.vtkPolyDataMapper())
    surf.GetMapper().SetInputData(poly); surf.SetTexture(tex)
    iso_actor = build_isolines(xyz)

    ren = vtk.vtkRenderer(); ren.SetBackground(0.05,0.1,0.15)
    ren.AddActor(surf); ren.AddActor(iso_actor)

    cam = ren.GetActiveCamera()
    cam.SetPosition(0, 0, DISTANCE)   # start outside
    cam.SetFocalPoint(0, 0, 0)
    cam.SetViewUp(0, 1, 0)
    cam.Azimuth(CAMERA_AZIM)
    cam.Elevation(CAMERA_ELEV)
    ren.ResetCameraClippingRange()

    win = vtk.vtkRenderWindow(); win.AddRenderer(ren); win.SetSize(1200,900)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetMotionFactor(2.0)
    iren.SetInteractorStyle(style)

    txt = vtk.vtkTextActor()
    txt.SetInput(f"Frozen at t = {SNAP_T:.2f} s")
    txt.GetTextProperty().SetFontSize(18); txt.GetTextProperty().SetColor(1,1,1)
    txt.SetDisplayPosition(20,20); ren.AddActor2D(txt)

    win.Render(); iren.Initialize(); iren.Start()

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
