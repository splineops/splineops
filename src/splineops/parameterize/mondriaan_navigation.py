"""
mondriaan_navigation.py  –  explore one frozen Mondriaan ‘breathing’ frame
===========================================================================

• Choose SNAP_T to lock any animation time-stamp (seconds).
• Builds the analytic monopole, applies the breathing bulge once,
  colours it with the Mondriaan texture, overlays u-/v-isolines.
• Uses a custom Trackball style with MotionFactor=2 → crisp, no inertia.
"""

import numpy as np, vtk
from  vtk.util import numpy_support as vtknp

from constants        import *
from morph            import Morph
from mondriaan_layers import MondriaanLayers

# ───────────── user-tweakable constants ──────────────────────────────────
SNAP_T        =  7.0     # seconds into the (hypothetical) animation
ISO_STEP      = 16       # isolines every N mesh samples

# breathing parameters (must match those used in Morph)
BREATH_FREQ   = 0.04     # Hz   (25-s cycle)
BREATH_AMP    = 0.12     # bulge amplitude (sphere-radius units)

# initial camera setting
CAMERA_AZIM   = 25       # deg
CAMERA_ELEV   = 12       # deg
# ------------------------------------------------------------------------

# ──────────────────────── geometry helpers ───────────────────────────────
def build_mesh(morph: Morph):
    """Return (vtkPolyData, ndarray xyz) for current morph state."""
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

    poly = vtk.vtkPolyData(); poly.SetPoints(pts)
    poly.GetPointData().SetNormals(vtk_nrm); poly.SetPolys(cells)
    return poly, xyz


def build_isolines(xyz: np.ndarray):
    """Return a black-line actor that re-uses the xyz array."""
    iso_idx = []
    for row in range(0, MESH_H, ISO_STEP):
        for col in range(MESH_W):
            iso_idx.append(row * MESH_W + col)
    row_break = len(iso_idx)
    for col_bundle in range(0, MESH_W, ISO_STEP):
        for row in range(MESH_H):
            iso_idx.append(row * MESH_W + col_bundle)

    iso_pts = vtk.vtkPoints(); iso_pts.SetData(vtknp.numpy_to_vtk(xyz[iso_idx]))

    lines = vtk.vtkCellArray(); off = 0
    for _ in range(0, MESH_H, ISO_STEP):                 # horizontal strips
        pl = vtk.vtkPolyLine(); pl.GetPointIds().SetNumberOfIds(MESH_W)
        for i in range(MESH_W):
            pl.GetPointIds().SetId(i, off + i)
        lines.InsertNextCell(pl); off += MESH_W
    off = row_break
    for _ in range(0, MESH_W, ISO_STEP):                 # vertical strips
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
    # freeze morph at desired time
    morph = Morph(freq_hz=BREATH_FREQ, amp=BREATH_AMP)
    morph.update(SNAP_T)
    poly, xyz = build_mesh(morph)

    layers = MondriaanLayers()
    tex    = build_texture(layers, SNAP_T)

    # surface actor
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
    surf = vtk.vtkActor(); surf.SetMapper(mapper); surf.SetTexture(tex)

    # isolines
    iso_actor = build_isolines(xyz)

    # renderer & scene
    ren = vtk.vtkRenderer(); ren.SetBackground(0.05,0.1,0.15)
    ren.AddActor(surf); ren.AddActor(iso_actor)

    cam = ren.GetActiveCamera()
    cam.Elevation(CAMERA_ELEV); cam.Azimuth(CAMERA_AZIM)
    cam.SetViewUp(0,1,0); cam.Dolly(1.1)
    ren.ResetCameraClippingRange()

    # window
    win = vtk.vtkRenderWindow(); win.AddRenderer(ren); win.SetSize(1200,900)

    # interactor with tight trackball
    iren  = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetMotionFactor(2.0)        # ↓ lower = slower = no overshoot
    iren.SetInteractorStyle(style)

    # caption
    txt = vtk.vtkTextActor(); txt.SetInput(f"Frozen at t = {SNAP_T:.2f} s")
    tprop = txt.GetTextProperty(); tprop.SetFontSize(18); tprop.SetColor(1,1,1)
    txt.SetDisplayPosition(20,20); ren.AddActor2D(txt)

    win.Render(); iren.Initialize(); iren.Start()

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
