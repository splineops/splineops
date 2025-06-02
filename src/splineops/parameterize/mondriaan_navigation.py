"""
mondriaan_navigation.py  –  inspect one frozen Mondriaan ‘breathing’ frame
===========================================================================

• Choose SNAP_T below (seconds in the animation timeline).
• Builds the analytic monopole, applies the breathing bulge **once**,
  colours it with the Mondriaan texture, overlays isolines.
• Opens a VTK window with full 3-button interactor (orbit, pan, zoom).
"""

import numpy as np, vtk
from vtk.util import numpy_support as vtknp
import time

from constants        import *
from morph            import Morph
from mondriaan_layers import MondriaanLayers

# ------------------------------------------------------------------- user
SNAP_T        = 7.0          # seconds into the animation to “freeze”
ISO_STEP      = 16           # isolines every N mesh samples

# breathing / camera parameters must match those used in Morph()
BREATH_FREQ   = 0.04         # Hz  (same as animated script)
BREATH_AMP    = 0.12         # amplitude (radius units)

CAMERA_AZIM   = 25           # initial azimuth   (deg)
CAMERA_ELEV   = 12           # initial elevation (deg)
# -------------------------------------------------------------------------

def build_mesh(morph: Morph):
    """Return vtkPolyData with positions+normals at current morph state."""
    s = np.linspace(0, 1, MESH_W, 'f4')
    t = np.linspace(0, 1, MESH_H, 'f4')
    st_grid = np.dstack(np.meshgrid(s, t, indexing='xy')).reshape(-1, 2)

    pts  = np.empty((st_grid.shape[0], 3), 'f4')
    nrm  = np.empty_like(pts)
    for p, st in enumerate(st_grid):
        pts[p], nrm[p] = morph.evaluate(st)

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(vtknp.numpy_to_vtk(pts))
    vtk_nrm = vtknp.numpy_to_vtk(nrm.ravel(), deep=1)
    vtk_nrm.SetNumberOfComponents(3)

    cells = vtk.vtkCellArray()
    for j in range(MESH_H - 1):
        for i in range(MESH_W - 1):
            a = j * MESH_W + i
            b = a + 1
            c = a + MESH_W
            d = c + 1
            for tri in ((a, c, b), (b, c, d)):
                tcell = vtk.vtkTriangle()
                for k, pid in enumerate(tri):
                    tcell.GetPointIds().SetId(k, pid)
                cells.InsertNextCell(tcell)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.GetPointData().SetNormals(vtk_nrm)
    poly.SetPolys(cells)
    return poly, pts   # pts reused for isolines


def build_isolines(pts_xyz: np.ndarray):
    iso_map = []
    for row in range(0, MESH_H, ISO_STEP):
        for col in range(MESH_W):
            iso_map.append(row * MESH_W + col)
    row_break = len(iso_map)
    for col_bundle in range(0, MESH_W, ISO_STEP):
        for row in range(MESH_H):
            iso_map.append(row * MESH_W + col_bundle)

    iso_pts = vtk.vtkPoints()
    iso_pts.SetData(vtknp.numpy_to_vtk(pts_xyz[iso_map]))

    lines = vtk.vtkCellArray(); off = 0
    for _ in range(0, MESH_H, ISO_STEP):      # horizontal
        poly = vtk.vtkPolyLine(); poly.GetPointIds().SetNumberOfIds(MESH_W)
        for i in range(MESH_W):
            poly.GetPointIds().SetId(i, off + i)
        lines.InsertNextCell(poly); off += MESH_W
    off = row_break                             # vertical
    for _ in range(0, MESH_W, ISO_STEP):
        poly = vtk.vtkPolyLine(); poly.GetPointIds().SetNumberOfIds(MESH_H)
        for j in range(MESH_H):
            poly.GetPointIds().SetId(j, off + j)
        lines.InsertNextCell(poly); off += MESH_H

    iso_poly = vtk.vtkPolyData(); iso_poly.SetPoints(iso_pts); iso_poly.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(iso_poly)
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    prop = actor.GetProperty(); prop.SetColor(0,0,0); prop.SetLineWidth(1.0); prop.LightingOff()
    return actor


def build_texture(layers: MondriaanLayers, t: float):
    rgb = layers.update(t)
    img = vtk.vtkImageData(); img.SetDimensions(TEX_W, TEX_H, 1)
    img.AllocateScalars(vtk.VTK_FLOAT, 3)
    img.GetPointData().SetScalars(vtknp.numpy_to_vtk(rgb[::-1].reshape(-1,3)))
    tex = vtk.vtkTexture(); tex.InterpolateOn(); tex.RepeatOn(); tex.SetInputData(img)
    return tex


def main():
    morph  = Morph(freq_hz=BREATH_FREQ, amp=BREATH_AMP)
    morph.update(SNAP_T)                       # lock breathing phase
    poly, pts = build_mesh(morph)

    layers = MondriaanLayers()
    tex    = build_texture(layers, SNAP_T)

    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
    actor  = vtk.vtkActor(); actor.SetMapper(mapper); actor.SetTexture(tex)
    iso_actor = build_isolines(pts)

    ren = vtk.vtkRenderer(); ren.SetBackground(0.05,0.1,0.15)
    ren.AddActor(actor); ren.AddActor(iso_actor)

    cam = ren.GetActiveCamera()
    cam.Elevation(CAMERA_ELEV); cam.Azimuth(CAMERA_AZIM)
    cam.SetViewUp(0,1,0); cam.Dolly(1.1)
    ren.ResetCameraClippingRange()

    win = vtk.vtkRenderWindow(); win.AddRenderer(ren); win.SetSize(1200, 900)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)

    txt = vtk.vtkTextActor()
    txt.SetInput(f"Frozen at t = {SNAP_T:.2f} s")
    txtprop = txt.GetTextProperty(); txtprop.SetFontSize(20); txtprop.SetColor(1,1,1)
    txt.SetDisplayPosition(20,20); ren.AddActor2D(txt)

    win.Render(); iren.Initialize(); iren.Start()


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
