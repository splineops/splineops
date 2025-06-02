"""
mondriaan_navigation.py  –  explore one frozen Mondriaan ‘breathing’ frame
===========================================================================

• SNAP_T picks the animation time-stamp (seconds).
• Uses the Mesh class (faithful port of Mesh.m) – the pole is now sealed.
• Camera starts at (0,0,DISTANCE) and rotates a little so you see the relief.
"""

import numpy as np
import vtk
from  vtk.util import numpy_support as vtknp

from constants        import *          # DISTANCE, MESH_W, MESH_H, …
from morph            import Morph
from mesh             import Mesh       # ← NEW: use the watertight mesh
from mondriaan_layers import MondriaanLayers

# ───────────── user-tweakable constants ─────────────────────────────────
SNAP_T        = 0.0        # seconds into the timeline (0 ⇒ no breathing)
ISO_STEP      = 16         # isolines every N mesh samples

BREATH_FREQ   = 0.00       # Hz – set >0 for animation
BREATH_AMP    = 0.00       # amplitude of bulge

CAMERA_AZIM   = 25         # deg
CAMERA_ELEV   = 12         # deg
# ------------------------------------------------------------------------


# ─────────────────────── VTK helpers ────────────────────────────────────
def mesh_to_vtk(mesh: Mesh):
    """Convert Mesh → vtkPolyData; return polydata and numpy xyz array."""
    # -- points ---------------------------------------------------------
    pts = vtk.vtkPoints()
    pts.SetData(vtknp.numpy_to_vtk(mesh.vertex3DCoordinates))

    # -- normals --------------------------------------------------------
    vtk_nrm = vtknp.numpy_to_vtk(mesh.vertex3DNormals.ravel(), deep=1)
    vtk_nrm.SetNumberOfComponents(3)

    # -- triangles ------------------------------------------------------
    tris = vtk.vtkCellArray()
    for i in range(0, mesh.numberOfEdges, 3):
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(mesh.edges[i    ]))
        tri.GetPointIds().SetId(1, int(mesh.edges[i + 1]))
        tri.GetPointIds().SetId(2, int(mesh.edges[i + 2]))
        tris.InsertNextCell(tri)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.GetPointData().SetNormals(vtk_nrm)
    poly.SetPolys(tris)
    return poly, mesh.vertex3DCoordinates

def build_isolines(xyz: np.ndarray):
    """
    Draw latitude / longitude grid lines.
    All stripes start at vertex-0 (the single south-pole vertex) so every
    pole-to-stripe segment is rendered exactly once.
    """
    iso_pts = vtk.vtkPoints()
    lines   = vtk.vtkCellArray()

    # we will push points stripe-by-stripe and remember their ids
    def add_polyline(pts_idx):
        """Helper: add one vtkPolyLine from a list of vertex indices."""
        first_id = iso_pts.InsertNextPoint(xyz[pts_idx[0]])
        ids = [first_id]
        for vidx in pts_idx[1:]:
            ids.append(iso_pts.InsertNextPoint(xyz[vidx]))
        pl = vtk.vtkPolyLine(); pl.GetPointIds().SetNumberOfIds(len(ids))
        for i, gid in enumerate(ids):
            pl.GetPointIds().SetId(i, gid)
        lines.InsertNextCell(pl)

    # ── horizontal stripes (row = k·ISO_STEP) ──────────────────────────
    for row in range(ISO_STEP, MESH_H, ISO_STEP):
        base = row * MESH_W
        stripe = [0]                                 # pole vertex once
        stripe.extend(base + np.arange(1, MESH_W))   # cols 1…127
        add_polyline(stripe)

    # ── vertical stripes (col = k·ISO_STEP) ────────────────────────────
    for col in range(ISO_STEP, MESH_W, ISO_STEP):
        stripe = [0]                                 # pole vertex once
        stripe.extend(col + MESH_W * np.arange(1, MESH_H))  # rows 1…127
        add_polyline(stripe)

    # ---- build actor ---------------------------------------------------
    iso_poly = vtk.vtkPolyData(); iso_poly.SetPoints(iso_pts); iso_poly.SetLines(lines)
    mapper   = vtk.vtkPolyDataMapper(); mapper.SetInputData(iso_poly)
    actor    = vtk.vtkActor(); actor.SetMapper(mapper)
    prop = actor.GetProperty(); prop.SetColor(0, 0, 0); prop.SetLineWidth(1.0); prop.LightingOff()
    return actor

def build_texture(layers: MondriaanLayers, t: float):
    rgb = layers.update(t)
    img = vtk.vtkImageData(); img.SetDimensions(TEX_W, TEX_H, 1)
    img.AllocateScalars(vtk.VTK_FLOAT, 3)
    img.GetPointData().SetScalars(vtknp.numpy_to_vtk(rgb[::-1].reshape(-1, 3)))
    tex = vtk.vtkTexture(); tex.RepeatOn(); tex.InterpolateOn(); tex.SetInputData(img)
    return tex


# ────────────────────────── main routine ────────────────────────────────
def main():
    # -- build breathing morph shared by Mesh & navigation label ----------
    morph = Morph(freq_hz=BREATH_FREQ, amp=BREATH_AMP)

    mesh = Mesh()
    mesh.morph = morph            # share the same Morph instance
    mesh.update(SNAP_T)           # freeze at desired timestamp

    poly, xyz = mesh_to_vtk(mesh)

    tex  = build_texture(MondriaanLayers(), SNAP_T)
    surf = vtk.vtkActor(); surf.SetMapper(vtk.vtkPolyDataMapper())
    surf.GetMapper().SetInputData(poly); surf.SetTexture(tex)

    iso_actor = build_isolines(xyz)

    # -- renderer & camera -----------------------------------------------
    ren = vtk.vtkRenderer(); ren.SetBackground(0.05, 0.1, 0.15)
    ren.AddActor(surf); ren.AddActor(iso_actor)

    cam = ren.GetActiveCamera()
    cam.SetPosition(0, 0, DISTANCE)
    cam.SetFocalPoint(0, 0, 0)
    cam.SetViewUp(0, 1, 0)
    cam.Azimuth(CAMERA_AZIM)
    cam.Elevation(CAMERA_ELEV)
    ren.ResetCameraClippingRange()

    # -- window / interaction --------------------------------------------
    win = vtk.vtkRenderWindow(); win.AddRenderer(ren); win.SetSize(1200, 900)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(win)
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetMotionFactor(2.0); iren.SetInteractorStyle(style)

    txt = vtk.vtkTextActor()
    txt.SetInput(f"Frozen at t = {SNAP_T:.2f} s")
    txt.GetTextProperty().SetFontSize(18); txt.GetTextProperty().SetColor(1, 1, 1)
    txt.SetDisplayPosition(20, 20); ren.AddActor2D(txt)

    win.Render(); iren.Initialize(); iren.Start()


# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()
