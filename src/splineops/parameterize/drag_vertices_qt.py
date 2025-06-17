#!/usr/bin/env python
# pip install vtk PySide6 numpy
import sys, numpy as np, vtk
from PySide6 import QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support as nps


# ---------- NumPy → vtkPolyData helper ---------------------------------
def numpy_mesh_to_polydata(vertices: np.ndarray, faces: np.ndarray):
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(vertices, deep=True))

    prefix = np.full((faces.shape[0], 1), faces.shape[1], int)
    flat   = np.hstack([prefix, faces]).ravel()

    cells = vtk.vtkCellArray()
    cells.SetCells(faces.shape[0],
                   nps.numpy_to_vtkIdTypeArray(flat, deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    return poly


# ---------- Main Qt window with an embedded VTK view -------------------
class MeshEditor(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SplineOps – vertex drag demo")
        self.resize(1000, 700)

        # ---- VTK widget
        self.vtk = QVTKRenderWindowInteractor(self)
        self.setCentralWidget(self.vtk)

        self.ren = vtk.vtkRenderer()
        self.vtk.GetRenderWindow().AddRenderer(self.ren)

        # ---- simple cube mesh
        verts = np.array(
            [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
             [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)

        faces = np.array(
            [[0,1,2,3],[4,5,6,7],[0,1,5,4],
             [1,2,6,5],[2,3,7,6],[3,0,4,7]], int)

        self.poly   = numpy_mesh_to_polydata(verts, faces)
        self.points = self.poly.GetPoints()          # mutable reference

        # ---- surface actor (semi-transparent, edges on)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.poly)
        actor  = vtk.vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn(); actor.GetProperty().SetOpacity(0.8)
        self.ren.AddActor(actor)

        # ------------------------------------------------------------------
        # **** OPTION 1 – BIG ORANGE POINT-SPRITES ON EVERY VERTEX ****
        #
        # 1. Convert each point into an explicit vertex cell
        v_filter = vtk.vtkVertexGlyphFilter()
        v_filter.SetInputData(self.poly)
        v_filter.Update()                           # run once now
        self.v_filter = v_filter                    # keep a handle for updates

        # 2. Map those vertices with large, sphere-shaded points
        v_mapper = vtk.vtkPolyDataMapper()
        v_mapper.SetInputConnection(v_filter.GetOutputPort())

        v_actor = vtk.vtkActor()
        v_actor.SetMapper(v_mapper)
        v_prop = v_actor.GetProperty()
        v_prop.SetColor(1, 0.4, 0)                  # orange
        v_prop.SetPointSize(14)                     # pixel radius
        v_prop.SetRenderPointsAsSpheres(1)          # nice round sprite

        self.ren.AddActor(v_actor)
        # ------------------------------------------------------------------

        self.ren.ResetCamera()

        # --- picker & interaction style ----------------------------------
        self.picker = vtk.vtkPointPicker(); self.picker.SetTolerance(0.02)
        iren = self.vtk.GetRenderWindow().GetInteractor()

        try:
            style = vtk.vtkInteractorStyleTerrain()
            if hasattr(style, "SetMotionFactor"):
                style.SetMotionFactor(0.3)
        except AttributeError:
            style = vtk.vtkInteractorStyleTrackballCamera()
            style.SetMotionFactor(0.25)
        iren.SetInteractorStyle(style)

        iren.SetPicker(self.picker)
        iren.AddObserver("LeftButtonPressEvent", self.on_left_press, 1.0)

        self.handle_widgets = {}   # pid -> vtkHandleWidget

        # ---- start interactor
        self.vtk.Initialize()
        self.show()

    # -----------------------------------------------------------------
    def on_left_press(self, iren, _evt):
        x, y = iren.GetEventPosition()
        if not self.picker.Pick(x, y, 0, self.ren):
            return
        pid = self.picker.GetPointId()
        if pid >= 0 and pid not in self.handle_widgets:
            self._spawn_handle(pid)

    # -----------------------------------------------------------------
    def _spawn_handle(self, pid: int):
        rep = vtk.vtkPointHandleRepresentation3D()
        rep.SetWorldPosition(self.points.GetPoint(pid))
        rep.SetRenderer(self.ren)
        rep.GetProperty().SetColor(1, .55, .2)

        widget = vtk.vtkHandleWidget()
        widget.SetInteractor(self.vtk.GetRenderWindow().GetInteractor())
        widget.SetRepresentation(rep)
        widget.EnabledOn()

        def drag_cb(*_):
            # update the moved vertex in place
            self.points.SetPoint(pid, rep.GetWorldPosition())
            self.points.Modified()
            # refresh glyph filter so sprites follow
            self.v_filter.Modified()
            self.v_filter.Update()
            self.vtk.GetRenderWindow().Render()

        widget.AddObserver("InteractionEvent", drag_cb)
        self.handle_widgets[pid] = widget


# ---------- Run the application ---------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MeshEditor()
    sys.exit(app.exec())
