#!/usr/bin/env python
# pip install vtk PySide6 numpy
import sys, numpy as np, vtk
from PySide6 import QtWidgets, QtCore
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

        self.renderer = vtk.vtkRenderer()
        self.vtk.GetRenderWindow().AddRenderer(self.renderer)

        # ---- simple cube mesh
        verts = np.array(
            [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
             [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)

        faces = np.array(
            [[0,1,2,3],[4,5,6,7],[0,1,5,4],
             [1,2,6,5],[2,3,7,6],[3,0,4,7]], int)

        self.poly   = numpy_mesh_to_polydata(verts, faces)
        self.points = self.poly.GetPoints()          # mutable

        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(self.poly)
        actor  = vtk.vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn(); actor.GetProperty().SetOpacity(0.8)

        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()

        # --- picker & mouse observer ------------------------------------------
        self.picker = vtk.vtkPointPicker(); self.picker.SetTolerance(0.02)

        iren = self.vtk.GetRenderWindow().GetInteractor()

        try:
            style = vtk.vtkInteractorStyleTerrain()
            # only call if it actually exists
            if hasattr(style, "SetMotionFactor"):
                style.SetMotionFactor(0.3)
        except AttributeError:
            # terrain not available / missing methods → use trackball instead
            style = vtk.vtkInteractorStyleTrackballCamera()
            style.SetMotionFactor(0.25)

        iren.SetInteractorStyle(style)

        # =======================================================================

        iren.AddObserver("LeftButtonPressEvent", self.on_left_press, 1.0)

        self.handle_widgets = {}   # pid -> vtkHandleWidget

        # ---- start interactor
        self.vtk.Initialize()
        self.show()

    # -----------------------------------------------------------------
    def on_left_press(self, obj, evt):
        x, y = obj.GetEventPosition()
        if not self.picker.Pick(x, y, 0, self.renderer):
            return
        pid = self.picker.GetPointId()
        if pid >= 0 and pid not in self.handle_widgets:
            self._spawn_handle(pid)

    # -----------------------------------------------------------------
    def _spawn_handle(self, pid: int):
        rep = vtk.vtkPointHandleRepresentation3D()
        rep.SetWorldPosition(self.points.GetPoint(pid))
        rep.SetRenderer(self.renderer)
        rep.GetProperty().SetColor(1, .55, .2)

        widget = vtk.vtkHandleWidget()
        widget.SetInteractor(self.vtk.GetRenderWindow().GetInteractor())  # <- FIX
        widget.SetRepresentation(rep)
        widget.EnabledOn()

        def drag_cb(_, __):
            self.points.SetPoint(pid, rep.GetWorldPosition())
            self.points.Modified()
            self.vtk.GetRenderWindow().Render()

        widget.AddObserver("InteractionEvent", drag_cb)
        self.handle_widgets[pid] = widget


# ---------- Run the application ---------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MeshEditor()
    sys.exit(app.exec())
