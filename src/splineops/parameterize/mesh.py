# mesh.py  –  watertight triangular grid for the “monopole” surface
# -----------------------------------------------------------------
import numpy as np
from constants import MESH_W, MESH_H
from morph     import Morph


class Mesh:
    """
    Holds a (MESH_W × MESH_H) vertex grid whose *entire boundary* collapses
    to a single south-pole vertex (index 0), exactly like the Objective-C
    screensaver.

    Attributes exposed for the VTK demo:
        • numberOfVertices          int
        • vertex3DCoordinates       ndarray (N,3) float32
        • vertex3DNormals           ndarray (N,3) float32
        • vertexStCoordinates       ndarray (N,2) float32   (cached (s,t))
        • numberOfEdges             int
        • edges                     ndarray (3×T) uint32    index buffer
    """

    # ───────────────────────── constructor ──────────────────────────────
    def __init__(self):
        # ---- vertex arrays --------------------------------------------
        self.numberOfVertices     = MESH_W * MESH_H
        self.vertex3DCoordinates  = np.zeros((self.numberOfVertices, 3),
                                             dtype=np.float32)
        self.vertex3DNormals      = np.zeros_like(self.vertex3DCoordinates)
        self.vertexStCoordinates  = np.zeros((self.numberOfVertices, 2),
                                             dtype=np.float32)

        # ---- element (index) buffer -----------------------------------
        body_tris = (MESH_W - 2) * (MESH_H - 2) * 2
        rim_tris  = 2 * (MESH_W - 2) + 2 * (MESH_H - 2)
        self.numberOfEdges = 3 * (body_tris + rim_tris)

        edges: list[int] = []                      # build → convert once

        # ── 1. interior -------------------------------------------------
        for k2 in range(1, MESH_H - 1):
            base = k2 * MESH_W
            for k1 in range(1, MESH_W - 1):
                ll = base + k1                     # lower-left
                lr = ll + 1                        # lower-right
                ul = ll + MESH_W                   # upper-left
                ur = ul + 1                        # upper-right
                edges.extend((ur, lr, ll))         # (ur,lr,ll)
                edges.extend((ll, ul, ur))         # (ll,ul,ur)

        # ── 2. bottom & top rows glued to vertex 0 ----------------------
        top_row = (MESH_H - 1) * MESH_W
        for k1 in range(1, MESH_W - 1):
            edges.extend((k1 + MESH_W,        k1 + 1 + MESH_W, 0))   # bottom
            edges.extend((top_row + k1 + 1,   top_row + k1,    0))   # top

        # ── 3. left & right columns glued to vertex 0 -------------------
        p_left  = 1 + MESH_W                      # first interior row, col 0
        p_right = 2 * MESH_W - 1                  # same row, last column
        for _ in range(1, MESH_H - 1):
            edges.extend((p_left + MESH_W,  p_left,        0))        # left
            edges.extend((p_right,         p_right + MESH_W, 0))      # right
            p_left  += MESH_W
            p_right += MESH_W

        # ---- freeze as ndarray & sanity-check --------------------------
        self.edges = np.asarray(edges, dtype=np.uint32)
        assert len(self.edges) == self.numberOfEdges, "edge count mismatch"

        # ---- cache (s,t) coordinates per vertex ------------------------
        p = 0
        for k2 in range(MESH_H):
            for k1 in range(MESH_W):
                if k1 == 0 or k2 == 0:                 # boundary → pole
                    self.vertexStCoordinates[p] = (0.0, 0.0)
                else:
                    self.vertexStCoordinates[p] = (float(k1), float(k2))
                p += 1

        # ---- initial geometry -----------------------------------------
        self.morph = Morph()
        self.update(0.0)               # fill coords & normals for t=0

    # ─────────────────────── per-frame refresh ─────────────────────────
    def update(self, elapsed: float) -> None:
        """
        Recompute vertex positions & normals for animation time *elapsed*.
        """
        self.morph.update(elapsed)

        for p, st in enumerate(self.vertexStCoordinates):
            xyz, nrm = self.morph.evaluate(st)
            self.vertex3DCoordinates[p] = xyz
            self.vertex3DNormals[p]     = nrm
