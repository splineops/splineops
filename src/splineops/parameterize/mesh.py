# mesh.py

import numpy as np
from constants import kMESH_WIDTH, kMESH_HEIGHT
from morph import Morph


class Mesh:
    """
    Port of the Objective-C Mesh class. Holds a triangular mesh of size
    (kMESH_WIDTH × kMESH_HEIGHT) that is pulled from Morph.morphedCoordinateAt().
    """

    def __init__(self):
        # Precompute number of triangles/edges
        body_triangles = (kMESH_WIDTH - 2) * (kMESH_HEIGHT - 2) * 2
        rim_triangles  = 2 * (kMESH_WIDTH - 2) + 2 * (kMESH_HEIGHT - 2)
        self.numberOfEdges = 3 * (body_triangles + rim_triangles)

        # Create a 1‐D array for edges (indices) of length = self.numberOfEdges
        self.edges = np.zeros(self.numberOfEdges, dtype=np.uint32)

        # Total number of vertices = kMESH_WIDTH * kMESH_HEIGHT
        self.numberOfVertices = kMESH_WIDTH * kMESH_HEIGHT

        # For each vertex we store: 3D position (x,y,z), 3D normal, and 2D st‐coordinate
        self.vertex3DCoordinates = np.zeros((self.numberOfVertices, 3), dtype=np.float32)
        self.vertex3DNormals     = np.zeros((self.numberOfVertices, 3), dtype=np.float32)
        self.vertexStCoordinates = np.zeros((self.numberOfVertices, 2), dtype=np.float32)

        # Build a Morph object to compute initial positions/normals
        self.morph = Morph()

        # Fill all vertices with the coordinate at st=(0,0) as a baseline
        st0 = np.array([0.0, 0.0], dtype=np.float32)
        coord0 = self.morph.morphed_coordinate_at(st0)
        norm0  = self.morph.morphed_normal_at(st0)
        for p in range(self.numberOfVertices):
            self.vertex3DCoordinates[p, :] = coord0
            self.vertex3DNormals[p, :]     = norm0
            self.vertexStCoordinates[p, :] = st0

        # Now fill interior vertices (skip the top row, then for each row skip leftmost col)
        p = 0
        for k2 in range(kMESH_HEIGHT):
            if k2 == 0:
                # Entire top row is left at st0
                p += kMESH_WIDTH
                continue
            # Leftmost column of each subsequent row is st0
            self.vertex3DCoordinates[p, :] = coord0
            self.vertex3DNormals[p, :]     = norm0
            self.vertexStCoordinates[p, :] = st0
            p += 1
            for k1 in range(1, kMESH_WIDTH):
                st = np.array([float(k1), float(k2)], dtype=np.float32)
                self.vertex3DCoordinates[p, :] = self.morph.morphed_coordinate_at(st)
                self.vertex3DNormals[p, :]     = self.morph.morphed_normal_at(st)
                self.vertexStCoordinates[p, :] = st
                p += 1

        # Build the index buffer (triangles → edges)
        idx = 0
        p2 = kMESH_WIDTH
        for k2 in range(1, kMESH_HEIGHT - 1):
            for k1 in range(1, kMESH_WIDTH - 1):
                v0 = p2 + k1        # lower-left
                v1 = p2 + k1 + 1    # lower-right
                v2 = p2 + k1 + kMESH_WIDTH       # upper-left
                v3 = p2 + k1 + 1 + kMESH_WIDTH   # upper-right

                # Triangle 1: (v3, v1, v0)
                self.edges[idx    ] = v3
                self.edges[idx + 1] = v1
                self.edges[idx + 2] = v0
                idx += 3
                # Triangle 2: (v0, v2, v3)
                self.edges[idx    ] = v0
                self.edges[idx + 1] = v2
                self.edges[idx + 2] = v3
                idx += 3
            p2 += kMESH_WIDTH

        # Rim triangles along top and bottom edges
        p2 = kMESH_WIDTH * (kMESH_HEIGHT - 1)
        # Bottom row (excluding corners) attached to vertex 0
        for k1 in range(1, kMESH_WIDTH - 1):
            vA = k1 + kMESH_WIDTH
            vB = k1 + 1 + kMESH_WIDTH
            self.edges[idx    ] = vA
            self.edges[idx + 1] = vB
            self.edges[idx + 2] = 0
            idx += 3
        # Top row (excluding corners) attached to vertex 0
        for k1 in range(1, kMESH_WIDTH - 1):
            vA = p2 + k1 + 1
            vB = p2 + k1
            self.edges[idx    ] = vA
            self.edges[idx + 1] = vB
            self.edges[idx + 2] = 0
            idx += 3

    def update(self, elapsed: float) -> None:
        """
        Recompute the interior vertices (positions & normals) each frame,
        using Morph.morphed_coordinate_at(). The “st” array never changes,
        so we only update indices > kMESH_WIDTH (i.e. skip top row).
        """
        self.morph.update(elapsed)

        st0 = np.array([0.0, 0.0], dtype=np.float32)
        coord0 = self.morph.morphed_coordinate_at(st0)
        norm0  = self.morph.morphed_normal_at(st0)

        p = 0
        for k1 in range(kMESH_WIDTH):
            self.vertex3DCoordinates[p, :] = coord0
            self.vertex3DNormals[p, :]     = norm0
            p += 1

        for k2 in range(1, kMESH_HEIGHT):
            self.vertex3DCoordinates[p, :] = coord0
            self.vertex3DNormals[p, :]     = norm0
            p += 1
            for k1 in range(1, kMESH_WIDTH):
                st = np.array([float(k1), float(k2)], dtype=np.float32)
                self.vertex3DCoordinates[p, :] = self.morph.morphed_coordinate_at(st)
                self.vertex3DNormals[p, :]     = self.morph.morphed_normal_at(st)
                p += 1
