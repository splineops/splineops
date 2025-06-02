import numpy as np

# ─── mesh resolution ──────────────────────────────────────────────────────
MESH_W = 128
MESH_H = 128

# ─── Mondriaan texture ─────────────────────────────────────────────────────
N_LAYERS = 40           # number of colour layers
TEX_W    = 128          # width   of RGB texture
TEX_H    = 128          # height  of RGB texture
LUT_LEN  = 1024         # colour look-up table length

# ─── camera ────────────────────────────────────────────────────────────────
DISTANCE   = 4.5        # distance from surface
ZOOM_ANGLE = 45.0       # field of view (deg)

# ─── misc ──────────────────────────────────────────────────────────────────
PI2 = 2.0 * np.pi
