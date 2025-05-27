# monopole.py ---------------------------------------------------------------
import numpy as np

def vertices_and_normals(M=128, N=128, eps=1.0e-4):
    """
    Sample Γ_U(u,v) on an (M+1)×(N+1) grid and return
    flat arrays: positions (x,y,z) and normals (nx,ny,nz).
    """
    u = np.linspace(eps, 1-eps, M+1, dtype=np.float32)
    v = np.linspace(eps, 1-eps, N+1, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing='ij')        # shape (M+1,N+1)

    su  = np.sin(np.pi * uu)
    sv  = np.sin(np.pi * vv)
    s2u = su * su

    x = -np.sin(2*np.pi*uu) * sv
    y = -s2u * np.sin(2*np.pi*vv)
    z = -(1.0 - 2.0 * s2u * sv*sv)

    # build flat positions
    pos = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # approximate normals by centred finite differences on the grid,
    # then normalise
    # (good enough for a first demo; later we'll use the analytic ∂Γ/∂u × ∂Γ/∂v)
    dzdu = np.gradient(pos.reshape(M+1, N+1, 3), axis=0)
    dzdv = np.gradient(pos.reshape(M+1, N+1, 3), axis=1)
    n    = np.cross(dzdu, dzdv, axis=-1)
    n   /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    nor  = n.reshape(-1, 3).astype(np.float32)

    return pos.astype(np.float32), nor

def indices(M=128, N=128):
    """
    Return a uint32 element array with 2 triangles per grid quad.
    """
    idx = []
    for i in range(M):
        for j in range(N):
            p0 =  i   * (N+1) +  j
            p1 = (i+1)* (N+1) +  j
            p2 = (i+1)* (N+1) + (j+1)
            p3 =  i   * (N+1) + (j+1)
            idx.extend((p0, p1, p2, p0, p2, p3))
    return np.array(idx, dtype=np.uint32)
