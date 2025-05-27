#!/usr/bin/env python3
"""
Monopole sphere – camera fixed, sphere drifts & spins (Brownian).
<Space> toggles wire-frame.
"""

import moderngl_window as mglw, moderngl, numpy as np, math, random
from monopole import vertices_and_normals, indices

# ---------- tiny math helpers (column-major output for GL) -----------------
def look_at(eye, target, up=np.array([0,1,0], dtype=np.float32)):
    f = target-eye; f/=np.linalg.norm(f)
    s = np.cross(f, up); s/=np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,:3], M[1,:3], M[2,:3] = s, u, -f
    M[:3,3] = -eye @ np.array([s,u,-f])
    return M
def perspective(fovy, aspect, znear, zfar):
    f = 1/np.tan(np.radians(fovy)*.5)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0], M[1,1] = f/aspect, f
    M[2,2], M[2,3] = (zfar+znear)/(znear-zfar), 2*zfar*znear/(znear-zfar)
    M[3,2] = -1
    return M
def quat(axis, ang):
    axis = axis/np.linalg.norm(axis); s = math.sin(ang/2)
    return np.array([*axis*s, math.cos(ang/2)], dtype=np.float32)
def qmul(a,b):
    x1,y1,z1,w1 = a; x2,y2,z2,w2 = b
    return np.array([
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-y1*w2+z1*x2-x1*z2,
        w1*z2+z1*w2+x1*y2-y1*x2,
        w1*w2-x1*x2-y1*y2-z1*z2], dtype=np.float32)
def qmat(q):
    x,y,z,w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 0],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 0],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), 0],
        [0,0,0,1]], dtype=np.float32)

# ------------------------------ shaders ------------------------------------
VERT = """
#version 330
in vec3 in_pos, in_nor;
uniform mat4 mvp;
out vec3 v_nor;
void main(){ gl_Position=mvp*vec4(in_pos,1); v_nor=in_nor; }
"""
FRAG = """
#version 330
in vec3 v_nor; out vec4 f;
void main(){
    vec3 N=normalize(v_nor), L=normalize(vec3(-.3,.4,1));
    float d=max(dot(N,L),0);
    f=vec4(vec3(.8,.85,.9)*(0.2+0.8*d),1);
}
"""

# ---------------------------- moderngl-window app --------------------------
class SphereDemo(mglw.WindowConfig):
    gl_version=(3,3); title="Monopole sphere"; window_size=(900,900)
    aspect_ratio=1; resizable=False

    # ---- Brownian parameters ---------------------------------------------
    TRAN_SIGMA = 0.25           # positional stdev
    TRAN_VEL   = 0.4            # translation rate

    ROT_SIGMA  = 0.06           # <<< 10× smaller than before
    ROT_DAMP   = 0.999          # very light damping
    MAX_STEP   = 0.1            # rad – cap per-frame rotation

    def __init__(self, **kw):
        super().__init__(**kw)
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

        # geometry ---------------------------------------------------------
        pos,_ = vertices_and_normals()            # unit sphere
        nor   = pos/np.linalg.norm(pos,axis=1,keepdims=True)
        idx   = indices()

        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.ctx.buffer(pos.astype('f4').tobytes()),'3f','in_pos'),
             (self.ctx.buffer(nor.astype('f4').tobytes()),'3f','in_nor')],
            self.ctx.buffer(idx.tobytes())
        )

        # GL state
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.ctx.wireframe=False
        self.wnd.key_event_func=self.on_key

        # camera = fixed
        self.proj = perspective(45,1,.1,20)
        self.view = look_at(np.array([0,0,6],dtype='f4'),
                            np.zeros(3,'f4'))

        # Brownian state for the sphere
        self.pos = np.zeros(3, dtype=np.float32)
        self.rot = np.array([0,0,0,1], dtype=np.float32)   # quaternion

    def on_key(self,key,action,mods):
        if key==self.wnd.keys.SPACE and action==self.wnd.keys.ACTION_PRESS:
            self.ctx.wireframe = not self.ctx.wireframe

    def on_render(self, time, dt):
        # --- Brownian update of translation ------------------------------
        self.pos += self.TRAN_VEL * np.random.normal(
                        scale=self.TRAN_SIGMA, size=3).astype('f4') * dt
        # softly pull back toward origin
        self.pos *= 0.999

        # --- Brownian rotation ------------------------------------------------
        axis = np.random.normal(size=3)
        axis /= np.linalg.norm(axis)

        dtheta = np.clip(
            self.ROT_SIGMA * math.sqrt(dt) * random.gauss(0, 1),
            -self.MAX_STEP,
            self.MAX_STEP,
        )

        self.rot = qmul(self.rot, quat(axis, dtheta))
        self.rot *= self.ROT_DAMP / np.linalg.norm(self.rot)

        model = np.eye(4,dtype='f4')
        model[:3,3] = self.pos
        model = model @ qmat(self.rot)

        mvp = self.proj @ self.view @ model
        self.prog['mvp'].write(mvp.T.tobytes())

        self.ctx.clear(.05,.10,.15)
        self.vao.render()

# --------------------------------------------------------------------------
if __name__=='__main__':
    mglw.run_window_config(SphereDemo)
