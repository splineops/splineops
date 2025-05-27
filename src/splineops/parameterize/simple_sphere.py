#!/usr/bin/env python3
"""
Analytical ‘monopole’ sphere – moderngl-window demo
  • unit sphere, camera orbits at radius 5
  • home-made look_at / perspective (column-major OK for GL)
  • <Space> toggles wire-frame
"""

import moderngl_window as mglw, moderngl, numpy as np
from monopole import vertices_and_normals, indices

# ------------------------------------------------------------------ helpers
def look_at(eye, target, up=np.array([0,1,0], dtype=np.float32)):
    f = target - eye;  f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0,:3], m[1,:3], m[2,:3] = s, u, -f
    m[:3,3] = -eye @ np.array([s,u,-f])
    return m

def perspective(fovy, aspect, znear, zfar):
    f = 1 / np.tan(np.radians(fovy) * .5)
    m = np.zeros((4,4), dtype=np.float32)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = (zfar+znear)/(znear-zfar)
    m[2,3] = 2*zfar*znear/(znear-zfar)
    m[3,2] = -1
    return m
# ---------------------------------------------------------------- shaders
VERT = """
#version 330
in  vec3 in_pos, in_nor;
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

class SphereDemo(mglw.WindowConfig):
    gl_version=(3,3)
    title="Monopole sphere"; window_size=(900,900)
    aspect_ratio=1; resizable=False

    def __init__(self,**kw):
        super().__init__(**kw)
        self.prog=self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

        # ---- geometry ----------------------------------------------------
        pos,_ = vertices_and_normals()           # radius ≈ 1
        nor   = pos/np.linalg.norm(pos,axis=1,keepdims=True)
        idx   = indices()
        self.vao=self.ctx.vertex_array(
            self.prog,
            [(self.ctx.buffer(pos.astype('f4').tobytes()),'3f','in_pos'),
             (self.ctx.buffer(nor.astype('f4').tobytes()),'3f','in_nor')],
            self.ctx.buffer(idx.tobytes())
        )

        # ---- GL state ----------------------------------------------------
        self.ctx.enable_only(moderngl.DEPTH_TEST)    # face-culling off
        self.ctx.wireframe=False
        self.theta=0.0
        self.wnd.key_event_func=self.on_key

    # toggle wire-frame
    def on_key(self,key,action,mods):
        if key==self.wnd.keys.SPACE and action==self.wnd.keys.ACTION_PRESS:
            self.ctx.wireframe=not self.ctx.wireframe

    # frame
    def on_render(self, time, dt):
        self.theta += dt * 0.4
        eye = np.array([5*np.sin(self.theta), 1.0, 5*np.cos(self.theta)],
                    dtype=np.float32)

        mvp = perspective(45, 1.0, 0.1, 20.0) @ look_at(eye, np.zeros(3, 'f4'))
        self.prog['mvp'].write(mvp.T.tobytes())   #  ← transpose here!

        self.ctx.clear(0.05, 0.10, 0.15)
        self.vao.render()


if __name__=='__main__':
    mglw.run_window_config(SphereDemo)
