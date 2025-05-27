#!/usr/bin/env python3
# simple_sphere.py ----------------------------------------------------------
"""
First-step demo: analytical ‘monopole’ sphere rendered with moderngl-window.
Rotate the camera, draw with Lambert shading, toggle wireframe on <space>.
"""

import moderngl_window as mglw
import moderngl
import numpy as np
import pyrr
from monopole import vertices_and_normals, indices   # same helper as before

VERT_SRC = """
#version 330 core
in vec3 in_pos;
in vec3 in_nor;
uniform mat4 mvp;
out vec3 v_normal;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_normal    = in_nor;
}
"""

FRAG_SRC = """
#version 330 core
in vec3 v_normal;
out vec4 f_color;
void main() {
    vec3  N    = normalize(v_normal);
    vec3  L    = normalize(vec3(-0.3, 0.4, 1.0));
    float diff = clamp(dot(N, L), 0.0, 1.0);
    vec3  base = vec3(0.80, 0.85, 0.90);
    f_color    = vec4(base * (0.20 + 0.80 * diff), 1.0);
}
"""

# ---------------------------------------------------------------------------

class SphereDemo(mglw.WindowConfig):
    """Draw the analytical monopole sphere, camera orbits around it."""
    gl_version   = (3, 3)
    title        = "Monopole sphere – moderngl"
    window_size  = (900, 900)
    aspect_ratio = 1.0
    resizable    = False
    resource_dir = '.'     # unused but required by moderngl-window

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. Compile shader program
        self.prog = self.ctx.program(vertex_shader=VERT_SRC,
                                     fragment_shader=FRAG_SRC)

        # 2. Geometry -------------------------------------------------------
        pos, _ = vertices_and_normals()          # ignore finite-diff normals
        pos *= 0.2                              # shrink sphere slightly
        nor = pos / np.linalg.norm(pos, axis=1, keepdims=True)

        idx = indices()
        vbo_pos = self.ctx.buffer(pos.astype('f4').tobytes())
        vbo_nor = self.ctx.buffer(nor.astype('f4').tobytes())
        ibo     = self.ctx.buffer(idx.tobytes())

        vao_content = [
            (vbo_pos, '3f', 'in_pos'),
            (vbo_nor, '3f', 'in_nor'),
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, ibo)

        # 3. State & camera -------------------------------------------------
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.front_face = 'cw'
        self.ctx.wireframe = False

        self.theta = 0.0                         # orbital angle
        self.wnd.key_event_func = self.on_key_press

    # ------------------------------------------------------------------ I/O
    def on_key_press(self, key, action, modifiers):
        if key == self.wnd.keys.SPACE and action == self.wnd.keys.ACTION_PRESS:
            self.ctx.wireframe = not self.ctx.wireframe

    # ---------------------------------------------------------------- render
    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.05, 0.10, 0.15)

        # camera: orbit outside the sphere (radius 3.5, slight elevation)
        self.theta += frame_time * 0.4
        eye    = np.array([3.5 * np.sin(self.theta), 0.8,
                           3.5 * np.cos(self.theta)], dtype='f4')
        target = np.array([0.0, 0.0, 0.0], dtype='f4')
        up     = np.array([0.0, 1.0, 0.0], dtype='f4')

        view = pyrr.matrix44.create_look_at(eye, target, up, dtype='f4')
        proj = pyrr.matrix44.create_perspective_projection(
            fovy=45.0, aspect=1.0, near=0.1, far=10.0, dtype='f4')
        mvp  = proj @ view
        self.prog['mvp'].write(mvp)

        self.vao.render()

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    mglw.run_window_config(SphereDemo)
