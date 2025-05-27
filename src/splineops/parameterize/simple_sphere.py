# simple_sphere.py ----------------------------------------------------------
import moderngl_window as mglw
import moderngl            
import numpy as np
from monopole import vertices_and_normals, indices
import pyrr

VERT_SRC = """
#version 330
in vec3 in_pos;
in vec3 in_nor;
uniform mat4 mvp;
out vec3 v_normal;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_normal = in_nor;
}
"""

FRAG_SRC = """
#version 330
in vec3 v_normal;
out vec4 f_color;
void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(-0.3, 0.4, 1.0));
    float diff = clamp(dot(N,L), 0.0, 1.0);
    vec3 base = vec3(0.8, 0.85, 0.9);
    f_color = vec4(base * (0.2 + 0.8*diff), 1.0);
}
"""

class SphereDemo(mglw.WindowConfig):
    """Draw the analytical monopole sphere. Rotate slowly."""
    gl_version = (3, 3)
    title      = "Monopole sphere – moderngl"
    window_size = (900, 900)
    aspect_ratio = 1.0
    resizable  = False
    resource_dir = '.'   # not used, but moderngl-window wants it

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 1. Compile shaders and build a simple program
        self.prog = self.ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)

        # 2. Geometry
        pos, nor = vertices_and_normals()
        idx      = indices()
        vbo_pos  = self.ctx.buffer(pos.tobytes())
        vbo_nor  = self.ctx.buffer(nor.tobytes())
        ibo      = self.ctx.buffer(idx.tobytes())

        vao_content = [
            (vbo_pos, '3f', 'in_pos'),
            (vbo_nor, '3f', 'in_nor')
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, ibo)

        # 3. Uniforms / camera
        self.theta = 0.0

    def on_render(self, time: float, frame_time: float):
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.ctx.clear(0.05, 0.10, 0.15)

        # Slowly spin the camera round the Y axis
        self.theta += frame_time * 0.4
        eye   = np.array([2.5*np.sin(self.theta),  0.8, 2.5*np.cos(self.theta)], dtype=np.float32)
        target= np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up    = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = pyrr.matrix44.create_look_at(eye, target, up, dtype='f4')
        proj = pyrr.matrix44.create_perspective_projection(
                fovy=45.0, aspect=1.0, near=0.1, far=10.0, dtype='f4')
        mvp = proj @ view

        self.prog['mvp'].write(mvp.astype('f4').tobytes())

        self.vao.render()

if __name__ == '__main__':
    mglw.run_window_config(SphereDemo)
