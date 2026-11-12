"""Transient geometric intrusion envelope for the optional Viser viewer."""

import time

import mujoco
import numpy as np

from .geometry import half_widths


class SoilSurface:
    """A display mesh only: it never changes the physical RFT surface."""

    def __init__(self, sand, spacing=0.00625):
        self.sand = sand
        self.spacing = spacing

    def mesh(self, data):
        sand = self.sand
        surface = self._rft_surface(data)
        if surface is None:
            x = y = np.array([-sand.soil_size / 2, sand.soil_size / 2])
            heights = np.full((2, 2), sand.surface_z, dtype=float)
        else:
            x, y, heights = surface
        xx, yy = np.meshgrid(x, y)
        vertices = np.column_stack((xx.ravel(), yy.ravel(), heights.ravel())).astype(np.float32)
        r, c = np.mgrid[:len(y) - 1, :len(x) - 1]
        a = (r * len(x) + c).ravel()
        faces = np.column_stack((a, a + 1, a + len(x) + 1,
                                 a, a + len(x) + 1, a + len(x))).reshape(-1, 3).astype(np.uint32)
        return vertices, faces, float(sand.surface_z - heights.min())

    def _rft_surface(self, data):
        sand = self.sand
        half = sand.soil_size / 2
        centers = data.geom_xpos[sand.geom_ids]
        widths = half_widths(sand.model, data, sand.geom_ids)
        near = centers[:, 2] - widths[:, 2] < sand.surface_z
        if not near.any():
            return None
        low = np.maximum(np.min(centers[near, :2] - widths[near, :2], axis=0) - .01, -half)
        high = np.minimum(np.max(centers[near, :2] + widths[near, :2], axis=0) + .01, half)
        if np.any(low >= high):
            return None
        # Cap mesh size while covering every immersed geom, including wide robots.
        counts = np.minimum(np.maximum(2, np.ceil((high - low) / self.spacing).astype(int) + 1), [385, 193])
        x = np.unique(np.r_[-half, np.linspace(low[0], high[0], counts[0]), half])
        y = np.unique(np.r_[-half, np.linspace(low[1], high[1], counts[1]), half])
        xx, yy = np.meshgrid(x, y)
        heights = np.full(xx.shape, sand.surface_z, dtype=float)
        xy = np.column_stack((xx[1:-1, 1:-1].ravel(), yy[1:-1, 1:-1].ravel()))
        heights[1:-1, 1:-1] = sand.contact_envelope_height(data, xy).reshape(len(y) - 2, len(x) - 2)
        return x, y, heights


class ViserSand:
    """mjviser render callback; Viser itself is an optional dependency."""

    def __init__(self, sand, data, server):
        self.surface = SoilSurface(sand)
        self.data, self.server = data, server
        self.handle = None
        self.last_time = None
        self.last_draw = 0.
        self.last_revision = None
        self.last_pose = None
        self.depth = server.gui.add_number("视野内沉陷 (cm)", initial_value=0., disabled=True)
        server.gui.add_markdown("**RFT**：显示当前几何接触下陷；抬起后恢复平面，不表示永久足印或颗粒流动。")

    def __call__(self, viewer_scene):
        # mj_step leaves poses at the start of the step. Update kinematics only:
        # mj_forward here would overwrite force-dependent sensors used by DDS.
        mujoco.mj_kinematics(self.surface.sand.model, self.data)
        viewer_scene.update_from_mjdata(self.data)
        visible = viewer_scene.geom_groups_visible[1]
        if self.handle is not None:
            self.handle.visible = visible
        revision = self.surface.sand.revision
        geom_ids = self.surface.sand.geom_ids
        pose = np.concatenate((self.data.geom_xpos[geom_ids].ravel(), self.data.geom_xmat[geom_ids].ravel()))
        if revision == self.last_revision and np.array_equal(pose, self.last_pose):
            return
        now = time.monotonic()
        # Reset and paused edits redraw immediately; running updates are capped at 20 Hz.
        if (revision == self.last_revision and self.last_time is not None
                and self.data.time > self.last_time and now - self.last_draw < 1 / 20):
            return
        vertices, faces, depth = self.surface.mesh(self.data)
        with self.server.atomic():
            if self.handle is None:
                self.handle = self.server.scene.add_mesh_simple(
                    "/fixed_bodies/sand_contact_surface", vertices, faces,
                    color=(209, 181, 125), side="double", flat_shading=False,
                    cast_shadow=True, receive_shadow=True)
            else:
                self.handle.vertices = vertices
                self.handle.faces = faces
            self.handle.visible = visible
            self.depth.value = round(100 * depth, 2)
        self.last_time = self.data.time
        self.last_draw = now
        self.last_revision = revision
        self.last_pose = pose
