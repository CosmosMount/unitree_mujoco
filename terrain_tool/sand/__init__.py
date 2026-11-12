"""Transient granular RFT for MuJoCo; z is up and all units are SI.

Use :class:`RFTSand` and call ``sand.step(data)`` once per physics step.
The terrain is a force model, not a MuJoCo collision geom or a heightfield.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


class SoilDomainError(RuntimeError):
    """A body has reached the limits of the finite RFT domain."""


@dataclass
class SandBase:
    surface_z: float = 0.0
    soil_size: float = 4.0
    depth: float = 0.5
    # None selects all dynamic, collision-enabled geoms. Explicit IDs can
    # be used for specialized scenes, but omitted bodies receive no soil force.
    geom_ids: object = None

    def __post_init__(self):
        if not np.isfinite([self.surface_z, self.soil_size, self.depth]).all():
            raise ValueError("sand dimensions must be finite")
        if self.soil_size <= 0 or self.depth <= 0:
            raise ValueError("sand size and depth must be positive")
        self.model = None
        self.revision = 0
        self._last_step_end = None
        self._last_qpos = None

    def bind(self, model):
        """Bind to a compiled model, without modifying its collision masks."""
        from .geometry import SUPPORTED

        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("RFTSand currently supports the Euler integrator only")
        probe = mujoco.MjData(model)
        mujoco.mj_forward(model, probe)
        bodies = model.geom_bodyid
        dynamic = model.body_weldid[bodies] != 0
        collidable = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        if self.geom_ids is None:
            ids = np.flatnonzero(dynamic & collidable)
        else:
            raw = np.asarray(self.geom_ids)
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError("geom_ids must be a one-dimensional sequence of integer IDs")
            ids = raw.astype(int, copy=True)
            if np.any(ids < 0) or np.any(ids >= model.ngeom) or len(np.unique(ids)) != len(ids):
                raise ValueError("geom_ids must contain distinct, valid geom IDs")
            if np.any(~dynamic[ids]) or np.any(~collidable[ids]):
                raise ValueError("sand geoms must be dynamic and collision-enabled")
        if not len(ids):
            raise ValueError("no dynamic collision geoms were selected for sand")
        unsupported = [int(i) for i in ids if int(model.geom_type[i]) not in SUPPORTED]
        if unsupported:
            raise ValueError(f"unsupported sand collision geom IDs: {unsupported}; use primitive geoms")

        # An infinite rigid floor would support the body before/inside the sand.
        # Other static obstacles require scene-specific inspection by the caller.
        for i in np.flatnonzero(~dynamic & collidable & (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)):
            normal = probe.geom_xmat[i].reshape(3, 3)[:, 2]
            if normal[2] > 1 - 1e-8 and probe.geom_xpos[i, 2] >= self.surface_z - self.depth:
                raise ValueError(f"rigid plane geom {i} overlaps the sand domain; remove it from the sand scene")

        self.model = model
        self.geom_ids = ids
        self.bodies = np.asarray(bodies[ids], dtype=int)
        self.geom_forces = np.zeros((len(ids), 3))
        self.geom_torques = np.zeros((len(ids), 3))
        self._qforce = np.zeros(model.nv)

    def reset(self):
        """Discard transient contact history after resetting/teleporting data."""
        self._qforce.fill(0)
        self.geom_forces.fill(0)
        self.geom_torques.fill(0)
        self._last_step_end = None
        self._last_qpos = None
        self.revision += 1

    def _finish_step(self, data, geoms):
        for geom in geoms:
            mujoco.mj_applyFT(self.model, data, self.geom_forces[geom], self.geom_torques[geom],
                              data.geom_xpos[self.geom_ids[geom]], int(self.bodies[geom]), self._qforce)
        return self._qforce

    def step(self, data):
        """Apply sand and take exactly one step, preserving user-applied forces."""
        if self.model is None:
            raise RuntimeError("bind the sand to a model before stepping")
        if self._last_step_end is not None:
            if (data.time < self._last_step_end - 1e-12
                    or not np.array_equal(data.qpos, self._last_qpos)):
                self.reset()
        soil_force = self.advance(data)
        applied = data.qfrc_applied.copy()
        try:
            data.qfrc_applied[:] = applied + soil_force
            mujoco.mj_step(self.model, data)
        finally:
            data.qfrc_applied[:] = applied
        self._last_step_end = float(data.time)
        self._last_qpos = data.qpos.copy()


from .rft import RFTMaterial, RFTSand, RFT_SAND  # noqa: E402

__all__ = ["RFTMaterial", "RFTSand", "RFT_SAND", "SandBase", "SoilDomainError"]
