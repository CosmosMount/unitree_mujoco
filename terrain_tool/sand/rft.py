"""Panel RFT: material, elastic contact history and articulated implicit solve.

Li (2013) / Treers (2021) sliding law with the reference inertial correction.
Elasticity and implicit damping are engineering extensions. No permanent soil
history. SI units; world z points up.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from . import SandBase, SoilDomainError
from .geometry import SurfacePanels, bottom_heights, half_widths, primitive_panels, inside_primitive


@dataclass(frozen=True)
class RFTMaterial:
    zeta: float
    inertial_lambda: float
    bulk_density: float
    elastic_displacement: float
    damping_ratio: float

    def __post_init__(self):
        values = (self.zeta, self.inertial_lambda, self.bulk_density,
                  self.elastic_displacement, self.damping_ratio)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("RFT material values must be finite and positive")


RFT_SAND = RFTMaterial(zeta=.6, inertial_lambda=1.1, bulk_density=1450.,
                       elastic_displacement=.00025, damping_ratio=.3)


@dataclass
class RFTSand(SandBase):
    soil_model = "rft"
    material: RFTMaterial = RFT_SAND
    # Quadrature resolution; results should be checked for convergence for each task.
    rings: int = 8
    sectors: int = 8
    length_scale: float = .022
    # Panels this far above the surface are skipped: dry now and, moving less than this in a
    # step (10 m/s at 2 ms), still dry in the step that would first need their history.
    near_margin: float = .02

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.material, dict):
            self.material = RFTMaterial(**self.material)
        if not isinstance(self.material, RFTMaterial):
            raise TypeError("RFT requires RFTMaterial or its parameter dictionary")
        if (not isinstance(self.rings, int) or not isinstance(self.sectors, int)
                or self.rings < 2 or self.sectors < 4 or self.sectors % 2
                or not np.isfinite(self.length_scale) or self.length_scale <= 0
                or not np.isfinite(self.near_margin) or self.near_margin <= 0):
            raise ValueError("RFT requires rings >= 2, even sectors >= 4 and positive finite length_scale and near_margin")

    def bind(self, model):
        super().bind(model)
        self._build_panels()
        self.panel_geoms = np.repeat(np.arange(len(self.geom_ids)),
                                     [sl.stop - sl.start for sl in self.panel_slices])
        self.reset()

    def _build_panels(self):
        model = self.model
        probe = mujoco.MjData(model)
        mujoco.mj_forward(model, probe)
        parts, self.panel_slices = [], []
        offset = 0
        for i, geom in enumerate(self.geom_ids):
            p = primitive_panels(int(model.geom_type[geom]), model.geom_size[geom], self.rings, self.sectors)
            rotation = probe.geom_xmat[geom].reshape(3, 3)
            world = p.positions @ rotation.T + probe.geom_xpos[geom]
            keep = np.ones(len(world), dtype=bool)
            for j, other in enumerate(self.geom_ids):
                if i == j or model.body_weldid[self.bodies[i]] != model.body_weldid[self.bodies[j]]:
                    continue
                q = (world - probe.geom_xpos[other]) @ probe.geom_xmat[other].reshape(3, 3)
                # Strict interior removal; deterministic ownership of coincident surfaces.
                keep &= ~inside_primitive(q, int(model.geom_type[other]), model.geom_size[other],
                                          1e-10 if j < i else -1e-10)
            p = SurfacePanels(p.positions[keep], p.normals[keep], p.areas[keep])
            parts.append(p)
            self.panel_slices.append(slice(offset, offset + len(p.areas)))
            offset += len(p.areas)
        if parts:
            self.panels = SurfacePanels(np.concatenate([p.positions for p in parts]),
                                        np.concatenate([p.normals for p in parts]),
                                        np.concatenate([p.areas for p in parts]))
        else:
            self.panels = SurfacePanels(np.empty((0, 3)), np.empty((0, 3)), np.empty(0))

    def reset(self):
        self.contact = PanelContact(self.material, self.panels, self.length_scale)
        super().reset()

    def surface_height(self, xy, *, plastic=True):
        """Reference plane: RFT has neither plastic nor evolving soil height."""
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("surface sampling requires finite (N,2) world positions")
        return np.full(len(xy), self.surface_z)

    def contact_envelope_height(self, data, xy):
        """Geometric intrusion envelope for display only; no force/history update."""
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("surface sampling requires finite (N,2) world positions")
        widths = half_widths(self.model, data, self.geom_ids)
        near = data.geom_xpos[self.geom_ids, 2] - widths[:, 2] < self.surface_z
        heights = bottom_heights(self.model, data, self.geom_ids[near], xy[:, 0], xy[:, 1])
        return np.minimum(heights.min(axis=0), self.surface_z) if len(heights) else np.full(len(xy), self.surface_z)

    def _check_domain(self, data):
        """Lowest point of each collision geom (world AABB), after the domain checks."""
        centers = data.geom_xpos[self.geom_ids]
        widths = half_widths(self.model, data, self.geom_ids)
        bottom = centers[:, 2] - widths[:, 2]
        near = bottom <= self.surface_z
        if np.any(bottom <= self.surface_z - self.depth):
            raise SoilDomainError("RFT collision shape reaches the hard bottom")
        if np.any(np.abs(centers[near, :2]) + widths[near, :2] > self.soil_size / 2):
            raise SoilDomainError("collision footprint leaves the finite sand domain")
        return bottom

    def advance(self, data):
        model = self.model
        self._qforce.fill(0)
        self.geom_forces.fill(0)
        self.geom_torques.fill(0)
        # Refresh free acceleration, including caller-applied forces and controls.
        mujoco.mj_forward(model, data)
        bottom = self._check_domain(data)
        if not len(self.panels.areas):
            return self._finish_step(data, ())
        inverse = np.zeros((model.nv, model.nv))
        mujoco.mj_solveM(model, data, inverse, np.eye(model.nv))
        free = data.qvel + model.opt.timestep * data.qacc
        # Only panels of near geoms are posed, except on the first step, which starts the history.
        start = self.contact.previous is None
        near = np.ones(len(self.geom_ids), dtype=bool) if start else bottom <= self.surface_z + self.near_margin
        candidates = np.flatnonzero(near[self.panel_geoms])
        rotations = data.geom_xmat[self.geom_ids].reshape(-1, 3, 3)[self.panel_geoms[candidates]]
        arms = (rotations @ self.panels.positions[candidates, :, None])[..., 0]  # from the geom center
        positions = arms + data.geom_xpos[self.geom_ids[self.panel_geoms[candidates]]]
        positions[:, 2] -= self.surface_z
        if start:
            self.contact.start(positions)
        keep = positions[:, 2] <= self.near_margin
        index, arms, positions, rotations = candidates[keep], arms[keep], positions[keep], rotations[keep]
        normals = (rotations @ self.panels.normals[index, :, None])[..., 0]
        geoms, owners = np.unique(self.panel_geoms[index], return_inverse=True)
        # Panel point velocity is J_g twist, J_g = [jacp; jacr] at the geom center (one per geom);
        # lever maps the twist to the panel: v + w x arm.
        jacobians = np.empty((len(geoms), 6, model.nv))
        for k, geom in enumerate(geoms):
            mujoco.mj_jac(model, data, jacobians[k, :3], jacobians[k, 3:], data.geom_xpos[self.geom_ids[geom]],
                          int(self.bodies[geom]))
        lever = np.zeros((len(index), 3, 6))
        lever[:, (0, 1, 2), (0, 1, 2)] = 1
        lever[:, 0, 4], lever[:, 0, 5] = arms[:, 2], -arms[:, 1]
        lever[:, 1, 3], lever[:, 1, 5] = -arms[:, 2], arms[:, 0]
        lever[:, 2, 3], lever[:, 2, 4] = arms[:, 1], -arms[:, 0]
        members = (owners == np.arange(len(geoms))[:, None]).astype(float)  # (geoms, panels)
        velocities = (lever @ (jacobians @ data.qvel)[owners, :, None])[..., 0]
        forces = self.contact.step(index, positions, normals, velocities, lever, owners, members, jacobians,
                                   inverse, free, model.opt.timestep)
        wrenches = members @ (forces[:, None] @ lever)[:, 0]  # force and torque about the geom center
        self.geom_forces[geoms], self.geom_torques[geoms] = wrenches[:, :3], wrenches[:, 3:]
        return self._finish_step(data, geoms)


def _sliding_forces(positions, normals, areas, velocities, material):
    """Li/Treers force and inertial correction on submerged panels (z < 0)."""
    v = np.asarray(velocities)
    speed = np.linalg.norm(v, axis=1)
    nh = np.linalg.norm(normals[:, :2], axis=1)
    e2 = np.zeros_like(v)
    np.divide(normals[:, :2], nh[:, None], out=e2[:, :2], where=nh[:, None] > 0)
    e1 = np.stack((-e2[:, 1], e2[:, 0], np.zeros(len(v))), axis=1)
    flat = nh == 0
    vh = np.linalg.norm(v[:, :2], axis=1)
    e1[flat] = [1., 0., 0.]
    moving_flat = flat & (vh > 0)
    e1[moving_flat, :2] = v[moving_flat, :2]/vh[moving_flat, None]
    e2[flat] = np.stack((-e1[flat, 1], e1[flat, 0], np.zeros(np.count_nonzero(flat))), axis=1)
    v1, v2 = np.sum(v*e1, axis=1), np.sum(v*e2, axis=1)
    speed23 = np.hypot(v2, v[:, 2])
    beta = np.arctan2(nh, -normals[:, 2])
    gamma = np.arctan2(-v[:, 2], v2)
    # Li et al. (2013), SI Table S2: N/cm^3 converted to Pa/m.
    az = (.206 + .169 * np.cos(2 * beta) + .212 * np.sin(2 * beta + gamma)
          + .358 * np.sin(gamma) + .055 * np.sin(-2 * beta + gamma)) * 1e6
    ax = (-.124 * np.cos(2 * beta + gamma) + .253 * np.cos(gamma)
          + .007 * np.cos(-2 * beta + gamma) + .088 * np.sin(2 * beta)) * 1e6
    u1 = np.divide(abs(v1), speed, out=np.zeros_like(speed), where=speed > 0)
    u23 = np.divide(speed23, speed, out=np.zeros_like(speed), where=speed > 0)
    a, b = 3.62263982880971, -1.60910808139526
    f1 = (np.tanh(a*u1+b)-np.tanh(b))/(np.tanh(a+b)-np.tanh(b))
    a, b = 1.61146827229181, -.973746396532650
    f23 = (np.arctanh(a*u23+b)-np.arctanh(b))/(np.arctanh(a+b)-np.arctanh(b))
    planar = -ax[:, None]*e2
    planar[:, 2] = az
    qs = f23[:, None]*planar - (f1*.136e6*np.sign(v1))[:, None]*e1
    depth = np.maximum(-positions[:, 2], 0)
    leading = np.sum(normals*v, axis=1)
    active = (depth > 0) & (leading >= 0) & (speed > 0)
    qs *= (material.zeta*depth*areas*active)[:, None]
    dynamic = -material.inertial_lambda*material.bulk_density*(np.maximum(leading, 0)*areas*(depth > 0))[:, None]*v
    return qs, dynamic


def _limit_force(force, normal_limit):
    """Apply the reference asymmetric normal and circular horizontal bounds in place."""
    force[:, 2] = np.clip(force[:, 2], -.25 * normal_limit, normal_limit)
    norm = np.linalg.norm(force[:, :2], axis=1)
    scale = np.divide(.136 * normal_limit, norm, out=np.ones_like(norm), where=norm > 0)
    force[:, :2] *= np.minimum(1., scale)[:, None]
    return force


class PanelContact:
    """Surface contact coupled through generalized (possibly articulated) inertia.

    Local state is stored in world coordinates for the current contact only.
    Damping mass is quadrature-partitioned by area, so refinement does not
    duplicate a body's inertia. Body-point Jacobians retain all cross-panel
    coupling in the backward solve. Robot-level validity is assessed separately.
    """
    def __init__(self, material, panels, length_scale):
        self.material, self.panels = material, panels
        self.stiffness = material.zeta * 1e6 * panels.areas * length_scale / material.elastic_displacement
        self.elastic = np.zeros_like(panels.positions)
        self.previous = None
        self.active = np.empty(0, dtype=int)  # panels of the last step; the only ones with elastic force

    def start(self, positions):
        """History before the first step: at rest, submerged panels at the surface."""
        self.previous = positions.copy()
        self.previous[:, 2] = np.maximum(positions[:, 2], 0)

    def step(self, index, positions, normals, velocities, lever, owners, members, jacobians,
             mass_inverse, free_velocity, dt):
        """Forces on panels ``index`` (per-panel arrays are of these panels).

        Panel ``i`` of geom ``owners[i]`` moves with ``lever[i] @ jacobians[owners[i]] @ qvel``;
        ``members`` is the (geoms, panels) indicator of ``owners``. Panels left out are dry and
        their history is not advanced; a dry panel's elastic force is zero, as the full step
        would leave it.
        """
        elastic = self.elastic[index]
        self.elastic[self.active] = 0
        self.active = index
        stiffness, areas = self.stiffness[index], self.panels.areas[index]
        depth = np.maximum(-positions[:, 2], 0)
        wet = depth > 0
        trial_force = elastic - stiffness[:, None] * (positions - self.previous[index])
        trial_force[~wet] = 0
        sliding_force = np.zeros_like(positions, dtype=float)
        inertial_force = np.zeros_like(positions, dtype=float)
        if wet.any():
            sliding_force[wet], inertial_force[wet] = _sliding_forces(
                positions[wet], normals[wet], areas[wet], velocities[wet], self.material)
        normal_limit = self.material.zeta * 1e6 * areas * depth
        # Use the sliding law only outside the elastic envelope on leading faces.
        yielding = ((trial_force[:, 2] > normal_limit)
                    | (trial_force[:, 2] < -.25 * normal_limit)
                    | (np.linalg.norm(trial_force[:, :2], axis=1) > .136 * normal_limit))
        leading = np.sum(normals * velocities, axis=1) >= 0
        sliding = yielding & leading & wet
        trial_force[sliding] = sliding_force[sliding]
        _limit_force(trial_force, normal_limit)

        # Panel mobility diag(J M^-1 J^T) from the 6 x 6 mobility of its geom.
        # Partition damping mass by wet area so finer quadrature adds no inertia.
        geom_mobility = jacobians @ mass_inverse @ jacobians.transpose(0, 2, 1)
        mobility_diag = ((lever @ geom_mobility[owners]) * lever).sum(-1) * wet[:, None]
        weights = areas * wet
        weights = weights / max(weights.sum(), np.finfo(float).tiny)
        effective_mass = np.divide(weights[:, None], mobility_diag,
                                   out=np.zeros_like(mobility_diag), where=mobility_diag > 0)
        damping = 2 * self.material.damping_ratio * np.sqrt(stiffness[:, None] * effective_mass)
        implicit_damping = (stiffness[:, None] * dt + damping) * wet[:, None]

        # Solve all panels together through the small generalized-velocity system; panel sums
        # are reduced to each geom's 6-D twist space before mapping to joint space.
        lever_t = lever.transpose(0, 2, 1)
        geom_damping = (members @ (lever_t @ (implicit_damping[:, :, None] * lever)).reshape(
            len(index), 36)).reshape(-1, 6, 6)
        stacked = jacobians.reshape(-1, len(free_velocity))  # (6 geoms, nv)
        damping_matrix = (jacobians.transpose(0, 2, 1) @ geom_damping @ jacobians).sum(0)

        def generalized(force):
            return stacked.T @ (members @ (lever_t @ force[:, :, None])[..., 0]).ravel()

        def panel_velocities(velocity):
            return (lever @ (jacobians @ velocity)[owners, :, None])[..., 0]

        lhs = np.eye(len(free_velocity)) + dt * mass_inverse @ damping_matrix
        rhs_velocity = free_velocity + dt * mass_inverse @ generalized(trial_force + inertial_force)
        predicted_velocity = panel_velocities(np.linalg.solve(lhs, rhs_velocity))
        force = trial_force - implicit_damping * predicted_velocity + inertial_force

        # Clamp support and remove horizontal work that would assist the motion.
        force[:, 2] = np.clip(force[:, 2], -.25 * normal_limit, normal_limit)
        horizontal_speed_sq = np.sum(velocities[:, :2]**2, axis=1)
        assisting_work = np.maximum(np.sum(force[:, :2] * velocities[:, :2], axis=1), 0)
        assist = np.divide(assisting_work, horizontal_speed_sq,
                           out=np.zeros_like(horizontal_speed_sq), where=horizontal_speed_sq > 0)
        force[:, :2] -= assist[:, None] * velocities[:, :2]
        _limit_force(force, normal_limit)

        # Advance history using the bounded force, once per physical step.
        predicted_velocity = panel_velocities(free_velocity + dt * mass_inverse @ generalized(force))
        elastic = trial_force - stiffness[:, None] * dt * predicted_velocity
        self.elastic[index] = _limit_force(elastic, normal_limit)
        self.previous[index] = positions + dt * predicted_velocity
        return force
