"""Primitive bounds, vertical intersections and closed-surface quadrature."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class SurfacePanels:
    positions: np.ndarray
    normals: np.ndarray
    areas: np.ndarray


SUPPORTED = {int(getattr(mujoco.mjtGeom, f"mjGEOM_{name}"))
             for name in ("SPHERE", "ELLIPSOID", "BOX", "CYLINDER", "CAPSULE")}


def half_widths(model, data, ids):
    """World AABB half extents; cylinder uses a sum, not an ellipsoid bound."""
    kinds, size = model.geom_type[ids], model.geom_size[ids]
    rotation = data.geom_xmat[ids].reshape(-1, 3, 3)
    axis = np.abs(rotation[:, :, 2])
    radius, half_length = size[:, :1], size[:, 1:2]
    widths = radius * np.sqrt(np.maximum(1 - axis ** 2, 0)) + half_length * axis  # cylinder
    capsule = kinds == mujoco.mjtGeom.mjGEOM_CAPSULE
    widths[capsule] = radius[capsule] + half_length[capsule] * axis[capsule]
    sphere = kinds == mujoco.mjtGeom.mjGEOM_SPHERE
    widths[sphere] = radius[sphere]
    box = kinds == mujoco.mjtGeom.mjGEOM_BOX
    widths[box] = (np.abs(rotation[box]) @ size[box, :, None])[..., 0]
    ellipsoid = kinds == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    widths[ellipsoid] = np.sqrt((rotation[ellipsoid] ** 2) @ (size[ellipsoid, :, None] ** 2))[..., 0]
    return widths


def bottom_heights(model, data, ids, x, y):
    """Exact world z of the first hit from below; inf outside each solid.

    Sphere/box/cylinder equations match the reference lower envelopes. Capsule
    is a cylinder union two spheres; ellipsoid uses a scaled sphere quadratic.
    """
    heights = np.full((len(ids), len(x)), np.inf)
    for i, geom in enumerate(ids):
        kind, size = int(model.geom_type[geom]), model.geom_size[geom]
        center = data.geom_xpos[geom]
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            rr = (x - center[0]) ** 2 + (y - center[1]) ** 2
            hit = rr < size[0] ** 2
            heights[i, hit] = center[2] - np.sqrt(size[0] ** 2 - rr[hit])
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        a = (np.column_stack((x, y, np.zeros(len(x)))) - center) @ rotation
        direction = rotation[2]
        lower, upper = np.full(len(x), -np.inf), np.full(len(x), np.inf)
        valid = np.ones(len(x), dtype=bool)

        def slab(axis, extent):
            nonlocal lower, upper, valid
            if abs(direction[axis]) < 1e-12:
                valid &= abs(a[:, axis]) <= extent
            else:
                first = (-extent - a[:, axis]) / direction[axis]
                second = (extent - a[:, axis]) / direction[axis]
                lower = np.maximum(lower, np.minimum(first, second))
                upper = np.minimum(upper, np.maximum(first, second))

        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            for axis in range(3):
                slab(axis, size[axis])
        else:
            if kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                b, d, radius = a / size, direction / size, 1.
            else:
                b, d, radius = a[:, :2], direction[:2], size[0]
            aa = np.sum(d ** 2)
            bb = 2 * (b @ d)
            cc = np.sum(b ** 2, axis=1) - radius ** 2
            if aa < 1e-12:
                valid &= cc <= 0
            else:
                disc = bb ** 2 - 4 * aa * cc
                valid &= disc >= 0
                root = np.sqrt(np.maximum(disc, 0))
                lower = np.maximum(lower, (-bb - root) / (2 * aa))
                upper = np.minimum(upper, (-bb + root) / (2 * aa))
            if kind != int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                slab(2, size[1])
        valid &= lower <= upper
        heights[i, valid] = lower[valid]
        if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            for sign in (-1, 1):
                cap = center + sign * size[1] * rotation[:, 2]
                rr = (x - cap[0]) ** 2 + (y - cap[1]) ** 2
                hit = rr < size[0] ** 2
                heights[i, hit] = np.minimum(heights[i, hit], cap[2] - np.sqrt(size[0] ** 2 - rr[hit]))
    return heights


def sphere_panels(radius, rings, sectors):
    """Gauss-Legendre area quadrature, exact sphere area and mirror pairs."""
    z, w = np.polynomial.legendre.leggauss(rings)
    phi = (np.arange(sectors)+.5)*2*np.pi/sectors
    normals = np.stack(np.broadcast_arrays(np.sqrt(1-z[:, None]**2)*np.cos(phi),
                       np.sqrt(1-z[:, None]**2)*np.sin(phi), z[:, None]), axis=-1).reshape(-1, 3)
    areas = np.broadcast_to(w[:, None]*2*np.pi*radius**2/sectors, (rings, sectors)).ravel().copy()
    return SurfacePanels(radius*normals, normals, areas)


def cylinder_panels(radius, half_length, rings, sectors):
    """Closed cylinder side and two disk caps, in the cylinder's local frame."""
    u, w = np.polynomial.legendre.leggauss(rings)
    phi = (np.arange(sectors)+.5)*2*np.pi/sectors
    side = np.stack(np.broadcast_arrays(radius*np.cos(phi)[None], radius*np.sin(phi)[None],
                                       half_length*u[:, None]), axis=-1).reshape(-1, 3)
    normal = side.copy()
    normal[:, 2] = 0.
    normal /= radius
    area = np.broadcast_to(w[:, None]*half_length*radius*2*np.pi/sectors, (rings, sectors)).ravel()
    points, normals, areas = [side], [normal], [area]
    rad = radius*np.sqrt((u+1)/2)
    for sign in (-1., 1.):
        cap = np.stack(np.broadcast_arrays(rad[:, None]*np.cos(phi), rad[:, None]*np.sin(phi),
                                           np.full((rings, sectors), sign*half_length)), axis=-1).reshape(-1, 3)
        n = np.zeros_like(cap)
        n[:, 2] = sign
        points.append(cap)
        normals.append(n)
        areas.append(np.broadcast_to(w[:, None]*np.pi*radius**2/(2*sectors), (rings, sectors)).ravel())
    return SurfacePanels(np.concatenate(points), np.concatenate(normals), np.concatenate(areas))


def box_panels(half_sizes, rings, sectors):
    """Tensor Gaussian quadrature on the six faces of an axis-aligned box."""
    u, wu = np.polynomial.legendre.leggauss(rings)
    v, wv = np.polynomial.legendre.leggauss(sectors)
    positions, normals, areas = [], [], []
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        for sign in (-1., 1.):
            point = np.zeros((rings, sectors, 3))
            normal = np.zeros_like(point)
            point[:, :, axis] = sign * half_sizes[axis]
            normal[:, :, axis] = sign
            point[:, :, other[0]] = u[:, None] * half_sizes[other[0]]
            point[:, :, other[1]] = v[None, :] * half_sizes[other[1]]
            positions.append(point.reshape(-1, 3))
            normals.append(normal.reshape(-1, 3))
            areas.append((wu[:, None] * wv[None, :] * half_sizes[other[0]] * half_sizes[other[1]]).ravel())
    return SurfacePanels(np.concatenate(positions), np.concatenate(normals), np.concatenate(areas))


def primitive_panels(kind, size, rings, sectors):
    """Closed primitive quadrature in the geom frame, including area normals."""
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return box_panels(size, rings, sectors)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return cylinder_panels(size[0], size[1], rings, sectors)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        cylinder = cylinder_panels(size[0], size[1], rings, sectors)
        nside = rings * sectors
        u, w = np.polynomial.legendre.leggauss(rings)
        phi = (np.arange(sectors) + .5) * 2 * np.pi / sectors
        points, normals, areas = [cylinder.positions[:nside]], [cylinder.normals[:nside]], [cylinder.areas[:nside]]
        for sign in (-1, 1):
            z = sign * (u + 1) / 2
            normal = np.stack(np.broadcast_arrays(np.sqrt(1-z[:, None]**2)*np.cos(phi),
                              np.sqrt(1-z[:, None]**2)*np.sin(phi), z[:, None]), axis=-1).reshape(-1, 3)
            point = size[0] * normal
            point[:, 2] += sign * size[1]
            points.append(point)
            normals.append(normal)
            areas.append(np.broadcast_to(w[:, None]*np.pi*size[0]**2/sectors, (rings, sectors)).ravel())
        return SurfacePanels(np.concatenate(points), np.concatenate(normals), np.concatenate(areas))
    sphere = sphere_panels(1., rings, sectors)
    axes = size if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else np.full(3, size[0])
    normal = sphere.normals / axes
    scale = np.linalg.norm(normal, axis=1)
    return SurfacePanels(sphere.positions * axes, normal / scale[:, None],
                         sphere.areas * np.prod(axes) * scale)


def inside_primitive(points, kind, size, tolerance=0.):
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return np.all(abs(points) <= size + tolerance, axis=1)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return ((np.linalg.norm(points[:, :2], axis=1) <= size[0] + tolerance)
                & (abs(points[:, 2]) <= size[1] + tolerance))
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        q = points.copy()
        q[:, 2] -= np.clip(q[:, 2], -size[1], size[1])
        return np.linalg.norm(q, axis=1) <= size[0] + tolerance
    axes = size if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else np.full(3, size[0])
    return np.linalg.norm(points / axes, axis=1) <= 1 + tolerance / min(axes)
