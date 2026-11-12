import contextlib
from pathlib import Path
import runpy
import sys
import threading
from types import ModuleType
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from terrain_tool.sand import RFTMaterial, RFTSand, SoilDomainError
from terrain_tool.sand.display import SoilSurface, ViserSand
from terrain_tool.sand.geometry import bottom_heights, half_widths, primitive_panels


ROOT = Path(__file__).resolve().parents[3]


def sphere_scene(z=0.045, extra=""):
    return mujoco.MjModel.from_xml_string(f"""
      <mujoco><option timestep="0.002"/>
      <worldbody>{extra}<body pos="0 0 {z}"><freejoint/>
      <geom name="ball" type="sphere" size="0.05" mass="1"/>
      </body></worldbody></mujoco>""")


@pytest.mark.parametrize("kind,size", [
    ("sphere", "0.07"), ("ellipsoid", "0.07 0.04 0.09"),
    ("box", "0.07 0.04 0.09"), ("cylinder", "0.05 0.08"),
    ("capsule", "0.05 0.08"),
])
def test_primitive_bottom_matches_mujoco_rays(kind, size):
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><worldbody><geom type="{kind}" size="{size}"
      pos="0.1 -0.02 0.2" euler="0.3 0.8 0.5"/></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(71)
    xy = rng.uniform([-0.06, -0.18], [0.26, 0.14], size=(200, 2))
    expected = []
    for x, y in xy:
        distance = mujoco.mj_ray(model, data, np.array([x, y, -1.]), np.array([0., 0., 1.]),
                                 None, True, -1, None)
        expected.append(-1 + distance if distance >= 0 else np.inf)
    actual = bottom_heights(model, data, np.array([0]), xy[:, 0], xy[:, 1])[0]
    np.testing.assert_allclose(actual, expected, atol=2e-9)
    bounds = half_widths(model, data, np.array([0]))[0]
    hit = np.isfinite(actual)
    assert hit.any()
    assert np.all(np.abs(xy[hit] - data.geom_xpos[0, :2]) <= bounds[:2] + 1e-10)
    assert np.all(actual[hit] >= data.geom_xpos[0, 2] - bounds[2] - 1e-10)


@pytest.mark.parametrize("kind,size,area", [
    (mujoco.mjtGeom.mjGEOM_SPHERE, [.05, 0, 0], 4 * np.pi * .05**2),
    (mujoco.mjtGeom.mjGEOM_CYLINDER, [.05, .08, 0], 2 * np.pi * .05 * .16 + 2 * np.pi * .05**2),
    (mujoco.mjtGeom.mjGEOM_CAPSULE, [.05, .08, 0], 2 * np.pi * .05 * .16 + 4 * np.pi * .05**2),
    (mujoco.mjtGeom.mjGEOM_BOX, [.05, .08, .1], 8 * (.05 * .08 + .08 * .1 + .1 * .05)),
])
def test_closed_panel_area_and_zero_net_normal(kind, size, area):
    panels = primitive_panels(kind, np.array(size), 8, 8)
    assert panels.areas.sum() == pytest.approx(area, rel=1e-12)
    np.testing.assert_allclose((panels.normals * panels.areas[:, None]).sum(axis=0), 0, atol=1e-12)


def test_sphere_contact_supports_and_preserves_user_force():
    model = sphere_scene()
    data = mujoco.MjData(model)
    sand = RFTSand()
    sand.bind(model)
    data.qvel[:3] = [.15, 0, -.1]
    data.qfrc_applied[:] = np.arange(model.nv) * .02
    applied = data.qfrc_applied.copy()
    sand.step(data)
    assert data.time == pytest.approx(.002)
    assert sand.geom_forces[0, 2] > 0
    assert sand.geom_forces[0, 0] <= 1e-12
    np.testing.assert_array_equal(data.qfrc_applied, applied)
    for _ in range(120):
        sand.step(data)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert data.qpos[2] > -.02
    np.testing.assert_array_equal(data.qfrc_applied, applied)


def test_reset_time_rebuilds_contact_history():
    model = sphere_scene()
    data = mujoco.MjData(model)
    sand = RFTSand()
    sand.bind(model)
    for _ in range(5):
        sand.step(data)
    revision = sand.revision
    mujoco.mj_resetData(model, data)
    sand.step(data)
    assert sand.revision == revision + 1
    expected = mujoco.MjData(model)
    fresh = RFTSand()
    fresh.bind(model)
    fresh.step(expected)
    np.testing.assert_allclose(data.qpos, expected.qpos, atol=1e-14)
    np.testing.assert_allclose(data.qvel, expected.qvel, atol=1e-14)


def test_teleport_without_time_reset_clears_history():
    model = sphere_scene()
    data = mujoco.MjData(model)
    sand = RFTSand()
    sand.bind(model)
    for _ in range(4):
        sand.step(data)
    data.qpos[2] += .01
    expected = mujoco.MjData(model)
    expected.qpos[:] = data.qpos
    expected.qvel[:] = data.qvel
    expected.time = data.time
    fresh = RFTSand()
    fresh.bind(model)
    revision = sand.revision
    sand.step(data)
    fresh.step(expected)
    assert sand.revision == revision + 1
    np.testing.assert_allclose(data.qpos, expected.qpos, atol=1e-14)
    np.testing.assert_allclose(data.qvel, expected.qvel, atol=1e-14)


def test_force_restored_even_when_mujoco_step_fails(monkeypatch):
    model = sphere_scene()
    data = mujoco.MjData(model)
    sand = RFTSand()
    sand.bind(model)
    data.qfrc_applied[:] = 1
    def fail(*args):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(mujoco, "mj_step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        sand.step(data)
    np.testing.assert_array_equal(data.qfrc_applied, np.ones(model.nv))


def test_domain_and_floor_are_not_silently_ignored():
    with pytest.raises(ValueError, match="rigid plane"):
        RFTSand().bind(sphere_scene(extra='<geom type="plane" size="1 1 .1"/>'))
    model = sphere_scene()
    data = mujoco.MjData(model)
    sand = RFTSand(soil_size=.4, depth=.2)
    sand.bind(model)
    data.qpos[0] = .19
    with pytest.raises(SoilDomainError, match="finite sand domain"):
        sand.step(data)
    data.qpos[0], data.qpos[2] = 0, -.16
    with pytest.raises(SoilDomainError, match="hard bottom"):
        sand.step(data)


def test_display_is_transient_and_does_not_change_physics():
    model = sphere_scene()
    data = mujoco.MjData(model)
    sand = RFTSand()
    sand.bind(model)
    mujoco.mj_forward(model, data)
    surface = SoilSurface(sand)
    before = data.qpos.copy()
    vertices, faces, depth = surface.mesh(data)
    assert depth == pytest.approx(.005, abs=2e-4)
    assert faces.shape[1] == 3 and np.max(faces) < len(vertices)
    np.testing.assert_array_equal(data.qpos, before)
    assert sand.contact.previous is None
    data.qpos[2] = .2
    mujoco.mj_forward(model, data)
    vertices, faces, depth = surface.mesh(data)
    assert depth == 0 and vertices.shape == (4, 3) and faces.shape == (2, 3)


def test_viser_callback_refreshes_paused_edits_and_visibility(monkeypatch):
    model = sphere_scene(z=.2)
    data = mujoco.MjData(model)
    sand = RFTSand()
    sand.bind(model)
    clock = [10.]
    monkeypatch.setattr("terrain_tool.sand.display.time.monotonic", lambda: clock[0])
    class Scene:
        geom_groups_visible = [True] * 6
        calls = 0
        def update_from_mjdata(self, data):
            self.calls += 1
    class Server:
        def __init__(self):
            self.handle = None
            self.gui = SimpleNamespace(add_number=lambda *a, **k: SimpleNamespace(value=0),
                                       add_markdown=lambda *a, **k: None)
            self.scene = SimpleNamespace(add_mesh_simple=self.add_mesh)
        def add_mesh(self, name, vertices, faces, **kwargs):
            self.handle = SimpleNamespace(vertices=vertices, faces=faces, visible=True)
            return self.handle
        def atomic(self):
            return contextlib.nullcontext()
    server, scene = Server(), Scene()
    draw = ViserSand(sand, data, server)
    data.qacc[:] = np.arange(model.nv)
    old_acc = data.qacc.copy()
    draw(scene)
    np.testing.assert_array_equal(data.qacc, old_acc)
    assert server.handle.vertices.shape == (4, 3)
    data.qpos[2] = .045  # paused pose edit must bypass the wall-time throttle
    draw(scene)
    assert server.handle.vertices.shape[0] > 4 and draw.depth.value > 0
    scene.geom_groups_visible[1] = False
    draw(scene)
    assert not server.handle.visible and scene.calls == 3
    data.qpos[2] = .2
    sand.reset()
    draw(scene)
    assert server.handle.vertices.shape == (4, 3) and draw.depth.value == 0


def test_go2_sand_scene_runs_without_rigid_ground():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "unitree_robots/go2/scene_sand.xml"))
    data = mujoco.MjData(model)
    sand = RFTSand()
    sand.bind(model)
    assert len(sand.geom_ids) == 23
    assert all(model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE for i in range(model.ngeom))
    for _ in range(200):
        sand.step(data)
    assert np.isfinite(data.qpos).all() and data.qpos[2] > 0
    assert np.max(sand.geom_forces[:, 2]) > 0


def test_material_validation():
    with pytest.raises(ValueError):
        RFTMaterial(0, 1, 1450, .00025, .3)


def fake_sdk(monkeypatch, config):
    calls = []
    package = ModuleType("unitree_sdk2py")
    package.__path__ = []
    core = ModuleType("unitree_sdk2py.core")
    core.__path__ = []
    channel = ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactoryInitialize = lambda domain, interface: calls.append((domain, interface))
    bridge = ModuleType("unitree_sdk2py_bridge")
    bridge.UnitreeSdk2Bridge = lambda model, data: SimpleNamespace(model=model, data=data)
    bridge.ElasticBand = object
    for name, module in [("unitree_sdk2py", package), ("unitree_sdk2py.core", core),
                         ("unitree_sdk2py.core.channel", channel), ("unitree_sdk2py_bridge", bridge),
                         ("config", config)]:
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def sdk_config():
    return SimpleNamespace(
        ROBOT="go2", ROBOT_SCENE=str(ROOT / "unitree_robots/go2/scene_sand.xml"),
        DOMAIN_ID=3, INTERFACE="lo", USE_JOYSTICK=False, PRINT_SCENE_INFORMATION=False,
        ENABLE_ELASTIC_BAND=False, SIMULATE_DT=.002, VIEWER_DT=.02,
        USE_RFT_SAND=True, SAND_SOIL_SIZE=4., SAND_SURFACE_Z=0., SAND_DEPTH=.5,
    )


@pytest.mark.parametrize("enable_sand", [False, True])
def test_native_simulator_steps_with_or_without_sand(monkeypatch, enable_sand):
    import mujoco.viewer
    config = sdk_config()
    config.USE_RFT_SAND = enable_sand
    calls = fake_sdk(monkeypatch, config)
    states = iter([True, False])
    viewer = SimpleNamespace(is_running=lambda: next(states))
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *args, **kwargs: viewer)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    program = runpy.run_path(str(ROOT / "simulate_python/unitree_mujoco.py"), run_name="native_sand_test")
    program["SimulationThread"]()
    assert program["mj_data"].time == pytest.approx(.002)
    assert (program["sand"] is not None) == enable_sand
    assert calls == [(3, "lo")]


def test_browser_sdk_entry_steps_and_resets_soil(monkeypatch):
    from terrain_tool.sand.__main__ import main

    config = sdk_config()
    calls = fake_sdk(monkeypatch, config)
    created = []
    class Server:
        def __init__(self, **kwargs):
            self.gui = SimpleNamespace(add_number=lambda *a, **k: SimpleNamespace(value=0),
                                       add_markdown=lambda *a, **k: None)
            self.scene = SimpleNamespace(add_mesh_simple=lambda name, vertices, faces, **kwargs:
                                         SimpleNamespace(vertices=vertices, faces=faces, visible=True))
        def atomic(self):
            return contextlib.nullcontext()
    class Viewer:
        def __init__(self, model, data, step_fn, render_fn, reset_fn, server):
            self.model, self.data = model, data
            self.step_fn, self.render_fn, self.reset_fn = step_fn, render_fn, reset_fn
            self._lock = threading.Lock()
            self.scene = SimpleNamespace(geom_groups_visible=[True] * 6, update_from_mjdata=lambda data: None)
            created.append(self)
        def _render(self):
            self.render_fn(self.scene)
        def run(self):
            self.step_fn(self.model, self.data)
            self._render()
            sand = self.render_fn.surface.sand
            rev = sand.revision
            self._set_joint_qpos(7, .2)
            assert sand.revision == rev + 1
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            key_qpos = self.data.qpos.copy()
            self.reset_fn(self.model, self.data)
            np.testing.assert_array_equal(self.data.qpos, key_qpos)
            self._reset()
            np.testing.assert_array_equal(self.data.qpos, self.model.qpos0)
    monkeypatch.setitem(sys.modules, "viser", SimpleNamespace(ViserServer=Server))
    monkeypatch.setitem(sys.modules, "mjviser", SimpleNamespace(Viewer=Viewer))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["sand", "--sdk"])
    assert main() == 0
    assert len(created) == 1 and calls == [(3, "lo")]
