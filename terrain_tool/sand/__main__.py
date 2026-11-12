"""Run the RFT sandbox with mjviser's browser viewer."""

import argparse
import sys
from pathlib import Path

import mujoco

from . import RFTSand, SoilDomainError
from .display import ViserSand


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Go2 transient RFT sand viewer")
    parser.add_argument("--scene", type=Path, default=ROOT / "unitree_robots/go2/scene_sand.xml")
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind address")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--soil-size", type=float, default=4., help="square domain width, m")
    parser.add_argument("--depth", type=float, default=.5, help="maximum intrusion depth, m")
    parser.add_argument("--surface-z", type=float, default=0., help="reference surface height, m")
    parser.add_argument("--timestep", type=float, default=.002, help="physics timestep, s")
    parser.add_argument("--sdk", action="store_true", help="connect the existing Unitree SDK2 Python bridge")
    args = parser.parse_args()
    if not 0 < args.timestep < float("inf"):
        parser.error("--timestep must be finite and positive")

    try:
        import viser
        from mjviser import Viewer
    except ImportError as exc:
        parser.error(f"browser viewer requires mjviser and viser: {exc}")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    model.opt.timestep = args.timestep
    data = mujoco.MjData(model)
    sand = RFTSand(surface_z=args.surface_z, soil_size=args.soil_size, depth=args.depth)
    sand.bind(model)
    mujoco.mj_forward(model, data)

    # Keep the bridge alive; its DDS workers publish sensors and set ctrl.
    bridge = None
    if args.sdk:
        sys.path.insert(0, str(ROOT / "simulate_python"))
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py_bridge import UnitreeSdk2Bridge
            import config
        except ImportError as exc:
            parser.error(f"--sdk needs the existing Python simulator dependencies: {exc}")
        ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
        bridge = UnitreeSdk2Bridge(model, data)
        if config.USE_JOYSTICK:
            bridge.SetupJoystick(device_id=config.JOYSTICK_DEVICE, js_type=config.JOYSTICK_TYPE)
        if config.PRINT_SCENE_INFORMATION:
            bridge.PrintSceneInformation()

    server = viser.ViserServer(host=args.host, port=args.port, label="Unitree RFT sand")
    render = ViserSand(sand, data, server)

    def reset_soil(model, data):
        # mjviser also calls this after loading a keyframe; preserve that pose.
        sand.reset()
        mujoco.mj_forward(model, data)

    class SandViewer(Viewer):
        def _reset(self):
            mujoco.mj_resetData(self.model, self.data)
            reset_soil(self.model, self.data)

        def _set_joint_qpos(self, qpos_adr, value):
            # Joint sliders are teleports, not physical steps.
            with self._lock:
                self.data.qpos[qpos_adr] = value
                reset_soil(self.model, self.data)
                self._render()

    viewer = SandViewer(model, data, step_fn=lambda model, data: sand.step(data),
                        render_fn=render, reset_fn=reset_soil, server=server)
    print(f"RFT: {len(sand.geom_ids)} collision geoms, {len(sand.panels.areas)} panels; "
          f"surface z={sand.surface_z:g} m, size={sand.soil_size:g} m, depth={sand.depth:g} m")
    if bridge is None:
        print("No controller is attached. Use --sdk for the Unitree bridge.")
    try:
        viewer.run()
    except SoilDomainError as exc:
        print(f"Sand simulation stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
