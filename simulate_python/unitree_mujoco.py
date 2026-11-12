import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

mj_model.opt.timestep = config.SIMULATE_DT
sand = None
if config.USE_RFT_SAND:
    # Import from the repository root when launched from simulate_python.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from terrain_tool.sand import RFTSand, SoilDomainError

    sand = RFTSand(soil_size=config.SAND_SOIL_SIZE,
                   surface_z=config.SAND_SURFACE_Z, depth=config.SAND_DEPTH)
    sand.bind(mj_model)


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        soil_error = None
        with locker:
            if config.ENABLE_ELASTIC_BAND:
                if elastic_band.enable:
                    mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                        mj_data.qpos[:3], mj_data.qvel[:3]
                    )
            if sand is None:
                mujoco.mj_step(mj_model, mj_data)
            else:
                try:
                    sand.step(mj_data)
                except SoilDomainError as exc:
                    soil_error = exc
        if soil_error is not None:
            print(f"Sand simulation stopped: {soil_error}", flush=True)
            viewer.close()
            return

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
