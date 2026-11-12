ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 1 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.005  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

# Optional RFT physics for the native viewer. Select a scene without a rigid
# ground plane, e.g. ../unitree_robots/go2/scene_sand.xml. Native MuJoCo does
# not render the dynamic envelope; use `python -m terrain_tool.sand --sdk`
# from the repository root for the browser view.
USE_RFT_SAND = False
SAND_SOIL_SIZE = 4.0  # square side length (m)
SAND_SURFACE_Z = 0.0  # reference plane (m)
SAND_DEPTH = 0.5  # maximum permitted intrusion (m)
