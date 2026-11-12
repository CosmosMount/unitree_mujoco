# RFT 瞬态沙地

本目录提供 Python 版 MuJoCo RFT 力模型和可选的 Viser 浏览器显示。沙地为有限正方形，世界坐标系 Z 轴向上，默认表面 `z=0 m`、边长 `4 m`、最大深度 `0.5 m`。`unitree_robots/go2/scene_sand.xml` 为不含刚性地面的 Go2 示例场景，Go2 脚端碰撞球半径为 0.022 m。

RFT 在每个物理步对碰撞形状表面的采样面元计算力，通过 MuJoCo 的雅可比映射到广义力。参考面的高度不演化；显示网格仅表示当前几何体侵入后的下包络，物体抬起后立即恢复平面。它不是永久足印、颗粒流或沙堆模拟，也不是可作为接触高度查询的变形地形。内部采用给定的滑移阻力公式和惯性修正，弹性历史与隐式阻尼是工程扩展；默认材料参数尚未针对特定沙样或机器人实验标定。

## 浏览器查看

在 `unitree_mujoco` 仓库根目录运行，安装依赖时使用所选 Python 环境：

```bash
python -m pip install numpy mujoco 'mjviser==0.0.14' 'viser==1.1.0'
python -m terrain_tool.sand
```

打开终端显示的 `http://127.0.0.1:8080`。浏览器提供暂停、单步、重置、关键帧及视角/模型分组控制；沙面对应 geom group 1，刷新上限为 20 Hz。可通过 `--port 端口`、`--scene 路径`、`--soil-size 米`、`--surface-z 米`、`--depth 米` 和 `--timestep 秒` 调整参数。默认只监听本机，如需在可信网络远程查看可显式指定 `--host 0.0.0.0`。

默认没有控制器，Go2 会在重力下倒下；浏览器显示不等于控制器验证。要通过已有 Unitree SDK2 Python bridge 使用控制程序，先按仓库主 README 安装 `unitree_sdk2py`、CycloneDDS 和 pygame，检查 `simulate_python/config.py` 中的 DDS 域、接口、机器人类型及手柄配置，再运行：

```bash
python -m terrain_tool.sand --sdk
```

浏览器入口的默认模型是 Go2，SDK 需与 `config.ROBOT` 和选用的场景匹配；按需关闭 `USE_JOYSTICK`。SDK 会正常收发该配置所指定 DDS 域中的消息。

## 原 Python 仿真器

可选地在 `simulate_python/config.py` 中设置：

```python
ROBOT = "go2"
ROBOT_SCENE = "../unitree_robots/go2/scene_sand.xml"
USE_RFT_SAND = True
SIMULATE_DT = 0.002
```

然后按仓库原说明从 `simulate_python` 运行 `python unitree_mujoco.py`。该入口仍使用 MuJoCo 原生查看器，只接入 RFT 物理，**不绘制**动态沙面；需要可视化时使用上面的浏览器入口。两种入口都调用同一 `RFTSand.step(data)`，不要在同一步同时调用普通 `mj_step` 或再次施加沙力。

## 作为库使用

```python
import mujoco
from terrain_tool.sand import RFTSand

model = mujoco.MjModel.from_xml_path("unitree_robots/go2/scene_sand.xml")
model.opt.timestep = 0.002
data = mujoco.MjData(model)
sand = RFTSand(soil_size=4.0, surface_z=0.0, depth=0.5)
sand.bind(model)

while data.time < 1.0:
    # 在这里设置 data.ctrl、data.qfrc_applied 等。
    sand.step(data)

# mj_resetData、传送或直接修改姿态后，清除接触历史。
mujoco.mj_resetData(model, data)
sand.reset()
```

`sand.step` 临时叠加沙力，步进后恢复调用者的 `qfrc_applied`；仿真时间倒退或两步之间直接修改 `qpos` 时，会自动清除历史。对于修改模型参数等其他不连续操作，调用者应显式 `sand.reset()`。`sand.surface_height(xy)` 返回参考平面，`sand.contact_envelope_height(data, xy)` 才返回仅供显示的瞬态几何包络；调用采样前需确保 MuJoCo 位置变换已更新。

每个并行的 `MjData` 都应使用独立的 `RFTSand` 实例，因为面元接触历史属于各自的仿真状态。默认材料参数为 `zeta=0.6`、`inertial_lambda=1.1`、`bulk_density=1450 kg/m³`、`elastic_displacement=0.00025 m`、`damping_ratio=0.3`；可以用 `RFTSand(material={...}, rings=8, sectors=8)` 替换整组材料参数和面元分辨率。

## 场景和模型限制

- 目前只支持 Euler 积分器和球、椭球、盒、圆柱、胶囊碰撞体。默认选择所有动态且 collision-enabled 的几何体；含其他碰撞类型的模型会报错，不能静默忽略。可用 `RFTSand(geom_ids=[...])` 显式限定，但未选中的几何体不会受到沙力。
- 必须移除与沙域重叠的刚性地面，否则会叠加两种支撑。模块会拒绝沙域中的水平刚性平面，但不能自动判断所有盒体、网格障碍或显式碰撞对是否重叠；自定义场景需要自行检查。示例场景没有 MuJoCo 地面碰撞，也没有实体硬底或侧墙。
- 几何体触及设定深度或浸入沙面时超出有限沙域会抛出 `SoilDomainError` 并停止仿真；不会自动切换到硬底或域外地面。显示网格按几何包围范围局部加密，采样过粗时最深点可能略低估。
- 默认 `near_margin=0.02 m`，2 ms 步长下假设单步位移不超过该缓冲距离（约 10 m/s）；高速运动应相应增大该参数或缩短步长。对实际任务应验证时间步长、面元密度、材料参数和控制稳定性，不应将示例参数直接视为真实沙地标定结果。
- 接触力模型只接入 Python 仿真；C++ `simulate` 入口不会自动使用此模块。

## 验证

```bash
python -m pip install pytest
python -m pytest terrain_tool/sand/tests -q
```

测试覆盖基本体与 MuJoCo 射线交点、面元面积、外力保留、瞬态沙面、边界和 Go2 无头步进。浏览器和 SDK 需要各自的运行时依赖；这些测试不替代控制器闭环、材料试验或完整行走场景验证。
