import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
import time

URDF_PATH = '/home/zxy/MujocoBin/mujoco-learning/model/franka_panda_description/robots/panda_description/urdf/panda.urdf'
XML_PATH = '/home/zxy/MujocoBin/mujoco_menagerie-main/franka_emika_panda/scene.xml'

# 初始化 MuJoCo 和 Pinocchio 模型
model_mj = mujoco.MjModel.from_xml_path(XML_PATH)
data_mj = mujoco.MjData(model_mj)

model_pin = pin.buildModelFromUrdf(URDF_PATH)
data_pin = model_pin.createData()

# 初始化 viewer
viewer = mujoco.viewer.launch_passive(model_mj, data_mj)

# 初始静态配置（合理姿态）
q_init = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8])
n_joints = 7

# 动态扰动参数
amplitude = 0.05  # 小幅度
frequency = 0.5   # Hz


start_time = time.time()
while viewer.is_running():
    t = time.time() - start_time

    # 在初始角度上叠加正弦扰动
    delta = amplitude * np.sin(2 * np.pi * frequency * t + np.linspace(0, np.pi, n_joints))
    q = q_init + delta

    # 构造完整 q，手指默认开口 0.04
    q_full = np.concatenate([q, [0.04, 0.04]])

    # Pinocchio FK
    pin.forwardKinematics(model_pin, data_pin, q_full)
    ee_transform = data_pin.oMi[7]  # panda_link8

    # 输出末端齐次变换
    print("\nTime: {:.2f}s".format(t))
    print("End-effector transform (Pinocchio):\n", ee_transform)

    # 更新 MuJoCo
    data_mj.qpos[:9] = q_full
    mujoco.mj_forward(model_mj, data_mj)

    viewer.sync()
    time.sleep(0.01)
