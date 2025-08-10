import numpy as np
import mujoco
import mujoco.viewer
import time

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser.urdf import treeFromUrdfModel

URDF_PATH = '/home/zxy/MujocoBin/mujoco-learning/model/franka_panda_description/robots/panda_description/urdf/panda.urdf'
XML_PATH = '/home/zxy/MujocoBin/mujoco_menagerie-main/franka_emika_panda/scene.xml'

# ---------- 构建 KDL 链 ----------
robot_urdf = URDF.from_xml_file(URDF_PATH)
ok, kdl_tree = treeFromUrdfModel(robot_urdf)
if not ok:
    raise RuntimeError("Failed to parse URDF into KDL Tree.")

# 从 base_link 到 panda_link8（你可以根据 URDF 中定义调整）
base_link = "panda_link0"
ee_link = "panda_link8"

kdl_chain = kdl_tree.getChain(base_link, ee_link)
fk_solver = kdl.ChainFkSolverPos_recursive(kdl_chain)

# ---------- 初始化 MuJoCo ----------
model_mj = mujoco.MjModel.from_xml_path(XML_PATH)
data_mj = mujoco.MjData(model_mj)

# ---------- Viewer ----------
viewer = mujoco.viewer.launch_passive(model_mj, data_mj)

# ---------- 初始关节位置 ----------
q_init = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8])  # 7 DoF
n_joints = 7

amplitude = 0.05
frequency = 0.5

# ---------- 主循环 ----------
start_time = time.time()
while viewer.is_running():
    t = time.time() - start_time

    # 添加正弦扰动
    delta = amplitude * np.sin(2 * np.pi * frequency * t + np.linspace(0, np.pi, n_joints))
    q = q_init + delta

    # 转换为 KDL joint array
    q_kdl = kdl.JntArray(n_joints)
    for i in range(n_joints):
        q_kdl[i] = q[i]

    # 使用 KDL FK
    ee_frame = kdl.Frame()
    fk_solver.JntToCart(q_kdl, ee_frame)

    # 获取齐次变换
    pos = ee_frame.p
    rot = ee_frame.M

    H = np.eye(4)
    H[:3, 3] = [pos[0], pos[1], pos[2]]
    H[:3, :3] = np.array([[rot[0, 0], rot[0, 1], rot[0, 2]],
                          [rot[1, 0], rot[1, 1], rot[1, 2]],
                          [rot[2, 0], rot[2, 1], rot[2, 2]]])

    # 打印末端变换
    print(f"\nTime: {t:.2f}s")
    print("End-effector transform (KDL):\n", H)

    # 更新 MuJoCo
    q_full = np.concatenate([q, [0.04, 0.04]])
    data_mj.qpos[:9] = q_full
    mujoco.mj_forward(model_mj, data_mj)

    viewer.sync()
    time.sleep(0.01)
