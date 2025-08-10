from pathlib import Path
from sys import argv
import mujoco
import mujoco_viewer
import numpy as np
import pinocchio
 
# Load the urdf model
pin_model = pinocchio.buildModelFromUrdf("model/franka_panda_description/robots/panda_description/urdf/panda.urdf")
print("model name: " + pin_model.name)

# Create data required by the algorithms
data = pin_model.createData()

# Sample a random configuration
q = pinocchio.randomConfiguration(pin_model)
print(f"q: {q.T}")
 
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(pin_model, data, q)

# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(pin_model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))

# 加载 MuJoCo 模型（假设已转换为 MJCF）
mj_path = "/home/zxy/MujocoBin/mujoco_menagerie-main/franka_emika_panda/scene.xml"  # 替换为实际 MJCF 文件路径
mj_model = mujoco.MjModel.from_xml_path(mj_path)
mj_data = mujoco.MjData(mj_model)

# Set joint positions (qpos)
mj_data.qpos[:mj_model.nq] = q[:mj_model.nq]
mujoco.mj_forward(mj_model, mj_data)

# Start viewer
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    print("Press ESC to exit viewer...")
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()

