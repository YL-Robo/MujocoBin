# PoseInterpolator - 机器人姿态插值器

## 概述

`PoseInterpolator` 是一个专为机器人应用设计的综合姿态插值类。它提供了多种插值方法，包括四元数SLERP和SO(3)测地线插值，用于生成平滑的机器人末端执行器轨迹。

## 主要功能

- **四元数SLERP插值**: 使用球面线性插值进行姿态插值
- **SO(3)测地线插值**: 在SO(3)群上进行测地线插值
- **SE(3)姿态插值**: 位置线性插值 + 姿态插值
- **多路点轨迹生成**: 支持通过多个路点生成完整轨迹
- **可视化功能**: 3D轨迹绘制和比较
- **灵活的输入格式**: 支持元组和字典格式的位姿表示

## 安装依赖

```bash
pip install numpy scipy matplotlib
```

## 基本用法

### 1. 创建插值器

```python
from pose_interpolator import PoseInterpolator

# 使用SO(3)测地线插值（默认）
interpolator = PoseInterpolator(interpolation_method="so3")

# 或使用四元数SLERP
interpolator = PoseInterpolator(interpolation_method="slerp")
```

### 2. 基本姿态插值

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# 定义起始和结束位姿
start_pos = np.array([0.0, 0.0, 0.0])
end_pos = np.array([1.0, 1.0, 1.0])

start_rot = np.eye(3)  # 单位旋转矩阵
end_rot = R.from_euler('z', 90, degrees=True).as_matrix()  # 绕Z轴旋转90度

# 生成轨迹
trajectory = interpolator.interpolate_pose(
    start_pose=(start_pos, start_rot),
    end_pose=(end_pos, end_rot),
    n_steps=50
)

# 可视化轨迹
interpolator.plot_trajectory(trajectory, title="My Trajectory")
```

### 3. 使用四元数

```python
# 使用四元数表示姿态
start_quat = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
end_quat = R.from_euler('x', 45, degrees=True).as_quat()

trajectory = interpolator.interpolate_pose(
    start_pose=(start_pos, start_quat),
    end_pose=(end_pos, end_quat),
    n_steps=50,
    method="slerp"
)
```

### 4. 使用字典格式

```python
# 使用字典格式定义位姿
start_pose = {
    'pos': np.array([0.0, 0.0, 0.0]),
    'rot': np.eye(3)
}

end_pose = {
    'pos': np.array([1.0, 0.0, 1.0]),
    'quat': R.from_euler('xyz', [30, 45, 60], degrees=True).as_quat()
}

trajectory = interpolator.interpolate_pose(
    start_pose=start_pose,
    end_pose=end_pose,
    n_steps=50
)
```



## 在机器人应用中的使用

### 与MuJoCo集成

```python
import mujoco
import pinocchio as pin

# 加载机器人模型
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# 创建插值器
interpolator = PoseInterpolator()

# 定义起始和结束关节角度
q_start = np.array([0.2, -0.5, 0.2, -1.7, 0.1, 1.6, 0.9])
q_end = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.8])

# 使用Pinocchio计算对应的位姿
model_pin = pin.buildModelFromUrdf("robot.urdf")
data_pin = model_pin.createData()

# 计算起始位姿
pin.forwardKinematics(model_pin, data_pin, q_start)
start_pos = data_pin.oMi[7].translation
start_rot = data_pin.oMi[7].rotation

# 计算结束位姿
pin.forwardKinematics(model_pin, data_pin, q_end)
end_pos = data_pin.oMi[7].translation
end_rot = data_pin.oMi[7].rotation

# 生成轨迹
trajectory = interpolator.interpolate_pose(
    start_pose=(start_pos, start_rot),
    end_pose=(end_pos, end_rot),
    n_steps=50
)

# 在MuJoCo中执行轨迹
for pos, rot in trajectory:
    # 使用逆运动学求解关节角度
    q_ik = solve_ik(model_pin, pos, rot, q_current)
    
    # 更新MuJoCo仿真
    data.qpos[:len(q_ik)] = q_ik
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
```



## API参考

### 主要方法

#### `__init__(interpolation_method="so3")`
初始化插值器。

**参数:**
- `interpolation_method`: 插值方法 ("slerp" 或 "so3")

#### `interpolate_pose(start_pose, end_pose, n_steps, method=None)`
在两个位姿之间进行插值。

**参数:**
- `start_pose`: 起始位姿（元组或字典格式）
- `end_pose`: 结束位姿（元组或字典格式）
- `n_steps`: 插值步数
- `method`: 插值方法（可选，默认使用初始化时的方法）

**返回:**
- 轨迹列表：`[(position, rotation_matrix), ...]`


#### `plot_trajectory(trajectory, step=5, axis_len=0.02, title="Pose Trajectory")`
绘制3D轨迹。


### 工具方法

#### `slerp_quaternion(q0, q1, t)`
四元数SLERP插值。

#### `rotmat_interp_geodesic(r1, r2, t)`
SO(3)测地线插值。

#### `hat(phi)`
斜对称算子。

#### `rodrigues_exp(axis, theta)`
Rodrigues公式。

#### `axis_angle_from_rotmat(rotmat)`
从旋转矩阵提取轴角表示。

## 数学背景

### 四元数SLERP
四元数SLERP（球面线性插值）在四元数空间中进行插值，确保插值路径在单位四元数球面上。

### SO(3)测地线插值
SO(3)测地线插值在旋转矩阵空间中进行插值，插值路径是SO(3)群上的测地线。

### SE(3)姿态插值
SE(3)姿态插值结合了位置的线性插值和姿态的插值，生成完整的6DOF轨迹。

## 示例运行

运行示例文件：

```bash
cd mujoco-learning
python3 pose_interpolator.py
```

这将展示所有基本用法和可视化效果。

## 注意事项

1. **数值稳定性**: 类中包含了数值稳定性的处理，如处理接近奇异的旋转。
2. **输入格式**: 支持多种输入格式，但建议保持一致性。
3. **性能**: 对于大量路点，考虑使用更高效的轨迹规划算法。
4. **可视化**: 可视化功能需要matplotlib，在无头环境中可能需要调整。

## 扩展

可以根据需要扩展以下功能：
- 添加更多插值方法（如样条插值）
- 支持速度约束
- 添加碰撞检测
- 集成更多机器人框架

