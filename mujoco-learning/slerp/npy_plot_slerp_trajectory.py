import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def set_axes_equal(ax):
    """让3D坐标轴比例一致，防止箭头长短错觉"""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    max_span = max(spans)
    for ctr, axis in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        axis([ctr - max_span / 2, ctr + max_span / 2])

def plot_trajectories_with_tcp(file_paths, labels,
                               arrow_interval=5,
                               axis_len=0.05,
                               axes='XYZ'):
    """
    从npy文件读取轨迹并绘制，包含TCP坐标系（X红，Y绿，Z蓝）

    file_paths: [direct_path, slerp_path]
    labels: ["direct", "slerp"]
    arrow_interval: 每隔多少个点画一次姿态
    axis_len: 坐标系箭头长度
    axes: 哪些轴要画，'XYZ'表示都画
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors_line = ['tab:blue', 'tab:orange']
    axis_colors = {'X': 'r', 'Y': 'g', 'Z': 'b'}

    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        traj_data = np.load(file_path, allow_pickle=True)  # [(pos, quat), ...]
        positions = np.array([p for p, _ in traj_data])
        quaternions = np.array([q for _, q in traj_data])

        # 画轨迹线
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                label=f"{label} trajectory",
                color=colors_line[idx % len(colors_line)],
                linewidth=2)

        # 每隔 arrow_interval 画 TCP 坐标系
        for i in range(0, len(positions), arrow_interval):
            pos = positions[i]
            rot_matrix = R.from_quat(quaternions[i]).as_matrix()
            for axis_name in axes:
                vec = rot_matrix[:, 'XYZ'.index(axis_name)] * axis_len
                ax.quiver(pos[0], pos[1], pos[2],
                          vec[0], vec[1], vec[2],
                          color=axis_colors[axis_name], linewidth=1.5)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Direct vs SLERP Trajectories with TCP Frames', fontsize=14)
    ax.legend()
    set_axes_equal(ax)
    plt.show()


# ---------------- 使用示例 ----------------
file_paths = [
    "/home/zxy/MujocoBin/Data/NPY/direct_trajectory_data.npy",
    "/home/zxy/MujocoBin/Data/NPY/slerp_trajectory_data.npy"
]
labels = ["direct_trajectory", "slerp_trajectory"]

plot_trajectories_with_tcp(
    file_paths,
    labels,
    arrow_interval=5,
    axis_len=0.02,
    axes='XYZ'  # 画全部三轴
)
