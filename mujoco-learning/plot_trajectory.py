# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial.transform import Rotation as R

# def plot_trajectories_with_orientation(file_paths, labels, arrow_interval=5):
#     # 创建 3D 图形
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     for file_path, label in zip(file_paths, labels):
#         traj_data = np.load(file_path, allow_pickle=True)  # 加载轨迹数据
#         positions = np.array([pos for pos, _ in traj_data])  # 提取位置
#         quaternions = np.array([quat for _, quat in traj_data])  # 提取四元数姿态

#         # 绘制位置曲线
#         ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f"{label} position")

#         # 绘制姿态箭头（每隔 arrow_interval 个点）
#         for i in range(0, len(positions), arrow_interval):
#             pos = positions[i]
#             rot = R.from_quat(quaternions[i]).as_matrix()  # 将四元数转为旋转矩阵
#             x_dir = rot[:, 0]  # 提取 X 轴方向
#             ax.quiver(pos[0], pos[1], pos[2], x_dir[0], x_dir[1], x_dir[2],
#                       length=0.05, color='r', label=f"{label} orientation" if i == 0 else "")

#     # 设置坐标轴标签和标题
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('direct_trajectory and slerp_trajectory')
#     ax.legend()
#     plt.show()

# # 使用示例
# file_paths = ["/home/zxy/MujocoBin/direct_trajectory_data.npy", "/home/zxy/MujocoBin/slerp_trajectory_data.npy"]  # 替换为你的文件路径
# labels = ["direct_trajectory", "slerp_trajectory"]
# plot_trajectories_with_orientation(file_paths, labels, arrow_interval=5)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def plot_trajectories_with_orientation(file_paths,
                                       labels,
                                       arrow_interval=5,
                                       arrow_len=0.05,
                                       axes='XYZ',   # 画哪几根轴：'X', 'Y', 'Z' 自由组合
                                       colors=None):
    """
    画 3D 轨迹并叠加朝向箭头。
    axes 举例：'XZ' 表示只画 X 轴和 Z 轴方向箭头。
    colors 举例：{'direct_trajectory':'tab:blue','slerp_trajectory':'tab:orange'}
    """
    if colors is None:
        colors = plt.cm.get_cmap('tab10').colors

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for idx, (file_path, label) in enumerate(zip(file_paths, labels)):
        traj_data = np.load(file_path, allow_pickle=True)
        positions = np.array([pos for pos, _ in traj_data])
        quaternions = np.array([quat for _, quat in traj_data])

        # —— 1. 画位置曲线 ——
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                label=f"{label} path", color=colors[idx % len(colors)])

        # —— 2. 画朝向箭头 ——
        for i in range(0, len(positions), arrow_interval):
            pos = positions[i]
            rot = R.from_quat(quaternions[i]).as_matrix()

            # 为每个要求的轴向画箭头
            for axis, color in zip('XYZ', ['r', 'g', 'b']):
                if axis not in axes:
                    continue
                vec = rot[:, 'XYZ'.index(axis)]  # 取出对应轴向
                ax.quiver(pos[0], pos[1], pos[2],
                          vec[0], vec[1], vec[2],
                          length=arrow_len, color=color, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    plt.show()

# ---------------- 使用示例 ----------------
file_paths = [
    "/home/zxy/MujocoBin/direct_trajectory_data.npy",
    "/home/zxy/MujocoBin/slerp_trajectory_data.npy"
]
labels = ["direct_trajectory", "slerp_trajectory"]

plot_trajectories_with_orientation(
    file_paths,
    labels,
    arrow_interval=5,
    arrow_len=0.05,
    axes='X',
    colors=['tab:blue', 'tab:orange']  # 列表而不是字典
)