"""
Test script for utils.py
Demonstrates SE(3) interpolation using Quaternion SLERP and SO(3) geodesic interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import QuaternionUtils, SO3Utils, SE3Interpolator
from scipy.spatial.transform import Rotation as R
from pose_interpolator import PoseInterpolator

def plot_3d_trajectory_with_tcp(direct_traj, slerp_traj, step=1, axis_len=0.05):
    """
    绘制 Direct 与 SLERP 轨迹，并在采样点绘制 TCP 坐标系（X红，Y绿，Z蓝）

    direct_traj, slerp_traj: [(position, rotation_matrix), ...]
    step: 绘制姿态的步长
    axis_len: 坐标系箭头长度
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取位置数据
    pos_direct = np.array([p for p, _ in direct_traj])
    pos_slerp = np.array([p for p, _ in slerp_traj])

    # 绘制轨迹线
    ax.plot(pos_direct[:, 0], pos_direct[:, 1], pos_direct[:, 2],
            'r-', label='Direct Trajectory', linewidth=2)
    ax.plot(pos_slerp[:, 0], pos_slerp[:, 1], pos_slerp[:, 2],
            'b-', label='SLERP Trajectory', linewidth=2)

    # 绘制 TCP 坐标系
    def draw_tcp_axes(ax, pos, rot_matrix, length=axis_len, alpha=0.8):
        rot = R.from_matrix(rot_matrix)
        # 三个方向单位向量
        axes = np.eye(3) * length
        colors = ['r', 'g', 'b']  # X红, Y绿, Z蓝
        for vec, c in zip(axes, colors):
            dir_vec = rot.apply(vec)  # 旋转到全局
            ax.quiver(pos[0], pos[1], pos[2],
                      dir_vec[0], dir_vec[1], dir_vec[2],
                      color=c, linewidth=1.5, alpha=alpha)

    for i in range(0, len(direct_traj), step):
        draw_tcp_axes(ax, *direct_traj[i], length=axis_len, alpha=0.8)

    for i in range(0, len(slerp_traj), step):
        draw_tcp_axes(ax, *slerp_traj[i], length=axis_len, alpha=0.5)

    # 设置标签与标题
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Direct vs SLERP Trajectories with TCP Frames', fontsize=14)
    ax.legend()

    # 让三个轴比例相同，避免箭头长度看起来不同
    def set_axes_equal(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        spans = limits[:, 1] - limits[:, 0]
        centers = np.mean(limits, axis=1)
        max_span = max(spans)
        for ctr, axis in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
            axis([ctr - max_span / 2, ctr + max_span / 2])

    set_axes_equal(ax)
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()

def utils_test(p0, p1, q0, q1, n_steps):
    # -------------------------
    # Interpolation with SLERP
    # -------------------------
    se3_slerp = SE3Interpolator.interpolate_se3_with_slerp(p0, p1, q0, q1, n_steps)
    positions_slerp = np.array([pos for pos, _ in se3_slerp])

    # -------------------------
    # Interpolation with SO(3) geodesic
    # -------------------------
    se3_so3 = SE3Interpolator.interpolate_se3_with_so3(p0, p1, r0, r1, n_steps)
    positions_so3 = np.array([pos for pos, _ in se3_so3])

    # -------------------------
    # Print some results
    # -------------------------
    print("SLERP interpolation:")
    for i, (pos, rot) in enumerate(se3_slerp):
        print(f"Step {i}: Position {pos}, Rotation matrix:\n{rot}\n")

    print("SO(3) geodesic interpolation:")
    for i, (pos, rot) in enumerate(se3_so3):
        print(f"Step {i}: Position {pos}, Rotation matrix:\n{rot}\n")

    # -------------------------
    # Optional: visualize position trajectories
    # -------------------------
    # 绘图
    plot_3d_trajectory_with_tcp(se3_so3, se3_slerp)

def pose_interpolator_test1(p0, p1, q0, q1, r0, r1, n_steps):
    slerp_inter = PoseInterpolator("slerp")
    so3_inter = PoseInterpolator("so3")

    trajectory_slerp = slerp_inter.interpolate_se3_with_slerp(p0, p1, q0, q1, n_steps)
    trajectory_so3 = so3_inter.interpolate_se3_with_so3(p0, p1, r0, r1, n_steps)

    plot_3d_trajectory_with_tcp(trajectory_slerp, trajectory_so3)

def pose_interpolator_test2(p0, p1, q0, q1, r0, r1, n_steps):
    slerp_inter = PoseInterpolator("slerp")
    so3_inter = PoseInterpolator("so3")

    start_pose_quat = (p0, q0)
    end_pose_quat = (p1, q1)

    start_pose_rot = (p0,r0)
    end_pose_rot = (p1,r1)


    trajectory_slerp = slerp_inter.interpolate_pose(start_pose_rot,end_pose_rot, n_steps, "slerp")
    trajectory_so3 = so3_inter.interpolate_pose(start_pose_quat,end_pose_quat, n_steps, "so3")

    plot_3d_trajectory_with_tcp(trajectory_slerp, trajectory_so3)

if __name__ == "__main__":
    # -------------------------
    # Define start and end poses
    # -------------------------
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 1.0, 1.0])

    # Quaternion: [x, y, z, w]
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = R.from_euler('xyz', [90, 45, 30], degrees=True).as_quat()

    # Rotation matrices for SO(3) test
    r0 = R.from_quat(q0).as_matrix()
    r1 = R.from_quat(q1).as_matrix()

    n_steps = 20

    # utils_test(p0, p1, q0, q1, r0, r1, n_steps)
    # pose_interpolator_test1(p0, p1, q0, q1, r0, r1, n_steps)
    pose_interpolator_test2(p0, p1, q0, q1, r0, r1, n_steps)

