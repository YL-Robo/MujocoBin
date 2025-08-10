import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import mujoco
import mujoco.viewer
import pinocchio as pin
from numpy.linalg import norm, solve
import time
import matplotlib.pyplot as plt

def Slerp_test(q0, q1, t):
    q0 = np.asarray(q0)
    q1 = np.asarray(q1)
    # t 转换为一个NumPy 数组，并将其形状调整为一列(column) 的二维数组
    t = np.asarray(t).reshape(-1, 1) 

    dot = np.dot(q0, q1)

    # If the quaternions are very close, use linear interpolation
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    # If the dot product is negative, negate q1 to take the shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    s0 = np.sin((1 - t) * theta) / sin_theta
    s1 = np.sin(t * theta) / sin_theta

    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)

# 插值函数
def interpolate_SE3(p0, p1, q0, q1, N, use_slerp=True):
    positions = np.linspace(p0, p1, N)
    if use_slerp:
        key_rots = R.from_quat([q0, q1])  # 起点和终点四元数转为旋转对象
        slerp = Slerp([0, 1], key_rots) # 创建 SLERP 插值器，关键帧时间为 0 和 1
        times = np.linspace(0, 1, N)    # 生成 N 个插值时间点
        interp_rots = slerp(times)   # 得到 N 个插值旋转
        return [(positions[i], interp_rots[i].as_matrix()) for i in range(N)]
    else:
        rot = R.from_quat(q0).as_matrix()  # 直接使用初始旋转
        return [(pos, rot) for pos in positions]

# IK 求解函数
def solve_ik_sequence(model_path, traj_interp, q_init=None):
    model = pin.buildModelFromUrdf(model_path)
    data = model.createData()
    JOINT_ID = 7  # 假设末端执行器关节ID为7

    eps = 1e-2      # 定义收敛阈值，当误差小于该值时认为算法收敛
    IT_MAX = 1000   # 定义最大迭代次数，防止算法陷入无限循环
    DT     = 1E-01  # 定义积分步长，用于更新关节角度
    damp   = 1e-14  # 定义阻尼因子，用于避免矩阵奇异

    if q_init is None:
        q_init = np.zeros(model.nq)
    q = q_init.copy()

    q_seq = []
    i = 0
    for pos, rot in traj_interp:
        oMdes = pin.SE3(rot, np.array(pos))
        success = False
        for i  in range(IT_MAX):  # 增加迭代次数到 1000
            pin.forwardKinematics(model, data, q)
            iMd = data.oMi[JOINT_ID].actInv(oMdes)
            err = pin.log(iMd).vector
            if norm(err) < eps:  # 减小收敛容差到 1e-2
                success = True
                break
            J = pin.computeJointJacobian(model, data, q, JOINT_ID)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(model, q, v * DT )

        if not success:
            print("Warning: IK did not converge.")
        q_seq.append(q.copy())
    return q_seq

# # 输入：轨迹数据（位置、旋转矩阵）元组列表
# def extract_euler_angles(traj_interp):
#     eulers = []
#     for pos, rot in traj_interp:
#         r = R.from_matrix(rot)
#         euler = r.as_euler('xyz', degrees=True)  # 使用 XYZ 欧拉角（单位：度）
#         eulers.append(euler)
#     return np.array(eulers)

# 绘制姿态曲线
def plot_3d_trajectory_with_orientations(direct_traj, slerp_traj, step=5):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract position data
    pos_direct = np.array([p for p, _ in direct_traj])
    pos_slerp = np.array([p for p, _ in slerp_traj])

    # Plot position trajectories
    ax.plot(pos_direct[:, 0], pos_direct[:, 1], pos_direct[:, 2], 'r-', label='Direct Trajectory', linewidth=2)
    ax.plot(pos_slerp[:, 0], pos_slerp[:, 1], pos_slerp[:, 2], 'b-', label='SLERP Trajectory', linewidth=2)

    # Plot single orientation vector (X-axis) for Direct trajectory
    for i in range(0, len(direct_traj), step):
        pos, rot_matrix = direct_traj[i]  # rot_matrix is a NumPy array
        rot = R.from_matrix(rot_matrix)   # Convert to Rotation object
        x_dir = rot.apply([0.03, 0.0, 0.0])   # X-axis vector, scaled for visibility
        ax.quiver(pos[0], pos[1], pos[2], x_dir[0], x_dir[1], x_dir[2], color='r', linewidth=1, alpha=0.8)

    # Plot single orientation vector (X-axis) for SLERP trajectory
    for i in range(0, len(slerp_traj), step):
        pos, rot_matrix = slerp_traj[i]  # rot_matrix is a NumPy array
        rot = R.from_matrix(rot_matrix)  # Convert to Rotation object
        x_dir = rot.apply([0.03, 0.0, 0.0])  # X-axis vector, scaled for visibility
        ax.quiver(pos[0], pos[1], pos[2], x_dir[0], x_dir[1], x_dir[2], color='b', linewidth=1, alpha=0.5)

    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Direct vs SLERP Trajectories with X-Axis Orientation', fontsize=14)

    # Add legend
    ax.legend()

    # Set view angle for better visualization
    ax.view_init(elev=30, azim=45)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# 自定义查看器类
class CustomViewer:
    def __init__(self, model, data, q_traj_direct, q_traj_slerp):
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.q_traj_direct = q_traj_direct  # 直接轨迹的关节角度序列
        self.q_traj_slerp = q_traj_slerp    # SLERP轨迹的关节角度序列
        self.model = model
        self.data = data
        self.traj_data_direct = []  # 存储直接轨迹的位置和姿态
        self.traj_data_slerp = []   # 存储SLERP轨迹的位置和姿态

    def run(self):
        try:
            # 播放直接轨迹
            print("Playing direct trajectory")
            for q in self.q_traj_direct:
                self.data.qpos[:len(q)] = q
                mujoco.mj_forward(self.model, self.data)
                # 记录末端执行器位置和姿态
                pos = self.data.xpos[-1].copy()  # 位置
                quat = self.data.xquat[-1].copy()  # 四元数
                self.traj_data_direct.append((pos, quat))
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(0.1)  # 增加延时到 0.1 秒

            # 保存直接轨迹数据
            np.save("direct_trajectory_data.npy", np.array(self.traj_data_direct, dtype=object))
            print("Direct trajectory data saved to 'direct_trajectory_data.npy'")

            # 重置模拟状态
            self.data.qpos[:] = 0  # 重置关节位置
            mujoco.mj_forward(self.model, self.data)

            # 播放SLERP轨迹
            print("Playing SLERP trajectory")
            for q in self.q_traj_slerp:
                self.data.qpos[:len(q)] = q
                mujoco.mj_forward(self.model, self.data)
                # 记录末端执行器位置和姿态
                pos = self.data.xpos[-1].copy()  # 位置
                quat = self.data.xquat[-1].copy()  # 四元数
                self.traj_data_slerp.append((pos, quat))
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(0.1)  # 增加延时到 0.1 秒

            # 保存SLERP轨迹数据
            np.save("slerp_trajectory_data.npy", np.array(self.traj_data_slerp, dtype=object))
            print("SLERP trajectory data saved to 'slerp_trajectory_data.npy'")

            # 检查轨迹数据的起点和终点
            print("Direct trajectory start:", self.traj_data_direct[0][0])
            print("Direct trajectory end:", self.traj_data_direct[-1][0])
            print("SLERP trajectory start:", self.traj_data_slerp[0][0])
            print("SLERP trajectory end:", self.traj_data_slerp[-1][0])

            # 保持窗口打开
            while self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.01)

        finally:
            self.viewer.close()

# 主程序
if __name__ == "__main__":
    urdf_path = "/home/zxy/MujocoBin/mujoco-learning/model/franka_panda_description/robots/panda_description/urdf/panda.urdf"
    xml_path = "/home/zxy/MujocoBin/mujoco_menagerie-main/franka_emika_panda/scene.xml"

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    p0 = [0.3, 0.2, 0.5]  # 起点位置
    q0 = [0, 0, 0, 1]     # 起点四元数
    p1 = [0.4, 0.2, 0.4]  # 终点位置
    q1 = [0.0, 0.7071, 0.0, 0.7071] # 终点四元数
    N = 60  # 插值点数

    # 生成两条轨迹
    traj_direct = interpolate_SE3(p0, p1, q0, q1, N, use_slerp=False)
    traj_slerp = interpolate_SE3(p0, p1, q0, q1, N, use_slerp=True)


    # 绘图
    plot_3d_trajectory_with_orientations(traj_direct, traj_slerp)

    # 求解逆运动学，生成关节角度序列
    q_traj_direct = solve_ik_sequence(urdf_path, traj_direct)
    q_traj_slerp = solve_ik_sequence(urdf_path, traj_slerp)

    # 创建并运行查看器
    viewer = CustomViewer(model, data, q_traj_direct, q_traj_slerp)
    viewer.run()