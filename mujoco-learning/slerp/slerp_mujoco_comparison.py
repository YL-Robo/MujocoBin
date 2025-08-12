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
    times = np.linspace(0, 1, N)        # 生成 N 个插值时间点
    normalized_times = (times - np.min(times)) / (np.max(times) - np.min(times))
    positions = np.linspace(p0, p1, N)
    if use_slerp:
        # # scipy 的 Slerp
        # key_rots = R.from_quat([q0, q1])    # 起点和终点四元数转为旋转对象
        # slerp = Slerp([0, 1], key_rots)     # 创建 SLERP 插值器，关键帧时间为 0 和 1
        # interp_rots = slerp(normalized_times)          # 得到 N 个插值旋转
        # custom Slerp
        quats_custom = Slerp_test(q0, q1, normalized_times)
        quats_custom = np.array(quats_custom)
        interp_rots = R.from_quat(quats_custom)
    else:
        interp_rots = [R.from_quat(q0)] * N

    return [(positions[i], interp_rots[i].as_matrix()) for i in range(N)]



# IK 求解函数
def solve_ik_sequence(model_path, pos, rot, q_c=None):
    # pinocchio 初始化
    model = pin.buildModelFromUrdf(model_path)
    data = model.createData()
    JOINT_ID = 7  # 假设末端执行器关节ID为7

    # 参数初始化
    eps = 1e-4      # 定义收敛阈值，当误差小于该值时认为算法收敛
    IT_MAX = 1000   # 定义最大迭代次数，防止算法陷入无限循环
    DT     = 1E-01  # 定义积分步长，用于更新关节角度
    damp   = 1e-12  # 定义阻尼因子，用于避免矩阵奇异
  
    oMdes = pin.SE3(rot, np.array(pos))
    q = q_c.copy()
    i = 0
    pin.forwardKinematics(model, data, q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)
    err = pin.log(iMd).vector

    if norm(err) < eps:
        return q_c
    else:
        while True:
            q = np.asarray(q, dtype=np.float32)
            J = pin.computeJointJacobian(model, data, q, JOINT_ID)
            # J = pin.getJointJacobian(model, data, JOINT_ID, pin.LOCAL_WORLD_ALIGNED)
            Jac = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -Jac.T.dot(solve(Jac.dot(Jac.T) + damp * np.eye(6), err))
            q = pin.integrate(model, q, v * DT)
            pin.forwardKinematics(model, data, q)
            iMd = data.oMi[JOINT_ID].actInv(oMdes)
            err = pin.log(iMd).vector
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                # print("IK did not converge after %d iterations." % IT_MAX)
                break
            if not i % 10:
                # print(f"Iteration {i}: error norm = {norm(err):.6f}, pos = {pos}")
                pass
            i += 1
        if success:
            return q
        else:
            return q_c
    

def plot_3d_trajectory_with_tcp(direct_traj, slerp_traj, step=5, axis_len=0.02):
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

# 自定义查看器类
class CustomViewer:
    def __init__(self, model, data, traj_direct, traj_slerp):
        self.model = model
        self.data = data
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.traj_direct = traj_direct  # 直接轨迹的关节角度序列
        self.traj_slerp = traj_slerp    # SLERP轨迹的关节角度序列
        self.traj_data_direct = []  # 存储直接轨迹的位置和姿态
        self.traj_data_slerp = []   # 存储SLERP轨迹的位置和姿态


    def run(self):
        try:
            # 播放直接轨迹
            print("Playing direct trajectory")
            for pos, rot in self.traj_direct:
                # 当前关节角度作为初值
                q_c = self.data.qpos[:self.model.nq].copy()
                # 求解IK
                q_new = solve_ik_sequence(urdf_path, pos, rot, q_c)
                # 更新到仿真
                self.data.qpos[:len(q_new)] = q_new
                mujoco.mj_forward(self.model, self.data)
                # 记录末端执行器位置和姿态
                ee_pos = self.data.xpos[8].copy()
                ee_quat = self.data.xquat[8].copy()
                ee_quat_xyzw = np.roll(ee_quat, -1)  # 从 [w, x, y, z] 转换为 [x, y, z, w]
                self.traj_data_direct.append((ee_pos, ee_quat_xyzw))
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(0.1)
 
            # 保存直接轨迹数据
            np.save("/home/zxy/MujocoBin/Data/NPY/slerp/direct_trajectory_data.npy", np.array(self.traj_data_direct, dtype=object))
            print("Direct trajectory data saved to 'direct_trajectory_data.npy'")
            
            # 重置模拟状态
            self.data.qpos[:] = q_start  # 重置关节位置
            mujoco.mj_forward(self.model, self.data)

            # 播放SLERP轨迹
            print("Playing SLERP trajectory with IK")
            for pos, rot in self.traj_slerp:
                q_c = self.data.qpos[:self.model.nq].copy()
                q_new = solve_ik_sequence(urdf_path, pos, rot, q_c)
                self.data.qpos[:len(q_new)] = q_new
                mujoco.mj_forward(self.model, self.data)
                ee_pos = self.data.xpos[8].copy()
                ee_quat = self.data.xquat[8].copy()
                ee_quat_xyzw = np.roll(ee_quat, -1)  # 从 [w, x, y, z] 转换为 [x, y, z, w]
                self.traj_data_slerp.append((ee_pos, ee_quat_xyzw))
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(0.1)
                
            np.save("/home/zxy/MujocoBin/Data/NPY/slerp/slerp_trajectory_data.npy", np.array(self.traj_data_slerp, dtype=object))
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
    # pinocchio 和 MuJoCo 模型路径
    urdf_path = "/home/zxy/MujocoBin/mujoco-learning/model/franka_panda_description/robots/panda_description/urdf/panda.urdf"
    xml_path = "/home/zxy/MujocoBin/mujoco_menagerie-main/franka_emika_panda/scene.xml"
    
    # MuJoCo 模型加载
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 用 pinocchio 正运动学计算对应的位姿
    model_pin = pin.buildModelFromUrdf(urdf_path)
    data_pin = model_pin.createData()


    # 关节零位附近，避免奇异姿态
    q_start = np.array([0.2, -0.5, 0.2, -1.7, 0.1, 1.6, 0.9, 0.0, 0.0])  # 起始关节角度
    q_end   = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.8, 0.0, 0.0])  # 结束关节角度
    
    # 起点位姿
    pin.forwardKinematics(model_pin, data_pin, q_start)
    p0 = data_pin.oMi[7].translation.tolist()
    q0 = R.from_matrix(data_pin.oMi[7].rotation).as_quat().tolist()

    # 终点位姿
    pin.forwardKinematics(model_pin, data_pin, q_end)
    p1 = data_pin.oMi[7].translation.tolist()
    q1 = R.from_matrix(data_pin.oMi[7].rotation).as_quat().tolist()

    # xyzw
    print("起点位置 p0:", p0)
    print("起点姿态 q0:", q0)
    print("终点位置 p1:", p1)
    print("终点姿态 q1:", q1)

    # ============================================================================
    #                                      路线测试 
    # # 启动可视化窗口
    # viewer = mujoco.viewer.launch_passive(model, data)

    # # 控制机械臂从起点慢慢运动到终点
    # steps = 200
    # for i in range(steps):
    #     alpha = i / (steps - 1)
    #     q_current = (1 - alpha) * q_start + alpha * q_end
    #     # 设置当前关节角度
    #     data.qpos[:] = q_current
    #     mujoco.mj_forward(model, data)  # 更新物理状态

    #     viewer.sync()
    #     time.sleep(0.01)  # 控制动画速度
    # ============================================================================

    # 更新到模拟
    data.qpos[:len(q_start)] = q_start
    mujoco.mj_forward(model, data)

    N = 50  # 插值点数

    # 生成两条轨迹(xyzw)
    traj_direct = interpolate_SE3(p0, p1, q0, q1, N, use_slerp=False)
    traj_slerp = interpolate_SE3(p0, p1, q0, q1, N, use_slerp=True)

    # 绘图
    plot_3d_trajectory_with_tcp(traj_direct, traj_slerp)

    # 求解逆运动学，生成关节角度序列
    q_c = data.qpos.copy()  # 当前关节角度作为初始值
    print("当前所有关节角度/位置:", q_c)

    # 创建并运行查看器
    viewer = CustomViewer(model, data, traj_direct, traj_slerp)
    viewer.run()