import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
import mujoco
import mujoco.viewer
from numpy.linalg import norm, solve
import time

# ------------------------------------------------------------------------------
# -------------------- Math helpers (formulas only) ----------------------------
# ------------------------------------------------------------------------------

def hat(phi):
    """
    Skew-symmetric (hat) operator: R^3 -> so(3).
    Given phi = [phi1, phi2, phi3]^T, returns:
        [  0  -phi3   phi2]
    [ phi3    0  -phi1]
    [ -phi2  phi1    0 ]
    """
    phi1, phi2, phi3 = phi
    return np.array([[0.0,   -phi3,   phi2],
                     [phi3,   0.0,   -phi1],
                     [-phi2,   phi1,   0.0]])

def rodrigues_exp(k, theta):
    """
    Matrix exponential on so(3) via Rodrigues' formula.
    exp([k]^ * theta) = I + sin(theta) [k]^  + (1 - cos(theta)) [k]^ ^2,
    where ||k|| = 1 (unit axis), theta >= 0.
    """
    # Normalize axis k to unit length (if nonzero)
    knorm = np.linalg.norm(k)
    if knorm < 1e-15:
        return np.eye(3)
    k = k / knorm
    K = hat(k)
    I = np.eye(3)
    s = np.sin(theta)
    c = np.cos(theta)
    return I + s * K + (1.0 - c) * (K @ K)

def axis_angle_from_R(R):
    """
    Extract axis k (unit) and angle theta from a proper rotation matrix R in SO(3).
    Formulas:
      cos(theta) = (trace(R) - 1)/2
      For theta in (0, pi): k = (1/(2*sin(theta))) * [R32 - R23, R13 - R31, R21 - R12]^T
      Handle theta ~ 0 and theta ~ pi carefully for numerical stability.
    """
    # Clamp to avoid numerical issues outside [-1, 1]
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-12:
        # No meaningful rotation: arbitrary axis
        return np.array([1.0, 0.0, 0.0]), 0.0

    # If close to pi, use a more stable method to extract the axis
    if np.pi - theta < 1e-6:
        # From diagonal elements: find the largest diagonal to compute axis reliably
        R_plus = (R + np.eye(3)) * 0.5
        # The axis is the (unit) eigenvector of R with eigenvalue 1. Here we derive from R_plus diagonals.
        k = np.array([R_plus[0,0], R_plus[1,1], R_plus[2,2]])
        # Pick the largest component to compute remaining components
        idx = np.argmax(k)
        if idx == 0:
            x = np.sqrt(max(R_plus[0,0], 0.0))
            if x < 1e-12:
                # Fallback
                k = np.array([1.0, 0.0, 0.0])
            else:
                kx = x
                ky = R_plus[1,0] / x
                kz = R_plus[2,0] / x
                k = np.array([kx, ky, kz])
        elif idx == 1:
            y = np.sqrt(max(R_plus[1,1], 0.0))
            if y < 1e-12:
                k = np.array([0.0, 1.0, 0.0])
            else:
                ky = y
                kx = R_plus[1,0] / y
                kz = R_plus[2,1] / y
                k = np.array([kx, ky, kz])
        else:
            z = np.sqrt(max(R_plus[2,2], 0.0))
            if z < 1e-12:
                k = np.array([0.0, 0.0, 1.0])
            else:
                kz = z
                kx = R_plus[2,0] / z
                ky = R_plus[2,1] / z
                k = np.array([kx, ky, kz])

        kn = np.linalg.norm(k)
        if kn < 1e-12:
            # Fallback: use off-diagonal formula (may be noisy near pi)
            k = np.array([R[2,1] - R[1,2],
                          R[0,2] - R[2,0],
                          R[1,0] - R[0,1]]) / (2.0 * np.sin(theta))
        else:
            k = k / kn
        return k, theta

    # General case (0 < theta < pi):
    k = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]]) / (2.0 * np.sin(theta))
    # Normalize to be safe
    kn = np.linalg.norm(k)
    if kn < 1e-15:
        # Degenerate; pick any
        k = np.array([1.0, 0.0, 0.0])
    else:
        k = k / kn
    return k, theta

def rotmat_interp_geodesic(R1, R2, t):
    """
    Geodesic interpolation on SO(3) using axis-angle + Rodrigues:
      R_rel = R1^T R2
      (k, theta) = axis_angle_from_R(R_rel)
      R(t) = R1 * exp( [k]^ x * (t * theta) )
    """
    R_rel = R1.T @ R2
    k, theta = axis_angle_from_R(R_rel)
    R_t = R1 @ rodrigues_exp(k, t * theta)
    # R_t = rodrigues_exp(k, t * theta) @ R1
    return R_t

def interpolate_SE3(p0, p1, R0, R1, N, use_so3=True):
    times = np.linspace(0, 1, N)        # 生成 N 个插值时间点
    positions = np.linspace(p0, p1, N)
    if use_so3:
        interp_so3 = [rotmat_interp_geodesic(R0, R1, t) for t in times] 
    else:
        # R0 复制 N 份
        interp_so3 = [R0.copy() for _ in range(N)]
    return [(positions[i], interp_so3[i]) for i in range(N)]

# ------------------------------------------------------------------------------
# -------------------- Viewer --------------------------------------------------
# ------------------------------------------------------------------------------

def plot_3d_trajectory_with_tcp(direct_traj, so3_traj, step=5, axis_len=0.02):
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取位置数据
    pos_direct = np.array([p for p, _ in direct_traj])
    pos_so3 = np.array([p for p, _ in so3_traj])

    # 绘制轨迹线
    ax.plot(pos_direct[:, 0], pos_direct[:, 1], pos_direct[:, 2],
            'r-', label='Direct Trajectory', linewidth=2)
    ax.plot(pos_so3[:, 0], pos_so3[:, 1], pos_so3[:, 2],
            'b-', label='SO3 Trajectory', linewidth=2)

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

    for i in range(0, len(so3_traj), step):
        draw_tcp_axes(ax, *so3_traj[i], length=axis_len, alpha=0.5)

    # 设置标签与标题
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Direct vs SO3 Trajectories with TCP Frames', fontsize=14)
    ax.legend()

    def set_axes_equal(ax):
        """让3D坐标轴比例一致，防止箭头长短错觉"""
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

# 自定义查看器类
class CustomViewer:
    def __init__(self, model, data, traj_direct, traj_so3):
        self.model = model
        self.data = data
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.traj_direct = traj_direct  # 直接轨迹的关节角度序列
        self.traj_so3 = traj_so3    # SO3轨迹的关节角度序列
        self.traj_data_direct = []  # 存储直接轨迹的位置和姿态
        self.traj_data_so3 = []   # 存储SO3轨迹的位置和姿态


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
                ee_rot = self.data.xmat[8].copy()
                self.traj_data_direct.append((ee_pos, ee_rot))
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(0.1)
 
            # 保存直接轨迹数据
            np.save("/home/zxy/MujocoBin/Data/NPY/so3/direct_trajectory_data.npy", np.array(self.traj_data_direct, dtype=object))
            print("Direct trajectory data saved to 'direct_trajectory_data.npy'")
            
            # 重置模拟状态
            self.data.qpos[:] = q_start  # 重置关节位置
            mujoco.mj_forward(self.model, self.data)

            # 播放SO3轨迹
            print("Playing SO3 trajectory with IK")
            for pos, rot in self.traj_so3:
                q_c = self.data.qpos[:self.model.nq].copy()
                q_new = solve_ik_sequence(urdf_path, pos, rot, q_c)
                self.data.qpos[:len(q_new)] = q_new
                mujoco.mj_forward(self.model, self.data)
                ee_pos = self.data.xpos[8].copy()
                ee_rot = self.data.xmat[8].copy()
                self.traj_data_so3.append((ee_pos, ee_rot))
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                time.sleep(0.1)
                
            np.save("/home/zxy/MujocoBin/Data/NPY/so3/so3_trajectory_data.npy", np.array(self.traj_data_so3, dtype=object))
            print("SO3 trajectory data saved to 'so3_trajectory_data.npy'")


            # 检查轨迹数据的起点和终点
            print("Direct trajectory start:", self.traj_data_direct[0][0])
            print("Direct trajectory end:", self.traj_data_direct[-1][0])
            print("SO3 trajectory start:", self.traj_data_so3[0][0])
            print("SO3 trajectory end:", self.traj_data_so3[-1][0])

            # 保持窗口打开
            while self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.01)

        finally:
            self.viewer.close()


if __name__ == '__main__':
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
    R0 = R.from_quat(q0).as_matrix()

    # 终点位姿
    pin.forwardKinematics(model_pin, data_pin, q_end)
    p1 = data_pin.oMi[7].translation.tolist()
    q1 = R.from_matrix(data_pin.oMi[7].rotation).as_quat().tolist()
    R1 = R.from_quat(q1).as_matrix()

    print("起点位置 p0:", p0)
    print("起点姿态 R0:", R0)
    print("终点位置 p1:", p1)
    print("终点姿态 R1:", R1)

    # 更新到模拟
    data.qpos[:len(q_start)] = q_start
    mujoco.mj_forward(model, data)

    N = 50  # 插值点数

    # 
    traj_direct = interpolate_SE3(p0, p1, R0, R1, N, use_so3=False)
    traj_so3 = interpolate_SE3(p0, p1, R0, R1, N, use_so3=True)

    # 绘图
    plot_3d_trajectory_with_tcp(traj_direct, traj_so3)
  
    # 求解逆运动学，生成关节角度序列
    q_c = data.qpos.copy()  # 当前关节角度作为初始值
    print("当前所有关节角度/位置:", q_c)

    # 创建并运行查看器
    viewer = CustomViewer(model, data, traj_direct, traj_so3)
    viewer.run()