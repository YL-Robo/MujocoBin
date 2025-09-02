import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import mujoco
import mujoco.viewer
import pinocchio as pin
from numpy.linalg import norm, solve
import time
import matplotlib.pyplot as plt
import casadi as cs
import casadi
from pinocchio import casadi as cpin
import os
# import sys
# from pathlib import Path
# # Allow running this file directly via `python IK_mujoco/IK_pinocchio_casadi.py`
# # by adding the project root (one level up) to sys.path so `utils` and `config` resolve.
# sys.path.append(str(Path(__file__).resolve().parents[1]))
from mujoco_learning.mujoco_learning.utils.pose_interpolator.pose_interpolator import PoseInterpolator
from mujoco_learning.mujoco_learning.config.default_config import Config


# 确保数据目录存在
os.makedirs(Config.DATA_DIR, exist_ok=True)

def generate_safe_trajectory(model_pin, q_start, q_end, N_points=50):
    """
    生成安全的轨迹，避免奇异配置
    
    Args:
        model_pin: Pinocchio模型
        q_start, q_end: 起始和结束关节角度
        N_points: 轨迹点数
    """
    # 计算起始和结束位姿
    data_pin = model_pin.createData()
    
    pin.forwardKinematics(model_pin, data_pin, q_start)
    p0 = data_pin.oMi[Config.JOINT_ID].translation.tolist()
    q0 = R.from_matrix(data_pin.oMi[Config.JOINT_ID].rotation).as_quat().tolist()

    pin.forwardKinematics(model_pin, data_pin, q_end)
    p1 = data_pin.oMi[Config.JOINT_ID].translation.tolist()
    q1 = R.from_matrix(data_pin.oMi[Config.JOINT_ID].rotation).as_quat().tolist()
    
    # 检查轨迹是否合理
    distance = np.linalg.norm(np.array(p1) - np.array(p0))
    if distance > 0.5:  # 如果距离太远，缩小目标
        print(f"Warning: Target distance {distance:.3f}m is too large, scaling down")
        scale = 0.5 / distance
        p1_scaled = np.array(p0) + scale * (np.array(p1) - np.array(p0))
        p1 = p1_scaled.tolist()
    
    # 生成轨迹
    interpolator = PoseInterpolator()
    traj_direct = interpolator.interpolate_se3_with_slerp(p0, p1, q0, q1, N_points, use_slerp=False)
    traj_slerp = interpolator.interpolate_se3_with_slerp(p0, p1, q0, q1, N_points, use_slerp=True)
    
    return traj_direct, traj_slerp, p0, p1

def solve_ik_sequence(model, pos, rot, q_c=None):
    """使用Jacobian方法求解逆运动学"""
    data = model.createData()
    JOINT_ID = Config.JOINT_ID

    # 参数初始化
    eps = Config.EPS
    IT_MAX = Config.IT_MAX
    DT = Config.DT
    damp = Config.DAMP
  
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
    
def plot_trajectory_comparison(traj_data_jacobian, traj_data_casadi, save_path=None):
    """
    绘制两种IK方法的轨迹对比图
    
    Args:
        traj_data_jacobian: Jacobian方法的轨迹数据 [(pos, quat), ...]
        traj_data_casadi: CasADi方法的轨迹数据 [(pos, quat), ...]
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 提取位置数据
    pos_jac = np.array([pos for pos, _ in traj_data_jacobian])
    pos_cas = np.array([pos for pos, _ in traj_data_casadi])
    
    # 3D轨迹对比
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(pos_jac[:, 0], pos_jac[:, 1], pos_jac[:, 2], 'r-', label='Jacobian', linewidth=2)
    ax1.plot(pos_cas[:, 0], pos_cas[:, 1], pos_cas[:, 2], 'b-', label='CasADi', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    
    # 各轴位置对比
    steps = np.arange(len(pos_jac))
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(steps, pos_jac[:, 0], 'r-', label='Jacobian X')
    ax2.plot(steps, pos_cas[:, 0], 'b-', label='CasADi X')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('X Position (m)')
    ax2.set_title('X Position Comparison')
    ax2.legend()
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(steps, pos_jac[:, 1], 'r-', label='Jacobian Y')
    ax3.plot(steps, pos_cas[:, 1], 'b-', label='CasADi Y')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Y Position (m)')
    ax3.set_title('Y Position Comparison')
    ax3.legend()
    
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(steps, pos_jac[:, 2], 'r-', label='Jacobian Z')
    ax4.plot(steps, pos_cas[:, 2], 'b-', label='CasADi Z')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Z Position (m)')
    ax4.set_title('Z Position Comparison')
    ax4.legend()
    
    # 轨迹误差
    ax5 = fig.add_subplot(2, 3, 5)
    pos_diff = np.linalg.norm(pos_jac - pos_cas, axis=1)
    ax5.plot(steps, pos_diff, 'g-', linewidth=2)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Position Error (m)')
    ax5.set_title('Trajectory Position Error')
    ax5.grid(True)
    
    # 统计信息
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = (
        "Trajectory Statistics:\n\n"
        f"Jacobian Start: {np.array2string(pos_jac[0], precision=4)}\n"
        f"Jacobian End:   {np.array2string(pos_jac[-1], precision=4)}\n"
        f"Jacobian Distance: {np.sum(np.linalg.norm(np.diff(pos_jac, axis=0), axis=1)):.4f} m\n\n"
        f"CasADi Start: {np.array2string(pos_cas[0], precision=4)}\n"
        f"CasADi End:   {np.array2string(pos_cas[-1], precision=4)}\n"
        f"CasADi Distance: {np.sum(np.linalg.norm(np.diff(pos_cas, axis=0), axis=1)):.4f} m\n\n"
        f"Average Position Error: {np.mean(pos_diff):.6f} m\n"
        f"Max Position Error: {np.max(pos_diff):.6f} m"
    )
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory comparison saved to {save_path}")
    
    plt.show()

# ----------------------------------------------------------------------------
# CasADi-based IK solver (Pinocchio + IPOPT)
# ----------------------------------------------------------------------------

class CasadiIKSolver(object):
    """
    CasADi-based IK solver that minimizes SE(3) log error to a target pose
    with joint limits and continuity regularization.
    """

    def __init__(self, pin_model: pin.Model, joint_id: int):
        self.model = pin_model
        self.joint_id = joint_id

        # Build CasADi symbolic model from Pinocchio model
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # Symbols
        self.cq = cs.SX.sym("q", self.cmodel.nq, 1)
        self.cTf = cs.SX.sym("tf", 4, 4)
        self.cq_prev = cs.SX.sym("q_prev", self.cmodel.nq, 1)

        # Forward kinematics (frames/joints)
        cpin.forwardKinematics(self.cmodel, self.cdata, self.cq)
        # cpin.updateFramePlacements(self.cmodel, self.cdata)

        # Use joint placement oMi for the specified joint id
        current_oMi = self.cdata.oMi[self.joint_id]
        error_se3 = cpin.log6(current_oMi.inverse() * cpin.SE3(self.cTf)).vector
        error = casadi.vertcat(error_se3)
        self.error = casadi.Function("error", [self.cq, self.cTf], [casadi.vertcat(error)])


        # Opti problem
        self.opti = cs.Opti()
        self.var_q = self.opti.variable(self.cmodel.nq)
        self.par_tf = self.opti.parameter(4, 4)
        self.par_q_prev = self.opti.parameter(self.cmodel.nq)

        total_error = self.error(self.var_q,self.par_tf)
        self.totalcost = casadi.sumsqr(total_error)
        self.regularization = casadi.sumsqr(self.var_q)

        # Joint bounds
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit, self.var_q, self.model.upperPositionLimit
        ))
        
        # 参数定义
        self.param_q_prev = self.opti.parameter(self.model.nq)  # 上一帧解
        
        # 连续性惩罚 - 关节角度变化量
        q_diff = self.var_q - self.param_q_prev
        continuity_penalty = casadi.sumsqr(q_diff)
        
        # 关节移动惩罚 - 不同关节的权重不同
        joint_move_penalty = 0
        for i in range(self.model.nq):
            # 前两个关节（基座关节）权重较大，避免大幅移动
            weight = 2.0 if i in [0,1] else 0.2
            joint_move_penalty = joint_move_penalty + weight * (self.var_q[i] - self.param_q_prev[i])**2
        
        # 正则化惩罚 - 避免关节角度过大
        regularization_penalty = 0.01 * casadi.sumsqr(self.var_q)
        
        # 总目标函数 - 平衡精度和连续性
        total_objective = (20.0 * self.totalcost + 
                          0.5 * joint_move_penalty + 
                          0.8 * continuity_penalty + 
                          0.1 * regularization_penalty)
        
        self.opti.minimize(total_objective)

        # Solver options - 改进收敛性和稳定性
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,  # 减少最大迭代次数，避免过度优化
                'tol': 1e-5,      # 放宽收敛条件
                'acceptable_tol': 1e-4,  # 可接受的收敛条件
                'hessian_approximation': 'limited-memory',  # 使用L-BFGS近似
                'mu_strategy': 'adaptive',  # 自适应障碍参数
                'bound_push': 1e-8,
                'bound_frac': 1e-8,
                'slack_bound_push': 1e-8,
                'slack_bound_frac': 1e-8,
                'recalc_y': 'yes',
                'max_cpu_time': 0.1  # 限制求解时间
            },
            'print_time': False,
            'expand': True
        }
        self.opti.solver('ipopt', opts)

    def solve(self, target_T: np.ndarray, q_init: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
        """Solve IK for target SE3 homogeneous 4x4 matrix."""
        try:
            # 设置初始值和参数
            self.opti.set_initial(self.var_q, q_init)
            self.opti.set_value(self.par_tf, target_T)
            self.opti.set_value(self.param_q_prev, q_prev)
            
            # 尝试求解
            sol = self.opti.solve_limited()
            q_sol = sol.value(self.var_q)
            
            # 检查解是否在关节限制范围内
            if np.any(q_sol < self.model.lowerPositionLimit) or np.any(q_sol > self.model.upperPositionLimit):
                print("Warning: CasADi solution out of joint limits, using previous solution")
                return q_prev.copy()
            
            # 数值健壮性检查
            if not np.all(np.isfinite(q_sol)):
                print("Warning: CasADi solution contains non-finite values, using previous solution")
                return q_prev.copy()
            
            # 步长限制，避免发散
            step_limit = 0.2  # 每个关节最大变化幅度（弧度）
            q_sol = np.clip(q_sol, q_prev - step_limit, q_prev + step_limit)
            
            return q_sol
            
        except Exception as e:
            # 如果求解失败，使用Jacobian方法作为后备
            print(f"CasADi solver failed: {e}, falling back to Jacobian method")
            try:
                # 使用Jacobian方法求解
                pos = target_T[:3, 3]
                rot = target_T[:3, :3]
                q_jac = solve_ik_sequence(self.model, pos, rot, q_prev)
                return q_jac
            except:
                # 如果Jacobian也失败，返回上一帧的解
                return q_prev.copy()

def compute_pose_error_norm(pin_model: pin.Model, pin_data: pin.Data,
                            joint_id: int, q: np.ndarray,
                            target_T: np.ndarray) -> float:
    pin.forwardKinematics(pin_model, pin_data, q)
    iMd = pin_data.oMi[joint_id].actInv(pin.SE3(target_T))
    return float(norm(pin.log(iMd).vector))

def plot_ik_runtime_and_error(times_jac: list, times_cas: list,
                              errs_jac: list, errs_cas: list,
                              title_suffix: str = "", save_path: str = None, show: bool = True):
    """绘制IK求解时间和误差对比图"""
    steps = np.arange(len(times_jac))
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 求解时间对比
    axs[0, 0].plot(steps, times_jac, 'r-', label='Jacobian IK', linewidth=2)
    axs[0, 0].plot(steps, times_cas, 'b-', label='CasADi IK', linewidth=2)
    axs[0, 0].set_title(f'Runtime per step {title_suffix}')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Time (ms)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 误差对比
    axs[0, 1].plot(steps, errs_jac, 'r-', label='Jacobian IK', linewidth=2)
    axs[0, 1].plot(steps, errs_cas, 'b-', label='CasADi IK', linewidth=2)
    axs[0, 1].set_title(f'Pose error norm {title_suffix}')
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('||log(iMd)||')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 平均时间统计
    axs[1, 0].bar(['Jacobian', 'CasADi'], [np.mean(times_jac), np.mean(times_cas)], 
                  color=['red', 'blue'], alpha=0.7)
    axs[1, 0].set_title('Average Runtime Comparison')
    axs[1, 0].set_ylabel('Time (ms)')
    axs[1, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate([np.mean(times_jac), np.mean(times_cas)]):
        axs[1, 0].text(i, v + 0.1, f'{v:.2f}ms', ha='center', va='bottom')

    # 平均误差统计
    axs[1, 1].bar(['Jacobian', 'CasADi'], [np.mean(errs_jac), np.mean(errs_cas)], 
                  color=['red', 'blue'], alpha=0.7)
    axs[1, 1].set_title('Average Error Comparison')
    axs[1, 1].set_ylabel('Error norm')
    axs[1, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate([np.mean(errs_jac), np.mean(errs_cas)]):
        axs[1, 1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
    if show:
        plt.show()
    return fig

def make_homogeneous(rot3x3: np.ndarray, pos3: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot3x3
    T[:3, 3] = pos3
    return T

def run_ik_comparison(trajectory, pin_model: pin.Model, joint_id: int, q_start: np.ndarray):
    """
    Run IK with Jacobian and CasADi solvers across a trajectory.
    Returns per-step runtimes (ms) and error norms.
    """
    pin_data = pin_model.createData()
    cas_solver = CasadiIKSolver(pin_model, joint_id)

    q_jac = q_start.copy()
    q_cas = q_start.copy()

    times_jac = []
    times_cas = []
    errs_jac = []
    errs_cas = []

    for pos, rot in trajectory:
        T = make_homogeneous(rot, np.asarray(pos))
        # Jacobian IK
        t0 = time.perf_counter()
        # q_jac = ik_step_jacobian(pin_model, pin_data, joint_id, T, q_jac)
        q_jac = solve_ik_sequence(pin_model, pos, rot, q_jac)
        t1 = time.perf_counter()
        err_j = compute_pose_error_norm(pin_model, pin_data, joint_id, q_jac, T)
        times_jac.append((t1 - t0) * 1000.0)
        errs_jac.append(err_j)

        # CasADi IK
        t0 = time.perf_counter()
        q_cas = cas_solver.solve(T, q_cas, q_cas)
        t1 = time.perf_counter()
        err_c = compute_pose_error_norm(pin_model, pin_data, joint_id, q_cas, T)
        times_cas.append((t1 - t0) * 1000.0)
        errs_cas.append(err_c)

    return times_jac, times_cas, errs_jac, errs_cas

class CustomViewer(object):
    """自定义MuJoCo查看器，用于实时显示IK求解过程"""
    
    def __init__(self, model, data, traj_direct, traj_slerp, model_pin):
        self.model = model
        self.model_pin = model_pin
        self.data = data
        self.viewer = mujoco.viewer.launch_passive(model, data)
        self.traj_direct = traj_direct
        self.traj_slerp = traj_slerp
        self.traj_data_jacobian = []
        self.traj_data_casadi = []
        
        # 性能统计
        self.jacobian_times = []
        self.casadi_times = []
        self.jacobian_errors = []
        self.casadi_errors = []


    def run_jacobian_ik(self):
        """运行Jacobian IK求解"""
        print("Running Jacobian IK...")
        for i, (pos, rot) in enumerate(self.traj_slerp):
            # 当前关节角度作为初值
            q_c = self.data.qpos[:self.model.nq].copy()
            
            # 记录开始时间
            start_time = time.perf_counter()
            
            # 求解IK
            q_new = solve_ik_sequence(self.model_pin, pos, rot, q_c)
            
            # 记录求解时间
            solve_time = (time.perf_counter() - start_time) * 1000.0
            self.jacobian_times.append(solve_time)
            
            # 更新到仿真
            self.data.qpos[:len(q_new)] = q_new
            mujoco.mj_forward(self.model, self.data)
            
            # 记录末端执行器位置和姿态
            ee_pos = self.data.xpos[8].copy()
            ee_quat = self.data.xquat[8].copy()
            ee_quat_xyzw = np.roll(ee_quat, -1)
            self.traj_data_jacobian.append((ee_pos, ee_quat_xyzw))
            
            # 计算误差
            target_T = make_homogeneous(rot, np.asarray(pos))
            error = compute_pose_error_norm(self.model_pin, self.model_pin.createData(), 
                                          Config.JOINT_ID, q_new, target_T)
            self.jacobian_errors.append(error)
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(Config.SIM_DT)
            
            # 显示进度
            if i % 10 == 0:
                print(f"Jacobian IK: {i}/{len(self.traj_slerp)} steps completed")
    
    def run_casadi_ik(self):
        """运行CasADi IK求解"""
        print("Running CasADi IK...")
        cas_solver = CasadiIKSolver(self.model_pin, Config.JOINT_ID)
        
        for i, (pos, rot) in enumerate(self.traj_slerp):
            q_c = self.data.qpos[:self.model.nq].copy()
            T = make_homogeneous(rot, np.asarray(pos))
            
            # 记录开始时间
            start_time = time.perf_counter()
            
            # 求解IK
            q_new = cas_solver.solve(T, q_c, q_c)
            
            # 记录求解时间
            solve_time = (time.perf_counter() - start_time) * 1000.0
            self.casadi_times.append(solve_time)
            
            # 更新到仿真
            self.data.qpos[:len(q_new)] = q_new
            mujoco.mj_forward(self.model, self.data)
            
            # 记录末端执行器位置和姿态
            ee_pos = self.data.xpos[8].copy()
            ee_quat = self.data.xquat[8].copy()
            ee_quat_xyzw = np.roll(ee_quat, -1)
            self.traj_data_casadi.append((ee_pos, ee_quat_xyzw))
            
            # 计算误差
            error = compute_pose_error_norm(self.model_pin, self.model_pin.createData(), 
                                          Config.JOINT_ID, q_new, T)
            self.casadi_errors.append(error)
            
            # 仿真步进
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(Config.SIM_DT)
            
            # 显示进度
            if i % 10 == 0:
                print(f"CasADi IK: {i}/{len(self.traj_slerp)} steps completed")
    
    def save_data(self):
        """保存轨迹数据和性能统计"""
        # 保存轨迹数据
        jacobian_path = os.path.join(Config.DATA_DIR, "jacobian_data.npy")
        casadi_path = os.path.join(Config.DATA_DIR, "casadi_data.npy")
        
        np.save(jacobian_path, np.array(self.traj_data_jacobian, dtype=object))
        np.save(casadi_path, np.array(self.traj_data_casadi, dtype=object))
        
        print(f"Jacobian data saved to {jacobian_path}")
        print(f"CasADi data saved to {casadi_path}")
        
        # 保存性能统计
        performance_data = {
            'jacobian_times': self.jacobian_times,
            'casadi_times': self.casadi_times,
            'jacobian_errors': self.jacobian_errors,
            'casadi_errors': self.casadi_errors
        }
        
        performance_path = os.path.join(Config.DATA_DIR, "performance_data.npy")
        np.save(performance_path, performance_data)
        print(f"Performance data saved to {performance_path}")
    
    def plot_results(self):
        """绘制结果对比图"""
        # 轨迹对比
        trajectory_path = os.path.join(Config.DATA_DIR, "trajectory_comparison.png")
        plot_trajectory_comparison(self.traj_data_jacobian, self.traj_data_casadi, trajectory_path)
        
        # 性能对比
        performance_path = os.path.join(Config.DATA_DIR, "performance_comparison.png")
        plot_ik_runtime_and_error(self.jacobian_times, self.casadi_times, 
                                 self.jacobian_errors, self.casadi_errors,
                                 title_suffix='(Real-time Simulation)', save_path=performance_path, show=False)
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("SIMULATION STATISTICS")
        print("="*50)
        
        print(f"Jacobian IK - Avg time: {np.mean(self.jacobian_times):.3f} ms, "
              f"Final error: {self.jacobian_errors[-1]:.6f}")
        print(f"CasADi IK   - Avg time: {np.mean(self.casadi_times):.3f} ms, "
              f"Final error: {self.casadi_errors[-1]:.6f}")
        
        print(f"\nTrajectory start (Jacobian): {self.traj_data_jacobian[0][0]}")
        print(f"Trajectory end   (Jacobian): {self.traj_data_jacobian[-1][0]}")
        print(f"Trajectory start (CasADi):   {self.traj_data_casadi[0][0]}")
        print(f"Trajectory end   (CasADi):   {self.traj_data_casadi[-1][0]}")
        print("="*50)
    
    def run(self):
        """主运行方法"""
        try:
            # 运行Jacobian IK
            self.run_jacobian_ik()
            
            # 重置模拟状态
            self.data.qpos[:] = Config.Q_START
            mujoco.mj_forward(self.model, self.data)
            
            # 运行CasADi IK
            self.run_casadi_ik()
            
            # 保存数据
            self.save_data()
            
            # 显示统计信息
            self.print_statistics()
            
            # 绘制结果
            self.plot_results()
            
            # 保持窗口打开
            print("Simulation completed. Press Ctrl+C to close viewer.")
            while self.viewer.is_running():
                self.viewer.sync()
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            self.viewer.close()

# 主程序
if __name__ == "__main__":
    print("="*60)
    print("IK COMPARISON: Jacobian vs CasADi Methods")
    print("="*60)
    
    # MuJoCo 模型加载
    model = mujoco.MjModel.from_xml_path(Config.XML_PATH)
    data = mujoco.MjData(model)

    # 用 pinocchio 正运动学计算对应的位姿
    model_pin = pin.buildModelFromUrdf(Config.URDF_PATH)
    data_pin = model_pin.createData()

    # 关节零位附近，避免奇异姿态
    q_start = Config.Q_START
    q_end = Config.Q_END
    
    # 更新到模拟
    data.qpos[:len(q_start)] = q_start
    mujoco.mj_forward(model, data)

    # 生成安全的轨迹
    print("Generating safe trajectory...")
    traj_direct, traj_slerp, p0, p1 = generate_safe_trajectory(model_pin, q_start, q_end, Config.N_POINTS)

    print(f"Generated trajectory with {Config.N_POINTS} points")
    print(f"Start position: {p0}")
    print(f"End position: {p1}")

    # 预运行IK比较（不显示仿真）
    print("\nRunning pre-simulation IK comparison...")
    times_jac, times_cas, errs_jac, errs_cas = run_ik_comparison(traj_slerp, model_pin, Config.JOINT_ID, q_start[:model_pin.nq])
        
    print(f"Jacobian IK avg time (ms): {np.mean(times_jac):.3f} | final err: {errs_jac[-1]:.6f}")
    print(f"CasADi  IK avg time (ms): {np.mean(times_cas):.3f} | final err: {errs_cas[-1]:.6f}")

    # 绘制预运行结果并保存
    pre_perf_path = os.path.join(Config.DATA_DIR, "pre_sim_performance.png")
    plot_ik_runtime_and_error(times_jac, times_cas, errs_jac, errs_cas,
                              title_suffix='(Pre-simulation)', save_path=pre_perf_path, show=False)
    plt.show()
    print(f"Pre-simulation performance saved to {pre_perf_path}")

    print("\nStarting real-time simulation...")
    print("Press Ctrl+C to stop simulation early")

    # 创建并运行查看器
    viewer = CustomViewer(model, data, traj_direct, traj_slerp, model_pin)
    viewer.run()

