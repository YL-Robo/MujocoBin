import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import time

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

if __name__ == '__main__':
    # start and end orientations
    start_quat = np.array([0.0, 0.0, 0.0, 1.0])
    end_quat = np.array([0.0, 0.7071, 0.0, 0.7071])

    # Number of interpolation steps
    N = 50
    times = np.linspace(0, 1, N)
    
    # Interpolate using slerp in scipy  
    # 起点和终点的四元数转换为旋转对象，得到关键帧的旋转。
    key_rots = R.from_quat([start_quat, end_quat]) 
    slerp = Slerp([0, 1], key_rots)

    start1 = time.perf_counter()
    rot_scipy = slerp(times)
    end1 = time.perf_counter()

    print(f"Scipy Slerp耗时: {end1 - start1:.6f}秒")
    euler_scipy = rot_scipy.as_euler('zyx', degrees=True)

    normalized_times = (times - np.min(times)) / (np.max(times) - np.min(times))
    
    start2 = time.perf_counter()
    quats_custom = Slerp_test(start_quat, end_quat, normalized_times)
    quats_custom = np.array(quats_custom)
    end2 = time.perf_counter()
    
    print(f"自定义Slerp耗时: {end2 - start2:.6f}秒")
    rot_custom = R.from_quat(quats_custom)
    euler_custom = rot_custom.as_euler('zyx', degrees=True)

    # ---------- 将旋转作用在单位向量 [1, 0, 0] 上 ----------
    v = np.array([1, 0, 0])
    vecs_scipy = rot_scipy.apply(v)
    vecs_custom = rot_custom.apply(v)

    # ---------- 绘制三维旋转路径 ----------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画单位球面
    u, w = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = np.cos(u) * np.sin(w)
    y = np.sin(u) * np.sin(w)
    z = np.cos(w)
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.2)

    # Scipy Slerp 路径
    ax.plot(vecs_scipy[:, 0], vecs_scipy[:, 1], vecs_scipy[:, 2],
            label='Scipy Slerp', color='blue', linewidth=2)

    # Custom Slerp 路径
    ax.plot(vecs_custom[:, 0], vecs_custom[:, 1], vecs_custom[:, 2],
            label='Custom Slerp', color='red', linestyle='--', linewidth=2)

    # 起点与终点
    ax.scatter(vecs_scipy[0, 0], vecs_scipy[0, 1], vecs_scipy[0, 2], color='green', label='Start')
    ax.scatter(vecs_scipy[-1, 0], vecs_scipy[-1, 1], vecs_scipy[-1, 2], color='black', label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Scipy Slerp and Custom Slerp')
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()