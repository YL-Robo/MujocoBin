import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp

# 起点与终点四元数
q0 = [0, 0, 0, 1]
q1 = R.from_euler('x', np.pi).as_quat()

# 创建关键帧 Rotation 对象
key_times = [0, 1]
key_rots = R.from_quat([q0, q1])

# 创建 Slerp 插值器
slerp = Slerp(key_times, key_rots)

# 插值时间点
interp_times = np.linspace(0, 1, 20)
interp_rots = slerp(interp_times)

# 可视化插值结果（Z轴方向）
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, 0, 0, 1, color='gray', linewidth=2, label='Z轴原始方向')

for rot in interp_rots:
    z_dir = rot.as_matrix()[:, 2]
    ax.quiver(0, 0, 0, z_dir[0], z_dir[1], z_dir[2], color='b', alpha=0.5)

ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("从 q0 到 q1 的四元数插值（Z轴方向）")
ax.legend()
plt.tight_layout()
plt.show()
