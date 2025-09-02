import numpy as np

class Config:
    """配置参数类"""
    # 文件路径
    URDF_PATH = "/home/zxy/MujocoBin/mujoco-learning/model/franka_panda_description/robots/panda_description/urdf/panda.urdf"
    XML_PATH = "/home/zxy/MujocoBin/mujoco_menagerie-main/franka_emika_panda/scene.xml"
    
    # 数据保存路径
    DATA_DIR = "/home/zxy/MujocoBin/Data/NPY/casadi"
    
    # IK求解参数
    JOINT_ID = 7  # 末端执行器关节ID
    EPS = 1e-4    # 收敛阈值
    IT_MAX = 1000 # 最大迭代次数
    DT = 1e-1     # 积分步长
    DAMP = 1e-12  # 阻尼因子
    
    # 轨迹参数
    N_POINTS = 50  # 插值点数
    
    # 仿真参数
    SIM_DT = 0.1  # 仿真时间步长
    
    # 起始和结束关节角度
    Q_START = np.array([0.2, -0.5, 0.2, -1.7, 0.1, 1.6, 0.9, 0.0, 0.0])
    Q_END = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.8, 0.0, 0.0])
