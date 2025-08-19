"""
pose_interpolator.py

Author: Xiangyu Zhu
Last Updated: 2025-08-18

A comprehensive pose interpolation class for robotic applications.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class PoseInterpolator:
    def __init__(self, interpolation_method = "slerp"):
        """
        Initialize the pose interpolator.
        
        Args:
            interpolation_method (str): Default interpolation method.
                Options: "slerp" (quaternion SLERP), "so3" (geodesic on SO(3))
        """
        self.interpolation_method = interpolation_method
        self._validate_method(interpolation_method)
    
    def _validate_method(self, method):
        """Validate the interpolation method."""
        valid_methods = ["slerp", "so3"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid interpolation method: {method}. "
                f"Valid options: {valid_methods}"
            )
    
    # --------------------------------------------------------------------------
    # Quaternion SLERP methods
    # --------------------------------------------------------------------------
    
    def slerp_quaternion(self, q0, q1, t):
        """Perform quaternion SLERP (spherical linear interpolation)."""
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        t = np.asarray(t).reshape(-1, 1)

        dot = np.dot(q0, q1)

        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / np.linalg.norm(q, axis=1, keepdims=True)

        if dot < 0.0:
            q1 = -q1
            dot = -dot

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        s0 = np.sin((1 - t) * theta) / sin_theta
        s1 = np.sin(t * theta) / sin_theta

        q = s0 * q0 + s1 * q1
        return q / np.linalg.norm(q, axis=1, keepdims=True)
    
    def interpolate_se3_with_slerp(self, p0, p1, q0, q1, n, use_slerp = True):
        """Interpolate SE(3) pose using linear position + quaternion SLERP."""
        times = np.linspace(0, 1, n)
        positions = np.linspace(p0, p1, n)

        if use_slerp:
            quats = self.slerp_quaternion(q0, q1, times)
            rots = R.from_quat(quats)
        else:
            rots = [R.from_quat(q0)] * n

        return [(positions[i], rots[i].as_matrix()) for i in range(n)]
    
    # --------------------------------------------------------------------------
    # SO(3) utility methods
    # --------------------------------------------------------------------------
    
    def hat(self, phi):
        """Hat (skew-symmetric) operator: R^3 -> so(3)."""
        phi1, phi2, phi3 = phi
        return np.array([
            [0.0, -phi3, phi2],
            [phi3, 0.0, -phi1],
            [-phi2, phi1, 0.0]
        ])
    
    def rodrigues_exp(self, axis, theta):
        """Rodrigues' formula: axis-angle -> rotation matrix."""
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-15:
            return np.eye(3)

        axis = axis / axis_norm
        K = self.hat(axis)
        I = np.eye(3)
        s, c = np.sin(theta), np.cos(theta)

        return I + s * K + (1.0 - c) * (K @ K)
    
    def axis_angle_from_rotmat(self, rotmat):
        """Extract axis-angle representation from a rotation matrix."""
        tr = np.trace(rotmat)
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if theta < 1e-12:
            return np.array([1.0, 0.0, 0.0]), 0.0

        if np.pi - theta < 1e-6:
            r_plus = (rotmat + np.eye(3)) * 0.5
            axis = np.array([r_plus[0, 0], r_plus[1, 1], r_plus[2, 2]])
            idx = np.argmax(axis)
            if idx == 0:
                x = np.sqrt(max(r_plus[0, 0], 0.0))
                axis = np.array([1.0, 0.0, 0.0]) if x < 1e-12 else np.array([x, r_plus[1, 0] / x, r_plus[2, 0] / x])
            elif idx == 1:
                y = np.sqrt(max(r_plus[1, 1], 0.0))
                axis = np.array([0.0, 1.0, 0.0]) if y < 1e-12 else np.array([r_plus[1, 0] / y, y, r_plus[2, 1] / y])
            else:
                z = np.sqrt(max(r_plus[2, 2], 0.0))
                axis = np.array([0.0, 0.0, 1.0]) if z < 1e-12 else np.array([r_plus[2, 0] / z, r_plus[2, 1] / z, z])
            return axis / (np.linalg.norm(axis) + 1e-15), theta

        axis = np.array([
            rotmat[2, 1] - rotmat[1, 2],
            rotmat[0, 2] - rotmat[2, 0],
            rotmat[1, 0] - rotmat[0, 1]
        ]) / (2.0 * np.sin(theta))
        return axis / (np.linalg.norm(axis) + 1e-15), theta
    
    def rotmat_interp_geodesic(self, r1, r2, t):
        """Interpolate between two rotation matrices along SO(3) geodesic."""
        r_rel = r1.T @ r2
        axis, theta = self.axis_angle_from_rotmat(r_rel)
        return r1 @ self.rodrigues_exp(axis, t * theta)
    
    def interpolate_se3_with_so3(self, p0, p1, r0, r1, n, use_so3 = True):
        """Interpolate SE(3) pose using linear position + SO(3) geodesic."""
        times = np.linspace(0, 1, n)
        positions = np.linspace(p0, p1, n)

        if use_so3:
            rots = [self.rotmat_interp_geodesic(r0, r1, t) for t in times]
        else:
            rots = [r0.copy() for _ in range(n)]

        return [(positions[i], rots[i]) for i in range(n)]
    

    
    # --------------------------------------------------------------------------
    # Main interpolation interface
    # --------------------------------------------------------------------------
    
    def interpolate_pose(self, start_pose, end_pose, n_steps, method = None):
        """
        Main interface for pose interpolation.
        
        Args:
            start_pose: Starting pose. Can be:
                - Tuple: (position, rotation_matrix) or (position, quaternion)
                - Dict: {'pos': position, 'rot': rotation_matrix} or 
                       {'pos': position, 'quat': quaternion}
            end_pose: Ending pose (same format as start_pose)
            n_steps: Number of interpolation steps
            method: Interpolation method ('slerp' or 'so3'). If None, uses default.
            
        Returns:
            List of tuples: [(position (3,), rotation matrix (3,3)), ...]
        """
        method = method or self.interpolation_method
        self._validate_method(method)
        
        # Parse start pose
        if isinstance(start_pose, tuple):
            p0, rot0 = start_pose
            if rot0.shape == (4,):  # quaternion
                q0 = rot0
                r0 = R.from_quat(q0).as_matrix()
            else:  # rotation matrix
                r0 = rot0
                q0 = R.from_matrix(r0).as_quat()
        else:  # dict
            p0 = start_pose['pos']
            if 'quat' in start_pose:
                q0 = start_pose['quat']
                r0 = R.from_quat(q0).as_matrix()
            else:
                r0 = start_pose['rot']
                q0 = R.from_matrix(r0).as_quat()
        
        # Parse end pose
        if isinstance(end_pose, tuple):
            p1, rot1 = end_pose
            if rot1.shape == (4,):  # quaternion
                q1 = rot1
                r1 = R.from_quat(q1).as_matrix()
            else:  # rotation matrix
                r1 = rot1
                q1 = R.from_matrix(r1).as_quat()
        else:  # dict
            p1 = end_pose['pos']
            if 'quat' in end_pose:
                q1 = end_pose['quat']
                r1 = R.from_quat(q1).as_matrix()
            else:
                r1 = end_pose['rot']
                q1 = R.from_matrix(r1).as_quat()
        
        # Convert to numpy arrays
        p0, p1 = np.asarray(p0), np.asarray(p1)
        q0, q1 = np.asarray(q0), np.asarray(q1)
        r0, r1 = np.asarray(r0), np.asarray(r1)
        
        # Perform interpolation
        if method == "slerp":
            return self.interpolate_se3_with_slerp(p0, p1, q0, q1, n_steps, use_slerp=True)
        else:  # so3
            return self.interpolate_se3_with_so3(p0, p1, r0, r1, n_steps, use_so3=True)

    # --------------------------------------------------------------------------
    # Visualization methods
    # --------------------------------------------------------------------------
    
    def plot_trajectory(self, trajectory, step = 5, axis_len = 0.02, title = "Pose Trajectory"):
        """Plot a 3D trajectory with coordinate frames."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        positions = np.array([p for p, _ in trajectory])
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'b-', linewidth=2, label='Trajectory')

        def draw_frame(ax, pos, rot_matrix, length=axis_len, alpha=0.8):
            rot = R.from_matrix(rot_matrix)
            axes = np.eye(3) * length
            colors = ['r', 'g', 'b']
            for vec, c in zip(axes, colors):
                dir_vec = rot.apply(vec)
                ax.quiver(pos[0], pos[1], pos[2],
                         dir_vec[0], dir_vec[1], dir_vec[2],
                         color=c, linewidth=1.5, alpha=alpha)

        for i in range(0, len(trajectory), step):
            draw_frame(ax, *trajectory[i], length=axis_len, alpha=0.8)

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()

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
def example_usage():
    """Example usage of the PoseInterpolator class."""
    
    # Create interpolator
    interpolator = PoseInterpolator(interpolation_method="slerp")
    
    # Define start and end poses
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([1.0, 1.0, 1.0])
    
    start_rot = np.eye(3)
    end_rot = R.from_euler('z', 90, degrees=True).as_matrix()
    
    # Method 1: Using so3
    start_pose_rot = (start_pos, start_rot)
    end_pose_rot = (end_pos, end_rot)
    
    trajectory_so3 = interpolator.interpolate_pose(start_pose_rot, end_pose_rot, n_steps = 50)
    
    # Method 2: Using quaternions
    start_quat = R.from_matrix(start_rot).as_quat()
    end_quat = R.from_matrix(end_rot).as_quat()
    
    start_pose_quat = (start_pos, start_quat)
    end_pose_quat = (end_pos, end_quat)
    
    trajectory_slerp = interpolator.interpolate_pose(start_pose_quat, end_pose_quat, n_steps=50, method="slerp")
    trajectory_so3 = interpolator.interpolate_pose(start_pose_quat, end_pose_quat, n_steps=50, method="so3")
    
    interpolator.plot_trajectory(trajectory_slerp, step = 5, axis_len = 0.05, title = "Pose Trajectory")
    interpolator.plot_trajectory(trajectory_so3, step = 5, axis_len = 0.05, title = "Pose Trajectory")

if __name__ == "__main__":
    example_usage()

    
