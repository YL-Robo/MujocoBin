"""
utils.py

Author: Xiangyu Zhu
Last Updated: 2025-08-18

General interpolation and math utility functions for pose representation.

Includes:
- QuaternionUtils: SLERP interpolation
- SO3Utils: Rodrigues' formula, axis-angle extraction, geodesic interpolation
- SE3Interpolator: SE(3) pose interpolation
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# ------------------------------------------------------------------------------
# Quaternion SLERP
# ------------------------------------------------------------------------------
class QuaternionUtils:
    """Utility functions for quaternion operations."""

    @staticmethod
    def quaternion_slerp(q0, q1, t):
        """Perform quaternion SLERP (spherical linear interpolation).

        Args:
            q0 (array-like): Start quaternion [x, y, z, w].
            q1 (array-like): End quaternion [x, y, z, w].
            t (float or ndarray): Interpolation parameter(s) in [0, 1].

        Returns:
            ndarray: Interpolated quaternion(s), normalized.
        """
        q0 = np.asarray(q0)
        q1 = np.asarray(q1)
        t = np.asarray(t).reshape(-1, 1)

        dot = np.dot(q0, q1)

        # Linear interpolation for nearly identical quaternions
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / np.linalg.norm(q, axis=1, keepdims=True)

        # Ensure shortest path
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        s0 = np.sin((1 - t) * theta) / sin_theta
        s1 = np.sin(t * theta) / sin_theta

        q = s0 * q0 + s1 * q1
        return q / np.linalg.norm(q, axis=1, keepdims=True)

# ------------------------------------------------------------------------------
# SO3 interpolation
# ------------------------------------------------------------------------------
class SO3Utils:
    """Utility functions for SO(3) operations."""
    @staticmethod
    def hat(phi):
        """Hat (skew-symmetric) operator: R^3 -> so(3).

        Args:
            phi (ndarray): 3D vector.

        Returns:
            ndarray: 3x3 skew-symmetric matrix.
        """
        phi1, phi2, phi3 = phi
        return np.array([
            [0.0, -phi3, phi2],
            [phi3, 0.0, -phi1],
            [-phi2, phi1, 0.0]
        ])

    @staticmethod
    def rodrigues_exp(axis, theta):
        """Rodrigues' formula: axis-angle -> rotation matrix.

        Args:
            axis (ndarray): Rotation axis (3,).
            theta (float): Rotation angle in radians.

        Returns:
            ndarray: 3x3 rotation matrix.
        """
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-15:
            return np.eye(3)

        axis = axis / axis_norm
        K = SO3Utils.hat(axis)
        I = np.eye(3)
        s, c = np.sin(theta), np.cos(theta)

        return I + s * K + (1.0 - c) * (K @ K)

    @staticmethod
    def axis_angle_from_rotmat(rotmat):
        """Extract axis-angle representation from a rotation matrix.

        Args:
            rotmat (ndarray): 3x3 rotation matrix.

        Returns:
            tuple:
                - axis (ndarray): Unit axis vector (3,).
                - theta (float): Rotation angle in radians.
        """
        tr = np.trace(rotmat)
        cos_theta = (tr - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # if theta close to zero, return arbitrary axis
        if theta < 1e-12:
            return np.array([1.0, 0.0, 0.0]), 0.0

        # if theta close to pi, handle numerically stable extraction
        if np.pi - theta < 1e-6:  # ~180 degrees
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
            
            if np.linalg.norm(axis) < 1e-12:
                axis = np.array([r_plus[2,1] - r_plus[1,2],
                                r_plus[0,2] - r_plus[2,0],
                                r_plus[1,0] - r_plus[0,1]]) / (2.0 * np.sin(theta))
            else:
                axis = axis / np.linalg.norm(axis)
            return axis, theta

        axis = np.array([
            rotmat[2, 1] - rotmat[1, 2],
            rotmat[0, 2] - rotmat[2, 0],
            rotmat[1, 0] - rotmat[0, 1]
        ]) / (2.0 * np.sin(theta))
        if np.linalg.norm(axis) < 1e-15:
            # Degenerate; pick any
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / np.linalg.norm(axis)
        return axis, theta

    @staticmethod
    def rotmat_interp_geodesic(r1, r2, t):
        """Interpolate between two rotation matrices along SO(3) geodesic.

        Args:
            r1 (ndarray): Start rotation matrix (3x3).
            r2 (ndarray): End rotation matrix (3x3).
            t (float): Interpolation parameter in [0, 1].

        Returns:
            ndarray: Interpolated rotation matrix (3x3).
        """
        r_rel = r1.T @ r2
        axis, theta = SO3Utils.axis_angle_from_rotmat(r_rel)
        return r1 @ SO3Utils.rodrigues_exp(axis, t * theta)

# ------------------------------------------------------------------------------
# position linear interpolation + rotation interpolation (slerp/SO3)
# ------------------------------------------------------------------------------
class SE3Interpolator:
    """Interpolation utilities for SE(3) poses."""
    
    @staticmethod
    def interpolate_se3_with_slerp(p0, p1, q0, q1, n, use_slerp=True):
        """Interpolate SE(3) pose using linear position + quaternion SLERP.

        Args:
            p0 (ndarray): Start position (3,).
            p1 (ndarray): End position (3,).
            q0 (ndarray): Start quaternion [x, y, z, w].
            q1 (ndarray): End quaternion [x, y, z, w].
            n (int): Number of interpolation steps.
            use_slerp (bool): Whether to apply SLERP for orientation.

        Returns:
            list of tuple: [(position (3,), rotation matrix (3,3)), ...]
        """
        times = np.linspace(0, 1, n)
        positions = np.linspace(p0, p1, n)

        if use_slerp:
            quats = QuaternionUtils.quaternion_slerp(q0, q1, times)
            rots = R.from_quat(quats)
        else:
            rots = [R.from_quat(q0)] * n

        return [(positions[i], rots[i].as_matrix()) for i in range(n)]

    @staticmethod
    def interpolate_se3_with_so3(p0, p1, r0, r1, n, use_so3=True):
        """Interpolate SE(3) pose using linear position + SO(3) geodesic.

        Args:
            p0 (ndarray): Start position (3,).
            p1 (ndarray): End position (3,).
            r0 (ndarray): Start rotation matrix (3,3).
            r1 (ndarray): End rotation matrix (3,3).
            n (int): Number of interpolation steps.
            use_so3 (bool): Whether to apply geodesic interpolation.

        Returns:
            list of tuple: [(position (3,), rotation matrix (3,3)), ...]
        """
        times = np.linspace(0, 1, n)
        positions = np.linspace(p0, p1, n)

        if use_so3:
            rots = [SO3Utils.rotmat_interp_geodesic(r0, r1, t) for t in times]
        else:
            rots = [r0.copy() for _ in range(n)]

        return [(positions[i], rots[i]) for i in range(n)]
