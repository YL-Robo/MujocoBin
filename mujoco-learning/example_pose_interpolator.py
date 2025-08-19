"""
example_pose_interpolator.py

Author: Xiangyu Zhu
Last Updated: 2025-08-18

Example usage of PoseInterpolator class for robotic pose interpolation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pose_interpolator import PoseInterpolator


def example_basic_usage():
    """Basic usage example of PoseInterpolator."""
    print("=== Basic Usage Example ===")
    
    # Create interpolator with SO(3) geodesic interpolation
    interpolator = PoseInterpolator(interpolation_method="so3")
    
    # Define start and end poses
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([1.0, 1.0, 1.0])
    
    # Start rotation (identity)
    start_rot = np.eye(3)
    
    # End rotation (90 degree rotation around Z-axis)
    end_rot = R.from_euler('z', 90, degrees=True).as_matrix()
    
    print(f"Start position: {start_pos}")
    print(f"End position: {end_pos}")
    print(f"Start rotation:\n{start_rot}")
    print(f"End rotation:\n{end_rot}")
    
    # Generate trajectory
    trajectory = interpolator.interpolate_pose(
        start_pose=(start_pos, start_rot),
        end_pose=(end_pos, end_rot),
        n_steps=50
    )
    
    print(f"Generated trajectory with {len(trajectory)} waypoints")
    
    # Plot the trajectory
    interpolator.plot_trajectory(trajectory, title="Basic SO(3) Trajectory")
    
    return trajectory


def example_quaternion_slerp():
    """Example using quaternion SLERP interpolation."""
    print("\n=== Quaternion SLERP Example ===")
    
    # Create interpolator with SLERP
    interpolator = PoseInterpolator(interpolation_method="slerp")
    
    # Define poses using quaternions
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([1.0, 0.0, 1.0])
    
    # Start quaternion (identity)
    start_quat = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
    
    # End quaternion (45 degree rotation around X-axis)
    end_quat = R.from_euler('x', 45, degrees=True).as_quat()
    
    print(f"Start quaternion: {start_quat}")
    print(f"End quaternion: {end_quat}")
    
    # Generate trajectory using quaternions
    trajectory = interpolator.interpolate_pose(
        start_pose=(start_pos, start_quat),
        end_pose=(end_pos, end_quat),
        n_steps=50,
        method="slerp"
    )
    
    print(f"Generated SLERP trajectory with {len(trajectory)} waypoints")
    
    # Plot the trajectory
    interpolator.plot_trajectory(trajectory, title="Quaternion SLERP Trajectory")
    
    return trajectory


def example_comparison():
    """Compare SO(3) geodesic vs quaternion SLERP."""
    print("\n=== Comparison Example ===")
    
    interpolator = PoseInterpolator()
    
    # Define poses
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([1.0, 1.0, 0.5])
    
    # Complex rotation: combination of rotations around multiple axes
    start_rot = np.eye(3)
    end_rot = R.from_euler('xyz', [30, 60, 45], degrees=True).as_matrix()
    
    # Generate SO(3) trajectory
    trajectory_so3 = interpolator.interpolate_pose(
        start_pose=(start_pos, start_rot),
        end_pose=(end_pos, end_rot),
        n_steps=50,
        method="so3"
    )
    
    # Generate SLERP trajectory
    start_quat = R.from_matrix(start_rot).as_quat()
    end_quat = R.from_matrix(end_rot).as_quat()
    
    trajectory_slerp = interpolator.interpolate_pose(
        start_pose=(start_pos, start_quat),
        end_pose=(end_pos, end_quat),
        n_steps=50,
        method="slerp"
    )
    
    # Compare trajectories
    interpolator.compare_trajectories(
        trajectory_so3, trajectory_slerp,
        labels=("SO(3) Geodesic", "Quaternion SLERP"),
        step=10
    )
    
    return trajectory_so3, trajectory_slerp


def example_multi_waypoint():
    """Example with multiple waypoints."""
    print("\n=== Multi-waypoint Example ===")
    
    interpolator = PoseInterpolator()
    
    # Define multiple waypoints
    waypoints = [
        # Waypoint 1: Start position
        (np.array([0.0, 0.0, 0.0]), np.eye(3)),
        
        # Waypoint 2: Move up and rotate around X
        (np.array([0.5, 0.0, 0.5]), R.from_euler('x', 30, degrees=True).as_matrix()),
        
        # Waypoint 3: Move right and rotate around Y
        (np.array([1.0, 0.5, 0.5]), R.from_euler('y', 45, degrees=True).as_matrix()),
        
        # Waypoint 4: Move back and rotate around Z
        (np.array([0.5, 1.0, 1.0]), R.from_euler('z', 60, degrees=True).as_matrix()),
        
        # Waypoint 5: Return to start with different orientation
        (np.array([0.0, 0.0, 0.0]), R.from_euler('xyz', [15, 30, 45], degrees=True).as_matrix()),
    ]
    
    print(f"Generating trajectory through {len(waypoints)} waypoints...")
    
    # Generate complete trajectory
    trajectory = interpolator.generate_trajectory(
        waypoints=waypoints,
        steps_per_segment=30
    )
    
    print(f"Generated multi-waypoint trajectory with {len(trajectory)} total waypoints")
    
    # Plot the trajectory
    interpolator.plot_trajectory(
        trajectory, 
        title="Multi-waypoint Trajectory",
        step=10
    )
    
    return trajectory


def example_dict_format():
    """Example using dictionary format for poses."""
    print("\n=== Dictionary Format Example ===")
    
    interpolator = PoseInterpolator()
    
    # Define poses using dictionary format
    start_pose = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'rot': np.eye(3)
    }
    
    end_pose = {
        'pos': np.array([1.0, 0.0, 1.0]),
        'quat': R.from_euler('xyz', [30, 45, 60], degrees=True).as_quat()
    }
    
    print("Start pose (dict):", start_pose)
    print("End pose (dict):", end_pose)
    
    # Generate trajectory
    trajectory = interpolator.interpolate_pose(
        start_pose=start_pose,
        end_pose=end_pose,
        n_steps=50
    )
    
    print(f"Generated trajectory with {len(trajectory)} waypoints")
    
    # Plot the trajectory
    interpolator.plot_trajectory(trajectory, title="Dictionary Format Trajectory")
    
    return trajectory


def example_robot_arm_trajectory():
    """Example simulating a robot arm trajectory."""
    print("\n=== Robot Arm Trajectory Example ===")
    
    interpolator = PoseInterpolator()
    
    # Simulate a robot arm picking and placing operation
    waypoints = [
        # Home position
        (np.array([0.0, 0.0, 0.5]), np.eye(3)),
        
        # Approach position above object
        (np.array([0.3, 0.2, 0.3]), R.from_euler('z', 45, degrees=True).as_matrix()),
        
        # Grasp position
        (np.array([0.3, 0.2, 0.1]), R.from_euler('z', 45, degrees=True).as_matrix()),
        
        # Lift object
        (np.array([0.3, 0.2, 0.4]), R.from_euler('z', 45, degrees=True).as_matrix()),
        
        # Move to target area
        (np.array([0.6, 0.4, 0.4]), R.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()),
        
        # Place position
        (np.array([0.6, 0.4, 0.15]), R.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()),
        
        # Return to home
        (np.array([0.0, 0.0, 0.5]), np.eye(3)),
    ]
    
    print("Generating robot arm pick-and-place trajectory...")
    
    # Generate trajectory with more steps for smoother motion
    trajectory = interpolator.generate_trajectory(
        waypoints=waypoints,
        steps_per_segment=40
    )
    
    print(f"Generated robot arm trajectory with {len(trajectory)} waypoints")
    
    # Plot the trajectory
    interpolator.plot_trajectory(
        trajectory, 
        title="Robot Arm Pick-and-Place Trajectory",
        step=15
    )
    
    return trajectory


def main():
    """Run all examples."""
    print("PoseInterpolator Examples")
    print("=" * 50)
    
    # Run all examples
    basic_traj = example_basic_usage()
    slerp_traj = example_quaternion_slerp()
    so3_traj, slerp_comp_traj = example_comparison()
    multi_traj = example_multi_waypoint()
    dict_traj = example_dict_format()
    robot_traj = example_robot_arm_trajectory()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("You can now use PoseInterpolator in your robotic applications.")
    
    # Keep plots open
    plt.show()


if __name__ == "__main__":
    main()
