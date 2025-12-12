"""
Plot comparison between sinusoidal trajectory and teleoperation trajectory.

This script:
1. Runs simulations for both sinusoidal and teleoperation trajectories
2. Plots:
   - V (Lyapunov function) over time for both
   - Per joint trajectory vs actual motion for all joints in right and left arm
   - Normal error (L2 norm of position error) for both over time
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import mujoco as mj
from pathlib import Path
import sys

# Import functions from follower.py
sys.path.insert(0, str(Path(__file__).parent))
from follower import (
    build_arm_maps,
    load_trajectory_from_npz,
    desired_arm_trajectories,
    desired_arm_trajectories_sinusoidal_shoulders,
    AdaptiveController,
    apply_adaptive_control,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    SLAVE_XML_PATH,
    TRAJECTORY_NPZ_PATH,
)

# ==========================
# CONFIGURATION
# ==========================

# Path to trajectory file
TRAJECTORY_PATH = Path(__file__).parent / "trajectories" / "traj_20251211_200112.npz"

# Simulation parameters
USE_ADAPTIVE_CONTROL = True
ADAPT_GAINS = False

# Sinusoidal trajectory parameters
SINUSOIDAL_AMPLITUDE = 0.5  # radians
SINUSOIDAL_FREQUENCY = 0.5  # Hz
SINUSOIDAL_DURATION = 20.0  # seconds


def run_simulation(trajectory_type='sinusoidal', save_data=True):
    """
    Run simulation for either sinusoidal or teleoperation trajectory.
    
    Args:
        trajectory_type: 'sinusoidal' or 'teleop'
        save_data: If True, save simulation data to file
    
    Returns:
        Dictionary with simulation results:
        - time: array of time values
        - q_left_des: desired left arm positions (T, n_left)
        - q_right_des: desired right arm positions (T, n_right)
        - q_left_actual: actual left arm positions (T, n_left)
        - q_right_actual: actual right arm positions (T, n_right)
        - V_left: Lyapunov function values for left arm (T,)
        - V_right: Lyapunov function values for right arm (T,)
        - error_left: L2 norm of position error for left arm (T,)
        - error_right: L2 norm of position error for right arm (T,)
    """
    print(f"\n{'='*60}")
    print(f"Running simulation: {trajectory_type.upper()}")
    print(f"{'='*60}")
    
    # Load model
    model = mj.MjModel.from_xml_path(SLAVE_XML_PATH)
    data = mj.MjData(model)
    dt = model.opt.timestep
    
    # Build arm maps
    left_qpos_idx, left_dof_idx, left_act_idx = build_arm_maps(model, LEFT_ARM_JOINT_NAMES)
    right_qpos_idx, right_dof_idx, right_act_idx = build_arm_maps(model, RIGHT_ARM_JOINT_NAMES)
    
    n_left = len(left_qpos_idx)
    n_right = len(right_qpos_idx)
    
    # Load trajectory if needed
    if trajectory_type == 'teleop':
        load_trajectory_from_npz(TRAJECTORY_PATH, left_qpos_idx, right_qpos_idx, dt)
        # Get trajectory duration
        traj_data = np.load(TRAJECTORY_PATH)
        sim_duration = traj_data['qpos'].shape[0] * traj_data['dt']
    else:
        sim_duration = SINUSOIDAL_DURATION
    
    # Initialize pose
    if trajectory_type == 'sinusoidal':
        q_left_des_0, q_right_des_0 = desired_arm_trajectories_sinusoidal_shoulders(
            0.0, n_left, n_right, SINUSOIDAL_AMPLITUDE, SINUSOIDAL_FREQUENCY
        )
    else:
        q_left_des_0, q_right_des_0 = desired_arm_trajectories(0.0, n_left, n_right)
    
    data.qpos[left_qpos_idx] = q_left_des_0
    data.qpos[right_qpos_idx] = q_right_des_0
    mj.mj_forward(model, data)
    
    # Initialize controllers
    if USE_ADAPTIVE_CONTROL:
        left_adaptive_ctrl = AdaptiveController(
            n_left, dt, model, left_qpos_idx, left_dof_idx,
            known_masses=None, known_inertias=None, adapt_gains=ADAPT_GAINS
        )
        right_adaptive_ctrl = AdaptiveController(
            n_right, dt, model, right_qpos_idx, right_dof_idx,
            known_masses=None, known_inertias=None, adapt_gains=ADAPT_GAINS
        )
    
    # Storage for results
    time_log = []
    q_left_des_log = []
    q_right_des_log = []
    q_left_actual_log = []
    q_right_actual_log = []
    V_left_log = []
    V_right_log = []
    error_left_log = []
    error_right_log = []
    
    # Variables for numerical differentiation
    q_left_des_prev = None
    q_right_des_prev = None
    qdot_left_des_prev = None
    qdot_right_des_prev = None
    t_prev = 0.0
    
    # Run simulation
    print(f"Running simulation for {sim_duration:.2f} seconds...")
    step_count = 0
    print_every = int(1.0 / dt)  # Print every second
    
    while data.time < sim_duration:
        t = data.time
        
        # Get desired trajectory
        if trajectory_type == 'sinusoidal':
            q_left_des, q_right_des = desired_arm_trajectories_sinusoidal_shoulders(
                t, n_left, n_right, SINUSOIDAL_AMPLITUDE, SINUSOIDAL_FREQUENCY
            )
        else:
            q_left_des, q_right_des = desired_arm_trajectories(t, n_left, n_right)
        
        # Calculate desired velocity and acceleration
        if q_left_des_prev is not None and t > t_prev:
            dt_traj = t - t_prev
            qdot_left_des = (q_left_des - q_left_des_prev) / dt_traj
            qdot_right_des = (q_right_des - q_right_des_prev) / dt_traj
            
            if qdot_left_des_prev is not None:
                qddot_left_des = (qdot_left_des - qdot_left_des_prev) / dt_traj
                qddot_right_des = (qdot_right_des - qdot_right_des_prev) / dt_traj
            else:
                qddot_left_des = np.zeros(n_left)
                qddot_right_des = np.zeros(n_right)
        else:
            qdot_left_des = np.zeros(n_left)
            qdot_right_des = np.zeros(n_right)
            qddot_left_des = np.zeros(n_left)
            qddot_right_des = np.zeros(n_right)
        
        # Save for next iteration
        q_left_des_prev = q_left_des.copy()
        q_right_des_prev = q_right_des.copy()
        qdot_left_des_prev = qdot_left_des.copy()
        qdot_right_des_prev = qdot_right_des.copy()
        t_prev = t
        
        # Apply control
        if USE_ADAPTIVE_CONTROL:
            V_left = apply_adaptive_control(
                model, data, left_qpos_idx, left_dof_idx, left_act_idx,
                q_left_des, qdot_left_des, qddot_left_des, left_adaptive_ctrl
            )
            V_right = apply_adaptive_control(
                model, data, right_qpos_idx, right_dof_idx, right_act_idx,
                q_right_des, qdot_right_des, qddot_right_des, right_adaptive_ctrl
            )
        else:
            # Simple PD control (not implemented here, would need to import)
            V_left = 0.0
            V_right = 0.0
            print("Warning: PD control not implemented, using adaptive control only")
        
        # Step simulation
        mj.mj_step(model, data)
        
        # Get actual positions
        q_left_actual = data.qpos[left_qpos_idx]
        q_right_actual = data.qpos[right_qpos_idx]
        
        # Calculate errors
        error_left = np.linalg.norm(q_left_des - q_left_actual)
        error_right = np.linalg.norm(q_right_des - q_right_actual)
        
        # Log data
        time_log.append(t)
        q_left_des_log.append(q_left_des.copy())
        q_right_des_log.append(q_right_des.copy())
        q_left_actual_log.append(q_left_actual.copy())
        q_right_actual_log.append(q_right_actual.copy())
        V_left_log.append(V_left)
        V_right_log.append(V_right)
        error_left_log.append(error_left)
        error_right_log.append(error_right)
        
        # Print progress
        if step_count % print_every == 0:
            print(f"  t = {t:.2f}s, error_left = {error_left:.5f}, error_right = {error_right:.5f}")
        
        step_count += 1
    
    # Convert to numpy arrays
    results = {
        'time': np.array(time_log),
        'q_left_des': np.array(q_left_des_log),
        'q_right_des': np.array(q_right_des_log),
        'q_left_actual': np.array(q_left_actual_log),
        'q_right_actual': np.array(q_right_actual_log),
        'V_left': np.array(V_left_log),
        'V_right': np.array(V_right_log),
        'error_left': np.array(error_left_log),
        'error_right': np.array(error_right_log),
    }
    
    # Save data if requested
    if save_data:
        output_path = Path(__file__).parent / f"simulation_data_{trajectory_type}.npz"
        np.savez(output_path, **results)
        print(f"Saved simulation data to: {output_path}")
    
    print(f"Simulation completed: {len(time_log)} timesteps")
    return results


def load_simulation_data(trajectory_type='sinusoidal'):
    """Load previously saved simulation data."""
    data_path = Path(__file__).parent / f"simulation_data_{trajectory_type}.npz"
    if data_path.exists():
        print(f"Loading simulation data from: {data_path}")
        data = np.load(data_path)
        return {
            'time': data['time'],
            'q_left_des': data['q_left_des'],
            'q_right_des': data['q_right_des'],
            'q_left_actual': data['q_left_actual'],
            'q_right_actual': data['q_right_actual'],
            'V_left': data['V_left'],
            'V_right': data['V_right'],
            'error_left': data['error_left'],
            'error_right': data['error_right'],
        }
    else:
        return None


def plot_comparison(sinusoidal_data, teleop_data):
    """
    Create plots comparing sinusoidal and teleoperation trajectories.
    
    Creates 3 separate images:
    1. Complex trajectory (teleop): All joints desired vs actual + total error
    2. Sinusoidal trajectory: Only joint 1 (shoulder_pitch) for both arms
    3. Lyapunov function over time for both trajectories
    
    Args:
        sinusoidal_data: Results dictionary from sinusoidal simulation
        teleop_data: Results dictionary from teleoperation simulation
    """
    # Get joint names
    n_left = teleop_data['q_left_des'].shape[1]
    n_right = teleop_data['q_right_des'].shape[1]
    
    left_joint_names = LEFT_ARM_JOINT_NAMES[:n_left]
    right_joint_names = RIGHT_ARM_JOINT_NAMES[:n_right]
    
    # ===== Image 1: Complex Trajectory (Teleop) =====
    # Right arm: all joints desired vs actual
    # Left arm: all joints desired vs actual
    # Total joint error
    fig1 = plt.figure(figsize=(16, 12))
    
    # Right arm joints (all 7 joints)
    for i in range(n_right):
        ax = plt.subplot(n_right + n_left + 1, 1, i + 1)
        joint_name = right_joint_names[i]
        short_name = joint_name.replace('right_', '').replace('_joint', '')
        ax.plot(teleop_data['time'], teleop_data['q_right_des'][:, i], 
                'b-', linewidth=2, label='Desired', color='blue')
        ax.plot(teleop_data['time'], teleop_data['q_right_actual'][:, i], 
                'b--', linewidth=2, label='Actual', color='blue')
        ax.set_ylabel(f'Joint {i+1}\n(rad)', fontsize=10)
        ax.set_title(f'Right Arm - {short_name}: Desired vs Actual', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        if i < n_right - 1:
            ax.set_xticklabels([])
    
    # Left arm joints (all 7 joints)
    for i in range(n_left):
        ax = plt.subplot(n_right + n_left + 1, 1, n_right + i + 1)
        joint_name = left_joint_names[i]
        short_name = joint_name.replace('left_', '').replace('_joint', '')
        ax.plot(teleop_data['time'], teleop_data['q_left_des'][:, i], 
                'b-', linewidth=2, label='Desired', color='blue')
        ax.plot(teleop_data['time'], teleop_data['q_left_actual'][:, i], 
                'b--', linewidth=2, label='Actual', color='blue')
        ax.set_ylabel(f'Joint {i+1}\n(rad)', fontsize=10)
        ax.set_title(f'Left Arm - {short_name}: Desired vs Actual', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        if i < n_left - 1:
            ax.set_xticklabels([])
    
    # Total joint error (sum of all joint errors)
    ax_total = plt.subplot(n_right + n_left + 1, 1, n_right + n_left + 1)
    # Calculate per-joint errors
    left_joint_errors = np.abs(teleop_data['q_left_des'] - teleop_data['q_left_actual'])
    right_joint_errors = np.abs(teleop_data['q_right_des'] - teleop_data['q_right_actual'])
    total_error = np.sum(left_joint_errors, axis=1) + np.sum(right_joint_errors, axis=1)
    ax_total.plot(teleop_data['time'], total_error, 'b-', linewidth=2, color='blue')
    ax_total.set_xlabel('Time (s)', fontsize=12)
    ax_total.set_ylabel('Total Joint Error (rad)', fontsize=12)
    ax_total.set_title('Total Joint Error (Sum of All Joint Errors)', fontsize=14, fontweight='bold')
    ax_total.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path1 = Path(__file__).parent / "trajectory_comparison_teleop.png"
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Saved teleop trajectory plots to: {output_path1}")
    plt.close()
    
    # ===== Image 2: Sinusoidal Trajectory (Only Joint 1) =====
    fig2 = plt.figure(figsize=(14, 6))
    
    # Right arm joint 1 (shoulder_pitch)
    ax1 = plt.subplot(2, 1, 1)
    joint_name = right_joint_names[0]
    short_name = joint_name.replace('right_', '').replace('_joint', '')
    ax1.plot(sinusoidal_data['time'], sinusoidal_data['q_right_des'][:, 0], 
            'r-', linewidth=2, label='Desired', color='red')
    ax1.plot(sinusoidal_data['time'], sinusoidal_data['q_right_actual'][:, 0], 
            'r--', linewidth=2, label='Actual', color='red')
    ax1.set_ylabel('Joint Angle (rad)', fontsize=12)
    ax1.set_title(f'Right Arm - {short_name}: Desired vs Actual (Sinusoidal)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Left arm joint 1 (shoulder_pitch)
    ax2 = plt.subplot(2, 1, 2)
    joint_name = left_joint_names[0]
    short_name = joint_name.replace('left_', '').replace('_joint', '')
    ax2.plot(sinusoidal_data['time'], sinusoidal_data['q_left_des'][:, 0], 
            'r-', linewidth=2, label='Desired', color='red')
    ax2.plot(sinusoidal_data['time'], sinusoidal_data['q_left_actual'][:, 0], 
            'r--', linewidth=2, label='Actual', color='red')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Joint Angle (rad)', fontsize=12)
    ax2.set_title(f'Left Arm - {short_name}: Desired vs Actual (Sinusoidal)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = Path(__file__).parent / "trajectory_comparison_sinusoidal.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved sinusoidal trajectory plots to: {output_path2}")
    plt.close()
    
    # ===== Image 3: Lyapunov Function over Time =====
    fig3 = plt.figure(figsize=(14, 6))
    ax = plt.subplot(1, 1, 1)
    ax.plot(sinusoidal_data['time'], sinusoidal_data['V_left'], 'r-', 
            label='Sinusoidal (Left)', linewidth=2, color='red')
    ax.plot(sinusoidal_data['time'], sinusoidal_data['V_right'], 'r--', 
            label='Sinusoidal (Right)', linewidth=2, color='red')
    ax.plot(teleop_data['time'], teleop_data['V_left'], 'b-', 
            label='Teleop (Left)', linewidth=2, color='blue')
    ax.plot(teleop_data['time'], teleop_data['V_right'], 'b--', 
            label='Teleop (Right)', linewidth=2, color='blue')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('V (Lyapunov Function)', fontsize=12)
    ax.set_title('Lyapunov Function V over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path3 = Path(__file__).parent / "trajectory_comparison_lyapunov.png"
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Saved Lyapunov function plot to: {output_path3}")
    plt.close()
    
    print("\nAll plots saved successfully!")


def main():
    """Main function to run simulations and create plots."""
    import argparse
    parser = argparse.ArgumentParser(description='Compare sinusoidal and teleoperation trajectories')
    parser.add_argument('--skip-sim', action='store_true', 
                       help='Skip simulation and load existing data')
    parser.add_argument('--sinusoidal-only', action='store_true',
                       help='Only run sinusoidal simulation')
    parser.add_argument('--teleop-only', action='store_true',
                       help='Only run teleoperation simulation')
    args = parser.parse_args()
    
    # Run or load simulations
    if args.skip_sim:
        print("Loading existing simulation data...")
        sinusoidal_data = load_simulation_data('sinusoidal')
        teleop_data = load_simulation_data('teleop')
        
        if sinusoidal_data is None or teleop_data is None:
            print("Error: Could not load simulation data. Run without --skip-sim first.")
            return
    else:
        sinusoidal_data = None
        teleop_data = None
        
        if not args.teleop_only:
            print("Running sinusoidal simulation...")
            sinusoidal_data = run_simulation('sinusoidal', save_data=True)
        
        if not args.sinusoidal_only:
            print("\nRunning teleoperation simulation...")
            teleop_data = run_simulation('teleop', save_data=True)
    
    # Create plots
    if sinusoidal_data is not None and teleop_data is not None:
        print("\nCreating comparison plots...")
        plot_comparison(sinusoidal_data, teleop_data)
    else:
        print("Error: Need both sinusoidal and teleoperation data to create comparison plots.")


if __name__ == "__main__":
    main()
