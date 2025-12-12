"""Teleoperate Unitree G1 humanoid robot arms using differential IK with hand tracking.

Requires: pip install 'mink[examples]'
Adapted from examples/11_diffik_aloha.py

This script is configured for the G1 model in assets/unitree_g1/.
It uses scene_teleop.xml which includes mocap target bodies for IK control.

The script maps Vision Pro hand tracking to G1's 7-DOF arms:
- Left/right hands control left/right arms
- Hand pose (from thumb/index finger) maps to end-effector pose
- Differential IK solves for joint velocities to reach target poses
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional, Sequence
from datetime import datetime

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import mink
    from loop_rate_limiters import RateLimiter
except ImportError:
    print("This example requires mink and its dependencies.")
    print("Install with: pip install 'mink[examples]'")
    sys.exit(1)

_HERE = Path(__file__).parent

# Path to G1 XML file (using scene_teleop.xml which includes mocap targets)
_XML = os.path.join(_HERE, "..", "assets", "unitree_g1", "scene_teleop.xml")

# G1 arm joint names (7 DOF per arm)
_LEFT_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

_RIGHT_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Velocity limits for G1 arms (rad/s) - adjust based on G1 specifications
_VELOCITY_LIMITS = {k: np.pi / 2 for k in _LEFT_JOINT_NAMES + _RIGHT_JOINT_NAMES}


def hand2pose(hand, side="right", euler=[0, 0, 0]):
    """
    Convert hand tracking data to a 4x4 pose matrix for robot control.
    
    Uses the wrist transform from hand tracking data with rotation correction.
    
    Args:
        hand: Dictionary containing hand tracking data from VisionProStreamer
        side: "left" or "right" hand
        euler: [x, y, z] rotation in degrees to align hand frame with robot frame
    
    Returns:
        4x4 pose matrix
    """
    wrist = np.array(hand[f"{side}_wrist"], dtype=np.float64)
    
    # Remove batch dimension if present (shape might be (1, 4, 4) instead of (4, 4))
    if wrist.ndim == 3 and wrist.shape[0] == 1:
        wrist = wrist[0]
    
    # Ensure wrist is 4x4
    if wrist.shape != (4, 4):
        raise ValueError(f"Expected wrist to be 4x4, got shape {wrist.shape}")
    
    # Apply rotation correction to align Vision Pro frame with robot frame
    # The euler angles rotate the hand orientation to match the robot's expected orientation
    correction = R.from_euler("xyz", euler, degrees=True).as_matrix()
    
    result = np.eye(4)
    result[:3, :3] = wrist[:3, :3] @ correction
    result[:3, 3] = wrist[:3, 3]
    
    return result


def save_trajectory(trajectory_log: dict, output_path: Path, model: mujoco.MjModel):
    """
    Save recorded trajectory to NPZ file (compatible with follower.py and 09_mujoco_streaming.py).
    
    Args:
        trajectory_log: Dictionary with lists of qpos, qvel, ctrl, mocap
        output_path: Path where to save the .npz file
        model: MuJoCo model (for validation)
    """
    print(f"\n💾 Saving trajectory to: {output_path}")
    
    # Convert lists to numpy arrays
    qpos_log = np.stack(trajectory_log['qpos'], axis=0)  # (T, nq)
    qvel_log = np.stack(trajectory_log['qvel'], axis=0)    # (T, nv)
    ctrl_log = np.stack(trajectory_log['ctrl'], axis=0)   # (T, nu)
    mocap_log = np.stack(trajectory_log['mocap'], axis=0) # (T, 14) - 2 mocap bodies
    
    T = qpos_log.shape[0]
    recording_dt = 1.0 / 200.0  # 200 Hz recording rate = 0.005 seconds
    duration = T * recording_dt
    
    print(f"   Trajectory length: {T} timesteps")
    print(f"   Recording rate: 200 Hz (dt = {recording_dt:.6f} s)")
    print(f"   Duration: {duration:.3f} seconds")
    print(f"   qpos shape: {qpos_log.shape}")
    print(f"   qvel shape: {qvel_log.shape}")
    print(f"   ctrl shape: {ctrl_log.shape}")
    print(f"   mocap shape: {mocap_log.shape}")
    
    # Validate shapes
    assert qpos_log.shape[0] == qvel_log.shape[0] == ctrl_log.shape[0] == mocap_log.shape[0], \
        "All trajectory arrays must have the same number of timesteps"
    assert qpos_log.shape[1] == model.nq, \
        f"qpos shape mismatch: expected {model.nq}, got {qpos_log.shape[1]}"
    assert qvel_log.shape[1] == model.nv, \
        f"qvel shape mismatch: expected {model.nv}, got {qvel_log.shape[1]}"
    assert ctrl_log.shape[1] == model.nu, \
        f"ctrl shape mismatch: expected {model.nu}, got {ctrl_log.shape[1]}"
    assert mocap_log.shape[1] == 7, \
        f"mocap shape mismatch: expected 7 (x,y,z,qx,qy,qz,qw), got {mocap_log.shape[1]}"
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to NPZ file
    # Store recording dt so follower.py can use the correct timestep
    recording_dt = 1.0 / 200.0  # 200 Hz recording rate
    np.savez(
        output_path,
        qpos=qpos_log,
        qvel=qvel_log,
        ctrl=ctrl_log,
        mocap=mocap_log,
        dt=recording_dt,  # Store actual recording timestep
    )
    
    print(f"✅ Trajectory saved successfully!")
    print(f"   File: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def setup_keyboard_listener(key_pressed_callback):
    """
    Set up a non-blocking keyboard listener in a separate thread.
    Detects 's' key presses and calls the callback.
    """
    import select
    import termios
    import tty
    
    # Check if stdin is a terminal
    if not sys.stdin.isatty():
        raise OSError("stdin is not a terminal")
    
    def read_key():
        # Set terminal to raw mode
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                try:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        char = sys.stdin.read(1)
                        if char == 's' or char == 'S':
                            key_pressed_callback()
                        elif char == '\x03':  # Ctrl+C
                            break
                except (OSError, ValueError):
                    # Terminal might have been closed or changed
                    break
        finally:
            # Restore terminal settings
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
    
    thread = threading.Thread(target=read_key, daemon=True)
    thread.start()
    return thread


def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: Optional[np.ndarray] = None,
) -> None:
    """Compute forces to counteract gravity for the given subtrees.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        subtree_ids: List of subtree ids. A subtree is defined as the kinematic tree
            starting at the body and including all its descendants. Gravity
            compensation forces will be applied to all bodies in the subtree.
        qfrc_applied: Optional array to store the computed forces. If not provided,
            the applied forces in `data` are used.
    """
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0  # Don't accumulate from previous calls.
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac


def main(args):
    """Main teleoperation loop for G1 humanoid robot."""
    
    # Load the G1 model
    if not os.path.exists(_XML):
        print(f"❌ G1 XML file not found at: {_XML}")
        print("Please download the G1 model from MuJoCo Menagerie and place it in:")
        print("  assets/mujoco_demos/unitree_g1/scene.xml")
        sys.exit(1)
    
    model = mujoco.MjModel.from_xml_path(str(_XML))
    data = mujoco.MjData(model)

    # Initialize VisionPro streamer
    from avp_stream import VisionProStreamer

    streamer = VisionProStreamer(ip=args.ip)

    # Configure simulation streaming
    # relative_to format: [x, y, z, yaw_degrees]
    # Adjust these values to position the robot in the AR view
    streamer.configure_sim(
        xml_path=str(_XML),
        model=model,
        data=data,
        relative_to=[0, 0.1, 0.6, 0],  # Adjust as needed
        force_reload=False
    )
    streamer.start_webrtc()

    # Bodies for which to apply gravity compensation
    # Use the shoulder pitch links as the base of each arm subtree
    try:
        left_subtree_id = model.body("left_shoulder_pitch_link").id
        right_subtree_id = model.body("right_shoulder_pitch_link").id
    except:
        print("⚠️  Could not find arm base bodies for gravity compensation.")
        print("   Available bodies:", [model.body(i).name for i in range(model.nbody)])
        print("   Skipping gravity compensation. You may want to add it manually.")
        left_subtree_id = None
        right_subtree_id = None

    # Get the dof and actuator ids for the joints we wish to control
    joint_names: List[str] = []
    velocity_limits: dict[str, float] = {}
    
    # Build joint names list - G1 uses exact joint names
    for n in _LEFT_JOINT_NAMES + _RIGHT_JOINT_NAMES:
        try:
            model.joint(n)
            joint_names.append(n)
            velocity_limits[n] = _VELOCITY_LIMITS.get(n, np.pi / 2)
        except:
            print(f"⚠️  Warning: Could not find joint: {n}")
            continue
    
    if len(joint_names) == 0:
        print("❌ No joints found!")
        print("   Available joints:", model.joint_names)
        sys.exit(1)
    
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    print(f"✅ Found {len(joint_names)} joints to control")
    print(f"   Joints: {joint_names}")

    configuration = mink.Configuration(model)

    # Create frame tasks for left and right end-effectors
    # Use the wrist sites defined in g1.xml
    left_ee_name = "left/wrist"
    right_ee_name = "right/wrist"
    
    tasks = [
        l_ee_task := mink.FrameTask(
            frame_name=left_ee_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,
            lm_damping=1.0,
        ),
        r_ee_task := mink.FrameTask(
            frame_name=right_ee_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.3,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-4),
    ]

    # Enable collision avoidance between arms
    try:
        # Get geoms from the wrist bodies (sites are attached to wrist_yaw_link bodies)
        l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left_wrist_yaw_link").id)
        r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right_wrist_yaw_link").id)
        l_arm_geoms = mink.get_subtree_geom_ids(model, model.body("left_shoulder_pitch_link").id)
        r_arm_geoms = mink.get_subtree_geom_ids(model, model.body("right_shoulder_pitch_link").id)
        torso_geoms = mink.get_subtree_geom_ids(model, model.body("torso_link").id)
        collision_pairs = [
            (l_wrist_geoms, r_wrist_geoms),
            (l_arm_geoms + r_arm_geoms, torso_geoms),
        ]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.05,
        )
        limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocity_limits),
            collision_avoidance_limit,
        ]
    except Exception as e:
        print(f"⚠️  Could not set up collision avoidance: {e}")
        print("   Using basic limits")
        limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocity_limits),
        ]

    # Get mocap body IDs for targets (using slash notation from scene_teleop.xml)
    try:
        l_mid = model.body("left/target").mocapid[0]
        r_mid = model.body("right/target").mocapid[0]
    except:
        print("❌ Could not find mocap target bodies!")
        print("   The scene_teleop.xml should include mocap bodies.")
        print("   Available bodies:", [model.body(i).name for i in range(model.nbody)])
        print("   Please ensure scene_teleop.xml includes:")
        print("     <body name=\"left/target\" mocap=\"true\" pos=\"0.5 0 .5\"/>")
        print("     <body name=\"right/target\" mocap=\"true\" pos=\"0.5 0 .5\"/>")
        sys.exit(1)
    
    solver = "daqp"
    pos_threshold = 5e-3
    ori_threshold = 5e-3
    max_iters = 5

    # Initialize to a neutral/home pose
    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    except:
        print("⚠️  No 'stand' keyframe found, using default pose")
        mujoco.mj_resetData(model, data)
    
    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)
    posture_task.set_target_from_configuration(configuration)

    # Initialize mocap targets at the end-effector sites
    mink.move_mocap_to_frame(model, data, "left/target", left_ee_name, "site")
    mink.move_mocap_to_frame(model, data, "right/target", right_ee_name, "site")

    rate = RateLimiter(frequency=200.0, warn=False)
    
    # Setup trajectory recording at 200 Hz
    # Record every iteration (loop runs at 200 Hz)
    recording_enabled = False
    recording_counter = 0
    recording_interval = int(200.0 / 200.0)  # Record every iteration (200 Hz)
    trajectory_log = {
        'qpos': [],
        'qvel': [],
        'ctrl': [],
        'mocap': [],
    }
    recording_start_time = None
    
    # Setup keyboard listener for 's' key to toggle recording
    def toggle_recording():
        nonlocal recording_enabled, recording_start_time, trajectory_log
        if not recording_enabled:
            # Start recording
            recording_enabled = True
            recording_start_time = time.time()
            trajectory_log = {
                'qpos': [],
                'qvel': [],
                'ctrl': [],
                'mocap': [],
            }
            print(f"\n📹 Recording STARTED")
            print(f"   Recording rate: 200 Hz (every {recording_interval} iteration at 200 Hz)")
            print(f"   Press 's' again to stop recording")
        else:
            # Stop recording
            recording_enabled = False
            print(f"\n🛑 Recording STOPPED")
            if len(trajectory_log['qpos']) > 0:
                # Save trajectory
                if args.output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path(_HERE) / "trajectories"
                    output_dir.mkdir(exist_ok=True)
                    output_path = output_dir / f"traj_{timestamp}.npz"
                else:
                    output_path = Path(args.output_path)
                save_trajectory(trajectory_log, output_path, model)
            else:
                print("   No data recorded")
    
    # Start keyboard listener
    try:
        setup_keyboard_listener(toggle_recording)
        print(f"\n⌨️  Keyboard controls:")
        print(f"   Press 's' to start/stop recording")
        print(f"   Press Ctrl+C to exit")
    except Exception as e:
        print(f"⚠️  Could not set up keyboard listener: {e}")
        print(f"   Recording will be disabled. Install termios support or use --record flag")
        recording_enabled = False

    try:
        while True:
            # Update task targets from mocap bodies
            l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
            r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

            # Get latest hand tracking data
            hand = streamer.get_latest()
            if hand is None:
                continue

            # Convert hand poses to robot end-effector poses
            # ADJUST euler angles [X, Y, Z] in degrees to align hand orientation with robot arm
            # G1's wrist site points along +X axis of the wrist_yaw_link
            # Vision Pro's wrist frame: Y toward fingers, Z out of back of hand
            # 
            # TUNING TIPS:
            # - Start with [0, 0, 0] to see the raw alignment
            # - If palm faces wrong direction: adjust X rotation (roll)
            # - If fingers point wrong direction: adjust Y rotation (pitch)  
            # - If thumb points wrong direction: adjust Z rotation (yaw)
            # - Left/right usually need opposite signs on some axes
            hand_left_pose = hand2pose(hand, side="left", euler=[180, 0, 0])
            hand_right_pose = hand2pose(hand, side="right", euler=[0, 0, 180])

            # Update mocap targets with hand poses
            data.mocap_pos[l_mid, :] = hand_left_pose[:3, 3]
            data.mocap_quat[l_mid, :] = R.from_matrix(hand_left_pose[:3, :3]).as_quat(scalar_first=True)

            data.mocap_pos[r_mid, :] = hand_right_pose[:3, 3]
            data.mocap_quat[r_mid, :] = R.from_matrix(hand_right_pose[:3, :3]).as_quat(scalar_first=True)

            # Solve inverse kinematics
            vel = mink.solve_ik(
                configuration,
                tasks,
                rate.dt,
                solver,
                # limits=limits,
                damping=1e-5,
            )
            configuration.integrate_inplace(vel, rate.dt)

            data.qpos[dof_ids] = configuration.q[dof_ids]
            mujoco.mj_forward(model, data)

            # Record trajectory at 200 Hz (every iteration)
            if recording_enabled:
                if recording_start_time is None:
                    recording_start_time = time.time()
                
                recording_counter += 1
                if recording_counter >= recording_interval:
                    recording_counter = 0
                    
                    # Record qpos (all joints)
                    trajectory_log['qpos'].append(data.qpos.copy())
                    
                    # Record qvel (all velocities)
                    trajectory_log['qvel'].append(data.qvel.copy())
                    
                    # Record ctrl (all actuators)
                    trajectory_log['ctrl'].append(data.ctrl.copy())
                    
                    # Record mocap pose (left target only, for compatibility with 09_mujoco_streaming.py)
                    # Format: [x, y, z, qx, qy, qz, qw] (7 values)
                    # MuJoCo stores quat as [w, x, y, z], convert to [x, y, z, w]
                    mocap_quat_l = data.mocap_quat[l_mid, :]  # [w, x, y, z]
                    mocap_data = np.zeros(7)
                    mocap_data[0:3] = data.mocap_pos[l_mid, :]  # [x, y, z]
                    mocap_data[3:6] = mocap_quat_l[1:4]  # [x, y, z] from quaternion
                    mocap_data[6] = mocap_quat_l[0]  # w
                    trajectory_log['mocap'].append(mocap_data)
            
            streamer.update_sim()
            rate.sleep()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
        
        # Save trajectory if recording was enabled
        if recording_enabled and len(trajectory_log['qpos']) > 0:
            if args.output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(_HERE) / "trajectories"
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"traj_{timestamp}.npz"
            else:
                output_path = Path(args.output_path)
            save_trajectory(trajectory_log, output_path, model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Mujoco G1 Humanoid DiffIK with VisionPro Hand Tracking"
    )
    parser.add_argument("--ip", type=str, required=True, help="Vision Pro IP address")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for trajectory NPZ file (default: Project/trajectories/traj_YYYYMMDD_HHMMSS.npz when recording stops)"
    )
    args = parser.parse_args()
    
    # Set default output path if provided, otherwise will be set when recording stops
    if args.output is not None:
        args.output_path = Path(args.output)
    else:
        args.output_path = None

    main(args)

