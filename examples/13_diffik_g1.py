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
from pathlib import Path
from typing import List, Optional, Sequence

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

    # # Enable collision avoidance between arms
    # try:
    #     # Get geoms from the wrist bodies (sites are attached to wrist_yaw_link bodies)
    #     l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left_wrist_yaw_link").id)
    #     r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right_wrist_yaw_link").id)
    #     l_arm_geoms = mink.get_subtree_geom_ids(model, model.body("left_shoulder_pitch_link").id)
    #     r_arm_geoms = mink.get_subtree_geom_ids(model, model.body("right_shoulder_pitch_link").id)
    #     torso_geoms = mink.get_subtree_geom_ids(model, model.body("torso_link").id)
    #     collision_pairs = [
    #         (l_wrist_geoms, r_wrist_geoms),
    #         (l_arm_geoms + r_arm_geoms, torso_geoms),
    #     ]
    #     collision_avoidance_limit = mink.CollisionAvoidanceLimit(
    #         model=model,
    #         geom_pairs=collision_pairs,  # type: ignore
    #         minimum_distance_from_collisions=0.05,
    #         collision_detection_distance=0.1,
    #     )
    #     limits = [
    #         mink.ConfigurationLimit(model=model),
    #         mink.VelocityLimit(model, velocity_limits),
    #         collision_avoidance_limit,
    #     ]
    # except Exception as e:
    #     print(f"⚠️  Could not set up collision avoidance: {e}")
    #     print("   Using basic limits")
    #     limits = [
    #         mink.ConfigurationLimit(model=model),
    #         mink.VelocityLimit(model, velocity_limits),
    #     ]

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

            streamer.update_sim()
            rate.sleep()

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Mujoco G1 Humanoid DiffIK with VisionPro Hand Tracking"
    )
    parser.add_argument("--ip", type=str, required=True, help="Vision Pro IP address")
    args = parser.parse_args()

    main(args)

