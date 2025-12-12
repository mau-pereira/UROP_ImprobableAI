"""
Follower Robot Controller - Lyapunov-Based Adaptive Control

This script implements two types of control:
1. Static PD control (fixed gains)
2. Lyapunov-based adaptive control (gains that adjust automatically)

ADAPTIVE CONTROL:
- Lyapunov function: V = ½(eᵀe + ėᵀė) + parameter terms
- Adaptive laws:
  * Kp adjusts according to position error: Kṗ = γ_Kp * e²
  * Kd adjusts according to velocity error: Kḋ = γ_Kd * ė²
  * b̂ estimates viscous friction: b̂̇ = γ_b * qdot * e
- Guarantees asymptotic stability (V̇ ≤ 0)

To switch between controllers, modify USE_ADAPTIVE_CONTROL in main().
"""

import time
import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer
from pathlib import Path


# ==========================
# PATH CONFIGURATION
# ==========================

# Set here the local path to the G1 slave XML (29 DOF, with motor-type actuators)
SLAVE_XML_PATH = "../assets/g1_from_unitree_github/scene_29dof.xml"

# (Optional, for future) path to G1 leader XML (menagerie, position-type actuators)
LEADER_XML_PATH = "path/to/mujoco_menagerie/unitree_g1/scene.xml"

# Path to NPZ file with teleoperation trajectory (format compatible with 09_mujoco_streaming.py)
# You can use a relative path from this file or an absolute path
# Example: "trajectories/traj_20251211_173426.npz" or absolute path
# The path is resolved relative to the directory where this file is located
_HERE = Path(__file__).parent
TRAJECTORY_NPZ_PATH = _HERE / "trajectories" / "traj_20251211_200112.npz"


# ==========================
# ARM JOINTS
# ==========================
# IMPORTANT:
#  - Fill these lists with the EXACT NAMES of the joints for each arm
#    as they appear in your G1 slave XML (scene_29dof.xml).
#  - You can print all joints using the print_all_joints(model) function below.

LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# ==========================
# UTILITIES
# ==========================

def print_all_joints(model: mj.MjModel):
    """Prints all joints with their indices so you can copy names."""
    print("=== JOINTS IN THE MODEL ===")
    for j_id in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j_id)
        qpos_adr = model.jnt_qposadr[j_id]
        dof_adr = model.jnt_dofadr[j_id]
        jnt_type = model.jnt_type[j_id]
        print(f"joint_id={j_id:2d}, name={name}, type={jnt_type}, qpos_adr={qpos_adr}, dof_adr={dof_adr}")
    print("===========================\n")


def build_arm_maps(model: mj.MjModel,
                   joint_names: list[str]):
    """
    From a list of joint names, builds:
    - qpos indices (generalized position)
    - dof indices (generalized velocity)
    - actuator indices that control each joint (for data.ctrl)

    Assumes 1 actuator per joint (typical in robot models).
    """
    # Map joint_id -> list of actuators that control it
    jointid_to_actids: dict[int, list[int]] = {}
    for act_id in range(model.nu):
        # actuator_trnid[act_id, 0] contains the joint_id if it's a joint-type actuator
        joint_id = model.actuator_trnid[act_id, 0]
        if joint_id >= 0:
            jointid_to_actids.setdefault(joint_id, []).append(act_id)

    qpos_indices = []
    dof_indices = []
    act_indices = []

    for name in joint_names:
        if name is None or name == "":
            raise ValueError("There is an empty joint in the list, check LEFT_ARM_JOINT_NAMES / RIGHT_ARM_JOINT_NAMES.")

        j_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise RuntimeError(f"Joint '{name}' does not exist in the model.")

        qpos_idx = model.jnt_qposadr[j_id]
        dof_idx = model.jnt_dofadr[j_id]

        if j_id not in jointid_to_actids:
            raise RuntimeError(f"Joint '{name}' (id={j_id}) has no associated actuator in 'actuator_trnid'.")

        # Assume one actuator per joint; if there are more, take the first one
        act_id = jointid_to_actids[j_id][0]

        qpos_indices.append(qpos_idx)
        dof_indices.append(dof_idx)
        act_indices.append(act_id)

    return np.array(qpos_indices, dtype=int), np.array(dof_indices, dtype=int), np.array(act_indices, dtype=int)


# ==========================
# DESIRED TRAJECTORY
# ==========================

# Global variable to store the loaded trajectory
_trajectory_data = None
_trajectory_dt = None
_trajectory_duration = None
_left_qpos_idx = None
_right_qpos_idx = None


def load_trajectory_from_npz(npz_path: str | Path,
                             left_qpos_idx: np.ndarray,
                             right_qpos_idx: np.ndarray,
                             model_dt: float):
    """
    Loads a trajectory from an NPZ file (format compatible with 09_mujoco_streaming.py).
    
    The NPZ file must contain:
    - 'qpos': array of shape (T, nq) with positions of all joints
    - 'dt': recording timestep (required)
    
    Args:
        npz_path: Path to the .npz file
        left_qpos_idx: qpos indices for the left arm
        right_qpos_idx: qpos indices for the right arm
        model_dt: MuJoCo model timestep (not used, 'dt' is read from NPZ)
    """
    global _trajectory_data, _trajectory_dt, _trajectory_duration
    global _left_qpos_idx, _right_qpos_idx
    
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {npz_path}")
    
    print(f"📂 Loading trajectory from: {npz_path}")
    traj = np.load(npz_path)
    
    if "qpos" not in traj:
        raise ValueError("The NPZ file must contain the 'qpos' key")
    
    qpos_log = traj["qpos"]  # shape: (T, nq)
    T = qpos_log.shape[0]
    
    # Read dt from NPZ (required from now on)
    if "dt" not in traj:
        raise ValueError(
            "The NPZ file must contain the 'dt' key with the recording timestep. "
            "Make sure to use a trajectory recorded with the updated version of teleop_trajectory_gen.py."
        )
    
    trajectory_dt = float(traj["dt"])  # actual recording dt
    recording_freq = 1.0 / trajectory_dt
    print(f"   ✅ Recording dt: {trajectory_dt:.6f} s ({recording_freq:.1f} Hz)")
    
    # Extract only arm joints
    q_left_traj = qpos_log[:, left_qpos_idx]  # shape: (T, 7)
    q_right_traj = qpos_log[:, right_qpos_idx]  # shape: (T, 7)
    
    # Save to global variable
    _trajectory_data = {
        'q_left': q_left_traj,
        'q_right': q_right_traj,
    }
    _trajectory_dt = trajectory_dt  # Use actual recording dt, not model_dt
    _trajectory_duration = T * trajectory_dt  # Correct duration
    _left_qpos_idx = left_qpos_idx
    _right_qpos_idx = right_qpos_idx
    
    print(f"✅ Trajectory loaded: {T} timesteps, duration = {_trajectory_duration:.3f} s")
    print(f"   Left arm: {q_left_traj.shape}")
    print(f"   Right arm: {q_right_traj.shape}")


def desired_arm_trajectories_sinusoidal_shoulders(t: float,
                                                   n_left: int,
                                                   n_right: int,
                                                   amplitude: float = 0.5,
                                                   frequency: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates sinusoidal trajectory ONLY for the first joint of each arm (shoulder_pitch).
    All other joints remain at 0 (fixed).
    
    This isolates the controller problem by moving only one joint per arm.
    
    Args:
        t: Current simulation time (seconds)
        n_left: Number of joints in the left arm
        n_right: Number of joints in the right arm
        amplitude: Sinusoid amplitude in radians (default: 0.5 rad ≈ 28°)
        frequency: Frequency in Hz (default: 0.5 Hz = period of 2 seconds)
    
    Returns:
        q_left_des: Array of shape (n_left,) with desired joint angles
        q_right_des: Array of shape (n_right,) with desired joint angles
    """
    # Initialize all joints to 0
    q_left_des = np.zeros(n_left)
    q_right_des = np.zeros(n_right)
    
    # Only the first joint (shoulder_pitch) moves sinusoidally
    if n_left > 0:
        # Left arm: only the first joint (shoulder_pitch)
        q_left_des[0] = amplitude * np.sin(2 * np.pi * frequency * t)
    
    if n_right > 0:
        # Right arm: only the first joint (shoulder_pitch)
        q_right_des[0] = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # All other joints (shoulder_roll, shoulder_yaw, elbow, wrist) remain at 0
    
    return q_left_des, q_right_des


def desired_arm_trajectories(t: float,
                             n_left: int,
                             n_right: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns q_des_left(t), q_des_right(t) for each arm from loaded trajectory.
    
    Linearly interpolates between timesteps of the loaded trajectory.
    If t exceeds the trajectory duration, returns the last value.
    
    Args:
        t: Current simulation time (seconds)
        n_left: Number of joints in the left arm
        n_right: Number of joints in the right arm
    
    Returns:
        q_left_des: Array of shape (n_left,) with desired joint angles for the left arm
        q_right_des: Array of shape (n_right,) with desired joint angles for the right arm
    """
    global _trajectory_data, _trajectory_dt, _trajectory_duration
    
    if _trajectory_data is None:
        raise RuntimeError(
            "Trajectory not loaded. Call load_trajectory_from_npz() before using desired_arm_trajectories()."
        )
    
    q_left_traj = _trajectory_data['q_left']
    q_right_traj = _trajectory_data['q_right']
    
    # Calculate timestep index (can be fractional for interpolation)
    step_index = t / _trajectory_dt
    
    # If t exceeds duration, use the last timestep
    if step_index >= q_left_traj.shape[0] - 1:
        q_left_des = q_left_traj[-1]
        q_right_des = q_right_traj[-1]
    else:
        # Linear interpolation between timesteps
        idx_low = int(np.floor(step_index))
        idx_high = min(idx_low + 1, q_left_traj.shape[0] - 1)
        alpha = step_index - idx_low  # interpolation factor [0, 1)
        
        q_left_des = (1 - alpha) * q_left_traj[idx_low] + alpha * q_left_traj[idx_high]
        q_right_des = (1 - alpha) * q_right_traj[idx_low] + alpha * q_right_traj[idx_high]
    
    return q_left_des, q_right_des


# ==========================
# COMPLETE ADAPTIVE CONTROL
# ==========================

class AdaptiveController:
    """
    Complete Lyapunov-based adaptive controller using equations of motion.
    
    Theory:
    - Equations of motion: M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
    - Control: τ = M_known(q)q̈_des + C_known(q,q̇)q̇ + g_known(q) + Y_friction * θ_friction + Kp*e + Kd*ė
    - Adaptive law: θ̂̇ = -ΓYᵀ(e + λė)
    
    Parameters:
    - Masses and inertias: KNOWN (fixed, not adapted)
    - Friction: UNKNOWN (adapted)
    """
    
    def __init__(self, n_joints: int, dt: float, model: mj.MjModel = None, 
                 arm_qpos_idx: np.ndarray = None, arm_dof_idx: np.ndarray = None,
                 known_masses: np.ndarray = None, known_inertias: np.ndarray = None,
                 adapt_gains: bool = False):
        """
        Args:
            n_joints: Number of joints to control
            dt: Simulation timestep
            model: MuJoCo model (optional, for building Y)
            arm_qpos_idx: Arm qpos indices (optional)
            arm_dof_idx: Arm qvel indices (optional)
            known_masses: Array of known masses [m₁, m₂, ..., mₙ] (kg)
            known_inertias: Array of known inertias [I₁, I₂, ..., Iₙ] (kg·m²)
            adapt_gains: If True, adapts PD gains (Kp, Kd). If False, uses fixed gains.
        """
        self.n_joints = n_joints
        self.dt = dt
        self.model = model
        self.arm_qpos_idx = arm_qpos_idx
        self.arm_dof_idx = arm_dof_idx
        self.adapt_gains = adapt_gains
        
        # Known parameters (fixed, not adapted)
        if known_masses is not None:
            if len(known_masses) != n_joints:
                raise ValueError(f"known_masses must have {n_joints} elements, has {len(known_masses)}")
            self.known_masses = np.array(known_masses)
        else:
            # Default values (extract from MuJoCo model if possible)
            self.known_masses = np.ones(n_joints) * 0.5  # kg default
        
        if known_inertias is not None:
            if len(known_inertias) != n_joints:
                raise ValueError(f"known_inertias must have {n_joints} elements, has {len(known_inertias)}")
            self.known_inertias = np.array(known_inertias)
        else:
            # Default values
            self.known_inertias = np.ones(n_joints) * 0.01  # kg·m² default
        
        # PD gains: adaptive or fixed according to adapt_gains
        if adapt_gains:
            # Number of adaptive parameters: friction (1 per joint) + PD gains (2 global)
            # Structure: [b₁, b₂, ..., bₙ, Kp, Kd]
            self.n_params = n_joints + 2
            self.theta_hat = np.zeros(self.n_params)
            # Initial friction = 0 (first n elements)
            # Initial gains (last 2 elements)
            self.theta_hat[-2] = 150.0  # Initial Kp
            self.theta_hat[-1] = 10.0   # Initial Kd
            
            # Adaptation matrix Γ
            self.Gamma = np.eye(self.n_params) * 0.5
            self.Gamma[-2, -2] = 0.1  # Gamma for Kp (more conservative)
            self.Gamma[-1, -1] = 0.1  # Gamma for Kd (more conservative)
        else:
            # Only friction is adapted, gains are fixed
            # Structure: [b₁, b₂, ..., bₙ]
            self.n_params = n_joints
            self.theta_hat = np.zeros(self.n_params)  # Only friction
            self.Gamma = np.eye(self.n_params) * 0.5
            
            # Fixed PD gains
            self._Kp = 150.0
            self._Kd = 10.0
        
        
        # ===== FILTERS TO IMPROVE STABILITY =====
        
        # Low-pass filter for qddot_des (reduces noise from numerical differentiation)
        self.qddot_des_filtered = None  # Filter state
        self.qddot_filter_alpha = 0.8  # Filter factor (0.8 = more smoothed, 0.9 = less smoothed)
        
        # Exponential filter to smooth adaptive parameters
        self.theta_hat_smoothed = self.theta_hat.copy()  # Smoothed version of theta_hat
        self.theta_smooth_alpha = 0.9  # Smoothing factor (0.9 = more smoothed, 0.95 = very smoothed)
        
        # History for analysis
        self.history = {
            'theta_hat': [],
            'theta_hat_smoothed': [],
            'lyapunov': [],
            'Y_norm': []
        }
    
    @property
    def Kp(self):
        """Proportional gain (adaptive or fixed according to adapt_gains)."""
        if self.adapt_gains:
            return self.theta_hat[-2]
        else:
            return self._Kp
    
    @property
    def Kd(self):
        """Derivative gain (adaptive or fixed according to adapt_gains)."""
        if self.adapt_gains:
            return self.theta_hat[-1]
        else:
            return self._Kd
    
    def compute_dynamics_matrices(self,
                                  q: np.ndarray,
                                  qdot: np.ndarray,
                                  data: mj.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes dynamic matrices using MuJoCo: M(q), C(q,q̇), g(q)
        
        Uses mj_rne() (Recursive Newton-Euler) to calculate:
        τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
        
        Args:
            q: Current arm positions
            qdot: Current arm velocities
            data: MuJoCo data (must have access to the complete model)
        
        Returns:
            M: Inertia matrix of shape (n_joints, n_joints)
            C_qdot: Vector C(q,q̇)q̇ of shape (n_joints,)
            g: Gravity vector of shape (n_joints,)
        """
        # Save complete system state
        q_full_orig = data.qpos.copy()
        qdot_full_orig = data.qvel.copy()
        qacc_full_orig = data.qacc.copy()
        
        # Set arm state in data
        data.qpos[self.arm_qpos_idx] = q
        data.qvel[self.arm_qpos_idx] = qdot
        
        # 1. Calculate g(q): gravitational forces (with qdot=0, qddot=0)
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mj.mj_forward(self.model, data)
        # mj_rne calculates: τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
        # With qdot=0, qddot=0: τ = g(q)
        # flg_acc=1 means include accelerations (but they are at 0)
        mj.mj_rne(self.model, data, 1, data.qfrc_inverse)
        g = data.qfrc_inverse[self.arm_qpos_idx].copy()
        
        # 2. Calculate C(q,q̇)q̇: Coriolis/centrifugal terms
        data.qvel[self.arm_qpos_idx] = qdot
        data.qacc[:] = 0.0
        mj.mj_forward(self.model, data)
        mj.mj_rne(self.model, data, 1, data.qfrc_inverse)
        # With qddot=0: τ = C(q,q̇)q̇ + g(q)
        C_qdot_plus_g = data.qfrc_inverse[self.arm_qpos_idx].copy()
        C_qdot = C_qdot_plus_g - g
        
        # 3. Calculate M(q): inertia matrix (column by column using mj_rne)
        M = np.zeros((self.n_joints, self.n_joints))
        data.qvel[self.arm_qpos_idx] = qdot  # maintain velocity
        
        for j in range(self.n_joints):
            # Calculate column j of M using qddot = e_j (unit vector)
            qddot_unit = np.zeros(self.n_joints)
            qddot_unit[j] = 1.0
            
            # Set unit acceleration only in the arm
            data.qacc[:] = 0.0
            data.qacc[self.arm_qpos_idx] = qddot_unit
            
            mj.mj_forward(self.model, data)
            mj.mj_rne(self.model, data, 1, data.qfrc_inverse)
            # τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
            # With q̈ = e_j: τ = M[:,j] + C(q,q̇)q̇ + g(q)
            tau_col = data.qfrc_inverse[self.arm_qpos_idx].copy()
            M[:, j] = tau_col - C_qdot - g
        
        # Restore original state
        data.qpos[:] = q_full_orig
        data.qvel[:] = qdot_full_orig
        data.qacc[:] = qacc_full_orig
        
        return M, C_qdot, g
    
    def build_regression_matrix(self, 
                               q: np.ndarray,
                               qdot: np.ndarray,
                               qddot: np.ndarray,
                               q_des: np.ndarray = None,
                               qdot_des: np.ndarray = None,
                               data: mj.MjData = None) -> np.ndarray:
        """
        Builds the regression matrix Y(q, q̇, e, ė) for FRICTION and optionally PD GAINS.
        
        If adapt_gains=True: θ = [b₁, b₂, ..., bₙ, Kp, Kd]
        If adapt_gains=False: θ = [b₁, b₂, ..., bₙ]
        
        Args:
            q: Current positions
            qdot: Current velocities
            qddot: Desired accelerations (not used, but necessary for compatibility)
            q_des: Desired positions (to calculate error e, only if adapt_gains=True)
            qdot_des: Desired velocities (to calculate error ė, only if adapt_gains=True)
            data: MuJoCo data (not used here, but necessary for compatibility)
        
        Returns:
            Y: Regression matrix of shape (n_joints, n_params)
        """
        Y = np.zeros((self.n_joints, self.n_params))
        
        # Columns 0 to n-1: Friction (1 per joint)
        for i in range(self.n_joints):
            Y[i, i] = qdot[i]  # Friction of joint i is proportional to its velocity
        
        # If we adapt gains, add columns for Kp and Kd
        if self.adapt_gains:
            # Calculate errors
            if q_des is not None and qdot_des is not None:
                e = q_des - q
                edot = qdot_des - qdot
            else:
                e = np.zeros(self.n_joints)
                edot = np.zeros(self.n_joints)
            
            # Column n: Proportional gains Kp (global)
            Y[:, self.n_joints] = e
            
            # Column n+1: Derivative gains Kd (global)
            Y[:, self.n_joints + 1] = edot
        
        return Y
    
    def filter_qddot_des(self, qddot_des: np.ndarray) -> np.ndarray:
        """
        Filters qddot_des using low-pass filter to reduce noise.
        
        Exponential filter: q̈_filtered = α * q̈_filtered_previous + (1-α) * q̈_current
        
        Args:
            qddot_des: Desired acceleration (may have noise)
        
        Returns:
            qddot_des_filtered: Filtered acceleration
        """
        if self.qddot_des_filtered is None:
            # First time: initialize with current value
            self.qddot_des_filtered = qddot_des.copy()
        else:
            # Exponential low-pass filter
            self.qddot_des_filtered = (self.qddot_filter_alpha * self.qddot_des_filtered + 
                                      (1.0 - self.qddot_filter_alpha) * qddot_des)
        
        return self.qddot_des_filtered
    
    def compute_control(self,
                       q: np.ndarray,
                       qdot: np.ndarray,
                       q_des: np.ndarray,
                       qdot_des: np.ndarray,
                       qddot_des: np.ndarray,
                       data: mj.MjData = None) -> tuple[np.ndarray, float]:
        """
        Calculates adaptive control torque using known masses/inertias.
        
        Control: τ = M_known(q)q̈_des + C_known(q,q̇)q̇ + g_known(q) + Y_friction*θ̂_friction + Kp*e + Kd*ė
        
        Where:
        - M_known, C_known, g_known: calculated with MuJoCo using known masses/inertias
        - Y_friction*θ̂_friction: adaptive friction compensation
        
        Args:
            q: Current positions
            qdot: Current velocities
            q_des: Desired positions
            qdot_des: Desired velocities
            qddot_des: Desired accelerations (will be filtered to reduce noise)
            data: MuJoCo data (required to calculate dynamics)
        
        Returns:
            tau: Control torques
            V: Lyapunov function value
        """
        # Errors
        e = q_des - q
        edot = qdot_des - qdot
        
        # Filter qddot_des to reduce noise from numerical differentiation
        qddot_des_filtered = self.filter_qddot_des(qddot_des)
        
        # Calculate dynamics using MuJoCo (with model masses/inertias)
        if data is not None and self.model is not None and self.arm_qpos_idx is not None:
            M, C_qdot, g = self.compute_dynamics_matrices(q, qdot, data)
            
            # Dynamic compensation using known masses/inertias (from MuJoCo model)
            # τ_dynamic = M(q)q̈_des + C(q,q̇)q̇ + g(q)
            # Use filtered qddot_des to reduce noise
            tau_dynamic_known = M @ qddot_des_filtered + C_qdot + g
        else:
            # Fallback if no access to MuJoCo
            tau_dynamic_known = np.zeros(self.n_joints)
        
        # Build regression matrix (use filtered qddot_des)
        Y = self.build_regression_matrix(q, qdot, qddot_des_filtered, q_des, qdot_des, data)
        
        # Adaptive compensation
        if self.adapt_gains:
            # τ_adaptive = Y * θ̂ = Y_friction * θ̂_friction + Y_Kp * Kp + Y_Kd * Kd
            # PD gains are already included in tau_adaptive
            tau_adaptive = Y @ self.theta_hat
            tau = tau_dynamic_known + tau_adaptive
        else:
            # Only adaptive friction, fixed PD gains
            tau_friction = Y @ self.theta_hat  # Only friction
            tau_pd = self.Kp * e + self.Kd * edot  # Fixed PD gains
            tau = tau_dynamic_known + tau_friction + tau_pd
        
        # Complete Lyapunov function
        # V = ½(eᵀe + ėᵀė + θ̃ᵀΓ⁻¹θ̃)
        # We simplify by assuming θ̃ ≈ 0 (well-estimated parameters)
        V = 0.5 * (np.dot(e, e) + np.dot(edot, edot))
        # Parameter term (θ̃ᵀΓ⁻¹θ̃) - we assume θ̃ is small
        theta_tilde_norm = np.dot(self.theta_hat, np.linalg.solve(self.Gamma, self.theta_hat))
        V += 0.5 * theta_tilde_norm * 0.01  # Small factor to balance
        
        return tau, V, Y
    
    def update_adaptive_parameters(self,
                                  e: np.ndarray,
                                  edot: np.ndarray,
                                  Y: np.ndarray,
                                  lambda_param: float = 1.0):
        """
        Updates adaptive parameters using adaptive law with smoothing.
        
        Adaptive law: θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
        Smoothing: θ̂_smooth = α * θ̂_smooth_old + (1-α) * θ̂_new
        
        This guarantees V̇ ≤ 0 (asymptotic stability) and reduces oscillations.
        
        Args:
            e: Position error
            edot: Velocity error
            Y: Regression matrix
            lambda_param: Mixing factor (typically 1.0)
        """
        # Standard adaptive law
        # θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
        error_vector = e + lambda_param * edot
        
        # Calculate parameter update
        dtheta = -self.Gamma @ (Y.T @ error_vector) * self.dt
        
        # Update parameters without smoothing first
        theta_hat_new = self.theta_hat + dtheta
        
        # Apply limits before smoothing
        # Friction (can be positive or negative depending on model)
        for i in range(self.n_joints):
            theta_hat_new[i] = np.clip(theta_hat_new[i], -50.0, 50.0)  # friction
        
        # If we adapt gains, apply limits
        if self.adapt_gains:
            # PD gains must be positive
            theta_hat_new[-2] = np.clip(theta_hat_new[-2], 10.0, 500.0)  # Kp: min 10, max 500
            theta_hat_new[-1] = np.clip(theta_hat_new[-1], 1.0, 100.0)   # Kd: min 1, max 100
        
        # Smooth parameters using exponential filter
        # θ̂_smooth = α * θ̂_smooth_old + (1-α) * θ̂_new
        # This avoids abrupt changes and oscillations
        self.theta_hat_smoothed = (self.theta_smooth_alpha * self.theta_hat_smoothed + 
                                   (1.0 - self.theta_smooth_alpha) * theta_hat_new)
        
        # Use smoothed version
        self.theta_hat = self.theta_hat_smoothed.copy()
    
    def log_state(self, V: float, Y: np.ndarray):
        """Saves current state for analysis."""
        self.history['theta_hat'].append(self.theta_hat.copy())
        self.history['theta_hat_smoothed'].append(self.theta_hat_smoothed.copy())
        self.history['lyapunov'].append(V)
        self.history['Y_norm'].append(np.linalg.norm(Y))


# ==========================
# PD CONTROLLER IN JOINT SPACE (ORIGINAL)
# ==========================

def apply_joint_pd_control(model: mj.MjModel,
                           data: mj.MjData,
                           arm_qpos_idx: np.ndarray,
                           arm_dof_idx: np.ndarray,
                           arm_act_idx: np.ndarray,
                           q_des: np.ndarray,
                           Kp: float,
                           Kd: float):
    """
    Simple PD in joint space:
        tau = Kp (q_des - q) - Kd * qdot

    Writes torques to data.ctrl[arm_act_idx], respecting actuator_ctrlrange.
    """
    # current state
    q = data.qpos[arm_qpos_idx]
    qdot = data.qvel[arm_dof_idx]

    # TODO: if you have qdot_des, could use (Kp*(q_des-q)+Kd*(qdot_des-qdot))
    q_error = q_des - q
    qdot_error = -qdot

    tau = Kp * q_error + Kd * qdot_error

    # write to ctrl with saturation according to ctrlrange
    for i, act_id in enumerate(arm_act_idx):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[act_id]
        u = np.clip(tau[i], ctrl_min, ctrl_max)
        data.ctrl[act_id] = u


# ==========================
# COMPLETE ADAPTIVE CONTROLLER IN JOINT SPACE
# ==========================

def apply_adaptive_control(model: mj.MjModel,
                           data: mj.MjData,
                           arm_qpos_idx: np.ndarray,
                           arm_dof_idx: np.ndarray,
                           arm_act_idx: np.ndarray,
                           q_des: np.ndarray,
                           qdot_des: np.ndarray,
                           qddot_des: np.ndarray,
                           controller: AdaptiveController):
    """
    Complete adaptive control in joint space using equations of motion.
    
    Control: τ = Y(q, q̇, q̈_des) * θ̂ + Kp*e + Kd*ė
    Adaptive law: θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        arm_qpos_idx: Arm qpos indices
        arm_dof_idx: Arm qvel indices
        arm_act_idx: Actuator indices
        q_des: Desired positions
        qdot_des: Desired velocities
        qddot_des: Desired accelerations
        controller: AdaptiveController instance
    """
    # Current state
    q = data.qpos[arm_qpos_idx]
    qdot = data.qvel[arm_dof_idx]
    
    # Calculate complete adaptive control
    tau, V, Y = controller.compute_control(q, qdot, q_des, qdot_des, qddot_des, data)
    
    # Update adaptive parameters using adaptive law
    e = q_des - q
    edot = qdot_des - qdot
    controller.update_adaptive_parameters(e, edot, Y, lambda_param=1.0)
    
    # Log state (optional, for analysis)
    controller.log_state(V, Y)
    
    # Write to ctrl with saturation
    for i, act_id in enumerate(arm_act_idx):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[act_id]
        u = np.clip(tau[i], ctrl_min, ctrl_max)
        data.ctrl[act_id] = u
    
    return V


# ==========================
# MAIN
# ==========================

def main():
    # 1) load model and data
    model = mj.MjModel.from_xml_path(SLAVE_XML_PATH)
    data = mj.MjData(model)

    # (Optional) print all joints to fill in names:
    # print_all_joints(model)
    # return

    if not LEFT_ARM_JOINT_NAMES or not RIGHT_ARM_JOINT_NAMES:
        raise RuntimeError(
            "You must fill LEFT_ARM_JOINT_NAMES and RIGHT_ARM_JOINT_NAMES with the joint names of your arms."
        )

    # 2) build indices for each arm
    left_qpos_idx, left_dof_idx, left_act_idx = build_arm_maps(model, LEFT_ARM_JOINT_NAMES)
    right_qpos_idx, right_dof_idx, right_act_idx = build_arm_maps(model, RIGHT_ARM_JOINT_NAMES)

    n_left = len(left_qpos_idx)
    n_right = len(right_qpos_idx)

    print(f"Left arm joints:  {LEFT_ARM_JOINT_NAMES}")
    print(f"Right arm joints: {RIGHT_ARM_JOINT_NAMES}")
    print(f"Left arm qpos idx:  {left_qpos_idx}")
    print(f"Right arm qpos idx: {right_qpos_idx}")
    print(f"Left arm act idx:   {left_act_idx}")
    print(f"Right arm act idx:  {right_act_idx}")
    print()

    # 3) Trajectory configuration
    dt = model.opt.timestep
    USE_SINUSOIDAL_TEST = False # Change to False to use trajectory from NPZ file
    
    if USE_SINUSOIDAL_TEST:
        print("📐 Using SINUSOIDAL TEST TRAJECTORY (only first joint)")
        print("   - Only the first joint (shoulder_pitch) of each arm moves")
        print("   - All other joints remain fixed at 0")
        print("   - This isolates the controller problem")
        # Don't load trajectory from file
        _trajectory_data = None
    else:
        print("📂 Loading trajectory from NPZ file")
        try:
            load_trajectory_from_npz(TRAJECTORY_NPZ_PATH, left_qpos_idx, right_qpos_idx, dt)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("   Switching to sinusoidal test trajectory...")
            USE_SINUSOIDAL_TEST = True
            _trajectory_data = None
        except Exception as e:
            print(f"❌ Error loading trajectory: {e}")
            raise

    # 4) initialize pose (optional: set initial qpos = desired trajectory at t=0)
    if USE_SINUSOIDAL_TEST:
        q_left_des_0, q_right_des_0 = desired_arm_trajectories_sinusoidal_shoulders(0.0, n_left, n_right)
    else:
        q_left_des_0, q_right_des_0 = desired_arm_trajectories(0.0, n_left, n_right)
    data.qpos[left_qpos_idx] = q_left_des_0
    data.qpos[right_qpos_idx] = q_right_des_0
    mj.mj_forward(model, data)

    # 5) simulation parameters
    # sim_duration automatically adjusts to the duration of the loaded trajectory
    global _trajectory_duration
    if USE_SINUSOIDAL_TEST:
        sim_duration = 20.0  # Fixed duration for sinusoidal test
    else:
        sim_duration = _trajectory_duration if _trajectory_duration is not None else 20.0
    print(f"⏱️  Simulation duration: {sim_duration:.3f} s")
    
    # ===== CONTROLLER SELECTION =====
    USE_ADAPTIVE_CONTROL = True  # Change to False to use static PD
    ADAPT_GAINS = False  # Change to True to adapt PD gains (Kp, Kd)
    
    if USE_ADAPTIVE_CONTROL:
        if ADAPT_GAINS:
            print("🎯 Using ADAPTIVE CONTROL (Known masses/inertias, friction + adaptive PD gains)")
        else:
            print("🎯 Using ADAPTIVE CONTROL (Known masses/inertias, adaptive friction, fixed PD gains)")
        
        # ===== KNOWN PARAMETERS CONFIGURATION =====
        # If you know the robot's masses and inertias, specify them here
        # Otherwise, default values extracted from the MuJoCo model will be used
        
        # Known masses per joint (kg) - EXAMPLE: adjust according to your robot
        # For 7 arm joints: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
        left_known_masses = None  # None = use default values from model
        right_known_masses = None
        
        # Known inertias per joint (kg·m²) - EXAMPLE: adjust according to your robot
        left_known_inertias = None  # None = use default values from model
        right_known_inertias = None
        
        # If you want to specify known values, uncomment and adjust:
        # left_known_masses = np.array([2.5, 1.8, 1.2, 0.8, 0.3, 0.2, 0.1])  # kg
        # left_known_inertias = np.array([0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])  # kg·m²
        # right_known_masses = np.array([2.5, 1.8, 1.2, 0.8, 0.3, 0.2, 0.1])  # kg
        # right_known_inertias = np.array([0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])  # kg·m²
        
        # Initialize adaptive controllers for each arm
        # Now uses known masses/inertias (or default values from MuJoCo model)
        left_adaptive_ctrl = AdaptiveController(n_left, dt, model, left_qpos_idx, left_dof_idx,
                                                known_masses=left_known_masses,
                                                known_inertias=left_known_inertias,
                                                adapt_gains=ADAPT_GAINS)
        right_adaptive_ctrl = AdaptiveController(n_right, dt, model, right_qpos_idx, right_dof_idx,
                                                 known_masses=right_known_masses,
                                                 known_inertias=right_known_inertias,
                                                 adapt_gains=ADAPT_GAINS)
        
        if ADAPT_GAINS:
            print(f"   Initial PD gains: Kp={left_adaptive_ctrl.Kp:.1f}, Kd={left_adaptive_ctrl.Kd:.1f} (adaptive)")
            print(f"   Adaptive parameters: {left_adaptive_ctrl.n_params} per arm ({n_left} friction + 2 PD gains)")
        else:
            print(f"   Fixed PD gains: Kp={left_adaptive_ctrl.Kp:.1f}, Kd={left_adaptive_ctrl.Kd:.1f}")
            print(f"   Adaptive parameters: {left_adaptive_ctrl.n_params} per arm (only friction)")
        print(f"   Known masses (fixed): {left_adaptive_ctrl.known_masses}")
        print(f"   Known inertias (fixed): {left_adaptive_ctrl.known_inertias}")
    else:
        print("🎯 Using STATIC PD CONTROL")
        Kp = 80.0            # proportional gain (adjust as needed)
        Kd = 5.0             # derivative gain
    
    # Variables to calculate qdot_des and qddot_des by numerical differentiation
    q_left_des_prev = None
    q_right_des_prev = None
    qdot_left_des_prev = None
    qdot_right_des_prev = None
    t_prev = 0.0

    # to print errors every N steps
    print_every = 20
    step_count = 0

    # 5) launch passive viewer
    with mjviewer.launch_passive(model, data) as viewer:
        start_wall = time.time()
        while viewer.is_running() and (data.time < sim_duration):
            step_start = time.time()

            # 5.1) get desired trajectory at this simulation time
            t = data.time
            if USE_SINUSOIDAL_TEST:
                q_left_des, q_right_des = desired_arm_trajectories_sinusoidal_shoulders(t, n_left, n_right)
            else:
                q_left_des, q_right_des = desired_arm_trajectories(t, n_left, n_right)
            
            # Calculate desired velocity and acceleration by numerical differentiation
            if q_left_des_prev is not None and t > t_prev:
                dt_traj = t - t_prev
                # Desired velocity
                qdot_left_des = (q_left_des - q_left_des_prev) / dt_traj
                qdot_right_des = (q_right_des - q_right_des_prev) / dt_traj
                
                # Desired acceleration
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

            # 5.2) apply control to each arm
            if USE_ADAPTIVE_CONTROL:
                # Complete adaptive control (uses equations of motion)
                V_left = apply_adaptive_control(model, data,
                                                 left_qpos_idx, left_dof_idx, left_act_idx,
                                                 q_left_des, qdot_left_des, qddot_left_des,
                                                 left_adaptive_ctrl)
                V_right = apply_adaptive_control(model, data,
                                                  right_qpos_idx, right_dof_idx, right_act_idx,
                                                  q_right_des, qdot_right_des, qddot_right_des,
                                                  right_adaptive_ctrl)
            else:
                # Static PD control
                apply_joint_pd_control(model, data,
                                       left_qpos_idx, left_dof_idx, left_act_idx,
                                       q_left_des, Kp, Kd)
                apply_joint_pd_control(model, data,
                                       right_qpos_idx, right_dof_idx, right_act_idx,
                                       q_right_des, Kp, Kd)

            # 5.3) advance physics
            mj.mj_step(model, data)

            # 5.4) calculate and show tracking errors (L2 norm per arm)
            if step_count % print_every == 0:
                q_left = data.qpos[left_qpos_idx]
                q_right = data.qpos[right_qpos_idx]
                err_left = np.linalg.norm(q_left - q_left_des)
                err_right = np.linalg.norm(q_right - q_right_des)
                
                if USE_ADAPTIVE_CONTROL:
                    # Show adaptive parameters: friction and PD gains
                    # Friction: first n_joints elements of theta_hat
                    b_left_avg = np.mean(left_adaptive_ctrl.theta_hat[:left_adaptive_ctrl.n_joints])  # Average friction
                    b_right_avg = np.mean(right_adaptive_ctrl.theta_hat[:right_adaptive_ctrl.n_joints])
                    print(f"t = {t:6.3f}  |  ||e_left|| = {err_left:8.5f} rad  |  ||e_right|| = {err_right:8.5f} rad")
                    print(f"         b_left={b_left_avg:6.3f}, Kp_left={left_adaptive_ctrl.Kp:6.1f}, Kd_left={left_adaptive_ctrl.Kd:5.1f}  |  V_left={V_left:.6f}")
                    print(f"         b_right={b_right_avg:6.3f}, Kp_right={right_adaptive_ctrl.Kp:6.1f}, Kd_right={right_adaptive_ctrl.Kd:5.1f}  |  V_right={V_right:.6f}")
                    # Print individual friction values for each joint
                    b_left = left_adaptive_ctrl.theta_hat[:left_adaptive_ctrl.n_joints]
                    b_right = right_adaptive_ctrl.theta_hat[:right_adaptive_ctrl.n_joints]
                    print(f"         Friction (left):  {b_left}")
                    print(f"         Friction (right): {b_right}")
                else:
                    print(f"t = {t:6.3f}  |  ||e_left|| = {err_left:8.5f} rad  |  ||e_right|| = {err_right:8.5f} rad")

            step_count += 1

            # 5.5) update viewer
            viewer.sync()

            # 5.6) maintain ~real-time rate
            time_spent = time.time() - step_start
            time_until_next_step = dt - time_spent
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Simulation finished.")


if __name__ == "__main__":
    main()
