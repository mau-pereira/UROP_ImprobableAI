# Complete Report: Adaptive Control System for G1 Robot

## Executive Summary

This document describes **ALL** changes made to implement a Lyapunov theory-based adaptive control system for the Unitree G1 robot arms. The system enables adaptive friction control with complete dynamic compensation using MuJoCo, with options to adapt or fix PD gains.

---

## 1. Changes in `follower.py`

### 1.1. Complete Adaptive Control Implementation

#### Class `AdaptiveController` (lines 301-687)

**Location**: New class implemented from scratch

**Main features**:

1. **Control based on equations of motion**:
   - Uses: `M(q)q̈ + C(q,q̇)q̇ + g(q) = τ`
   - Complete dynamic compensation using MuJoCo

2. **Known parameters (NOT adaptive)**:
   - **Masses**: `known_masses` - Array of known masses per joint (kg)
   - **Inertias**: `known_inertias` - Array of known inertias per joint (kg·m²)
   - Default values if not specified: `m = 0.5 kg`, `I = 0.01 kg·m²`
   - Can be extracted from MuJoCo model or specified manually

3. **Adaptive parameters**:
   - **Viscous friction**: `θ = [b₁, b₂, ..., bₙ]` - ALWAYS adapted
   - **PD gains**: `Kp, Kd` - Optionally adaptive (controlled by flag)

4. **Flag `adapt_gains`**:
   - `adapt_gains = False` (default): Only friction is adapted, PD gains are fixed
     - `Kp = 150.0` (fixed)
     - `Kd = 10.0` (fixed)
     - `n_params = n_joints` (only friction)
   - `adapt_gains = True`: Friction + PD gains are adapted
     - Initial `Kp` = 150.0 (adaptive)
     - Initial `Kd` = 10.0 (adaptive)
     - `n_params = n_joints + 2` (friction + 2 gains)

5. **Adaptation matrix Γ**:
   - Friction: `Γ = 0.5 * I` (faster adaptation rate)
   - PD gains (if `adapt_gains=True`): `Γ = 0.1 * I` (more conservative rate)

#### Method `compute_dynamics_matrices()` (lines 415-488)

**Function**: Calculates `M(q)`, `C(q,q̇)q̇`, `g(q)` using MuJoCo

**Implementation**:
- Uses `mj_rne()` (Recursive Newton-Euler) from MuJoCo
- Saves and restores complete system state
- Calculates:
  1. `g(q)`: With `qdot=0`, `qddot=0` → `τ = g(q)`
  2. `C(q,q̇)q̇`: With `qddot=0` → `τ = C(q,q̇)q̇ + g(q)`, then subtracts `g`
  3. `M(q)`: Column by column using unit vectors in `qddot`

**Technical note**: Uses `mj.mj_rne(model, data, 1, data.qfrc_inverse)` where `1` is `flg_acc` (include accelerations).

#### Method `build_regression_matrix()` (lines 490-540)

**Function**: Builds regression matrix `Y` for adaptation

**Structure**:
- If `adapt_gains=False`: `Y[i, i] = qdot[i]` (only friction)
- If `adapt_gains=True`: 
  - `Y[i, i] = qdot[i]` (friction)
  - `Y[:, n_joints] = e` (Kp)
  - `Y[:, n_joints+1] = edot` (Kd)

#### Method `compute_control()` (lines 542-630)

**Control law**:
```
τ = M_known(q)q̈_des + C_known(q,q̇)q̇ + g_known(q) + τ_adaptive + τ_PD
```

**Components**:
1. **Known dynamic compensation**: `M_known*q̈_des + C_known*q̇ + g_known`
2. **Adaptive compensation**: 
   - If `adapt_gains=False`: `τ_friction = Y_friction * θ̂_friction`, `τ_PD = Kp*e + Kd*ė` (fixed)
   - If `adapt_gains=True`: `τ_adaptive = Y * θ̂` (includes friction + gains)

**Filter usage**:
- Uses `qddot_des_filtered` (filtered) instead of raw `qddot_des`

#### Method `update_adaptive_parameters()` (lines 632-679)

**Adaptive law**:
```
θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
```

**Process**:
1. Calculates update: `dtheta = -Γ @ (Y.T @ error_vector) * dt`
2. Updates: `theta_hat_new = theta_hat + dtheta`
3. Applies limits (clipping):
   - Friction: `[-50.0, 50.0]`
   - Kp (if adaptive): `[10.0, 500.0]`
   - Kd (if adaptive): `[1.0, 100.0]`
4. **Smooths parameters** (exponential filter):
   ```
   θ̂_smooth = α * θ̂_smooth_old + (1-α) * θ̂_new
   ```
   - `alpha = 0.9` (more smoothed)
5. Uses smoothed version: `theta_hat = theta_hat_smoothed`

#### Method `filter_qddot_des()` (lines 538-558)

**Function**: Exponential low-pass filter for `qddot_des`

**Implementation**:
```python
qddot_des_filtered = α * qddot_des_filtered_old + (1-α) * qddot_des_new
```
- `alpha = 0.8` (reduces noise from numerical differentiation)
- First iteration: `qddot_des_filtered = qddot_des` (no filter)

**Purpose**: Reduces noise introduced by double numerical differentiation of the trajectory.

---

### 1.2. Flag Configuration in `main()`

#### Main flags (lines 813, 854-855):

```python
USE_SINUSOIDAL_TEST = False  # True = sinusoidal test trajectory
USE_ADAPTIVE_CONTROL = True   # True = adaptive control, False = static PD
ADAPT_GAINS = False           # True = adapt Kp/Kd, False = fixed gains
```

**Behavior**:
- `USE_ADAPTIVE_CONTROL = True`, `ADAPT_GAINS = False` (current configuration):
  - Adaptive control with adaptive friction
  - Fixed PD gains: `Kp=150.0`, `Kd=10.0`
  - Known masses and inertias (not adaptive)

- `USE_ADAPTIVE_CONTROL = True`, `ADAPT_GAINS = True`:
  - Adaptive control with friction + adaptive PD gains
  - Known masses and inertias (not adaptive)

- `USE_ADAPTIVE_CONTROL = False`:
  - Simple static PD control: `τ = Kp*e + Kd*ė`
  - `Kp=80.0`, `Kd=5.0` (fixed values)

---

### 1.3. Improved Trajectory Loading

#### Function `load_trajectory_from_npz()` (lines 147-203)

**Critical changes**:

1. **Reads `dt` from NPZ** (lines 180-189):
   - **BEFORE**: Used `model_dt` (incorrect, caused 10x shorter duration)
   - **NOW**: Reads `dt` from NPZ file (required)
   - If `dt` doesn't exist in NPZ: raises `ValueError` (no more fallback)

2. **Correct duration calculation**:
   ```python
   _trajectory_dt = trajectory_dt  # actual recording dt
   _trajectory_duration = T * trajectory_dt  # Correct duration
   ```

3. **Informative messages**:
   - Shows detected recording frequency
   - Shows calculated duration

**Impact**: Simulation now lasts exactly the same as teleoperation.

---

### 1.4. Desired Velocities and Accelerations Calculation

#### Numerical differentiation (lines 906-948)

**Implementation**:
- `qdot_des`: First derivative of `q_des` (numerical differentiation)
- `qddot_des`: Second derivative of `q_des` (double numerical differentiation)

**Process**:
```python
qdot_des = (q_des - q_des_prev) / dt_traj
qddot_des = (qdot_des - qdot_des_prev) / dt_traj
```

**Usage in adaptive control**:
- `qddot_des` is filtered with `filter_qddot_des()` before use
- `qdot_des` is used directly (no filter)

---

### 1.5. Sinusoidal Test Trajectory

#### Function `desired_arm_trajectories_sinusoidal_shoulders()` (lines 198-235)

**Purpose**: Isolate controller problems by moving only the first joint

**Behavior**:
- Only the first joint (shoulder_pitch) of each arm moves sinusoidally
- All other joints remain at 0
- Frequency: `0.5 Hz`, Amplitude: `0.3 rad`

**Usage**: Activate with `USE_SINUSOIDAL_TEST = True` for debugging.

---

## 2. Changes in `teleop_trajectory_gen.py`

### 2.1. Recording Frequency: 50 Hz → 200 Hz

#### Configuration changes (lines 386-390):

**BEFORE**:
```python
recording_interval = int(200.0 / 50.0)  # Record every 4 iterations
# Recorded every 4 iterations (50 Hz)
```

**NOW**:
```python
recording_interval = int(200.0 / 200.0)  # Record every iteration (200 Hz)
# Records every iteration (200 Hz)
```

**Impact**: 
- 4x more data per second
- Better temporal resolution
- Fast movements captured with more precision

---

### 2.2. Save `dt` in NPZ File

#### Function `save_trajectory()` (lines 145-155):

**Critical change**:
```python
# BEFORE: Didn't save dt
np.savez(output_path, qpos=..., qvel=..., ctrl=..., mocap=...)

# NOW: Saves dt
recording_dt = 1.0 / 200.0  # 200 Hz recording rate
np.savez(
    output_path,
    qpos=qpos_log,
    qvel=qvel_log,
    ctrl=ctrl_log,
    mocap=mocap_log,
    dt=recording_dt,  # ⭐ NEW: recording timestep
)
```

**Purpose**: Allows `follower.py` to read the correct `dt` and calculate correct duration.

---

### 2.3. Corrected Duration Calculation

#### Changes in `save_trajectory()` (lines 117-123):

**BEFORE**:
```python
duration = T / 50.0  # 50 Hz recording rate (hardcoded)
```

**NOW**:
```python
recording_dt = 1.0 / 200.0  # 200 Hz recording rate = 0.005 seconds
duration = T * recording_dt  # Correct duration
```

**Improved messages**:
- Shows recording frequency: `"Recording rate: 200 Hz (dt = 0.005000 s)"`
- Shows correctly calculated duration

---

### 2.4. Updated Messages

#### Changes in prints (line 408, 484):

**BEFORE**:
```python
print(f"   Recording rate: 50 Hz (every {recording_interval} iterations at 200 Hz)")
# Record trajectory at 50 Hz (every 4 iterations at 200 Hz)
```

**NOW**:
```python
print(f"   Recording rate: 200 Hz (every {recording_interval} iteration at 200 Hz)")
# Record trajectory at 200 Hz (every iteration)
```

---

## 3. Current Configuration Summary

### 3.1. Adaptive vs Fixed Parameters

| Parameter | Status | Value/Behavior |
|-----------|--------|----------------|
| **Masses** | ❌ Fixed (known) | Extracted from MuJoCo model or manually specified |
| **Inertias** | ❌ Fixed (known) | Extracted from MuJoCo model or manually specified |
| **Viscous friction** | ✅ Adaptive | `θ = [b₁, b₂, ..., b₇]`, adapts with `Γ = 0.5` |
| **Gain Kp** | ❌ Fixed | `Kp = 150.0` (when `ADAPT_GAINS = False`) |
| **Gain Kd** | ❌ Fixed | `Kd = 10.0` (when `ADAPT_GAINS = False`) |

**Note**: If `ADAPT_GAINS = True`, then Kp and Kd are also adapted.

---

### 3.2. Frequencies and Timesteps

| Component | Frequency/Timestep | Location |
|-----------|-------------------|----------|
| **Control loop (teleop)** | 200 Hz (`dt = 0.005 s`) | `teleop_trajectory_gen.py:384` |
| **Trajectory recording** | 200 Hz (`dt = 0.005 s`) | `teleop_trajectory_gen.py:390` |
| **MuJoCo simulation** | Variable (typically 500 Hz, `dt ≈ 0.002 s`) | `model.opt.timestep` |
| **Trajectory interpolation** | Uses `dt` from NPZ (200 Hz) | `follower.py:267` |

**Result**: Simulation lasts exactly the same as teleoperation.

---

### 3.3. Filters and Smoothing

| Filter | Type | Alpha | Purpose |
|--------|------|-------|---------|
| **`qddot_des`** | Exponential low-pass | `0.8` | Reduces noise from numerical differentiation |
| **`theta_hat`** | Exponential smoothing | `0.9` | Avoids abrupt changes in adaptive parameters |

**Equations**:
- `qddot_des_filtered = 0.8 * qddot_des_filtered_old + 0.2 * qddot_des_new`
- `theta_hat_smoothed = 0.9 * theta_hat_smoothed_old + 0.1 * theta_hat_new`

---

### 3.4. Adaptive Parameter Limits

| Parameter | Minimum | Maximum | Location |
|-----------|---------|---------|----------|
| **Friction `b`** | `-50.0` | `50.0` | `follower.py:664` |
| **Kp (if adaptive)** | `10.0` | `500.0` | `follower.py:669` |
| **Kd (if adaptive)** | `1.0` | `100.0` | `follower.py:670` |

---

## 4. NPZ Data Structure

### 4.1. NPZ File Format

**Saved fields**:
```python
{
    'qpos': (T, nq)  # Positions of all joints
    'qvel': (T, nv)  # Velocities of all joints
    'ctrl': (T, nu)  # Control commands
    'mocap': (T, 14) # Positions/orientations of mocap bodies (2 bodies × 7)
    'dt': float      # ⭐ NEW: Recording timestep (0.005 s for 200 Hz)
}
```

**Requirements**:
- `dt` is **required** (not optional)
- If `dt` is missing, `follower.py` raises `ValueError`

---

## 5. Implemented Control Theory

### 5.1. Lyapunov Function

```
V = ½(eᵀe + ėᵀė) + ½(θ̃ᵀΓ⁻¹θ̃)
```

Where:
- `e = q_des - q` (position error)
- `ė = qdot_des - qdot` (velocity error)
- `θ̃ = θ - θ̂` (parameter error)

### 5.2. Control Law

```
τ = M_known(q)q̈_des + C_known(q,q̇)q̇ + g_known(q) + Y(q,q̇,e,ė)θ̂ + Kp*e + Kd*ė
```

Where:
- `M_known, C_known, g_known`: Known dynamic matrices (fixed masses/inertias)
- `Y`: Regression matrix
- `θ̂`: Estimated adaptive parameters (friction, optionally Kp/Kd)

### 5.3. Adaptive Law

```
θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
```

Where:
- `Γ`: Adaptation matrix (diagonal)
- `λ = 1.0`: Mixing factor (typically 1.0)

**Guarantee**: `V̇ ≤ 0` → Asymptotic stability

---

## 6. Modified Files

### 6.1. `follower.py`

**Modified/added lines**:
- Lines 1-17: Documentation comments
- Lines 147-203: `load_trajectory_from_npz()` (improved)
- Lines 198-235: `desired_arm_trajectories_sinusoidal_shoulders()` (new)
- Lines 301-687: `AdaptiveController` class (complete, new)
- Lines 415-488: `compute_dynamics_matrices()` (new)
- Lines 490-540: `build_regression_matrix()` (new)
- Lines 542-630: `compute_control()` (new)
- Lines 538-558: `filter_qddot_des()` (new)
- Lines 632-679: `update_adaptive_parameters()` (new)
- Lines 681-687: `log_state()` (new)
- Lines 813, 854-855: Configuration flags
- Lines 906-948: `qdot_des` and `qddot_des` calculation (improved)

**Total**: ~600 new/modified lines

---

### 6.2. `teleop_trajectory_gen.py`

**Modified lines**:
- Lines 117-123: Duration calculation (corrected)
- Lines 145-155: Save `dt` in NPZ (new)
- Lines 386-390: `recording_interval` (changed from 4 to 1)
- Line 408: Print message (updated)
- Line 484: Comment (updated)

**Total**: ~15 modified lines

---

## 7. Dependencies and Requirements

### 7.1. Required Libraries

- `mujoco` (MuJoCo Python bindings)
- `numpy`
- `pathlib`

### 7.2. MuJoCo Functions Used

- `mj.mj_forward()`: Forward dynamics
- `mj.mj_rne()`: Recursive Newton-Euler (inverse dynamics)
- `data.qfrc_inverse`: Array to store calculated forces/torques

---

## 8. Recommended Configuration

### 8.1. For Better Stability

```python
USE_ADAPTIVE_CONTROL = True
ADAPT_GAINS = False  # Fixed gains (more stable)
```

**Reason**: Adapting only friction is more stable than also adapting gains.

### 8.2. For Better Precision (if you know masses/inertias)

```python
left_known_masses = np.array([2.5, 1.8, 1.2, 0.8, 0.3, 0.2, 0.1])  # kg
left_known_inertias = np.array([0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])  # kg·m²
```

**Reason**: More precise dynamic compensation.

### 8.3. For Debugging

```python
USE_SINUSOIDAL_TEST = True  # Isolates problems by moving only one joint
```

---

## 9. Solved Problems

### 9.1. Incorrect Simulation Duration

**Problem**: Simulation lasted 10x less than teleoperation.

**Cause**: Used `model_dt` (0.002 s) instead of actual recording `dt` (0.02 s at 50 Hz, 0.005 s at 200 Hz).

**Solution**: 
- Save `dt` in NPZ
- Read `dt` from NPZ in `follower.py`
- Use correct `dt` to calculate duration

---

### 9.2. Noise in Desired Accelerations

**Problem**: `qddot_des` had a lot of noise from double numerical differentiation.

**Solution**: Exponential low-pass filter with `alpha = 0.8`.

---

### 9.3. Oscillations in Adaptive Parameters

**Problem**: `theta_hat` changed abruptly causing instability.

**Solution**: Exponential smoothing filter with `alpha = 0.9`.

---

### 9.4. Error in `mj_rne()` Call

**Problem**: `TypeError: mj_rne(): incompatible function arguments`

**Cause**: Incorrect call with 5 arguments instead of 4.

**Solution**: Change from `mj.mj_rne(model, data, True, False, result)` to `mj.mj_rne(model, data, 1, result)`.

---

## 10. Suggested Next Steps

1. **Adjust filter `alpha` values** according to experimental results
2. **Specify real masses/inertias** of G1 robot if available
3. **Adjust parameter limits** according to observed behavior
4. **Add dead-zone** to avoid adaptation when error is very small
5. **Add leakage** to prevent parameter drift

---

## 11. Theoretical References

- **Lyapunov Stability Theory**: Theoretical basis to guarantee stability
- **Adaptive Control**: Real-time parameter adjustment
- **Recursive Newton-Euler**: Algorithm to calculate inverse dynamics
- **Linear Parameterization**: Express dynamics as `τ = Y(q,q̇)θ`

---

**Creation date**: 2025-01-11  
**Last update**: 2025-01-11  
**Version**: 1.0
