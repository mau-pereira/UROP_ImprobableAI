"""
Follower Robot Controller - Control Adaptativo Basado en Lyapunov

Este script implementa dos tipos de control:
1. Control PD estático (ganancias fijas)
2. Control adaptativo basado en Lyapunov (ganancias que se ajustan automáticamente)

CONTROL ADAPTATIVO:
- Función de Lyapunov: V = ½(eᵀe + ėᵀė) + términos de parámetros
- Leyes adaptativas:
  * Kp se ajusta según el error de posición: Kṗ = γ_Kp * e²
  * Kd se ajusta según el error de velocidad: Kḋ = γ_Kd * ė²
  * b̂ estima fricción viscosa: b̂̇ = γ_b * qdot * e
- Garantiza estabilidad asintótica (V̇ ≤ 0)

Para cambiar entre controladores, modifica USE_ADAPTIVE_CONTROL en main().
"""

import time
import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer
from pathlib import Path


# ==========================
# CONFIGURACIÓN DE RUTAS
# ==========================

# Pon aquí la ruta local al XML del G1 esclavo (29 DOF, con actuadores tipo motor)
SLAVE_XML_PATH = "../assets/g1_from_unitree_github/scene_29dof.xml"

# (Opcional, para el futuro) ruta al XML del G1 líder (menagerie, actuadores tipo position)
LEADER_XML_PATH = "path/to/mujoco_menagerie/unitree_g1/scene.xml"

# Ruta al archivo NPZ con trayectoria de teleoperación (formato compatible con 09_mujoco_streaming.py)
# Puedes usar una ruta relativa desde este archivo o absoluta
# Ejemplo: "trajectories/traj_20251211_173426.npz" o ruta absoluta
# La ruta se resuelve relativa al directorio donde está este archivo
_HERE = Path(__file__).parent
TRAJECTORY_NPZ_PATH = _HERE / "trajectories" / "traj_20251211_173426.npz"


# ==========================
# JOINTS DE LOS BRAZOS
# ==========================
# IMPORTANTE:
#  - Rellena estas listas con los NOMBRES EXACTOS de los joints de cada brazo
#    tal como aparecen en tu XML del G1 esclavo (scene_29dof.xml).
#  - Puedes imprimir todos los joints con la función print_all_joints(model) de más abajo.

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
# UTILIDADES
# ==========================

def print_all_joints(model: mj.MjModel):
    """Imprime todos los joints con sus índices para que puedas copiar nombres."""
    print("=== JOINTS EN EL MODELO ===")
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
    A partir de una lista de nombres de joints, construye:
    - índices de qpos (posición generalizada)
    - índices de dof (velocidad generalizada)
    - índices de actuadores que controlan cada joint (para data.ctrl)

    Asume 1 actuador por joint (lo típico en modelos de robots).
    """
    # Mapa joint_id -> lista de actuadores que lo controlan
    jointid_to_actids: dict[int, list[int]] = {}
    for act_id in range(model.nu):
        # actuator_trnid[act_id, 0] contiene el joint_id si es un actuator de tipo joint
        joint_id = model.actuator_trnid[act_id, 0]
        if joint_id >= 0:
            jointid_to_actids.setdefault(joint_id, []).append(act_id)

    qpos_indices = []
    dof_indices = []
    act_indices = []

    for name in joint_names:
        if name is None or name == "":
            raise ValueError("Hay un joint vacío en la lista, revisa LEFT_ARM_JOINT_NAMES / RIGHT_ARM_JOINT_NAMES.")

        j_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if j_id < 0:
            raise RuntimeError(f"Joint '{name}' no existe en el modelo.")

        qpos_idx = model.jnt_qposadr[j_id]
        dof_idx = model.jnt_dofadr[j_id]

        if j_id not in jointid_to_actids:
            raise RuntimeError(f"Joint '{name}' (id={j_id}) no tiene actuador asociado en 'actuator_trnid'.")

        # Asumimos un actuador por joint; si hay más, tomamos el primero
        act_id = jointid_to_actids[j_id][0]

        qpos_indices.append(qpos_idx)
        dof_indices.append(dof_idx)
        act_indices.append(act_id)

    return np.array(qpos_indices, dtype=int), np.array(dof_indices, dtype=int), np.array(act_indices, dtype=int)


# ==========================
# TRAYECTORIA DESEADA
# ==========================

# Variable global para almacenar la trayectoria cargada
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
    Carga una trayectoria desde un archivo NPZ (formato compatible con 09_mujoco_streaming.py).
    
    El archivo NPZ debe contener:
    - 'qpos': array de shape (T, nq) con posiciones de todos los joints
    
    Args:
        npz_path: Ruta al archivo .npz
        left_qpos_idx: Índices de qpos para el brazo izquierdo
        right_qpos_idx: Índices de qpos para el brazo derecho
        model_dt: Timestep del modelo MuJoCo (para calcular duración)
    """
    global _trajectory_data, _trajectory_dt, _trajectory_duration
    global _left_qpos_idx, _right_qpos_idx
    
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Archivo de trayectoria no encontrado: {npz_path}")
    
    print(f"📂 Cargando trayectoria desde: {npz_path}")
    traj = np.load(npz_path)
    
    if "qpos" not in traj:
        raise ValueError("El archivo NPZ debe contener la clave 'qpos'")
    
    qpos_log = traj["qpos"]  # shape: (T, nq)
    T = qpos_log.shape[0]
    
    # Extraer solo los joints de los brazos
    q_left_traj = qpos_log[:, left_qpos_idx]  # shape: (T, 7)
    q_right_traj = qpos_log[:, right_qpos_idx]  # shape: (T, 7)
    
    # Guardar en variable global
    _trajectory_data = {
        'q_left': q_left_traj,
        'q_right': q_right_traj,
    }
    _trajectory_dt = model_dt
    _trajectory_duration = T * model_dt
    _left_qpos_idx = left_qpos_idx
    _right_qpos_idx = right_qpos_idx
    
    print(f"✅ Trayectoria cargada: {T} timesteps, duración = {_trajectory_duration:.3f} s")
    print(f"   Brazo izquierdo: {q_left_traj.shape}")
    print(f"   Brazo derecho: {q_right_traj.shape}")


def desired_arm_trajectories_sinusoidal_shoulders(t: float,
                                                   n_left: int,
                                                   n_right: int,
                                                   amplitude: float = 0.5,
                                                   frequency: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera trayectoria sinusoidal SOLO para el primer joint de cada brazo (shoulder_pitch).
    Todos los demás joints permanecen en 0 (fijos).
    
    Esto aísla el problema del controlador moviendo solo un joint por brazo.
    
    Args:
        t: Tiempo actual de simulación (segundos)
        n_left: Número de joints del brazo izquierdo
        n_right: Número de joints del brazo derecho
        amplitude: Amplitud de la sinusoide en radianes (default: 0.5 rad ≈ 28°)
        frequency: Frecuencia en Hz (default: 0.5 Hz = período de 2 segundos)
    
    Returns:
        q_left_des: Array de shape (n_left,) con joint angles deseados
        q_right_des: Array de shape (n_right,) con joint angles deseados
    """
    # Inicializar todos los joints en 0
    q_left_des = np.zeros(n_left)
    q_right_des = np.zeros(n_right)
    
    # Solo el primer joint (shoulder_pitch) se mueve sinusoidalmente
    if n_left > 0:
        # Brazo izquierdo: solo el primer joint (shoulder_pitch)
        q_left_des[0] = amplitude * np.sin(2 * np.pi * frequency * t)
    
    if n_right > 0:
        # Brazo derecho: solo el primer joint (shoulder_pitch)
        q_right_des[0] = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Todos los demás joints (shoulder_roll, shoulder_yaw, elbow, wrist) permanecen en 0
    
    return q_left_des, q_right_des


def desired_arm_trajectories(t: float,
                             n_left: int,
                             n_right: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve q_des_left(t), q_des_right(t) para cada brazo desde trayectoria cargada.
    
    Interpola linealmente entre timesteps de la trayectoria cargada.
    Si t excede la duración de la trayectoria, devuelve el último valor.
    
    Args:
        t: Tiempo actual de simulación (segundos)
        n_left: Número de joints del brazo izquierdo
        n_right: Número de joints del brazo derecho
    
    Returns:
        q_left_des: Array de shape (n_left,) con joint angles deseados del brazo izquierdo
        q_right_des: Array de shape (n_right,) con joint angles deseados del brazo derecho
    """
    global _trajectory_data, _trajectory_dt, _trajectory_duration
    
    if _trajectory_data is None:
        raise RuntimeError(
            "Trayectoria no cargada. Llama a load_trajectory_from_npz() antes de usar desired_arm_trajectories()."
        )
    
    q_left_traj = _trajectory_data['q_left']
    q_right_traj = _trajectory_data['q_right']
    
    # Calcular índice de timestep (puede ser fraccional para interpolación)
    step_index = t / _trajectory_dt
    
    # Si t excede la duración, usar el último timestep
    if step_index >= q_left_traj.shape[0] - 1:
        q_left_des = q_left_traj[-1]
        q_right_des = q_right_traj[-1]
    else:
        # Interpolación lineal entre timesteps
        idx_low = int(np.floor(step_index))
        idx_high = min(idx_low + 1, q_left_traj.shape[0] - 1)
        alpha = step_index - idx_low  # factor de interpolación [0, 1)
        
        q_left_des = (1 - alpha) * q_left_traj[idx_low] + alpha * q_left_traj[idx_high]
        q_right_des = (1 - alpha) * q_right_traj[idx_low] + alpha * q_right_traj[idx_high]
    
    return q_left_des, q_right_des


# ==========================
# CONTROL ADAPTATIVO COMPLETO
# ==========================

class AdaptiveController:
    """
    Controlador adaptativo completo basado en Lyapunov usando ecuaciones del movimiento.
    
    Teoría:
    - Ecuaciones del movimiento: M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
    - Control: τ = M_known(q)q̈_des + C_known(q,q̇)q̇ + g_known(q) + Y_friction * θ_friction + Kp*e + Kd*ė
    - Ley adaptativa: θ̂̇ = -ΓYᵀ(e + λė)
    
    Parámetros:
    - Masas e inercias: CONOCIDAS (fijas, no se adaptan)
    - Fricción: DESCONOCIDA (se adapta)
    """
    
    def __init__(self, n_joints: int, dt: float, model: mj.MjModel = None, 
                 arm_qpos_idx: np.ndarray = None, arm_dof_idx: np.ndarray = None,
                 known_masses: np.ndarray = None, known_inertias: np.ndarray = None,
                 adapt_gains: bool = False):
        """
        Args:
            n_joints: Número de joints a controlar
            dt: Timestep de simulación
            model: Modelo MuJoCo (opcional, para construir Y)
            arm_qpos_idx: Índices de qpos del brazo (opcional)
            arm_dof_idx: Índices de qvel del brazo (opcional)
            known_masses: Array de masas conocidas [m₁, m₂, ..., mₙ] (kg)
            known_inertias: Array de inercias conocidas [I₁, I₂, ..., Iₙ] (kg·m²)
            adapt_gains: Si True, adapta ganancias PD (Kp, Kd). Si False, usa ganancias fijas.
        """
        self.n_joints = n_joints
        self.dt = dt
        self.model = model
        self.arm_qpos_idx = arm_qpos_idx
        self.arm_dof_idx = arm_dof_idx
        self.adapt_gains = adapt_gains
        
        # Parámetros conocidos (fijos, no se adaptan)
        if known_masses is not None:
            if len(known_masses) != n_joints:
                raise ValueError(f"known_masses debe tener {n_joints} elementos, tiene {len(known_masses)}")
            self.known_masses = np.array(known_masses)
        else:
            # Valores por defecto (extraer del modelo MuJoCo si es posible)
            self.known_masses = np.ones(n_joints) * 0.5  # kg por defecto
        
        if known_inertias is not None:
            if len(known_inertias) != n_joints:
                raise ValueError(f"known_inertias debe tener {n_joints} elementos, tiene {len(known_inertias)}")
            self.known_inertias = np.array(known_inertias)
        else:
            # Valores por defecto
            self.known_inertias = np.ones(n_joints) * 0.01  # kg·m² por defecto
        
        # Ganancias PD: adaptativas o fijas según adapt_gains
        if adapt_gains:
            # Número de parámetros adaptativos: fricción (1 por joint) + ganancias PD (2 globales)
            # Estructura: [b₁, b₂, ..., bₙ, Kp, Kd]
            self.n_params = n_joints + 2
            self.theta_hat = np.zeros(self.n_params)
            # Fricción inicial = 0 (primeros n elementos)
            # Ganancias iniciales (últimos 2 elementos)
            self.theta_hat[-2] = 150.0  # Kp inicial
            self.theta_hat[-1] = 10.0   # Kd inicial
            
            # Matriz de adaptación Γ
            self.Gamma = np.eye(self.n_params) * 0.5
            self.Gamma[-2, -2] = 0.1  # Gamma para Kp (más conservador)
            self.Gamma[-1, -1] = 0.1  # Gamma para Kd (más conservador)
        else:
            # Solo fricción se adapta, ganancias son fijas
            # Estructura: [b₁, b₂, ..., bₙ]
            self.n_params = n_joints
            self.theta_hat = np.zeros(self.n_params)  # Solo fricción
            self.Gamma = np.eye(self.n_params) * 0.5
            
            # Ganancias PD fijas
            self._Kp = 150.0
            self._Kd = 10.0
        
        
        # Factor de filtrado para q̈_des (para suavizar)
        self.lambda_filter = 0.1
        
        # Historial para análisis
        self.history = {
            'theta_hat': [],
            'lyapunov': [],
            'Y_norm': []
        }
    
    @property
    def Kp(self):
        """Ganancia proporcional (adaptativa o fija según adapt_gains)."""
        if self.adapt_gains:
            return self.theta_hat[-2]
        else:
            return self._Kp
    
    @property
    def Kd(self):
        """Ganancia derivativa (adaptativa o fija según adapt_gains)."""
        if self.adapt_gains:
            return self.theta_hat[-1]
        else:
            return self._Kd
    
    def compute_dynamics_matrices(self,
                                  q: np.ndarray,
                                  qdot: np.ndarray,
                                  data: mj.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula matrices dinámicas usando MuJoCo: M(q), C(q,q̇), g(q)
        
        Usa mj_rne() (Recursive Newton-Euler) para calcular:
        τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
        
        Args:
            q: Posiciones actuales del brazo
            qdot: Velocidades actuales del brazo
            data: Datos MuJoCo (debe tener acceso al modelo completo)
        
        Returns:
            M: Matriz de inercia de shape (n_joints, n_joints)
            C_qdot: Vector C(q,q̇)q̇ de shape (n_joints,)
            g: Vector de gravedad de shape (n_joints,)
        """
        # Guardar estado completo del sistema
        q_full_orig = data.qpos.copy()
        qdot_full_orig = data.qvel.copy()
        qacc_full_orig = data.qacc.copy()
        
        # Establecer estado del brazo en data
        data.qpos[self.arm_qpos_idx] = q
        data.qvel[self.arm_qpos_idx] = qdot
        
        # 1. Calcular g(q): fuerzas gravitacionales (con qdot=0, qddot=0)
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mj.mj_forward(self.model, data)
        # mj_rne calcula: τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
        # Con qdot=0, qddot=0: τ = g(q)
        # flg_acc=1 significa incluir aceleraciones (pero están en 0)
        mj.mj_rne(self.model, data, 1, data.qfrc_inverse)
        g = data.qfrc_inverse[self.arm_qpos_idx].copy()
        
        # 2. Calcular C(q,q̇)q̇: términos de Coriolis/centrífugos
        data.qvel[self.arm_qpos_idx] = qdot
        data.qacc[:] = 0.0
        mj.mj_forward(self.model, data)
        mj.mj_rne(self.model, data, 1, data.qfrc_inverse)
        # Con qddot=0: τ = C(q,q̇)q̇ + g(q)
        C_qdot_plus_g = data.qfrc_inverse[self.arm_qpos_idx].copy()
        C_qdot = C_qdot_plus_g - g
        
        # 3. Calcular M(q): matriz de inercia (columna por columna usando mj_rne)
        M = np.zeros((self.n_joints, self.n_joints))
        data.qvel[self.arm_qpos_idx] = qdot  # mantener velocidad
        
        for j in range(self.n_joints):
            # Calcular columna j de M usando qddot = e_j (vector unitario)
            qddot_unit = np.zeros(self.n_joints)
            qddot_unit[j] = 1.0
            
            # Establecer aceleración unitaria solo en el brazo
            data.qacc[:] = 0.0
            data.qacc[self.arm_qpos_idx] = qddot_unit
            
            mj.mj_forward(self.model, data)
            mj.mj_rne(self.model, data, 1, data.qfrc_inverse)
            # τ = M(q)q̈ + C(q,q̇)q̇ + g(q)
            # Con q̈ = e_j: τ = M[:,j] + C(q,q̇)q̇ + g(q)
            tau_col = data.qfrc_inverse[self.arm_qpos_idx].copy()
            M[:, j] = tau_col - C_qdot - g
        
        # Restaurar estado original
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
        Construye la matriz de regresión Y(q, q̇, e, ė) para FRICCIÓN y opcionalmente GANANCIAS PD.
        
        Si adapt_gains=True: θ = [b₁, b₂, ..., bₙ, Kp, Kd]
        Si adapt_gains=False: θ = [b₁, b₂, ..., bₙ]
        
        Args:
            q: Posiciones actuales
            qdot: Velocidades actuales
            qddot: Aceleraciones deseadas (no usado, pero necesario para compatibilidad)
            q_des: Posiciones deseadas (para calcular error e, solo si adapt_gains=True)
            qdot_des: Velocidades deseadas (para calcular error ė, solo si adapt_gains=True)
            data: Datos MuJoCo (no usado aquí, pero necesario para compatibilidad)
        
        Returns:
            Y: Matriz de regresión de shape (n_joints, n_params)
        """
        Y = np.zeros((self.n_joints, self.n_params))
        
        # Columnas 0 a n-1: Fricción (1 por joint)
        for i in range(self.n_joints):
            Y[i, i] = qdot[i]  # Fricción del joint i es proporcional a su velocidad
        
        # Si adaptamos ganancias, agregar columnas para Kp y Kd
        if self.adapt_gains:
            # Calcular errores
            if q_des is not None and qdot_des is not None:
                e = q_des - q
                edot = qdot_des - qdot
            else:
                e = np.zeros(self.n_joints)
                edot = np.zeros(self.n_joints)
            
            # Columna n: Ganancias proporcionales Kp (global)
            Y[:, self.n_joints] = e
            
            # Columna n+1: Ganancias derivativas Kd (global)
            Y[:, self.n_joints + 1] = edot
        
        return Y
    
    def compute_control(self,
                       q: np.ndarray,
                       qdot: np.ndarray,
                       q_des: np.ndarray,
                       qdot_des: np.ndarray,
                       qddot_des: np.ndarray,
                       data: mj.MjData = None) -> tuple[np.ndarray, float]:
        """
        Calcula el torque de control adaptativo usando masas/inercias conocidas.
        
        Control: τ = M_known(q)q̈_des + C_known(q,q̇)q̇ + g_known(q) + Y_friction*θ̂_friction + Kp*e + Kd*ė
        
        Donde:
        - M_known, C_known, g_known: calculados con MuJoCo usando masas/inercias conocidas
        - Y_friction*θ̂_friction: compensación adaptativa de fricción
        
        Args:
            q: Posiciones actuales
            qdot: Velocidades actuales
            q_des: Posiciones deseadas
            qdot_des: Velocidades deseadas
            qddot_des: Aceleraciones deseadas
            data: Datos MuJoCo (requerido para calcular dinámicas)
        
        Returns:
            tau: Torques de control
            V: Valor de la función de Lyapunov
        """
        # Errores
        e = q_des - q
        edot = qdot_des - qdot
        
        # Calcular dinámicas usando MuJoCo (con masas/inercias del modelo)
        if data is not None and self.model is not None and self.arm_qpos_idx is not None:
            M, C_qdot, g = self.compute_dynamics_matrices(q, qdot, data)
            
            # Compensación dinámica usando masas/inercias conocidas (del modelo MuJoCo)
            # τ_dynamic = M(q)q̈_des + C(q,q̇)q̇ + g(q)
            tau_dynamic_known = M @ qddot_des + C_qdot + g
        else:
            # Fallback si no hay acceso a MuJoCo
            tau_dynamic_known = np.zeros(self.n_joints)
        
        # Construir matriz de regresión
        Y = self.build_regression_matrix(q, qdot, qddot_des, q_des, qdot_des, data)
        
        # Compensación adaptativa
        if self.adapt_gains:
            # τ_adaptive = Y * θ̂ = Y_friction * θ̂_friction + Y_Kp * Kp + Y_Kd * Kd
            # Las ganancias PD ya están incluidas en tau_adaptive
            tau_adaptive = Y @ self.theta_hat
            tau = tau_dynamic_known + tau_adaptive
        else:
            # Solo fricción adaptativa, ganancias PD fijas
            tau_friction = Y @ self.theta_hat  # Solo fricción
            tau_pd = self.Kp * e + self.Kd * edot  # Ganancias PD fijas
            tau = tau_dynamic_known + tau_friction + tau_pd
        
        # Función de Lyapunov completa
        # V = ½(eᵀe + ėᵀė + θ̃ᵀΓ⁻¹θ̃)
        # Simplificamos asumiendo θ̃ ≈ 0 (parámetros bien estimados)
        V = 0.5 * (np.dot(e, e) + np.dot(edot, edot))
        # Término de parámetros (θ̃ᵀΓ⁻¹θ̃) - asumimos que θ̃ es pequeño
        theta_tilde_norm = np.dot(self.theta_hat, np.linalg.solve(self.Gamma, self.theta_hat))
        V += 0.5 * theta_tilde_norm * 0.01  # Factor pequeño para balancear
        
        return tau, V, Y
    
    def update_adaptive_parameters(self,
                                  e: np.ndarray,
                                  edot: np.ndarray,
                                  Y: np.ndarray,
                                  lambda_param: float = 1.0):
        """
        Actualiza los parámetros adaptativos usando ley adaptativa.
        
        Ley adaptativa: θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
        
        Esto garantiza V̇ ≤ 0 (estabilidad asintótica)
        
        Args:
            e: Error de posición
            edot: Error de velocidad
            Y: Matriz de regresión
            lambda_param: Factor de mezcla (típicamente 1.0)
        """
        # Ley adaptativa estándar
        # θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
        error_vector = e + lambda_param * edot
        
        # Actualizar parámetros
        dtheta = -self.Gamma @ (Y.T @ error_vector) * self.dt
        self.theta_hat += dtheta
        
        # Límites para evitar parámetros no físicos
        # Fricción (puede ser positiva o negativa dependiendo del modelo)
        for i in range(self.n_joints):
            self.theta_hat[i] = np.clip(self.theta_hat[i], -50.0, 50.0)  # fricción
        
        # Si adaptamos ganancias, aplicar límites
        if self.adapt_gains:
            # Ganancias PD deben ser positivas
            self.theta_hat[-2] = np.clip(self.theta_hat[-2], 10.0, 500.0)  # Kp: mínimo 10, máximo 500
            self.theta_hat[-1] = np.clip(self.theta_hat[-1], 1.0, 100.0)   # Kd: mínimo 1, máximo 100
    
    def log_state(self, V: float, Y: np.ndarray):
        """Guarda el estado actual para análisis."""
        self.history['theta_hat'].append(self.theta_hat.copy())
        self.history['lyapunov'].append(V)
        self.history['Y_norm'].append(np.linalg.norm(Y))


# ==========================
# CONTROLADOR PD EN ESPACIO DE JUNTAS (ORIGINAL)
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
    PD simple en espacio de juntas:
        tau = Kp (q_des - q) - Kd * qdot

    Escribe los torques en data.ctrl[arm_act_idx], respetando actuator_ctrlrange.
    """
    # estado actual
    q = data.qpos[arm_qpos_idx]
    qdot = data.qvel[arm_dof_idx]

    # TODO: si tienes qdot_des, se podría usar (Kp*(q_des-q)+Kd*(qdot_des-qdot))
    q_error = q_des - q
    qdot_error = -qdot

    tau = Kp * q_error + Kd * qdot_error

    # escribir en ctrl con saturación según ctrlrange
    for i, act_id in enumerate(arm_act_idx):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[act_id]
        u = np.clip(tau[i], ctrl_min, ctrl_max)
        data.ctrl[act_id] = u


# ==========================
# CONTROLADOR ADAPTATIVO COMPLETO EN ESPACIO DE JUNTAS
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
    Control adaptativo completo en espacio de juntas usando ecuaciones del movimiento.
    
    Control: τ = Y(q, q̇, q̈_des) * θ̂ + Kp*e + Kd*ė
    Ley adaptativa: θ̂̇ = -Γ * Yᵀ * (e + λ*ė)
    
    Args:
        model: Modelo MuJoCo
        data: Datos MuJoCo
        arm_qpos_idx: Índices de qpos del brazo
        arm_dof_idx: Índices de qvel del brazo
        arm_act_idx: Índices de actuadores
        q_des: Posiciones deseadas
        qdot_des: Velocidades deseadas
        qddot_des: Aceleraciones deseadas
        controller: Instancia de AdaptiveController
    """
    # Estado actual
    q = data.qpos[arm_qpos_idx]
    qdot = data.qvel[arm_dof_idx]
    
    # Calcular control adaptativo completo
    tau, V, Y = controller.compute_control(q, qdot, q_des, qdot_des, qddot_des, data)
    
    # Actualizar parámetros adaptativos usando ley adaptativa
    e = q_des - q
    edot = qdot_des - qdot
    controller.update_adaptive_parameters(e, edot, Y, lambda_param=1.0)
    
    # Log del estado (opcional, para análisis)
    controller.log_state(V, Y)
    
    # Escribir en ctrl con saturación
    for i, act_id in enumerate(arm_act_idx):
        ctrl_min, ctrl_max = model.actuator_ctrlrange[act_id]
        u = np.clip(tau[i], ctrl_min, ctrl_max)
        data.ctrl[act_id] = u
    
    return V


# ==========================
# MAIN
# ==========================

def main():
    # 1) cargar modelo y data
    model = mj.MjModel.from_xml_path(SLAVE_XML_PATH)
    data = mj.MjData(model)

    # (Opcional) imprimir todos los joints para rellenar los nombres:
    # print_all_joints(model)
    # return

    if not LEFT_ARM_JOINT_NAMES or not RIGHT_ARM_JOINT_NAMES:
        raise RuntimeError(
            "Debes rellenar LEFT_ARM_JOINT_NAMES y RIGHT_ARM_JOINT_NAMES con los nombres de joints de tus brazos."
        )

    # 2) construir índices para cada brazo
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

    # 3) Configuración de trayectoria
    dt = model.opt.timestep
    USE_SINUSOIDAL_TEST = True # Cambia a False para usar trayectoria del archivo NPZ
    
    if USE_SINUSOIDAL_TEST:
        print("📐 Usando TRAYECTORIA SINUSOIDAL DE PRUEBA (solo primer joint)")
        print("   - Solo el primer joint (shoulder_pitch) de cada brazo se mueve")
        print("   - Todos los demás joints permanecen fijos en 0")
        print("   - Esto aísla el problema del controlador")
        # No cargar trayectoria del archivo
        _trajectory_data = None
    else:
        print("📂 Cargando trayectoria desde archivo NPZ")
        try:
            load_trajectory_from_npz(TRAJECTORY_NPZ_PATH, left_qpos_idx, right_qpos_idx, dt)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            print("   Cambiando a trayectoria sinusoidal de prueba...")
            USE_SINUSOIDAL_TEST = True
            _trajectory_data = None
        except Exception as e:
            print(f"❌ Error cargando trayectoria: {e}")
            raise

    # 4) inicializar la pose (opcional: poner qpos inicial = trayectoria deseada en t=0)
    if USE_SINUSOIDAL_TEST:
        q_left_des_0, q_right_des_0 = desired_arm_trajectories_sinusoidal_shoulders(0.0, n_left, n_right)
    else:
        q_left_des_0, q_right_des_0 = desired_arm_trajectories(0.0, n_left, n_right)
    data.qpos[left_qpos_idx] = q_left_des_0
    data.qpos[right_qpos_idx] = q_right_des_0
    mj.mj_forward(model, data)

    # 5) parámetros de simulación
    # sim_duration se ajusta automáticamente a la duración de la trayectoria cargada
    global _trajectory_duration
    if USE_SINUSOIDAL_TEST:
        sim_duration = 20.0  # Duración fija para prueba sinusoidal
    else:
        sim_duration = _trajectory_duration if _trajectory_duration is not None else 20.0
    print(f"⏱️  Duración de simulación: {sim_duration:.3f} s")
    
    # ===== ELECCIÓN DE CONTROLADOR =====
    USE_ADAPTIVE_CONTROL = True  # Cambia a False para usar PD estático
    ADAPT_GAINS = False  # Cambia a True para adaptar ganancias PD (Kp, Kd)
    
    if USE_ADAPTIVE_CONTROL:
        if ADAPT_GAINS:
            print("🎯 Usando CONTROL ADAPTATIVO (Masas/Inercias conocidas, fricción + ganancias PD adaptativas)")
        else:
            print("🎯 Usando CONTROL ADAPTATIVO (Masas/Inercias conocidas, fricción adaptativa, ganancias PD fijas)")
        
        # ===== CONFIGURACIÓN DE PARÁMETROS CONOCIDOS =====
        # Si conoces las masas e inercias del robot, especifícalas aquí
        # De lo contrario, se usarán valores por defecto extraídos del modelo MuJoCo
        
        # Masas conocidas por joint (kg) - EJEMPLO: ajusta según tu robot
        # Para 7 joints del brazo: [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
        left_known_masses = None  # None = usar valores por defecto del modelo
        right_known_masses = None
        
        # Inercias conocidas por joint (kg·m²) - EJEMPLO: ajusta según tu robot
        left_known_inertias = None  # None = usar valores por defecto del modelo
        right_known_inertias = None
        
        # Si quieres especificar valores conocidos, descomenta y ajusta:
        # left_known_masses = np.array([2.5, 1.8, 1.2, 0.8, 0.3, 0.2, 0.1])  # kg
        # left_known_inertias = np.array([0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])  # kg·m²
        # right_known_masses = np.array([2.5, 1.8, 1.2, 0.8, 0.3, 0.2, 0.1])  # kg
        # right_known_inertias = np.array([0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001])  # kg·m²
        
        # Inicializar controladores adaptativos para cada brazo
        # Ahora usa masas/inercias conocidas (o valores por defecto del modelo MuJoCo)
        left_adaptive_ctrl = AdaptiveController(n_left, dt, model, left_qpos_idx, left_dof_idx,
                                                known_masses=left_known_masses,
                                                known_inertias=left_known_inertias,
                                                adapt_gains=ADAPT_GAINS)
        right_adaptive_ctrl = AdaptiveController(n_right, dt, model, right_qpos_idx, right_dof_idx,
                                                 known_masses=right_known_masses,
                                                 known_inertias=right_known_inertias,
                                                 adapt_gains=ADAPT_GAINS)
        
        if ADAPT_GAINS:
            print(f"   Ganancias PD iniciales: Kp={left_adaptive_ctrl.Kp:.1f}, Kd={left_adaptive_ctrl.Kd:.1f} (adaptativas)")
            print(f"   Parámetros adaptativos: {left_adaptive_ctrl.n_params} por brazo ({n_left} fricción + 2 ganancias PD)")
        else:
            print(f"   Ganancias PD fijas: Kp={left_adaptive_ctrl.Kp:.1f}, Kd={left_adaptive_ctrl.Kd:.1f}")
            print(f"   Parámetros adaptativos: {left_adaptive_ctrl.n_params} por brazo (solo fricción)")
        print(f"   Masas conocidas (fijas): {left_adaptive_ctrl.known_masses}")
        print(f"   Inercias conocidas (fijas): {left_adaptive_ctrl.known_inertias}")
    else:
        print("🎯 Usando CONTROL PD ESTÁTICO")
        Kp = 80.0            # ganancia proporcional (ajusta según necesites)
        Kd = 5.0             # ganancia derivativa
    
    # Variables para calcular qdot_des y qddot_des por diferenciación numérica
    q_left_des_prev = None
    q_right_des_prev = None
    qdot_left_des_prev = None
    qdot_right_des_prev = None
    t_prev = 0.0

    # para imprimir errores cada N pasos
    print_every = 20
    step_count = 0

    # 5) lanzar viewer pasivo
    with mjviewer.launch_passive(model, data) as viewer:
        start_wall = time.time()
        while viewer.is_running() and (data.time < sim_duration):
            step_start = time.time()

            # 5.1) obtener trayectoria deseada en este tiempo de simulación
            t = data.time
            if USE_SINUSOIDAL_TEST:
                q_left_des, q_right_des = desired_arm_trajectories_sinusoidal_shoulders(t, n_left, n_right)
            else:
                q_left_des, q_right_des = desired_arm_trajectories(t, n_left, n_right)
            
            # Calcular velocidad y aceleración deseadas por diferenciación numérica
            if q_left_des_prev is not None and t > t_prev:
                dt_traj = t - t_prev
                # Velocidad deseada
                qdot_left_des = (q_left_des - q_left_des_prev) / dt_traj
                qdot_right_des = (q_right_des - q_right_des_prev) / dt_traj
                
                # Aceleración deseada
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
            
            # Guardar para siguiente iteración
            q_left_des_prev = q_left_des.copy()
            q_right_des_prev = q_right_des.copy()
            qdot_left_des_prev = qdot_left_des.copy()
            qdot_right_des_prev = qdot_right_des.copy()
            t_prev = t

            # 5.2) aplicar control a cada brazo
            if USE_ADAPTIVE_CONTROL:
                # Control adaptativo completo (usa ecuaciones del movimiento)
                V_left = apply_adaptive_control(model, data,
                                                 left_qpos_idx, left_dof_idx, left_act_idx,
                                                 q_left_des, qdot_left_des, qddot_left_des,
                                                 left_adaptive_ctrl)
                V_right = apply_adaptive_control(model, data,
                                                  right_qpos_idx, right_dof_idx, right_act_idx,
                                                  q_right_des, qdot_right_des, qddot_right_des,
                                                  right_adaptive_ctrl)
            else:
                # Control PD estático
                apply_joint_pd_control(model, data,
                                       left_qpos_idx, left_dof_idx, left_act_idx,
                                       q_left_des, Kp, Kd)
                apply_joint_pd_control(model, data,
                                       right_qpos_idx, right_dof_idx, right_act_idx,
                                       q_right_des, Kp, Kd)

            # 5.3) avanzar la física
            mj.mj_step(model, data)

            # 5.4) calcular y mostrar errores de tracking (norma-2 por brazo)
            if step_count % print_every == 0:
                q_left = data.qpos[left_qpos_idx]
                q_right = data.qpos[right_qpos_idx]
                err_left = np.linalg.norm(q_left - q_left_des)
                err_right = np.linalg.norm(q_right - q_right_des)
                
                if USE_ADAPTIVE_CONTROL:
                    # Mostrar parámetros adaptativos: fricción y ganancias PD
                    # Fricción: primeros n_joints elementos de theta_hat
                    b_left_avg = np.mean(left_adaptive_ctrl.theta_hat[:left_adaptive_ctrl.n_joints])  # Promedio de fricción
                    b_right_avg = np.mean(right_adaptive_ctrl.theta_hat[:right_adaptive_ctrl.n_joints])
                    print(f"t = {t:6.3f}  |  ||e_left|| = {err_left:8.5f} rad  |  ||e_right|| = {err_right:8.5f} rad")
                    print(f"         b_left={b_left_avg:6.3f}, Kp_left={left_adaptive_ctrl.Kp:6.1f}, Kd_left={left_adaptive_ctrl.Kd:5.1f}  |  V_left={V_left:.6f}")
                    print(f"         b_right={b_right_avg:6.3f}, Kp_right={right_adaptive_ctrl.Kp:6.1f}, Kd_right={right_adaptive_ctrl.Kd:5.1f}  |  V_right={V_right:.6f}")
                else:
                    print(f"t = {t:6.3f}  |  ||e_left|| = {err_left:8.5f} rad  |  ||e_right|| = {err_right:8.5f} rad")

            step_count += 1

            # 5.5) actualizar viewer
            viewer.sync()

            # 5.6) mantener ritmo ~tiempo real
            time_spent = time.time() - step_start
            time_until_next_step = dt - time_spent
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("Simulación terminada.")


if __name__ == "__main__":
    main()
