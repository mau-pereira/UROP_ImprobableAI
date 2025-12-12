"""
Análisis paramétrico rápido para control adaptativo.
Ejecuta simulaciones sin visualización y guarda resultados.
"""

import time
import numpy as np
import mujoco as mj
from pathlib import Path
import json
from datetime import datetime
import itertools

# Importar del follower.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from follower import (
    AdaptiveController, build_arm_maps, load_trajectory_from_npz,
    desired_arm_trajectories, desired_arm_trajectories_sinusoidal_shoulders,
    apply_adaptive_control, apply_joint_pd_control,
    LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, SLAVE_XML_PATH, TRAJECTORY_NPZ_PATH
)


def run_simulation(config: dict, save_data: bool = True):
    """
    Ejecuta una simulación con configuración dada (sin visualización).
    
    Args:
        config: Diccionario con parámetros:
            - gamma_friction: float
            - gamma_gains: float (si adapt_gains=True)
            - alpha_qddot: float
            - alpha_theta: float
            - adapt_gains: bool
            - trajectory_type: 'sinusoidal' o 'teleop'
            - Kp_init: float
            - Kd_init: float
            - lambda_param: float
        save_data: Si True, guarda datos completos (más lento)
    
    Returns:
        dict con métricas y datos
    """
    # Cargar modelo
    model = mj.MjModel.from_xml_path(SLAVE_XML_PATH)
    data = mj.MjData(model)
    dt = model.opt.timestep
    
    # Build arm maps
    left_qpos_idx, left_dof_idx, left_act_idx = build_arm_maps(model, LEFT_ARM_JOINT_NAMES)
    right_qpos_idx, right_dof_idx, right_act_idx = build_arm_maps(model, RIGHT_ARM_JOINT_NAMES)
    n_left = len(left_qpos_idx)
    n_right = len(right_qpos_idx)
    
    # Configurar trayectoria
    USE_SINUSOIDAL = (config['trajectory_type'] == 'sinusoidal')
    if USE_SINUSOIDAL:
        sim_duration = 20.0
        _trajectory_data = None
    else:
        try:
            load_trajectory_from_npz(TRAJECTORY_NPZ_PATH, left_qpos_idx, right_qpos_idx, dt)
            from follower import _trajectory_duration
            sim_duration = _trajectory_duration if _trajectory_duration is not None else 20.0
        except:
            USE_SINUSOIDAL = True
            sim_duration = 20.0
    
    # Inicializar pose
    if USE_SINUSOIDAL:
        q_left_des_0, q_right_des_0 = desired_arm_trajectories_sinusoidal_shoulders(0.0, n_left, n_right)
    else:
        q_left_des_0, q_right_des_0 = desired_arm_trajectories(0.0, n_left, n_right)
    data.qpos[left_qpos_idx] = q_left_des_0
    data.qpos[right_qpos_idx] = q_right_des_0
    mj.mj_forward(model, data)
    
    # Inicializar controladores adaptativos
    left_ctrl = AdaptiveController(n_left, dt, model, left_qpos_idx, left_dof_idx,
                                   adapt_gains=config.get('adapt_gains', False))
    right_ctrl = AdaptiveController(n_right, dt, model, right_qpos_idx, right_dof_idx,
                                    adapt_gains=config.get('adapt_gains', False))
    
    # Configurar parámetros del controlador
    # Gamma
    gamma_friction = config.get('gamma_friction', 0.5)
    gamma_gains = config.get('gamma_gains', 0.1)
    left_ctrl.Gamma[:n_left, :n_left] = np.eye(n_left) * gamma_friction
    right_ctrl.Gamma[:n_right, :n_right] = np.eye(n_right) * gamma_friction
    if config.get('adapt_gains', False):
        left_ctrl.Gamma[-2, -2] = gamma_gains
        left_ctrl.Gamma[-1, -1] = gamma_gains
        right_ctrl.Gamma[-2, -2] = gamma_gains
        right_ctrl.Gamma[-1, -1] = gamma_gains
    
    # Filtros
    left_ctrl.qddot_filter_alpha = config.get('alpha_qddot', 0.8)
    right_ctrl.qddot_filter_alpha = config.get('alpha_qddot', 0.8)
    left_ctrl.theta_smooth_alpha = config.get('alpha_theta', 0.9)
    right_ctrl.theta_smooth_alpha = config.get('alpha_theta', 0.9)
    
    # Kp/Kd iniciales
    if not config.get('adapt_gains', False):
        left_ctrl._Kp = config.get('Kp_init', 150.0)
        left_ctrl._Kd = config.get('Kd_init', 10.0)
        right_ctrl._Kp = config.get('Kp_init', 150.0)
        right_ctrl._Kd = config.get('Kd_init', 10.0)
    else:
        left_ctrl.theta_hat[-2] = config.get('Kp_init', 150.0)
        left_ctrl.theta_hat[-1] = config.get('Kd_init', 10.0)
        right_ctrl.theta_hat[-2] = config.get('Kp_init', 150.0)
        right_ctrl.theta_hat[-1] = config.get('Kd_init', 10.0)
    
    # Variables para diferenciación numérica
    q_left_des_prev = None
    q_right_des_prev = None
    qdot_left_des_prev = None
    qdot_right_des_prev = None
    t_prev = 0.0
    
    # Datos para guardar
    lambda_param = config.get('lambda_param', 1.0)
    errors_left = []
    errors_right = []
    errors_vel_left = []
    errors_vel_right = []
    lyapunov_left = []
    lyapunov_right = []
    friction_left = []
    friction_right = []
    Kp_left = []
    Kp_right = []
    Kd_left = []
    Kd_right = []
    times = []
    
    # Simulación (sin visualización)
    step_count = 0
    while data.time < sim_duration:
        t = data.time
        
        # Obtener trayectoria deseada
        if USE_SINUSOIDAL:
            q_left_des, q_right_des = desired_arm_trajectories_sinusoidal_shoulders(t, n_left, n_right)
        else:
            q_left_des, q_right_des = desired_arm_trajectories(t, n_left, n_right)
        
        # Calcular velocidades y aceleraciones deseadas
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
        
        q_left_des_prev = q_left_des.copy()
        q_right_des_prev = q_right_des.copy()
        qdot_left_des_prev = qdot_left_des.copy()
        qdot_right_des_prev = qdot_right_des.copy()
        t_prev = t
        
        # Aplicar control (esto ya actualiza parámetros internamente)
        V_left = apply_adaptive_control(model, data,
                                        left_qpos_idx, left_dof_idx, left_act_idx,
                                        q_left_des, qdot_left_des, qddot_left_des,
                                        left_ctrl)
        V_right = apply_adaptive_control(model, data,
                                         right_qpos_idx, right_dof_idx, right_act_idx,
                                         q_right_des, qdot_right_des, qddot_right_des,
                                         right_ctrl)
        
        # Calcular errores para logging
        e_left = q_left_des - data.qpos[left_qpos_idx]
        edot_left = qdot_left_des - data.qvel[left_dof_idx]
        e_right = q_right_des - data.qpos[right_qpos_idx]
        edot_right = qdot_right_des - data.qvel[right_dof_idx]
        
        # Guardar datos (cada N pasos para ahorrar memoria)
        if save_data and step_count % 10 == 0:
            errors_left.append(np.linalg.norm(e_left))
            errors_right.append(np.linalg.norm(e_right))
            errors_vel_left.append(np.linalg.norm(edot_left))
            errors_vel_right.append(np.linalg.norm(edot_right))
            lyapunov_left.append(V_left)
            lyapunov_right.append(V_right)
            friction_left.append(np.mean(left_ctrl.theta_hat[:n_left]))
            friction_right.append(np.mean(right_ctrl.theta_hat[:n_right]))
            Kp_left.append(left_ctrl.Kp)
            Kp_right.append(right_ctrl.Kp)
            Kd_left.append(left_ctrl.Kd)
            Kd_right.append(right_ctrl.Kd)
            times.append(t)
        
        # Avanzar simulación
        mj.mj_step(model, data)
        step_count += 1
    
    # Calcular métricas finales
    if save_data and len(errors_left) > 0:
        errors_left = np.array(errors_left)
        errors_right = np.array(errors_right)
        
        results = {
            'config': config,
            'metrics': {
                'rmse_left': float(np.sqrt(np.mean(errors_left**2))),
                'rmse_right': float(np.sqrt(np.mean(errors_right**2))),
                'max_error_left': float(np.max(errors_left)),
                'max_error_right': float(np.max(errors_right)),
                'mean_error_left': float(np.mean(errors_left)),
                'mean_error_right': float(np.mean(errors_right)),
                'final_error_left': float(errors_left[-1]),
                'final_error_right': float(errors_right[-1]),
                'std_error_left': float(np.std(errors_left)),
                'std_error_right': float(np.std(errors_right)),
            },
            'data': {
                'times': times,
                'errors_left': errors_left.tolist(),
                'errors_right': errors_right.tolist(),
                'errors_vel_left': errors_vel_left,
                'errors_vel_right': errors_vel_right,
                'lyapunov_left': lyapunov_left,
                'lyapunov_right': lyapunov_right,
                'friction_left': friction_left,
                'friction_right': friction_right,
                'Kp_left': Kp_left,
                'Kp_right': Kp_right,
                'Kd_left': Kd_left,
                'Kd_right': Kd_right,
            }
        }
    else:
        # Solo métricas básicas (más rápido)
        results = {
            'config': config,
            'metrics': {
                'rmse_left': 0.0,  # Placeholder
                'rmse_right': 0.0,
            }
        }
    
    return results


def run_parametric_sweep(output_dir: Path = None, quick_mode: bool = True):
    """
    Ejecuta barrido paramétrico completo.
    
    Args:
        output_dir: Directorio donde guardar resultados
        quick_mode: Si True, solo guarda métricas (más rápido)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "parametric_results"
    output_dir.mkdir(exist_ok=True)
    
    # Definir rangos de parámetros
    gamma_friction_values = [0.1, 0.5, 1.0, 2.0]
    alpha_qddot_values = [0.7, 0.8, 0.9]
    alpha_theta_values = [0.85, 0.9, 0.95]
    trajectory_types = ['sinusoidal', 'teleop']
    
    # Generar todas las combinaciones
    configs = []
    for gamma, alpha_q, alpha_t, traj_type in itertools.product(
        gamma_friction_values, alpha_qddot_values, alpha_theta_values, trajectory_types
    ):
        configs.append({
            'gamma_friction': gamma,
            'alpha_qddot': alpha_q,
            'alpha_theta': alpha_t,
            'trajectory_type': traj_type,
            'adapt_gains': False,  # Fijo para simplificar
            'Kp_init': 150.0,
            'Kd_init': 10.0,
            'lambda_param': 1.0,
        })
    
    print(f"🚀 Ejecutando {len(configs)} configuraciones...")
    print(f"   Modo rápido: {'Sí' if quick_mode else 'No'}")
    
    all_results = []
    start_time = time.time()
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Gamma={config['gamma_friction']:.1f}, "
              f"α_qddot={config['alpha_qddot']:.2f}, α_θ={config['alpha_theta']:.2f}, "
              f"Trayectoria={config['trajectory_type']}")
        
        try:
            result = run_simulation(config, save_data=not quick_mode)
            all_results.append(result)
            
            if not quick_mode:
                print(f"   RMSE left: {result['metrics']['rmse_left']:.6f}, "
                      f"right: {result['metrics']['rmse_right']:.6f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completado en {elapsed:.1f}s")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"parametric_sweep_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_configs': len(configs),
            'successful_runs': len(all_results),
            'elapsed_time_seconds': elapsed,
            'results': all_results
        }, f, indent=2)
    
    print(f"💾 Resultados guardados en: {output_file}")
    
    return all_results, output_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Análisis paramétrico de control adaptativo")
    parser.add_argument("--quick", action="store_true", help="Modo rápido (solo métricas)")
    parser.add_argument("--output", type=str, help="Directorio de salida")
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    run_parametric_sweep(output_dir, quick_mode=args.quick)
