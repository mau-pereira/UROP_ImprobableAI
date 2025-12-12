"""
Genera gráficas clave del análisis paramétrico.
Ejecuta después de parametric_analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from collections import defaultdict

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(json_file: Path):
    """Carga resultados del análisis paramétrico."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def plot_error_vs_gamma(results_data, output_dir: Path):
    """Gráfica: Error RMSE vs Gamma para diferentes trayectorias."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Separar por tipo de trayectoria
    sinusoidal = defaultdict(list)
    teleop = defaultdict(list)
    
    for result in results_data['results']:
        config = result['config']
        metrics = result['metrics']
        gamma = config['gamma_friction']
        traj_type = config['trajectory_type']
        
        if traj_type == 'sinusoidal':
            sinusoidal['gamma'].append(gamma)
            sinusoidal['rmse_left'].append(metrics['rmse_left'])
            sinusoidal['rmse_right'].append(metrics['rmse_right'])
        else:
            teleop['gamma'].append(gamma)
            teleop['rmse_left'].append(metrics['rmse_left'])
            teleop['rmse_right'].append(metrics['rmse_right'])
    
    # Plot sinusoidal
    ax = axes[0]
    if sinusoidal['gamma']:
        ax.scatter(sinusoidal['gamma'], sinusoidal['rmse_left'], 
                  label='Brazo Izquierdo', alpha=0.6, s=50)
        ax.scatter(sinusoidal['gamma'], sinusoidal['rmse_right'], 
                  label='Brazo Derecho', alpha=0.6, s=50, marker='^')
        ax.set_xlabel('Gamma (fricción)')
        ax.set_ylabel('RMSE (rad)')
        ax.set_title('Trayectoria Sinusoidal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    # Plot teleop
    ax = axes[1]
    if teleop['gamma']:
        ax.scatter(teleop['gamma'], teleop['rmse_left'], 
                  label='Brazo Izquierdo', alpha=0.6, s=50)
        ax.scatter(teleop['gamma'], teleop['rmse_right'], 
                  label='Brazo Derecho', alpha=0.6, s=50, marker='^')
        ax.set_xlabel('Gamma (fricción)')
        ax.set_ylabel('RMSE (rad)')
        ax.set_title('Trayectoria Teleoperada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_gamma.png', dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: error_vs_gamma.png")
    plt.close()


def plot_filter_impact(results_data, output_dir: Path):
    """Gráfica: Impacto de filtros (alpha_qddot, alpha_theta) en error."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Agrupar por tipo de trayectoria
    for traj_idx, traj_type in enumerate(['sinusoidal', 'teleop']):
        # Filtrar resultados
        traj_results = [r for r in results_data['results'] 
                       if r['config']['trajectory_type'] == traj_type]
        
        if not traj_results:
            continue
        
        # Alpha qddot vs error
        ax = axes[0, traj_idx]
        alpha_q = [r['config']['alpha_qddot'] for r in traj_results]
        rmse = [(r['metrics']['rmse_left'] + r['metrics']['rmse_right'])/2 
                for r in traj_results]
        ax.scatter(alpha_q, rmse, alpha=0.6, s=50)
        ax.set_xlabel('Alpha (filtro qddot)')
        ax.set_ylabel('RMSE promedio (rad)')
        ax.set_title(f'Impacto α_qddot - {traj_type.capitalize()}')
        ax.grid(True, alpha=0.3)
        
        # Alpha theta vs error
        ax = axes[1, traj_idx]
        alpha_t = [r['config']['alpha_theta'] for r in traj_results]
        rmse = [(r['metrics']['rmse_left'] + r['metrics']['rmse_right'])/2 
                for r in traj_results]
        ax.scatter(alpha_t, rmse, alpha=0.6, s=50)
        ax.set_xlabel('Alpha (filtro parámetros)')
        ax.set_ylabel('RMSE promedio (rad)')
        ax.set_title(f'Impacto α_θ - {traj_type.capitalize()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'filter_impact.png', dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: filter_impact.png")
    plt.close()


def plot_heatmap_gamma_alpha(results_data, output_dir: Path):
    """Heatmap: Error vs Gamma y Alpha (promedio de ambos filtros)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for traj_idx, traj_type in enumerate(['sinusoidal', 'teleop']):
        traj_results = [r for r in results_data['results'] 
                       if r['config']['trajectory_type'] == traj_type]
        
        if not traj_results:
            continue
        
        # Crear matriz para heatmap
        gamma_values = sorted(set(r['config']['gamma_friction'] for r in traj_results))
        alpha_values = sorted(set((r['config']['alpha_qddot'] + r['config']['alpha_theta'])/2 
                                  for r in traj_results))
        
        # Agrupar por gamma y alpha promedio
        error_matrix = defaultdict(lambda: defaultdict(list))
        for r in traj_results:
            gamma = r['config']['gamma_friction']
            alpha_avg = (r['config']['alpha_qddot'] + r['config']['alpha_theta'])/2
            rmse_avg = (r['metrics']['rmse_left'] + r['metrics']['rmse_right'])/2
            error_matrix[gamma][alpha_avg].append(rmse_avg)
        
        # Promediar múltiples valores
        matrix = np.zeros((len(gamma_values), len(alpha_values)))
        for i, gamma in enumerate(gamma_values):
            for j, alpha in enumerate(alpha_values):
                if gamma in error_matrix and alpha in error_matrix[gamma]:
                    matrix[i, j] = np.mean(error_matrix[gamma][alpha])
                else:
                    matrix[i, j] = np.nan
        
        # Plot heatmap
        ax = axes[traj_idx]
        im = ax.imshow(matrix, aspect='auto', cmap='viridis_r', origin='lower')
        ax.set_xticks(range(len(alpha_values)))
        ax.set_xticklabels([f'{a:.2f}' for a in alpha_values])
        ax.set_yticks(range(len(gamma_values)))
        ax.set_yticklabels([f'{g:.1f}' for g in gamma_values])
        ax.set_xlabel('Alpha promedio (filtros)')
        ax.set_ylabel('Gamma (fricción)')
        ax.set_title(f'Error RMSE - {traj_type.capitalize()}')
        plt.colorbar(im, ax=ax, label='RMSE (rad)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_gamma_alpha.png', dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: heatmap_gamma_alpha.png")
    plt.close()


def plot_trajectory_comparison(results_data, output_dir: Path):
    """Compara rendimiento entre trayectorias simple vs compleja."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Agrupar por tipo
    sinusoidal_metrics = {'rmse_left': [], 'rmse_right': [], 'max_error_left': [], 'max_error_right': []}
    teleop_metrics = {'rmse_left': [], 'rmse_right': [], 'max_error_left': [], 'max_error_right': []}
    
    for result in results_data['results']:
        config = result['config']
        metrics = result['metrics']
        
        if config['trajectory_type'] == 'sinusoidal':
            for key in sinusoidal_metrics:
                sinusoidal_metrics[key].append(metrics[key])
        else:
            for key in teleop_metrics:
                teleop_metrics[key].append(metrics[key])
    
    # Box plots comparativos
    metrics_to_plot = [
        ('rmse_left', 'RMSE Izquierdo (rad)'),
        ('rmse_right', 'RMSE Derecho (rad)'),
        ('max_error_left', 'Error Máximo Izquierdo (rad)'),
        ('max_error_right', 'Error Máximo Derecho (rad)'),
    ]
    
    for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        data_to_plot = [
            sinusoidal_metrics[metric_key] if sinusoidal_metrics[metric_key] else [0],
            teleop_metrics[metric_key] if teleop_metrics[metric_key] else [0]
        ]
        
        bp = ax.boxplot(data_to_plot, labels=['Sinusoidal', 'Teleoperada'], 
                       patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric_label)
        ax.set_title(f'Comparación: {metric_label}')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: trajectory_comparison.png")
    plt.close()


def plot_best_configurations(results_data, output_dir: Path):
    """Identifica y muestra las mejores configuraciones."""
    # Ordenar por RMSE promedio
    sorted_results = sorted(
        results_data['results'],
        key=lambda r: (r['metrics']['rmse_left'] + r['metrics']['rmse_right'])/2
    )
    
    # Top 10
    top_10 = sorted_results[:10]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    configs_str = []
    rmse_values = []
    for i, r in enumerate(top_10):
        c = r['config']
        config_str = (f"γ={c['gamma_friction']:.1f}, "
                     f"α_q={c['alpha_qddot']:.2f}, α_θ={c['alpha_theta']:.2f}, "
                     f"{c['trajectory_type'][:4]}")
        configs_str.append(config_str)
        rmse_values.append((r['metrics']['rmse_left'] + r['metrics']['rmse_right'])/2)
    
    y_pos = np.arange(len(configs_str))
    bars = ax.barh(y_pos, rmse_values, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs_str, fontsize=8)
    ax.set_xlabel('RMSE Promedio (rad)')
    ax.set_title('Top 10 Configuraciones (Menor Error)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_configurations.png', dpi=150, bbox_inches='tight')
    print(f"✅ Guardado: best_configurations.png")
    plt.close()


def generate_all_plots(json_file: Path, output_dir: Path = None):
    """Genera todas las gráficas."""
    if output_dir is None:
        output_dir = json_file.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Cargando resultados de: {json_file}")
    results_data = load_results(json_file)
    
    print(f"📈 Generando gráficas en: {output_dir}")
    
    plot_error_vs_gamma(results_data, output_dir)
    plot_filter_impact(results_data, output_dir)
    plot_heatmap_gamma_alpha(results_data, output_dir)
    plot_trajectory_comparison(results_data, output_dir)
    plot_best_configurations(results_data, output_dir)
    
    print(f"\n✅ Todas las gráficas generadas en: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genera gráficas del análisis paramétrico")
    parser.add_argument("json_file", type=str, help="Archivo JSON con resultados")
    parser.add_argument("--output", type=str, help="Directorio de salida para gráficas")
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ Archivo no encontrado: {json_path}")
        exit(1)
    
    output_dir = Path(args.output) if args.output else None
    generate_all_plots(json_path, output_dir)
