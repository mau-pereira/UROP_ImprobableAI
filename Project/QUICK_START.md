# Guía Rápida: Análisis Paramétrico (2 horas)

## Paso 1: Ejecutar Análisis Paramétrico (30-45 min)

```bash
cd Project
python parametric_analysis.py --quick
```

**Modo rápido (`--quick`):**
- Solo guarda métricas (no datos completos)
- Más rápido, suficiente para gráficas comparativas
- ~30-45 minutos para ~72 configuraciones

**Modo completo (sin `--quick`):**
- Guarda todos los datos (errores, Lyapunov, parámetros)
- Más lento pero permite gráficas detalladas
- ~1-2 horas

**Salida:** `parametric_results/parametric_sweep_[TIMESTAMP].json`

## Paso 2: Generar Gráficas (2-5 min)

```bash
python generate_plots.py parametric_results/parametric_sweep_[TIMESTAMP].json
```

**Gráficas generadas:**
- `error_vs_gamma.png` - Error vs tasa de adaptación
- `filter_impact.png` - Impacto de filtros
- `heatmap_gamma_alpha.png` - Mapa de calor combinado
- `trajectory_comparison.png` - Comparación simple vs compleja
- `best_configurations.png` - Top 10 configuraciones

**Salida:** `parametric_results/plots/`

## Paso 3: Llenar Reporte (60-90 min)

1. Abrir `REPORT_TEMPLATE.md`
2. Revisar gráficas generadas
3. Llenar secciones con resultados observados
4. Agregar conclusiones basadas en datos

## Opciones Avanzadas

### Ejecutar solo configuraciones específicas

Editar `parametric_analysis.py` línea ~200 para cambiar rangos:

```python
gamma_friction_values = [0.5, 1.0]  # Reducir para más rápido
alpha_qddot_values = [0.8]  # Solo un valor
alpha_theta_values = [0.9]  # Solo un valor
```

### Análisis más rápido: menos configuraciones

Reducir combinaciones en `run_parametric_sweep()` para ejecutar más rápido.

### Ver resultados mientras corre

Los resultados se guardan en JSON. Puedes generar gráficas parciales editando el script para guardar después de cada configuración.

## Troubleshooting

**Error: "Trajectory file not found"**
- Verificar que `trajectories/traj_20251211_200112.npz` existe
- O cambiar a modo solo sinusoidal editando el código

**Muy lento:**
- Usar `--quick`
- Reducir número de configuraciones
- Reducir duración de simulación (editar `sim_duration`)

**Falta matplotlib/seaborn:**
```bash
pip install matplotlib seaborn
```

## Estructura de Tiempo Sugerida

- **0:00-0:45:** Ejecutar análisis paramétrico
- **0:45-0:50:** Generar gráficas
- **0:50-1:30:** Analizar resultados y llenar reporte
- **1:30-2:00:** Revisar, agregar conclusiones, formatear
