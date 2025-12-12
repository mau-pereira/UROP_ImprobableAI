#!/bin/bash
# Script rápido para ejecutar análisis completo

echo "🚀 Iniciando análisis paramétrico..."
echo ""

# Paso 1: Ejecutar análisis
echo "📊 Paso 1/2: Ejecutando simulaciones..."
python parametric_analysis.py --quick

# Encontrar el archivo JSON más reciente
LATEST_JSON=$(ls -t parametric_results/parametric_sweep_*.json 2>/dev/null | head -1)

if [ -z "$LATEST_JSON" ]; then
    echo "❌ No se encontró archivo de resultados"
    exit 1
fi

echo ""
echo "✅ Análisis completado: $LATEST_JSON"
echo ""

# Paso 2: Generar gráficas
echo "📈 Paso 2/2: Generando gráficas..."
python generate_plots.py "$LATEST_JSON"

echo ""
echo "✅ ¡Completado! Revisa:"
echo "   - Resultados: $LATEST_JSON"
echo "   - Gráficas: parametric_results/plots/"
echo "   - Template: REPORT_TEMPLATE.md"
