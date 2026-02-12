#!/bin/bash
# Generate all paper figures from pre-computed evaluation data.
#
# Usage:
#   cd reproduction/
#   pip install -r requirements.txt
#   bash generate_all_figures.sh
#
# Output files will be written to output/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Generating all paper figures"
echo "============================================"
echo ""

# Figure 2: Scaling curves (2x2 grid, 4 tasks)
echo "[1/5] Figure 2: Scaling curves..."
python scripts/plotting/plot_figure2_scaling_curves.py \
    --linear-probe-dir data/linear_probe_sweep \
    --resnet-results data/final_data/resnet_results.json \
    --output output/figure2_scaling_curves.png
echo ""

# Figure 3A: Cross-species transfer (1x4 grouped bars)
echo "[2/5] Figure 3A: Cross-species transfer..."
python scripts/plotting/plot_figure3a_subplots.py \
    --resnet-results data/final_data/resnet_cross_species_results.json \
    --linear-probe-results data/final_data/linear_probe_cross_species_results.json \
    --output output/figure3a_cross_species.png
echo ""

# Figure 3B: Calibration ECE vs data scale
echo "[3/5] Figure 3B: ECE vs data scale..."
python scripts/plotting/plot_figure3b_ece_vs_data.py
echo ""

# Figure 3C: Cross-species calibration ECE
echo "[4/5] Figure 3C: Cross-species calibration..."
python scripts/plotting/plot_figure3c_calibration.py \
    --calibration-results data/final_data/calibration_cross_species_results.json \
    --output output/figure3c_calibration.png
echo ""

# Table 1: Benchmark table (LaTeX)
echo "[5/5] Table 1: Benchmark table..."
python scripts/plotting/generate_benchmark_table.py
echo ""

echo "============================================"
echo "All figures generated! Check output/ for:"
echo "  - figure2_scaling_curves.png/.pdf"
echo "  - figure3a_cross_species.png/.pdf"
echo "  - figure3b_ece_vs_data.png/.pdf"
echo "  - figure3c_calibration.png/.pdf"
echo "  - benchmark_table.tex"
echo "============================================"
