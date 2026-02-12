#!/usr/bin/env python3
"""
Plot Figure 3C: Cross-species calibration (ECE) for all models.

Shows Expected Calibration Error (ECE) across tasks, species, and model types.

Usage:
    cd reproduction/
    python scripts/plotting/plot_figure3c_calibration.py \
        --calibration-results data/final_data/calibration_cross_species_results.json \
        --output output/figure3c_calibration.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Task display names and order (easiest to hardest, split-related then merge-related)
TASK_ORDER = [
    "merge_action",
    "endpoint_error_identification_with_em",
    "merge_error_identification",
    "split_action",
]

TASK_DISPLAY_NAMES = {
    "merge_action": "Split Error\nCorrection",
    "endpoint_error_identification_with_em": "Split Error\nIdentification",
    "merge_error_identification": "Merge Error\nIdentification",
    "split_action": "Split Action\nEvaluation",
}

# Species display names and order
SPECIES_ORDER = ["mouse", "fly", "zebrafish", "human"]

SPECIES_DISPLAY_NAMES = {
    "mouse": "Mouse",
    "fly": "Fly",
    "zebrafish": "Zebrafish",
    "human": "Human",
}

# Default path: relative to reproduction/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # reproduction/


def load_results(json_path: Path) -> dict:
    """Load calibration results JSON."""
    with open(json_path) as f:
        return json.load(f)


def plot_calibration_subplots(
    calibration_results: dict,
    output_path: Path,
):
    """
    Plot cross-species calibration with subplots per task.

    Args:
        calibration_results: Calibration results (ECE, MCE, etc.) with structure:
                            {task: {species: {model_type: {ece: value}}}}
        output_path: Where to save figure
    """
    # Set up plot style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Create 1x4 subplot grid
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Model colors (grayscale matching Figure 3A)
    model_colors = {
        'resnet': '0.3',      # Dark gray
        'linear_probe': '0.6', # Medium gray
        'vlm': 'white',        # White (placeholder)
    }

    model_labels = {
        'resnet': 'ResNet (finetuned)',
        'linear_probe': 'Linear Probe',
        'vlm': 'VLM',
    }

    for task_idx, task in enumerate(TASK_ORDER):
        ax = axes[task_idx]

        # Get results for this task
        task_data = calibration_results.get(task, {})

        # Determine which species to show for this task
        if task == "endpoint_error_identification_with_em":
            task_species = ["mouse", "zebrafish", "human"]
        else:
            task_species = SPECIES_ORDER

        # Prepare data
        n_species = len(task_species)
        bar_width = 0.25
        species_gap = 0.15

        x_positions = []
        current_x = 0

        for species_idx, species in enumerate(task_species):
            x_base = current_x
            x_positions.append(x_base)

            # Plot bars for each model
            models = ['resnet', 'linear_probe', 'vlm']

            for model_idx, model in enumerate(models):
                x_pos = x_base + model_idx * bar_width

                # Get ECE for this model
                species_data = task_data.get(species, {})

                # Try new format first: {species: {model: {ece: value}}}
                model_data = species_data.get(model, {})
                ece = model_data.get("ece")

                # Fall back to old format for ResNet: {species: {ece: value}}
                if ece is None and model == 'resnet' and 'ece' in species_data:
                    ece = species_data.get("ece")

                if ece is not None:
                    # Real data
                    ax.bar(
                        x_pos, ece * 100, bar_width,
                        color=model_colors[model],
                        edgecolor='black',
                        linewidth=0.8,
                        alpha=0.85,
                    )

                    # Add value label on top
                    if ece * 100 > 5:  # Only show if bar is tall enough
                        ax.text(
                            x_pos, ece * 100 + 2,
                            f'{ece*100:.0f}',
                            ha='center', va='bottom',
                            fontsize=7,
                        )
                else:
                    # Placeholder - empty bar with dashed outline
                    ax.bar(
                        x_pos, 100, bar_width,
                        color=model_colors[model],
                        edgecolor='0.7',
                        linestyle='--',
                        linewidth=1,
                    )
                    # Add "?" for placeholders
                    ax.text(
                        x_pos, 50, '?',
                        ha='center', va='center',
                        fontsize=12, color='0.6',
                    )

            current_x += 3 * bar_width + species_gap

        # Configure subplot
        ax.set_title(TASK_DISPLAY_NAMES[task], fontsize=16, fontweight='bold', pad=10)
        ax.set_ylabel('Expected Calibration Error (%)', fontsize=14)
        ax.set_ylim(0, 105)

        # Set x-ticks at center of each species group
        species_centers = [x + 1.5 * bar_width for x in x_positions]
        ax.set_xticks(species_centers)
        ax.set_xticklabels(
            [SPECIES_DISPLAY_NAMES[s] for s in task_species],
            fontsize=12,
            rotation=0,
        )

        # Grid and reference lines
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.axhline(y=10, color='green', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.axhline(y=20, color='orange', linestyle='--', linewidth=0.8, alpha=0.4)

    # Create shared legend outside subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=model_colors['resnet'], edgecolor='black',
              linewidth=0.8, label=model_labels['resnet'], alpha=0.85),
        Patch(facecolor=model_colors['linear_probe'], edgecolor='0.7',
              linestyle='--', linewidth=1, label=model_labels['linear_probe']),
        Patch(facecolor=model_colors['vlm'], edgecolor='black',
              linewidth=0.8, label=model_labels['vlm'], alpha=0.85),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.04),
               ncol=3, framealpha=0.95, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)  # Make room for legend

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved to {output_path}")
    print(f"Saved to {output_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 3C - Cross-species calibration")
    parser.add_argument("--calibration-results", type=Path,
                       default=REPO_DIR / "data" / "final_data" / "calibration_cross_species_results.json",
                       help="Calibration results JSON")
    parser.add_argument("--output", type=Path,
                       default=REPO_DIR / "output" / "figure3c_calibration.png",
                       help="Output path for figure")

    args = parser.parse_args()

    calibration_results = load_results(args.calibration_results)

    plot_calibration_subplots(
        calibration_results,
        args.output,
    )

    # Print summary
    print("\nCross-Species Calibration Summary (ECE - ResNet Finetuned):")
    print("="*80)
    for task in TASK_ORDER:
        print(f"\n{TASK_DISPLAY_NAMES[task].replace(chr(10), ' ')}:")

        task_data = calibration_results.get(task, {})

        # Determine which species to show for this task
        if task == "endpoint_error_identification_with_em":
            task_species = ["mouse", "zebrafish", "human"]
        else:
            task_species = SPECIES_ORDER

        for species in task_species:
            species_data = task_data.get(species, {})

            # Handle both old format {ece: value} and new format {resnet: {ece: value}}
            if isinstance(species_data, dict) and 'ece' in species_data:
                ece = species_data.get("ece")
            else:
                ece = species_data.get("resnet", {}).get("ece")

            if ece is not None:
                ece_str = f"{ece*100:5.1f}%"
            else:
                ece_str = "  N/A"

            print(f"  {SPECIES_DISPLAY_NAMES[species]:12s}: ECE={ece_str}")


if __name__ == "__main__":
    main()
