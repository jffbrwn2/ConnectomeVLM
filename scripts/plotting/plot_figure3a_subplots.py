#!/usr/bin/env python3
"""
Plot Figure 3A: Cross-species transfer with subplots per task.

Each subplot shows one task with species grouped on x-axis and
3 bars per species: ResNet, Linear Probe, VLM (placeholder).

Usage:
    cd reproduction/
    python scripts/plotting/plot_figure3a_subplots.py \
        --resnet-results data/final_data/resnet_cross_species_results.json \
        --linear-probe-results data/final_data/linear_probe_cross_species_results.json \
        --output output/figure3a_cross_species.png
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
    """Load results JSON."""
    with open(json_path) as f:
        return json.load(f)

def plot_cross_species_subplots(
    resnet_results: dict,
    linear_probe_results: dict,
    output_path: Path,
):
    """
    Plot cross-species transfer with subplots per task.

    Args:
        resnet_results: ResNet cross-species results
        linear_probe_results: Linear probe cross-species results
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

    # Model colors (grayscale)
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
        resnet_task_data = resnet_results.get(task, {})
        lp_task_data = linear_probe_results.get(task, {})

        # Determine which species to show for this task
        # Exclude fly for endpoint_error_identification
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

            # Get accuracies
            resnet_species_data = resnet_task_data.get(species, {})
            resnet_acc = resnet_species_data.get("finetuned", {}).get("balanced_accuracy")

            lp_acc = lp_task_data.get(species, {}).get("balanced_accuracy")

            vlm_data = resnet_species_data.get("vlm", {})
            vlm_maj_acc = vlm_data.get("majority_accuracy")
            vlm_ind_acc = vlm_data.get("individual_accuracy")

            # Plot bars for each model
            models = ['resnet', 'linear_probe', 'vlm']
            accs = [resnet_acc, lp_acc, vlm_ind_acc]  # Use individual (Pass@1) as primary

            for model_idx, (model, acc) in enumerate(zip(models, accs)):
                x_pos = x_base + model_idx * bar_width

                if acc is None:
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
                else:
                    # For VLM, draw majority voting first (behind) with hatching
                    if model == 'vlm' and vlm_maj_acc is not None:
                        ax.bar(
                            x_pos, vlm_maj_acc * 100, bar_width,
                            color='none',
                            edgecolor='black',
                            linewidth=1.2,
                            hatch='///',
                            alpha=1.0,
                            zorder=2,
                        )

                    # Draw main bar (solid for non-VLM, or individual/Pass@1 for VLM)
                    # Use full opacity for VLM to hide hatching underneath
                    bar_alpha = 1.0 if model == 'vlm' else 0.85
                    ax.bar(
                        x_pos, acc * 100, bar_width,
                        color=model_colors[model],
                        edgecolor='black',
                        linewidth=0.8,
                        alpha=bar_alpha,
                        zorder=3,
                    )

                    # Add value label on top - use max of individual or majority for VLM
                    if model == 'vlm' and vlm_maj_acc is not None:
                        label_height = max(acc * 100, vlm_maj_acc * 100)
                    else:
                        label_height = acc * 100

                    if label_height > 10:  # Only show if bar is tall enough
                        ax.text(
                            x_pos, label_height + 2,
                            f'{label_height:.1f}',
                            ha='center', va='bottom',
                            fontsize=7,
                        )

            current_x += 3 * bar_width + species_gap

        # Configure subplot
        ax.set_title(TASK_DISPLAY_NAMES[task], fontsize=16, fontweight='bold', pad=10)
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=14)
        ax.set_ylim(0, 105)

        # Set x-ticks at center of each species group
        species_centers = [x + 1.5 * bar_width for x in x_positions]
        ax.set_xticks(species_centers)
        ax.set_xticklabels(
            [SPECIES_DISPLAY_NAMES[s] for s in task_species],
            fontsize=12,
            rotation=0,
        )

        # Grid and reference line
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=0.8, alpha=0.4, zorder=10)

    # Create shared legend outside subplots
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=model_colors['resnet'], edgecolor='black',
              linewidth=0.8, label=model_labels['resnet'], alpha=0.85),
        Patch(facecolor=model_colors['linear_probe'], edgecolor='black',
              linewidth=0.8, label=model_labels['linear_probe'], alpha=0.85),
        Patch(facecolor=model_colors['vlm'], edgecolor='black',
              linewidth=0.8, label='VLM (Pass@1)', alpha=0.85),
        Patch(facecolor='none', edgecolor='black',
              linewidth=1.2, hatch='///', label='VLM (Majority Voting)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.04),
               ncol=4, framealpha=0.95, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)  # Make room for legend

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved to {output_path}")
    print(f"Saved to {output_path.with_suffix('.pdf')}")

def main():
    parser = argparse.ArgumentParser(description="Plot Figure 3A with subplots")
    parser.add_argument("--resnet-results", type=Path,
                       default=REPO_DIR / "data" / "final_data" / "resnet_cross_species_results.json",
                       help="ResNet cross-species results JSON")
    parser.add_argument("--linear-probe-results", type=Path,
                       default=REPO_DIR / "data" / "final_data" / "linear_probe_cross_species_results.json",
                       help="Linear probe cross-species results JSON")
    parser.add_argument("--output", type=Path,
                       default=REPO_DIR / "output" / "figure3a_cross_species.png",
                       help="Output path for figure")

    args = parser.parse_args()

    resnet_results = load_results(args.resnet_results)
    lp_results = load_results(args.linear_probe_results)

    plot_cross_species_subplots(
        resnet_results,
        lp_results,
        args.output,
    )

    # Print summary
    print("\nCross-Species Transfer Summary:")
    print("="*80)
    for task in TASK_ORDER:
        print(f"\n{TASK_DISPLAY_NAMES[task].replace(chr(10), ' ')}:")

        resnet_task_data = resnet_results.get(task, {})
        lp_task_data = lp_results.get(task, {})

        # Use task-specific species list
        if task == "split_error_identification":
            task_species = ["mouse", "zebrafish", "human"]
        else:
            task_species = SPECIES_ORDER

        for species in task_species:
            resnet_species_data = resnet_task_data.get(species, {})
            resnet_acc = resnet_species_data.get("finetuned", {}).get("balanced_accuracy")
            lp_acc = lp_task_data.get(species, {}).get("balanced_accuracy")

            resnet_str = f"{resnet_acc*100:5.1f}%" if resnet_acc is not None else "  N/A"
            lp_str = f"{lp_acc*100:5.1f}%" if lp_acc is not None else "  N/A"

            print(f"  {SPECIES_DISPLAY_NAMES[species]:12s}: ResNet={resnet_str}  LP={lp_str}")

if __name__ == "__main__":
    main()
