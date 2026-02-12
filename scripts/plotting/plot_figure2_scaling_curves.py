#!/usr/bin/env python3
"""
Plot scaling curves for all tasks (Figure 2).

Shows how performance scales with training data size, comparing:
- Linear Probe (scaling curve)
- ResNet-50 Frozen (point)
- ResNet-50 Fine-tuned (point)

Usage:
    cd reproduction/
    python scripts/plotting/plot_figure2_scaling_curves.py \
        --linear-probe-dir data/linear_probe_sweep \
        --resnet-results data/final_data/resnet_results.json \
        --output output/figure2_scaling_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Task order and display names (easiest to hardest, split-related then merge-related)
TASK_ORDER = [
    "merge_action",
    "endpoint_error_identification_with_em",
    "merge_error_identification",
    "split_action",
]

TASK_DISPLAY_NAMES = {
    "merge_action": "Split Error Correction",
    "endpoint_error_identification_with_em": "Split Error Identification",
    "merge_error_identification": "Merge Error Identification",
    "split_action": "Split Action Evaluation",
}

# Default path: relative to reproduction/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # reproduction/

HUMAN_EVAL_FILES = {
    "endpoint_error_identification_with_em": REPO_DIR / "data/human_eval/human_eval_endpoint_error_identification_with_em_anonymous_12149c9ecfa7.json",
    "merge_action": REPO_DIR / "data/human_eval/human_eval_split-error-correction_anonymous_453d4b109747.json",
    "merge_error_identification": REPO_DIR / "data/human_eval/human_eval_merge-error-identification_anonymous_29e820ea8eb1.json",
    "split_action": REPO_DIR / "data/human_eval/human_eval_split-correction-evaluation_anonymous_ef37eba8df98.json",
}

def load_linear_probe_results(sweep_dir: Path) -> dict:
    """Load all linear probe sweep results."""
    results = {}

    for task in TASK_ORDER:
        # Use the test1024 sweep files (with 80% train/10% val/10% test split)
        filename = f"data_sweep_{task}_linear_test1024_repeats10_seed42.json"
        filepath = sweep_dir / filename

        if not filepath.exists():
            print(f"Warning: No linear probe results found for {task} at {filepath}")
            continue

        # Load the sweep data
        with open(filepath) as f:
            data = json.load(f)

        # Extract scaling curve and compute balanced accuracy from TPR+TNR
        aggregated = data["aggregated_results"]
        train_sizes = [r["n_train"] for r in aggregated]

        # Compute balanced accuracy from TPR and TNR
        balanced_accs = [(r["test_tpr_mean"] + r["test_tnr_mean"]) / 2 * 100 for r in aggregated]

        # Compute std for balanced accuracy using error propagation
        # Std((TPR+TNR)/2) = sqrt((Std(TPR)^2 + Std(TNR)^2)/4)
        balanced_stds = [np.sqrt((r["test_tpr_std"]**2 + r["test_tnr_std"]**2) / 4) * 100 for r in aggregated]

        results[task] = {
            "train_sizes": train_sizes,
            "accuracies": balanced_accs,
            "stds": balanced_stds,
        }

    return results

def load_resnet_results(resnet_path: Path) -> dict:
    """Load ResNet results."""
    with open(resnet_path) as f:
        return json.load(f)

def load_human_baselines() -> dict:
    """Load human baseline accuracies."""
    baselines = {}
    for task_key, filepath in HUMAN_EVAL_FILES.items():
        if not filepath.exists():
            baselines[task_key] = None
            continue

        with open(filepath) as f:
            data = json.load(f)

        # Average human and tim accuracy
        human_acc = data.get("accuracy")
        tim_acc = data.get("tim_accuracy")

        if human_acc is not None and tim_acc is not None:
            baselines[task_key] = (human_acc + tim_acc) / 2 * 100
        elif human_acc is not None:
            baselines[task_key] = human_acc * 100
        else:
            baselines[task_key] = None

    return baselines

def plot_scaling_curves(
    linear_probe_results: dict,
    resnet_results: dict,
    human_baselines: dict,
    output_path: Path,
):
    """
    Create 2x2 subplot figure showing scaling curves for all tasks.
    """
    # Set up plot style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, task in enumerate(TASK_ORDER):
        ax = axes[idx]

        # Get linear probe data
        lp_data = linear_probe_results.get(task)
        if lp_data is None:
            ax.text(0.5, 0.5, f'No data for {task}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(TASK_DISPLAY_NAMES[task])
            continue

        # Plot linear probe scaling curve
        ax.errorbar(
            lp_data["train_sizes"], lp_data["accuracies"], yerr=lp_data["stds"],
            marker='o',
            markersize=5,
            color='black',
            markerfacecolor='0.6',
            linewidth=1.5,
            capsize=3,
            capthick=1,
            label='Linear Probe' if idx == 0 else '',
            zorder=3,
        )

        # Get ResNet results
        task_key = f"{task}_resnet"
        resnet_task_data = resnet_results.get(task_key, {})

        # Plot ResNet frozen
        frozen_data = resnet_task_data.get('frozen_model')
        if frozen_data:
            frozen_acc = frozen_data['balanced_accuracy'] * 100
            frozen_samples = frozen_data.get('train_samples', max(lp_data["train_sizes"]))
            ax.scatter(
                [frozen_samples], [frozen_acc],
                marker='s',
                s=120,
                color='0.4',
                edgecolor='black',
                linewidth=1.5,
                label='ResNet-Frozen' if idx == 0 else '',
                zorder=4,
            )

        # Plot ResNet fine-tuned
        finetuned_data = resnet_task_data.get('fully_finetuned_model')
        if finetuned_data:
            finetuned_acc = finetuned_data['balanced_accuracy'] * 100
            finetuned_samples = finetuned_data.get('train_samples', max(lp_data["train_sizes"]))
            ax.scatter(
                [finetuned_samples], [finetuned_acc],
                marker='^',
                s=140,
                color='0.2',
                edgecolor='black',
                linewidth=1.5,
                label='ResNet-Finetuned' if idx == 0 else '',
                zorder=4,
            )

        # Plot ResNet small (512 samples)
        small_data = resnet_task_data.get('small_model')
        if small_data:
            small_acc = small_data['balanced_accuracy'] * 100
            small_samples = small_data.get('train_samples', 512)
            ax.scatter(
                [small_samples], [small_acc],
                marker='o',
                s=100,
                color='0.7',
                edgecolor='black',
                linewidth=1.5,
                label='ResNet (512 Training Samples)' if idx == 0 else '',
                zorder=4,
            )

        # Plot VLM (if available)
        vlm_data = resnet_task_data.get('vlm')
        if vlm_data:
            vlm_acc = vlm_data['balanced_accuracy'] * 100
            # VLM trained on same data as finetuned model
            vlm_samples = finetuned_data.get('train_samples', max(lp_data["train_sizes"])) if finetuned_data else max(lp_data["train_sizes"])
            ax.scatter(
                [vlm_samples], [vlm_acc],
                marker='D',
                s=140,
                color='white',
                edgecolor='black',
                linewidth=1.5,
                label='VLM' if idx == 0 else '',
                zorder=5,
            )

        # Plot human baseline (if available)
        human_baseline = human_baselines.get(task)
        if human_baseline is not None:
            ax.axhline(
                y=human_baseline,
                color='gray',
                linestyle=':',
                linewidth=2,
                label='Human' if idx == 0 else '',
                alpha=0.8,
                zorder=1,
            )

        # Configure axes
        ax.set_xlabel('Training Samples', fontsize=14, fontweight='bold')
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(45, 105)

        # Add horizontal gridlines
        ax.yaxis.grid(True, alpha=0.3, linestyle='-', zorder=0)
        ax.set_axisbelow(True)

        # Add 50% chance line
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.4, linewidth=1.5)

        # Add title
        ax.set_title(TASK_DISPLAY_NAMES[task], fontsize=16, fontweight='bold', pad=10)

    # Create shared legend outside subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=6, frameon=True, fancybox=False, edgecolor='black', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.02)  # Make room for legend

    # Save as PNG and PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print(f"Saved to {output_path}")
    print(f"Saved to {pdf_path}")

    # Print summary
    print("\nScaling Curves Summary:")
    print("="*80)
    for task in TASK_ORDER:
        print(f"\n{TASK_DISPLAY_NAMES[task]}:")

        lp_data = linear_probe_results.get(task)
        if lp_data:
            min_samples = min(lp_data["train_sizes"])
            max_samples = max(lp_data["train_sizes"])
            min_acc = min(lp_data["accuracies"])
            max_acc = max(lp_data["accuracies"])
            print(f"  Linear Probe: {min_acc:.1f}% -> {max_acc:.1f}% ({min_samples}-{max_samples} samples)")

        task_key = f"{task}_resnet"
        resnet_task_data = resnet_results.get(task_key, {})

        frozen_data = resnet_task_data.get('frozen_model')
        if frozen_data:
            frozen_acc = frozen_data['balanced_accuracy'] * 100
            frozen_samples = frozen_data.get('train_samples', 'unknown')
            print(f"  ResNet-Frozen: {frozen_acc:.1f}% @ {frozen_samples} samples")

        finetuned_data = resnet_task_data.get('fully_finetuned_model')
        if finetuned_data:
            finetuned_acc = finetuned_data['balanced_accuracy'] * 100
            finetuned_samples = finetuned_data.get('train_samples', 'unknown')
            print(f"  ResNet-Finetuned: {finetuned_acc:.1f}% @ {finetuned_samples} samples")

        small_data = resnet_task_data.get('small_model')
        if small_data:
            small_acc = small_data['balanced_accuracy'] * 100
            small_samples = small_data.get('train_samples', 512)
            print(f"  ResNet-Small: {small_acc:.1f}% @ {small_samples} samples")

def main():
    parser = argparse.ArgumentParser(
        description="Plot scaling curves for all tasks (Figure 2)"
    )
    parser.add_argument(
        "--linear-probe-dir",
        type=Path,
        default=REPO_DIR / "data" / "linear_probe_sweep",
        help="Directory containing linear probe sweep results",
    )
    parser.add_argument(
        "--resnet-results",
        type=Path,
        default=REPO_DIR / "data" / "final_data" / "resnet_results.json",
        help="Path to ResNet results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_DIR / "output" / "figure2_scaling_curves.png",
        help="Output path for plot (PNG)",
    )

    args = parser.parse_args()

    # Load data
    linear_probe_results = load_linear_probe_results(args.linear_probe_dir)
    resnet_results = load_resnet_results(args.resnet_results)
    human_baselines = load_human_baselines()

    # Create plot
    plot_scaling_curves(linear_probe_results, resnet_results, human_baselines, args.output)

if __name__ == "__main__":
    main()
