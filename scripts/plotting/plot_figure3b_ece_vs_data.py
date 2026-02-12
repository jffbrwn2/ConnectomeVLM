#!/usr/bin/env python3
"""
Generate Figure 3B: ECE vs Training Samples (4 subplots like Figure 2).

Shows how calibration quality changes with training data scale.

Usage:
    cd reproduction/
    python scripts/plotting/plot_figure3b_ece_vs_data.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Default path: relative to reproduction/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # reproduction/

# Task display names and order (easiest to hardest, split-related then merge-related)
TASKS = [
    ("merge_action", "Split Error\nCorrection"),
    ("endpoint_error_identification_with_em", "Split Error\nIdentification"),
    ("merge_error_identification", "Merge Error\nIdentification"),
    ("split_action", "Split Action\nEvaluation"),
]

# Model colors (grayscale matching other figures)
COLORS = {
    "linear_probe": "black",  # Black
    "resnet_frozen": "0.4",  # Dark gray
    "resnet_finetuned": "0.2",  # Very dark gray
    "resnet_small": "0.7",  # Light gray
    "vlm": "white",  # White
}

MODEL_NAMES = {
    "linear_probe": "Linear Probe",
    "resnet_frozen": "ResNet-Frozen",
    "resnet_finetuned": "ResNet-Finetuned",
    "resnet_small": "ResNet-Small (512)",
    "vlm": "VLM",
}


def load_linear_probe_ece(task_key):
    """Load linear probe ECE scaling data for a task."""
    sweep_files = {
        "endpoint_error_identification_with_em":
            REPO_DIR / "data/linear_probe_sweep/data_sweep_endpoint_error_identification_with_em_linear_test1024_repeats10_seed42.json",
        "merge_action":
            REPO_DIR / "data/linear_probe_sweep/data_sweep_merge_action_linear_test1024_repeats10_seed42.json",
        "merge_error_identification":
            REPO_DIR / "data/linear_probe_sweep/data_sweep_merge_error_identification_linear_test1024_repeats10_seed42.json",
        "split_action":
            REPO_DIR / "data/linear_probe_sweep/data_sweep_split_action_linear_test1024_repeats10_seed42.json",
    }

    filepath = sweep_files.get(task_key)
    if filepath is None or not filepath.exists():
        print(f"Warning: Linear probe ECE sweep not found for {task_key}")
        return None

    with open(filepath) as f:
        data = json.load(f)

    results = data["aggregated_results"]
    train_sizes = [r["n_train"] for r in results]
    ece_means = [r["test_ece_mean"] * 100 for r in results]  # Convert to percentage
    ece_stds = [r["test_ece_std"] * 100 for r in results]

    return {
        "train_sizes": train_sizes,
        "ece_means": ece_means,
        "ece_stds": ece_stds,
    }


def load_metadata():
    """Load model metadata."""
    with open(REPO_DIR / "data/model_metadata.json") as f:
        return json.load(f)


def load_resnet_calibration():
    """Load ResNet calibration data."""
    with open(REPO_DIR / "data/calibration_results_all/calibration_summary.json") as f:
        resnet_calib = json.load(f)

    metadata = load_metadata()
    train_samples_map = metadata["train_samples"]

    checkpoint_mapping = {
        "resnet50_resnet50_endpoint_error_identification_with_em_test_frozen": {
            "task": "endpoint_error_identification_with_em",
            "model": "resnet_frozen",
        },
        "resnet50_resnet50_endpoint_em_test": {
            "task": "endpoint_error_identification_with_em",
            "model": "resnet_finetuned",
        },
        "resnet50_resnet50_merge_action_test_frozen": {
            "task": "merge_action",
            "model": "resnet_frozen",
        },
        "resnet50_resnet50_merge_action_test": {
            "task": "merge_action",
            "model": "resnet_finetuned",
        },
        "resnet50_resnet50_merge_error_identification_test_frozen": {
            "task": "merge_error_identification",
            "model": "resnet_frozen",
        },
        "resnet50_resnet50_merge_error_identification_test": {
            "task": "merge_error_identification",
            "model": "resnet_finetuned",
        },
        "resnet50_resnet50_split_action_test_frozen": {
            "task": "split_action",
            "model": "resnet_frozen",
        },
        "resnet50_resnet50_split_action_test": {
            "task": "split_action",
            "model": "resnet_finetuned",
        },
    }

    results = {}
    for task_key, _ in TASKS:
        results[task_key] = {}

    for checkpoint_name, info in checkpoint_mapping.items():
        if checkpoint_name in resnet_calib:
            task = info["task"]
            model = info["model"]

            ece = resnet_calib[checkpoint_name]["metrics"]["ece"] * 100
            train_samples = train_samples_map.get(task)

            results[task][model] = {
                "ece": ece,
                "train_samples": train_samples,
            }

    return results


def load_vlm_calibration():
    """Load VLM calibration data."""
    vlm_calib_path = REPO_DIR / "data/final_data/vlm_calibration_all_results.json"
    if not vlm_calib_path.exists():
        return {}

    with open(vlm_calib_path) as f:
        vlm_data = json.load(f)

    metadata = load_metadata()
    train_samples_map = metadata["train_samples"]

    # Map from VLM task names to figure task names
    task_name_mapping = {
        "merge_action": "merge_action",
        "split_action": "split_action",
        "merge_error_identification": "merge_error_identification",
        "endpoint_error_identification_with_em": "endpoint_error_identification_with_em",
    }

    results = {}
    for task_key, _ in TASKS:
        results[task_key] = {}

    # Get VLM data for mouse test split
    aggregated = vlm_data.get("aggregated", {})
    for vlm_task, species_data in aggregated.items():
        # Map task name
        mapped_task = task_name_mapping.get(vlm_task)
        if mapped_task and "mouse" in species_data:
            mouse_data = species_data["mouse"]
            if "original" in mouse_data:
                results[mapped_task]["vlm"] = {
                    "ece": mouse_data["original"]["ece"] * 100,
                    "train_samples": train_samples_map.get(mapped_task),
                }

    return results


def plot_task_ece(ax, task_key, task_name, linear_probe_data, resnet_data, vlm_data, is_first_subplot=False):
    """Plot ECE scaling for one task."""

    # Plot linear probe scaling curve
    if linear_probe_data is not None:
        ax.errorbar(
            linear_probe_data["train_sizes"],
            linear_probe_data["ece_means"],
            yerr=linear_probe_data["ece_stds"],
            marker='o',
            markersize=5,
            color=COLORS["linear_probe"],
            markerfacecolor='0.6',
            linewidth=1.5,
            capsize=3,
            capthick=1,
            label=MODEL_NAMES["linear_probe"] if is_first_subplot else '',
            zorder=3,
        )

    # Plot ResNet points
    resnet_task_data = resnet_data.get(task_key, {})

    frozen_data = resnet_task_data.get("resnet_frozen", {})
    if frozen_data:
        ax.scatter(
            [frozen_data["train_samples"]], [frozen_data["ece"]],
            marker='s', s=120,
            color=COLORS["resnet_frozen"],
            edgecolor='black',
            linewidth=1.5,
            label=MODEL_NAMES["resnet_frozen"] if is_first_subplot else '',
            zorder=4,
        )

    finetuned_data = resnet_task_data.get("resnet_finetuned", {})
    if finetuned_data:
        ax.scatter(
            [finetuned_data["train_samples"]], [finetuned_data["ece"]],
            marker='^', s=140,
            color=COLORS["resnet_finetuned"],
            edgecolor='black',
            linewidth=1.5,
            label=MODEL_NAMES["resnet_finetuned"] if is_first_subplot else '',
            zorder=4,
        )

    # Plot VLM point
    vlm_task_data = vlm_data.get(task_key, {})
    vlm_point = vlm_task_data.get("vlm", {})
    if vlm_point:
        ax.scatter(
            [vlm_point["train_samples"]], [vlm_point["ece"]],
            marker='D', s=140,
            color=COLORS["vlm"],
            edgecolor='black',
            linewidth=1.5,
            label=MODEL_NAMES["vlm"] if is_first_subplot else '',
            zorder=5,
        )

    # Styling (match Figure 2)
    ax.set_xscale('log')
    ax.set_xlabel('Training Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('ECE (%)', fontsize=14, fontweight='bold')
    ax.set_title(task_name, fontsize=16, fontweight='bold', pad=10)

    ax.set_ylim(0, 35)  # ECE can be higher for small data

    # Add horizontal gridlines only (like Figure 2)
    ax.yaxis.grid(True, alpha=0.3, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    # Add 5% calibration threshold line
    ax.axhline(y=5, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)


def main():
    print("Loading calibration data...")

    resnet_data = load_resnet_calibration()
    vlm_data = load_vlm_calibration()

    # Set up plot style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Create 2x2 subplot figure (match Figure 2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    print("\nGenerating ECE scaling curves...")

    for idx, (task_key, task_name) in enumerate(TASKS):
        print(f"  Task {idx+1}/4: {task_name.replace(chr(10), ' ')}")

        # Load linear probe ECE data
        linear_probe_data = load_linear_probe_ece(task_key)

        # Plot
        plot_task_ece(
            axes[idx],
            task_key,
            task_name,
            linear_probe_data,
            resnet_data,
            vlm_data,
            is_first_subplot=(idx == 0),
        )

    # Create shared legend outside subplots (match Figure 2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=4, frameon=True, fancybox=False, edgecolor='black', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.02)  # Make room for legend

    # Save
    output_dir = REPO_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    png_path = output_dir / "figure3b_ece_vs_data.png"
    pdf_path = output_dir / "figure3b_ece_vs_data.pdf"

    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    print(f"\nSaved to {png_path}")
    print(f"Saved to {pdf_path}")

    # Summary
    print("\n" + "=" * 60)
    print("ECE Scaling Summary:")
    print("=" * 60)
    for task_key, task_name in TASKS:
        print(f"\n{task_name.replace(chr(10), ' ')}:")
        linear_probe_data = load_linear_probe_ece(task_key)
        if linear_probe_data:
            min_ece = min(linear_probe_data['ece_means'])
            max_ece = max(linear_probe_data['ece_means'])
            print(f"  Linear Probe: {linear_probe_data['train_sizes'][0]}-{linear_probe_data['train_sizes'][-1]} samples")
            print(f"    ECE: {max_ece:.1f}% -> {min_ece:.1f}%")
        else:
            print(f"  Linear Probe: MISSING DATA")

        resnet_task = resnet_data.get(task_key, {})
        frozen = resnet_task.get("resnet_frozen", {})
        finetuned = resnet_task.get("resnet_finetuned", {})

        if frozen:
            print(f"  ResNet-Frozen: {frozen['ece']:.2f}% @ {frozen['train_samples']:,} samples")
        if finetuned:
            print(f"  ResNet-Finetuned: {finetuned['ece']:.2f}% @ {finetuned['train_samples']:,} samples")

    print("\n" + "=" * 60)
    print("Note: Dashed line at 5% ECE shows well-calibrated threshold")


if __name__ == "__main__":
    main()
