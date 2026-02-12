"""
ResNet Calibration Analysis Script

Computes Expected Calibration Error (ECE) and other calibration metrics for trained
ResNet/ViT/ConvNeXt models. Generates reliability diagrams and comparison plots.

Usage:
    # Local execution (for analysis/plotting):
    python dev/compute_resnet_calibration.py --checkpoint resnet50_merge_action_20240101_120000

    # Modal execution (for compute):
    modal run dev/compute_resnet_calibration.py --checkpoint resnet50_merge_action_20240101_120000

    # Analyze all checkpoints in a directory:
    modal run dev/compute_resnet_calibration.py --analyze-all
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    print("Warning: modal not available, running in local mode only")


# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_ece(confidences: np.ndarray, predictions: np.ndarray,
                labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE)

    Measures average calibration across confidence bins.
    ECE = sum over bins: |accuracy_in_bin - avg_confidence_in_bin| * proportion_in_bin

    Args:
        confidences: Max softmax probabilities (shape: [N])
        predictions: Predicted class indices (shape: [N])
        labels: True class indices (shape: [N])
        n_bins: Number of bins for calibration

    Returns:
        ECE value (0.0 = perfect calibration, higher = worse)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = (predictions == labels).astype(float)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_mce(confidences: np.ndarray, predictions: np.ndarray,
                labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Maximum Calibration Error (MCE)

    Maximum calibration gap across all bins.
    MCE = max over bins: |accuracy_in_bin - avg_confidence_in_bin|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = (predictions == labels).astype(float)
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_gap = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = max(mce, calibration_gap)

    return float(mce)


def compute_brier_score(confidences: np.ndarray, predictions: np.ndarray,
                        labels: np.ndarray) -> float:
    """
    Brier Score (mean squared error between confidence and correctness)

    Lower is better. Perfect calibration + accuracy = 0.0
    """
    correctness = (predictions == labels).astype(float)
    brier = np.mean((confidences - correctness) ** 2)
    return float(brier)


def compute_adaptive_ece(confidences: np.ndarray, predictions: np.ndarray,
                         labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Adaptive ECE - bins are created to have equal number of samples
    More robust when confidence distribution is skewed.
    """
    # Sort by confidence
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    sorted_accuracies = (sorted_predictions == sorted_labels).astype(float)

    n_samples = len(confidences)
    samples_per_bin = n_samples // n_bins

    ece = 0.0
    for i in range(n_bins):
        start_idx = i * samples_per_bin
        end_idx = (i + 1) * samples_per_bin if i < n_bins - 1 else n_samples

        bin_confidences = sorted_confidences[start_idx:end_idx]
        bin_accuracies = sorted_accuracies[start_idx:end_idx]

        if len(bin_confidences) > 0:
            avg_confidence = bin_confidences.mean()
            avg_accuracy = bin_accuracies.mean()
            prop_in_bin = len(bin_confidences) / n_samples
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

    return float(ece)


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics"""
    ece: float
    mce: float
    brier_score: float
    adaptive_ece: float
    accuracy: float
    avg_confidence: float
    n_samples: int

    # Per-bin statistics for plotting
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    bin_boundaries: List[float]

    def to_dict(self) -> Dict:
        return {
            'ece': self.ece,
            'mce': self.mce,
            'brier_score': self.brier_score,
            'adaptive_ece': self.adaptive_ece,
            'accuracy': self.accuracy,
            'avg_confidence': self.avg_confidence,
            'n_samples': self.n_samples,
            'bin_accuracies': self.bin_accuracies,
            'bin_confidences': self.bin_confidences,
            'bin_counts': self.bin_counts,
            'bin_boundaries': self.bin_boundaries,
        }


def compute_all_metrics(confidences: np.ndarray, predictions: np.ndarray,
                        labels: np.ndarray, n_bins: int = 10) -> CalibrationMetrics:
    """Compute all calibration metrics for a model"""

    # Core metrics
    ece = compute_ece(confidences, predictions, labels, n_bins)
    mce = compute_mce(confidences, predictions, labels, n_bins)
    brier = compute_brier_score(confidences, predictions, labels)
    adaptive_ece = compute_adaptive_ece(confidences, predictions, labels, n_bins)
    accuracy = (predictions == labels).mean()
    avg_confidence = confidences.mean()

    # Bin statistics for plotting
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_accuracies.append(float((predictions[in_bin] == labels[in_bin]).mean()))
            bin_confidences.append(float(confidences[in_bin].mean()))
            bin_counts.append(int(in_bin.sum()))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((bin_boundaries[i] + bin_boundaries[i+1]) / 2))
            bin_counts.append(0)

    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier_score=brier,
        adaptive_ece=adaptive_ece,
        accuracy=float(accuracy),
        avg_confidence=float(avg_confidence),
        n_samples=len(confidences),
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        bin_boundaries=bin_boundaries.tolist(),
    )


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_reliability_diagram(metrics: CalibrationMetrics, title: str = "",
                             save_path: Path = None) -> plt.Figure:
    """
    Create reliability diagram (calibration plot)

    Shows:
    - Bar plot: confidence vs accuracy per bin
    - Perfect calibration line (diagonal)
    - Confidence histogram
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bin_centers = [(metrics.bin_boundaries[i] + metrics.bin_boundaries[i+1]) / 2
                   for i in range(len(metrics.bin_accuracies))]

    # Plot 1: Reliability diagram
    ax = axes[0]
    bar_width = 0.08
    ax.bar(bin_centers, metrics.bin_accuracies, width=bar_width,
           alpha=0.7, color='steelblue', label='Accuracy')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Reliability Diagram', fontsize=12, fontweight='bold')

    # Add ECE text
    textstr = f'ECE: {metrics.ece:.3f}\n'
    textstr += f'MCE: {metrics.mce:.3f}\n'
    textstr += f'Acc: {metrics.accuracy:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Calibration gap
    ax = axes[1]
    gaps = [abs(acc - conf) for acc, conf in
            zip(metrics.bin_accuracies, metrics.bin_confidences)]
    colors = ['green' if gap < 0.05 else 'orange' if gap < 0.1 else 'red'
              for gap in gaps]
    ax.bar(bin_centers, gaps, width=bar_width, color=colors, alpha=0.7)
    ax.axhline(y=metrics.ece, color='blue', linestyle='--',
               linewidth=2, label=f'ECE: {metrics.ece:.3f}')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('|Accuracy - Confidence|', fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Calibration Gap per Bin', fontsize=12, fontweight='bold')

    # Plot 3: Sample distribution
    ax = axes[2]
    ax.bar(bin_centers, metrics.bin_counts, width=bar_width,
           alpha=0.7, color='steelblue')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Number of samples', fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")

    return fig


def plot_confidence_histogram(confidences: np.ndarray, predictions: np.ndarray,
                               labels: np.ndarray, title: str = "",
                               save_path: Path = None) -> plt.Figure:
    """
    Plot confidence distribution split by correct/incorrect predictions
    """
    correct_mask = predictions == labels

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 21)
    ax.hist(confidences[correct_mask], bins=bins, alpha=0.6,
            label=f'Correct ({correct_mask.sum()})', color='green', density=True)
    ax.hist(confidences[~correct_mask], bins=bins, alpha=0.6,
            label=f'Incorrect ({(~correct_mask).sum()})', color='red', density=True)

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add vertical lines for mean confidences
    ax.axvline(confidences[correct_mask].mean(), color='green',
               linestyle='--', linewidth=2, alpha=0.7,
               label=f'Mean (correct): {confidences[correct_mask].mean():.3f}')
    ax.axvline(confidences[~correct_mask].mean(), color='red',
               linestyle='--', linewidth=2, alpha=0.7,
               label=f'Mean (incorrect): {confidences[~correct_mask].mean():.3f}')
    ax.legend(fontsize=10)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")

    return fig


def plot_comparison_across_models(results: Dict[str, CalibrationMetrics],
                                  save_path: Path = None) -> plt.Figure:
    """
    Compare ECE across multiple models/tasks

    Args:
        results: Dict mapping "model_task" -> CalibrationMetrics
    """
    # Group by model and task
    by_task = defaultdict(dict)
    for key, metrics in results.items():
        # Parse key like "resnet50_merge_action" or "vit_l_16_split_action"
        parts = key.split('_')
        # Find where task name starts (after model name)
        # Task names: merge_action, split_action, segment_classification, etc.
        task_keywords = ['merge', 'split', 'segment', 'endpoint']
        task_start_idx = None
        for i, part in enumerate(parts):
            if part in task_keywords:
                task_start_idx = i
                break

        if task_start_idx:
            model_name = '_'.join(parts[:task_start_idx])
            task_name = '_'.join(parts[task_start_idx:])
        else:
            # Fallback: assume first part is model
            model_name = parts[0]
            task_name = '_'.join(parts[1:])

        by_task[task_name][model_name] = metrics

    # Create comparison plot
    n_tasks = len(by_task)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6*n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for ax, (task, model_metrics) in zip(axes, by_task.items()):
        models = list(model_metrics.keys())
        eces = [model_metrics[m].ece for m in models]
        accuracies = [model_metrics[m].accuracy for m in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, eces, width, label='ECE', alpha=0.8, color='coral')
        bars2 = ax.bar(x + width/2, [1-acc for acc in accuracies], width,
                       label='Error Rate', alpha=0.8, color='steelblue')

        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(task.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Calibration Comparison Across Models and Tasks',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison plot to: {save_path}")

    return fig


# ============================================================================
# Model Loading and Evaluation (Modal-compatible)
# ============================================================================

if MODAL_AVAILABLE:
    sys.path.insert(0, str(Path(__file__).parent.parent / "model-post-training"))
    from modal_resnet_finetune import (
        app, image, MODEL_DIR, DATASET_DIR, CHECKPOINT_DIR,
        checkpoint_volume, model_volume, dataset_volume,
        create_resnet_model, get_transforms, ImageGridDataset,
        ResNetTrainingConfig
    )
    from environment.task_configs import get_task

    @app.function(
        gpu="A10G",
        timeout=3600,
        volumes={
            MODEL_DIR: model_volume,
            DATASET_DIR: dataset_volume,
            CHECKPOINT_DIR: checkpoint_volume,
        },
    )
    def evaluate_checkpoint_calibration(checkpoint_path: str) -> Dict:
        """
        Evaluate calibration for a single checkpoint (Modal function)

        Returns:
            Dict with metrics and arrays needed for plotting
        """
        import torch
        from torch.utils.data import DataLoader
        import os

        os.environ["HF_HOME"] = str(MODEL_DIR)

        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_path}")
        print(f"{'='*60}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint info
        checkpoint_dir = CHECKPOINT_DIR / checkpoint_path
        config_path = checkpoint_dir / "training_config.json"
        test_indices_path = checkpoint_dir / "test_indices.json"
        best_model_path = checkpoint_dir / "best_model.pt"
        final_model_path = checkpoint_dir / "final_model.pt"

        if not config_path.exists():
            raise FileNotFoundError(f"No training_config.json in {checkpoint_dir}")

        # Load config
        with open(str(config_path), "r") as f:
            config_dict = json.load(f)
        config = ResNetTrainingConfig(**config_dict)

        # Load task
        task = get_task(config.task_name)
        print(f"Task: {task.name}")

        # Load class mapping from checkpoint
        model_path = best_model_path if best_model_path.exists() else final_model_path
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {checkpoint_dir}")

        checkpoint = torch.load(str(model_path), map_location=device)

        # Parse label mappings
        def parse_label(label_str):
            """Parse label string to original type"""
            if label_str in ["True", "False"]:
                return label_str == "True"
            try:
                return int(label_str)
            except:
                return label_str

        if "label_to_idx" in checkpoint:
            label_to_idx = {parse_label(k): v for k, v in checkpoint["label_to_idx"].items()}
            idx_to_label = {int(k): parse_label(v) for k, v in checkpoint["idx_to_label"].items()}
            num_classes = checkpoint["num_classes"]
        else:
            # Load from file
            class_mapping_path = checkpoint_dir / "class_mapping.json"
            with open(str(class_mapping_path), "r") as f:
                mapping = json.load(f)
            label_to_idx = {parse_label(k): v for k, v in mapping["label_to_idx"].items()}
            idx_to_label = {int(k): parse_label(v) for k, v in mapping["idx_to_label"].items()}
            num_classes = len(label_to_idx)

        print(f"Classes: {num_classes}")

        # Create and load model
        model = create_resnet_model(config.model_name, num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Load dataset
        print(f"Loading dataset...")
        dataset = task.load_dataset(cache_dir=str(DATASET_DIR))
        dataset = task.filter_dataset(dataset)
        print(f"  Full dataset: {len(dataset)} samples")

        # Filter to test set
        if test_indices_path.exists():
            with open(str(test_indices_path), "r") as f:
                indices_data = json.load(f)
            test_indices = indices_data.get("test_indices", [])

            if test_indices:
                # Check if indices are parquet indices or direct dataset indices
                has_parquet_idx = '_original_parquet_idx' in dataset.column_names
                if has_parquet_idx:
                    # Map parquet indices to current dataset indices
                    parquet_to_idx = {
                        sample['_original_parquet_idx']: i
                        for i, sample in enumerate(dataset)
                    }
                    test_indices = [parquet_to_idx[idx] for idx in test_indices
                                   if idx in parquet_to_idx]

                class IndexedDataset:
                    def __init__(self, dataset, indices):
                        self.dataset = dataset
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.dataset[self.indices[idx]]

                dataset = IndexedDataset(dataset, test_indices)
                print(f"  Using test set: {len(dataset)} samples")

        # Create dataloader
        val_transform = get_transforms(config.image_size, is_training=False)
        eval_ds = ImageGridDataset(
            dataset, task, label_to_idx,
            image_size=config.image_size,
            grid_layout=config.grid_layout,
            transform=val_transform
        )
        eval_loader = DataLoader(
            eval_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )

        # Run inference to collect logits
        print(f"Running inference...")
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute probabilities and predictions
        probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)  # softmax
        confidences = probs.max(axis=1)  # max probability
        predictions = probs.argmax(axis=1)

        print(f"  Collected {len(confidences)} predictions")
        print(f"  Accuracy: {(predictions == all_labels).mean():.3f}")
        print(f"  Mean confidence: {confidences.mean():.3f}")

        # Compute all calibration metrics
        print(f"Computing calibration metrics...")
        metrics = compute_all_metrics(confidences, predictions, all_labels, n_bins=10)

        print(f"  ECE: {metrics.ece:.4f}")
        print(f"  MCE: {metrics.mce:.4f}")
        print(f"  Brier: {metrics.brier_score:.4f}")
        print(f"  Adaptive ECE: {metrics.adaptive_ece:.4f}")

        # Return serializable results
        return {
            'checkpoint_path': checkpoint_path,
            'task_name': config.task_name,
            'model_name': config.model_name,
            'metrics': metrics.to_dict(),
            'confidences': confidences.tolist(),
            'predictions': predictions.tolist(),
            'labels': all_labels.tolist(),
        }


# ============================================================================
# Local Analysis and Plotting
# ============================================================================

def analyze_checkpoint_local(results: Dict, output_dir: Path):
    """
    Analyze results locally and generate plots

    Args:
        results: Output from evaluate_checkpoint_calibration
        output_dir: Where to save plots
    """
    checkpoint_name = Path(results['checkpoint_path']).name
    task_name = results['task_name']
    model_name = results['model_name'].split('/')[-1] if '/' in results['model_name'] else results['model_name']

    # Reconstruct arrays
    confidences = np.array(results['confidences'])
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])

    # Reconstruct metrics
    metrics_dict = results['metrics']
    metrics = CalibrationMetrics(**metrics_dict)

    # Create output directories
    diagrams_dir = output_dir / "reliability_diagrams"
    histograms_dir = output_dir / "confidence_histograms"
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    histograms_dir.mkdir(parents=True, exist_ok=True)

    # Generate reliability diagram
    title = f"{model_name} on {task_name}\n"
    title += f"Acc: {metrics.accuracy:.3f} | ECE: {metrics.ece:.3f} | MCE: {metrics.mce:.3f}"

    plot_reliability_diagram(
        metrics,
        title=title,
        save_path=diagrams_dir / f"{checkpoint_name}_reliability.png"
    )
    plt.close()

    # Generate confidence histogram
    plot_confidence_histogram(
        confidences, predictions, labels,
        title=f"Confidence Distribution: {model_name} on {task_name}",
        save_path=histograms_dir / f"{checkpoint_name}_confidence_hist.png"
    )
    plt.close()

    print(f"\n✓ Generated plots for {checkpoint_name}")


def analyze_all_checkpoints(checkpoint_paths: List[str], output_dir: Path):
    """
    Analyze multiple checkpoints and create comparison plots
    """
    if not MODAL_AVAILABLE:
        print("Error: Modal is required for evaluation. Install with: pip install modal")
        return

    print(f"\n{'='*60}")
    print(f"Analyzing {len(checkpoint_paths)} checkpoints...")
    print(f"{'='*60}")

    # Run evaluations on Modal
    all_results = {}
    for checkpoint_path in checkpoint_paths:
        try:
            result = evaluate_checkpoint_calibration.remote(checkpoint_path)
            key = f"{result['model_name'].split('/')[-1]}_{result['task_name']}"
            all_results[key] = result

            # Generate individual plots
            analyze_checkpoint_local(result, output_dir)

        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {e}")
            continue

    if not all_results:
        print("No results to analyze!")
        return

    # Save summary JSON
    summary = {
        key: result['metrics']
        for key, result in all_results.items()
    }
    summary_path = output_dir / "calibration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary to: {summary_path}")

    # Create comparison plot
    metrics_dict = {
        key: CalibrationMetrics(**result['metrics'])
        for key, result in all_results.items()
    }

    plot_comparison_across_models(
        metrics_dict,
        save_path=output_dir / "calibration_comparison.png"
    )
    plt.close()

    # Print summary table
    print(f"\n{'='*60}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model/Task':<40} {'Acc':>6} {'ECE':>6} {'MCE':>6} {'Brier':>6}")
    print(f"{'-'*60}")
    for key, metrics in sorted(metrics_dict.items()):
        print(f"{key:<40} {metrics.accuracy:>6.3f} {metrics.ece:>6.3f} "
              f"{metrics.mce:>6.3f} {metrics.brier_score:>6.3f}")
    print(f"{'='*60}\n")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute calibration metrics for ResNet models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single checkpoint
  modal run dev/compute_resnet_calibration.py --checkpoint resnet50_merge_action_20240101

  # Multiple checkpoints
  modal run dev/compute_resnet_calibration.py \\
    --checkpoint resnet50_merge_action_20240101 \\
    --checkpoint vit_l_16_split_action_20240102

  # Auto-discover all checkpoints
  modal run dev/compute_resnet_calibration.py --analyze-all

  # Specify output directory
  modal run dev/compute_resnet_calibration.py --analyze-all --output calibration_results_v2
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        action='append',
        help='Checkpoint directory name (can specify multiple times)'
    )
    parser.add_argument(
        '--analyze-all',
        action='store_true',
        help='Analyze all checkpoints in CHECKPOINT_DIR'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration_results',
        help='Output directory for results (default: calibration_results)'
    )

    args = parser.parse_args()

    # Determine output directory
    if MODAL_AVAILABLE:
        # Running on Modal - results will be in Modal volume
        from scripts.model_post_training.modal_resnet_finetune import CHECKPOINT_DIR
        output_dir = CHECKPOINT_DIR / args.output
    else:
        # Local execution
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which checkpoints to analyze
    if args.analyze_all:
        if not MODAL_AVAILABLE:
            print("Error: --analyze-all requires Modal. Specify checkpoints manually for local mode.")
            return

        # List all checkpoint directories
        from scripts.model_post_training.modal_resnet_finetune import CHECKPOINT_DIR
        checkpoint_paths = [
            p.name for p in CHECKPOINT_DIR.iterdir()
            if p.is_dir() and (p / "training_config.json").exists()
        ]

        if not checkpoint_paths:
            print(f"No checkpoints found in {CHECKPOINT_DIR}")
            return

        print(f"Found {len(checkpoint_paths)} checkpoints")

    elif args.checkpoint:
        checkpoint_paths = args.checkpoint
    else:
        parser.print_help()
        return

    # Run analysis
    analyze_all_checkpoints(checkpoint_paths, output_dir)

    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
