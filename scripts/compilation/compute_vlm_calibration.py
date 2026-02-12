#!/usr/bin/env python3
"""
Compute calibration metrics (ECE, MCE, Brier score) for VLM predictions.

Computes metrics for both original answers and derived answers (from analysis).

Usage:
    python scripts/analysis/compute_vlm_calibration.py \
        --input evaluation_results/final_data/vlm_evaluations_with_derived/*.json \
        --output evaluation_results/final_data/vlm_calibration_results.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict


def extract_confidence(response: str) -> Optional[float]:
    """Extract confidence score from response (0-100 scale)."""
    if not response:
        return None

    match = re.search(r'<confidence>\s*(\d+)\s*</confidence>', response, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0  # Convert to 0-1 scale
    return None


def compute_ece(confidences: np.ndarray, predictions: np.ndarray,
                ground_truths: np.ndarray, n_bins: int = 10) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error.

    Args:
        confidences: Array of confidence scores (0-1)
        predictions: Array of binary predictions
        ground_truths: Array of binary ground truths
        n_bins: Number of bins for calibration

    Returns:
        ece: Expected calibration error
        mce: Maximum calibration error
        bin_accuracies: Accuracy in each bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n_in_bin = in_bin.sum()

        if n_in_bin > 0:
            # Compute accuracy and average confidence in bin
            accuracy_in_bin = (predictions[in_bin] == ground_truths[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Update ECE and MCE
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += (n_in_bin / len(confidences)) * calibration_error
            mce = max(mce, calibration_error)

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(n_in_bin)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return ece, mce, np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_counts)


def compute_brier_score(confidences: np.ndarray, ground_truths: np.ndarray) -> float:
    """Compute Brier score."""
    return np.mean((confidences - ground_truths) ** 2)


def process_vlm_file(file_path: Path) -> Dict:
    """Process a single VLM evaluation file with derived answers."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    task_name = data.get('task_name', '')
    predictions = data.get('predictions', [])
    dataset_source = data.get('dataset_source', '')

    # Parse species from dataset_source
    species = None
    dataset_lower = dataset_source.lower()

    if 'fly' in dataset_lower:
        species = 'fly'
    elif 'zebrafish' in dataset_lower:
        species = 'zebrafish'
    elif 'human' in dataset_lower:
        species = 'human'
    else:
        # Default datasets without species prefix are mouse test set
        species = 'mouse'

    print(f"  Processing: {file_path.name}")
    print(f"    Task: {task_name}, Species: {species} (from: {dataset_source})")

    # Collect per-vote data for original predictions
    original_confidences = []
    original_predictions = []
    original_ground_truths = []

    # Collect per-vote data for derived predictions
    derived_confidences = []
    derived_predictions = []
    derived_ground_truths = []

    for pred in predictions:
        gt = pred['ground_truth']
        all_responses = pred.get('all_responses', [pred.get('response', '')])
        all_preds = pred.get('all_predictions', [pred.get('predicted')])
        all_derived = pred.get('all_derived_answers', [])

        for vote_idx, (response, original_pred) in enumerate(zip(all_responses, all_preds)):
            confidence = extract_confidence(response)

            if confidence is not None and original_pred is not None:
                # Original predictions
                original_confidences.append(confidence)
                original_predictions.append(original_pred)
                original_ground_truths.append(gt)

                # Derived predictions (use same confidence as original)
                if vote_idx < len(all_derived) and all_derived[vote_idx] is not None:
                    derived_confidences.append(confidence)
                    derived_predictions.append(all_derived[vote_idx])
                    derived_ground_truths.append(gt)

    # Convert to numpy arrays
    original_confidences = np.array(original_confidences)
    original_predictions = np.array(original_predictions, dtype=float)
    original_ground_truths = np.array(original_ground_truths, dtype=float)

    derived_confidences = np.array(derived_confidences)
    derived_predictions = np.array(derived_predictions, dtype=float)
    derived_ground_truths = np.array(derived_ground_truths, dtype=float)

    results = {
        'task_name': task_name,
        'species': species,
        'file': file_path.name,
    }

    # Compute original metrics
    if len(original_confidences) > 0:
        ece, mce, bin_accs, bin_confs, bin_counts = compute_ece(
            original_confidences, original_predictions, original_ground_truths
        )
        brier = compute_brier_score(original_confidences, original_ground_truths)
        accuracy = (original_predictions == original_ground_truths).mean()

        results['original'] = {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier),
            'accuracy': float(accuracy),
            'n_samples': len(original_confidences),
            'bin_accuracies': bin_accs.tolist(),
            'bin_confidences': bin_confs.tolist(),
            'bin_counts': bin_counts.tolist(),
        }

        print(f"    Original - ECE: {ece:.4f}, Acc: {accuracy:.4f}")

    # Compute derived metrics
    if len(derived_confidences) > 0:
        ece, mce, bin_accs, bin_confs, bin_counts = compute_ece(
            derived_confidences, derived_predictions, derived_ground_truths
        )
        brier = compute_brier_score(derived_confidences, derived_ground_truths)
        accuracy = (derived_predictions == derived_ground_truths).mean()

        results['derived'] = {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier),
            'accuracy': float(accuracy),
            'n_samples': len(derived_confidences),
            'bin_accuracies': bin_accs.tolist(),
            'bin_confidences': bin_confs.tolist(),
            'bin_counts': bin_counts.tolist(),
        }

        print(f"    Derived  - ECE: {ece:.4f}, Acc: {accuracy:.4f}")

    return results


def aggregate_by_task_species(results: List[Dict]) -> Dict:
    """Aggregate results by task and species."""
    aggregated = defaultdict(lambda: defaultdict(dict))

    for result in results:
        task = result['task_name']
        species = result['species']

        if species is None:
            continue

        if 'original' in result:
            aggregated[task][species]['original'] = {
                'ece': result['original']['ece'],
                'mce': result['original']['mce'],
                'brier_score': result['original']['brier_score'],
                'accuracy': result['original']['accuracy'],
            }

        if 'derived' in result:
            aggregated[task][species]['derived'] = {
                'ece': result['derived']['ece'],
                'mce': result['derived']['mce'],
                'brier_score': result['derived']['brier_score'],
                'accuracy': result['derived']['accuracy'],
            }

    return dict(aggregated)


def main():
    parser = argparse.ArgumentParser(
        description="Compute calibration metrics for VLM predictions"
    )
    parser.add_argument(
        "--input", "-i",
        nargs='+',
        required=True,
        help="Path(s) to VLM evaluation files with derived answers"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output JSON file for calibration results"
    )

    args = parser.parse_args()

    # Expand input files
    input_files = []
    for pattern in args.input:
        matching_files = list(Path().glob(pattern))
        if matching_files:
            input_files.extend(matching_files)
        else:
            p = Path(pattern)
            if p.exists():
                input_files.append(p)

    print(f"Processing {len(input_files)} files...\n")

    # Process each file
    all_results = []
    for file_path in input_files:
        try:
            result = process_vlm_file(file_path)
            all_results.append(result)
            print()
        except Exception as e:
            print(f"  Error processing {file_path}: {e}\n")
            continue

    # Aggregate by task and species
    aggregated = aggregate_by_task_species(all_results)

    # Save results
    output_data = {
        'per_file_results': all_results,
        'aggregated': aggregated,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved calibration results to: {args.output}")

    # Print summary
    print("\n" + "="*80)
    print("VLM Calibration Summary")
    print("="*80)

    for task, species_data in sorted(aggregated.items()):
        print(f"\n{task}:")
        for species, metrics in sorted(species_data.items()):
            print(f"  {species}:")
            if 'original' in metrics:
                orig = metrics['original']
                print(f"    Original - ECE: {orig['ece']:.4f}, Acc: {orig['accuracy']:.4f}")
            if 'derived' in metrics:
                deriv = metrics['derived']
                print(f"    Derived  - ECE: {deriv['ece']:.4f}, Acc: {deriv['accuracy']:.4f}")


if __name__ == "__main__":
    main()
