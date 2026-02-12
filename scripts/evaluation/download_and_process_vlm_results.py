#!/usr/bin/env python3
"""
Download VLM evaluation results from Modal and add to final data files.

This script:
1. Reads final_final_vlm_data.json to get list of VLM evaluations
2. Downloads evaluation files from Modal if not already present
3. Computes majority vote accuracy for each evaluation
4. Updates relevant data files with VLM results
"""

import json
import subprocess
from pathlib import Path
from collections import Counter


def download_vlm_results():
    """Download VLM evaluation results from Modal."""

    # Load VLM data manifest
    manifest_path = Path("evaluation_results/final_data/final_final_vlm_data.json")
    with open(manifest_path) as f:
        vlm_data = json.load(f)

    # Create output directory
    output_dir = Path("evaluation_results/final_data/vlm_evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all filenames
    all_files = []
    for species, tasks in vlm_data.items():
        for task, filename in tasks.items():
            if filename:  # Skip empty strings
                all_files.append(filename)

    print(f"Found {len(all_files)} VLM evaluation files to download")

    # Download each file from Modal
    downloaded_files = []
    for filename in all_files:
        local_path = output_dir / filename

        if local_path.exists():
            print(f"✓ Already exists: {filename}")
            downloaded_files.append(filename)
        else:
            print(f"Downloading: {filename}")
            try:
                # Try to download from qwen-finetune-results volume
                cmd = [
                    "modal", "volume", "get", "qwen-finetune-results",
                    filename, str(local_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"✓ Downloaded: {filename}")
                    downloaded_files.append(filename)
                else:
                    print(f"✗ Failed to download {filename}: {result.stderr}")
            except Exception as e:
                print(f"✗ Error downloading {filename}: {e}")

    print(f"\nDownloaded {len(downloaded_files)}/{len(all_files)} files")
    return output_dir, downloaded_files


def compute_majority_vote_accuracy(eval_file_path):
    """
    Compute majority vote accuracy from VLM evaluation file.

    Args:
        eval_file_path: Path to evaluation JSON file

    Returns:
        dict with accuracy, balanced_accuracy, correct, total, individual_accuracy, majority_accuracy
    """
    with open(eval_file_path) as f:
        data = json.load(f)

    # Extract individual and majority accuracies if available
    individual_accuracy = data.get("individual_accuracy")
    majority_accuracy = data.get("majority_accuracy")

    # Check if top-level structure has accuracy already computed
    if "accuracy" in data and "num_correct" in data and "num_samples" in data:
        # Use precomputed values
        correct = data["num_correct"]
        total = data["num_samples"]
        accuracy = data["accuracy"]
    else:
        # Fallback: compute from predictions array
        correct = 0
        total = 0

        predictions = data.get("predictions", [])
        for item in predictions:
            if item.get("correct") is not None:
                total += 1
                if item["correct"]:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

    # Compute balanced accuracy from predictions
    per_class_correct = {}
    per_class_total = {}

    predictions = data.get("predictions", [])
    for item in predictions:
        ground_truth = item.get("ground_truth")
        is_correct = item.get("correct")

        if ground_truth is None or is_correct is None:
            continue

        # Convert ground_truth to hashable type
        gt_key = str(ground_truth)

        # Track per-class accuracy
        if gt_key not in per_class_correct:
            per_class_correct[gt_key] = 0
            per_class_total[gt_key] = 0

        per_class_total[gt_key] += 1
        if is_correct:
            per_class_correct[gt_key] += 1

    # Compute balanced accuracy (average of per-class accuracies)
    class_accuracies = []
    for gt_class in per_class_total:
        if per_class_total[gt_class] > 0:
            class_acc = per_class_correct[gt_class] / per_class_total[gt_class]
            class_accuracies.append(class_acc)

    balanced_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0

    result = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "correct": correct,
        "total": total,
    }

    # Add individual and majority accuracies if available
    if individual_accuracy is not None:
        result["individual_accuracy"] = individual_accuracy
    if majority_accuracy is not None:
        result["majority_accuracy"] = majority_accuracy

    return result


def update_resnet_results(vlm_results):
    """Update resnet_results.json with VLM data."""

    results_path = Path("evaluation_results/final_data/resnet_results.json")
    with open(results_path) as f:
        resnet_results = json.load(f)

    # Map task names
    task_mapping = {
        "split_error_correction": "merge_action_resnet",
        "merge_error_identification": "merge_error_identification_resnet",
        "split_action": "split_action_resnet",
        "split_error_identification": "endpoint_error_identification_with_em_resnet",
    }

    # Add VLM results for mouse (test set)
    for task, task_key in task_mapping.items():
        if task in vlm_results.get("mouse", {}):
            if task_key not in resnet_results:
                resnet_results[task_key] = {}

            resnet_results[task_key]["vlm"] = vlm_results["mouse"][task]

    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(resnet_results, f, indent=2)

    print(f"✓ Updated {results_path}")


def update_cross_species_results(vlm_results):
    """Update resnet_cross_species_results.json with VLM data."""

    results_path = Path("evaluation_results/final_data/resnet_cross_species_results.json")
    with open(results_path) as f:
        cross_species_results = json.load(f)

    # Map internal task names to file keys
    task_mapping = {
        "split_error_correction": "merge_action",
        "merge_error_identification": "merge_error_identification",
        "split_action": "split_action",
        "split_error_identification": "endpoint_error_identification",
    }

    # Add VLM results for each species
    for species, tasks in vlm_results.items():
        for task_internal, metrics in tasks.items():
            task_key = task_mapping.get(task_internal)
            if not task_key:
                continue

            if task_key not in cross_species_results:
                cross_species_results[task_key] = {}

            if species not in cross_species_results[task_key]:
                cross_species_results[task_key][species] = {}

            cross_species_results[task_key][species]["vlm"] = metrics

    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(cross_species_results, f, indent=2)

    print(f"✓ Updated {results_path}")


def main():
    print("=" * 80)
    print("VLM Results Download and Processing")
    print("=" * 80)

    # Step 1: Download files from Modal
    print("\nStep 1: Downloading VLM evaluation files from Modal...")
    vlm_dir, downloaded_files = download_vlm_results()

    if not downloaded_files:
        print("\n✗ No files downloaded. Exiting.")
        return

    # Step 2: Load manifest and process each file
    print("\n" + "=" * 80)
    print("Step 2: Computing majority vote accuracies...")
    print("=" * 80)

    manifest_path = Path("evaluation_results/final_data/final_final_vlm_data.json")
    with open(manifest_path) as f:
        vlm_manifest = json.load(f)

    vlm_results = {}

    for species, tasks in vlm_manifest.items():
        vlm_results[species] = {}

        for task, filename in tasks.items():
            if not filename:
                continue

            eval_path = vlm_dir / filename
            if not eval_path.exists():
                print(f"✗ File not found: {filename}")
                continue

            print(f"\nProcessing: {species} - {task}")
            print(f"  File: {filename}")

            try:
                metrics = compute_majority_vote_accuracy(eval_path)
                vlm_results[species][task] = metrics

                print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
                print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
            except Exception as e:
                print(f"  ✗ Error processing file: {e}")

    # Step 3: Update data files
    print("\n" + "=" * 80)
    print("Step 3: Updating data files...")
    print("=" * 80)

    update_resnet_results(vlm_results)
    update_cross_species_results(vlm_results)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary of VLM Results")
    print("=" * 80)

    for species in ["mouse", "fly", "zebrafish", "human", "liconn"]:
        if species not in vlm_results or not vlm_results[species]:
            continue

        print(f"\n{species.capitalize()}:")
        for task, metrics in vlm_results[species].items():
            task_display = task.replace("_", " ").title()
            print(f"  {task_display}: {metrics['balanced_accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

    print("\n" + "=" * 80)
    print("✓ Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
