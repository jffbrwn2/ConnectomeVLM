#!/usr/bin/env python3
"""
Cross-species evaluation script for linear probe and ResNet/ViT/ConvNeXt models.

Evaluates all models on fly, mouse, human, and zebrafish datasets for Figure 3A.

Usage:
    python scripts/analysis/evaluate_cross_species.py --model-type linear_probe --species fly --task merge_action
    python scripts/analysis/evaluate_cross_species.py --model-type resnet --checkpoint resnet50_resnet50_merge_action_test --species zebrafish --task merge_action

    # Batch evaluation
    python scripts/analysis/evaluate_cross_species.py --batch --output cross_species_results.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Species to dataset path mapping (task-specific)
# Format: SPECIES_DATASETS[species][task] = dataset_path
# None means use default (mouse) dataset
SPECIES_DATASETS = {
    "fly": {
        "merge_action": "fly-merge-parquet",
        "merge_error_identification": "fly_merge_identification_jan27",  # Will be extracted from tar.zst
        "split_action": "split_generalization_parquets/fly-splits",
        "endpoint_error_identification_with_em": None,  # Not available
    },
    "zebrafish": {
        "merge_action": "zebrafish-merge-parquet",
        "merge_error_identification": "zebrafish_merge_identification_jan27",  # Will be extracted from tar.zst
        "split_action": "fish-jan24",  # Will be extracted from tar.zst
        "endpoint_error_identification_with_em": "endpoint_zebrafish_parquet",
    },
    "human": {
        "merge_action": "human-merge-parquet",
        "merge_error_identification": "human_merge_identification_jan27",  # Will be extracted from tar.zst
        "split_action": "split_generalization_parquets/human-splits",
        "endpoint_error_identification_with_em": "endpoint_human_parquet",
    },
    "liconn": {
        "merge_action": None,  # Not available
        "merge_error_identification": None,  # Not available
        "split_action": None,  # junction_evaluation_parquet_jan25-train (maybe?)
        "endpoint_error_identification_with_em": "endpoint_identification-jan25-train",  # Will be extracted (41GB!)
    },
    "mouse": {  # Training data (default)
        "merge_action": None,
        "merge_error_identification": None,
        "split_action": None,
        "endpoint_error_identification_with_em": None,
    },
}

# Task name mappings
TASKS = [
    "endpoint_error_identification_with_em",
    "merge_action",
    "merge_error_identification",
    "split_action",
]

# ResNet checkpoints (frozen and finetuned)
RESNET_CHECKPOINTS = {
    "endpoint_error_identification_with_em": {
        "frozen": "resnet50_resnet50_endpoint_error_identification_with_em_test_frozen",
        "finetuned": "resnet50_resnet50_endpoint_em_test",
        "small": "resnet50_endpoint_error_identification_with_em_20260128_034320",
    },
    "merge_action": {
        "frozen": "resnet50_resnet50_merge_action_test_frozen",
        "finetuned": "resnet50_resnet50_merge_action_test",
        "small": "resnet50_resnet50_merge_action_test_small",  # Will be replaced with class-balanced version
    },
    "merge_error_identification": {
        "frozen": "resnet50_resnet50_merge_error_identification_test_frozen",
        "finetuned": "resnet50_resnet50_merge_error_identification_test",
        "small": "resnet50_merge_error_identification_20260128_040048",
    },
    "split_action": {
        "frozen": "resnet50_resnet50_split_action_test_frozen",
        "finetuned": "resnet50_resnet50_split_action_test",
        "small": "resnet50_resnet50_split_action_test_small",  # Will be replaced with class-balanced version
    },
}

# ViT checkpoints (if available)
VIT_CHECKPOINTS = {
    "merge_action": {
        "frozen": "vit_l_32_resnet50_merge_action_test_frozen_wide_resnet101_2",
    },
}

# ConvNeXt checkpoints (if available)
CONVNEXT_CHECKPOINTS = {
    "merge_action": {
        "frozen": "convnext_large_resnet50_merge_action_test_frozen_convnext_large",
    },
}


def evaluate_resnet_cross_species(
    checkpoint: str,
    task: str,
    species: str,
    output_dir: Path,
) -> Dict:
    """
    Evaluate a ResNet checkpoint on a cross-species dataset.

    Calls Modal evaluation function with dataset_path parameter.
    """
    from datetime import datetime
    import subprocess

    # Get task-specific dataset path for this species
    species_datasets = SPECIES_DATASETS.get(species, {})
    dataset_path = species_datasets.get(task)

    if dataset_path is None and species != "mouse":
        print(f"  ⚠️  No dataset available for {task} on {species}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nEvaluating {checkpoint} on {species} dataset...")
    print(f"  Task: {task}")
    print(f"  Dataset: {dataset_path or 'default (mouse)'}")

    # Build modal command
    cmd = [
        "modal", "run",
        "scripts/model-post-training/modal_resnet_finetune.py::evaluate",
        "--checkpoint", checkpoint,
        "--task", task,
    ]

    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])

    # Run evaluation
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"  ✗ Failed: {result.stderr}")
            return None

        print(f"  ✓ Completed")

        # Find the saved result file
        eval_dir = Path("evaluation_results/resnet_evaluations")
        if eval_dir.exists():
            # Find most recent file matching this checkpoint
            files = sorted(eval_dir.glob(f"{checkpoint}*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                with open(files[0]) as f:
                    data = json.load(f)
                return {
                    "checkpoint": checkpoint,
                    "task": task,
                    "species": species,
                    "dataset_path": dataset_path,
                    "accuracy": data.get("accuracy"),
                    "balanced_accuracy": data.get("balanced_accuracy"),
                    "timestamp": timestamp,
                }
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout after 10 minutes")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    return None


def evaluate_linear_probe_cross_species(
    task: str,
    species: str,
    output_dir: Path,
) -> Dict:
    """
    Evaluate linear probe on a cross-species dataset.

    This requires:
    1. Extracting features using frozen SigLIP encoder
    2. Loading saved linear probe weights
    3. Computing predictions and accuracy

    For now, this would need to be implemented via Modal function.
    """
    # Get task-specific dataset path for this species
    species_datasets = SPECIES_DATASETS.get(species, {})
    dataset_path = species_datasets.get(task)

    if dataset_path is None and species != "mouse":
        print(f"  ⚠️  No dataset available for {task} on {species}")
        return None

    print(f"\nEvaluating linear probe on {species} dataset...")
    print(f"  Task: {task}")
    print(f"  Dataset: {dataset_path or 'default (mouse)'}")

    # TODO: Implement linear probe cross-species evaluation
    # This would involve:
    # 1. modal run with feature extraction + saved probe weights
    # 2. Evaluate on cross-species dataset

    print(f"  ⚠️  Linear probe cross-species evaluation not yet implemented")
    return None


def batch_evaluate(output_file: str = "evaluation_results/cross_species_results.json"):
    """
    Run batch evaluation of all models on all species for all tasks.
    """
    results = []
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Cross-Species Evaluation - Batch Mode")
    print("="*80)

    # Evaluate ResNet models
    for task in TASKS:
        if task not in RESNET_CHECKPOINTS:
            continue

        for model_type, checkpoint in RESNET_CHECKPOINTS[task].items():
            for species in SPECIES_DATASETS.keys():
                if species == "fly":  # Skip fly for now (training data)
                    continue

                result = evaluate_resnet_cross_species(
                    checkpoint=checkpoint,
                    task=task,
                    species=species,
                    output_dir=output_path.parent,
                )

                if result:
                    result["model_architecture"] = "resnet50"
                    result["model_type"] = model_type
                    results.append(result)

                    # Save intermediate results
                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)

    # Evaluate ViT models
    for task, checkpoints in VIT_CHECKPOINTS.items():
        for model_type, checkpoint in checkpoints.items():
            for species in SPECIES_DATASETS.keys():
                if species == "fly":
                    continue

                result = evaluate_resnet_cross_species(
                    checkpoint=checkpoint,
                    task=task,
                    species=species,
                    output_dir=output_path.parent,
                )

                if result:
                    result["model_architecture"] = "vit_l_32"
                    result["model_type"] = model_type
                    results.append(result)

                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)

    # Evaluate ConvNeXt models
    for task, checkpoints in CONVNEXT_CHECKPOINTS.items():
        for model_type, checkpoint in checkpoints.items():
            for species in SPECIES_DATASETS.keys():
                if species == "fly":
                    continue

                result = evaluate_resnet_cross_species(
                    checkpoint=checkpoint,
                    task=task,
                    species=species,
                    output_dir=output_path.parent,
                )

                if result:
                    result["model_architecture"] = "convnext_large"
                    result["model_type"] = model_type
                    results.append(result)

                    with open(output_path, 'w') as f:
                        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"Batch evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"Total evaluations: {len(results)}")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-species model evaluation")
    parser.add_argument("--model-type", choices=["resnet", "vit", "convnext", "linear_probe"],
                        help="Model architecture to evaluate")
    parser.add_argument("--checkpoint", help="Checkpoint name (for ResNet/ViT/ConvNeXt)")
    parser.add_argument("--task", choices=TASKS, help="Task to evaluate")
    parser.add_argument("--species", choices=list(SPECIES_DATASETS.keys()), help="Species dataset")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation on all models/species/tasks")
    parser.add_argument("--output", default="evaluation_results/cross_species_results.json",
                        help="Output JSON file for results")

    args = parser.parse_args()

    if args.batch:
        batch_evaluate(args.output)
    elif args.model_type and args.task and args.species:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.model_type == "linear_probe":
            result = evaluate_linear_probe_cross_species(args.task, args.species, output_dir)
        elif args.checkpoint:
            result = evaluate_resnet_cross_species(args.checkpoint, args.task, args.species, output_dir)
        else:
            print("Error: --checkpoint required for ResNet/ViT/ConvNeXt evaluation")
            sys.exit(1)

        if result:
            print("\nResult:")
            print(json.dumps(result, indent=2))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
