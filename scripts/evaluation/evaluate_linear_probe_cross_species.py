#!/usr/bin/env python3
"""
Evaluate linear probes on cross-species datasets.

Since we don't have pretrained weights saved, we'll train linear probes
directly on the cross-species datasets using their features.

Usage:
    python scripts/analysis/evaluate_linear_probe_cross_species.py \
        --task merge_action \
        --species fly \
        --output evaluation_results/linear_probe_cross_species/

Note: This script assumes SigLIP features have been extracted and cached.
If not, you'll need to extract them first using the feature extraction script.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

# Dataset mappings (same as in evaluate_cross_species.py)
SPECIES_DATASETS = {
    "fly": {
        "merge_action": "fly-merge-parquet",
        "split_action": "split_generalization_parquets/fly-splits",
    },
    "zebrafish": {
        "merge_action": "zebrafish-merge-parquet",
        "split_action": "fish-jan24",
    },
    "human": {
        "merge_action": "human-merge-parquet",
        "split_action": "split_generalization_parquets/human-splits",
    },
}

def evaluate_linear_probe(task: str, species: str, output_dir: Path):
    """
    Train and evaluate a linear probe on cross-species data.

    This requires feature extraction first. For now, print what would be done.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = SPECIES_DATASETS.get(species, {}).get(task)
    if not dataset_path:
        print(f"No dataset available for {task} on {species}")
        return None

    print(f"\nLinear Probe Evaluation:")
    print(f"  Task: {task}")
    print(f"  Species: {species}")
    print(f"  Dataset: {dataset_path}")
    print()

    # TODO: Implement this with Modal
    # Steps needed:
    # 1. Extract SigLIP features from cross-species dataset
    # 2. Train linear probe on those features
    # 3. Evaluate and return results

    print("  ⚠️  Feature extraction and training not yet implemented")
    print("  This requires:")
    print("    1. Modal function to extract SigLIP features from dataset")
    print("    2. Train logistic regression on features")
    print("    3. Evaluate on holdout set")
    print()
    print("  Alternative approach: Just use the full dataset for training")
    print("  (no train/test split needed for cross-species generalization test)")

    return None

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate linear probes on cross-species datasets"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["merge_action", "split_action"],
        help="Task to evaluate",
    )
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        choices=["fly", "zebrafish", "human"],
        help="Species to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results/linear_probe_cross_species"),
        help="Output directory for results",
    )

    args = parser.parse_args()

    result = evaluate_linear_probe(args.task, args.species, args.output)

    if result:
        output_file = args.output / f"linear_probe_{args.task}_{args.species}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
