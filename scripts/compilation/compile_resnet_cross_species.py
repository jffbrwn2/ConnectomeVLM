#!/usr/bin/env python3
"""
Compile ResNet cross-species evaluation results.

Reads evaluation_results/resnet_evaluations/*.json and creates
a summary of cross-species transfer results.

Usage:
    python scripts/analysis/compile_resnet_cross_species.py
"""

import json
from pathlib import Path
from collections import defaultdict

# Dataset path to species mapping
DATASET_TO_SPECIES = {
    "fly-merge-parquet": "fly",
    "zebrafish-merge-parquet": "zebrafish",
    "human-merge-parquet": "human",
    "split_generalization_parquets/fly-splits": "fly",
    "split_generalization_parquets-fly-splits": "fly",
    "split_generalization_parquets/fish-jan24": "zebrafish",
    "split_generalization_parquets-fish-jan24": "zebrafish",
    "split_generalization_parquets/human-splits": "human",
    "split_generalization_parquets-human-splits": "human",
}

SPECIES_DISPLAY_NAMES = {
    "mouse": "Mouse",
    "fly": "Fly",
    "zebrafish": "Zebrafish",
    "human": "Human",
}

def main():
    eval_dir = Path("evaluation_results/resnet_evaluations")
    output_file = Path("evaluation_results/final_data/resnet_cross_species_results.json")

    # Load mouse (training dataset) results
    mouse_results = {}
    for json_file in eval_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # Skip cross-species
        if data.get('dataset_path'):
            continue

        task = data['task']
        checkpoint = data['checkpoint']

        # Determine model type
        if 'frozen' in checkpoint:
            model_type = 'frozen'
        elif 'small' in checkpoint or '20260128' in checkpoint:
            model_type = 'small'
        else:
            model_type = 'finetuned'

        if task not in mouse_results:
            mouse_results[task] = {}

        mouse_results[task][model_type] = {
            'balanced_accuracy': data['balanced_accuracy'],
            'accuracy': data['accuracy'],
            'correct': data['correct'],
            'total': data['total'],
        }

    # Load cross-species results
    cross_species_results = defaultdict(lambda: defaultdict(lambda: {}))

    for json_file in eval_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # Only process cross-species
        dataset_path = data.get('dataset_path')
        if not dataset_path:
            continue

        # Map dataset to species
        species = DATASET_TO_SPECIES.get(dataset_path)
        if not species:
            print(f"Warning: Unknown dataset path: {dataset_path}")
            continue

        task = data['task']
        checkpoint = data['checkpoint']

        # Determine model type
        if 'frozen' in checkpoint:
            model_type = 'frozen'
        elif 'small' in checkpoint or '20260128' in checkpoint:
            model_type = 'small'
        else:
            model_type = 'finetuned'

        cross_species_results[task][species][model_type] = {
            'balanced_accuracy': data['balanced_accuracy'],
            'accuracy': data['accuracy'],
            'correct': data['correct'],
            'total': data['total'],
        }

    # Compile final results structure
    final_results = {}

    for task in sorted(set(list(mouse_results.keys()) + list(cross_species_results.keys()))):
        final_results[task] = {
            'mouse': mouse_results.get(task, {}),
        }

        # Add cross-species results
        for species in ['fly', 'zebrafish', 'human']:
            final_results[task][species] = cross_species_results[task].get(species, {})

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Saved cross-species results to {output_file}")

    # Print summary
    print("\nCross-Species Transfer Summary (Balanced Accuracy):")
    print("="*100)

    for task in sorted(final_results.keys()):
        print(f"\n{task}:")
        print(f"  {'Species':12s} {'Frozen':>8s} {'Finetuned':>10s} {'Small':>8s}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")

        for species in ['mouse', 'fly', 'zebrafish', 'human']:
            species_data = final_results[task].get(species, {})

            frozen_acc = species_data.get('frozen', {}).get('balanced_accuracy')
            finetuned_acc = species_data.get('finetuned', {}).get('balanced_accuracy')
            small_acc = species_data.get('small', {}).get('balanced_accuracy')

            frozen_str = f"{frozen_acc*100:6.1f}%" if frozen_acc is not None else "   -"
            finetuned_str = f"{finetuned_acc*100:8.1f}%" if finetuned_acc is not None else "     -"
            small_str = f"{small_acc*100:6.1f}%" if small_acc is not None else "   -"

            print(f"  {SPECIES_DISPLAY_NAMES[species]:12s} {frozen_str:>8s} {finetuned_str:>10s} {small_str:>8s}")

if __name__ == "__main__":
    main()
