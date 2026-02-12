#!/usr/bin/env python3
"""
Compile ResNet evaluation results with balanced accuracy.

Reads evaluation_results/resnet_evaluations/*.json and creates
a summary JSON file for plotting.

Usage:
    python scripts/analysis/compile_resnet_results.py
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    eval_dir = Path("evaluation_results/resnet_evaluations")
    output_file = Path("evaluation_results/final_data/resnet_results.json")
    train_samples_file = Path("evaluation_results/final_data/resnet_train_samples.json")

    # Load training sample counts
    with open(train_samples_file) as f:
        train_samples_data = json.load(f)

    # Parse all mouse dataset evaluations (exclude cross-species)
    # Keep track of timestamps to use most recent for each (task, model_type)
    results = defaultdict(lambda: {})
    result_timestamps = defaultdict(lambda: {})

    for json_file in eval_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # Skip cross-species (has dataset_path that's not None/default)
        if data.get('dataset_path'):
            continue

        checkpoint = data['checkpoint']

        # Extract task from checkpoint name if task field is None
        task = data['task']
        if task is None:
            # Parse from checkpoint name like "resnet50_merge_action_20260128_081833"
            parts = checkpoint.split('_')
            if 'merge' in checkpoint and 'error' in checkpoint:
                task = 'merge_error_identification'
            elif 'split' in checkpoint:
                task = 'split_action'
            elif 'merge' in checkpoint:
                task = 'merge_action'
            elif 'endpoint' in checkpoint:
                if 'em' in checkpoint:
                    task = 'endpoint_error_identification_with_em'
                else:
                    task = 'endpoint_error_identification'

        # Determine model type - check training config for more accurate classification
        model_type = None

        # Try to load training config from checkpoint volume to determine frozen vs finetuned
        # For now, use heuristics:
        # - If has '_frozen' suffix or timestamp + 082101 -> frozen
        # - If has timestamp 081833/081834/081841 -> small (512 samples)
        # - Otherwise finetuned

        if 'frozen' in checkpoint or checkpoint.endswith('082101'):
            model_type = 'frozen_model'
        elif '081833' in checkpoint or '081834' in checkpoint or '081841' in checkpoint:
            model_type = 'small_model'
        elif 'small' in checkpoint:
            model_type = 'small_model'
        elif '082058' in checkpoint or ('20260128' in checkpoint and model_type is None):
            model_type = 'fully_finetuned_model'
        else:
            model_type = 'fully_finetuned_model'

        # Store result (only if newer timestamp or no existing result)
        task_key = f"{task}_resnet"
        timestamp = data.get('timestamp', '00000000_000000')

        # Check if we should use this result (most recent timestamp)
        existing_timestamp = result_timestamps.get(task_key, {}).get(model_type, '00000000_000000')

        if timestamp >= existing_timestamp:
            result_data = {
                'accuracy': data['accuracy'],
                'balanced_accuracy': data['balanced_accuracy'],
                'correct': data['correct'],
                'total': data['total'],
            }

            # Add training samples
            if model_type == 'small_model':
                # Small models were trained on 512 samples
                result_data['train_samples'] = 512
            elif task in train_samples_data:
                # Frozen/finetuned use full dataset
                result_data['train_samples'] = train_samples_data[task]['train_samples']

            results[task_key][model_type] = result_data
            result_timestamps[task_key][model_type] = timestamp

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(dict(results), f, indent=2)

    print(f"Saved results to {output_file}")

    # Print summary
    print("\nSummary:")
    print("="*80)
    for task_key in sorted(results.keys()):
        task = task_key.replace('_resnet', '')
        print(f"\n{task}:")
        for model_type in ['frozen_model', 'fully_finetuned_model', 'small_model']:
            if model_type in results[task_key]:
                data = results[task_key][model_type]
                print(f"  {model_type:25s}: acc={data['accuracy']:.4f} "
                      f"balanced={data['balanced_accuracy']:.4f} "
                      f"({data['correct']}/{data['total']})")
            else:
                print(f"  {model_type:25s}: NOT AVAILABLE")

if __name__ == "__main__":
    main()
