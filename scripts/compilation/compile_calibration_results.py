#!/usr/bin/env python3
"""
Compile calibration results from cross-species evaluations.
"""
import json
from pathlib import Path

def compile_calibration_results(base_dir: Path) -> dict:
    """
    Compile calibration results from directory structure.

    Returns:
        dict with structure: {task: {species: {metric: value}}}
    """
    results = {}

    # Define the structure we expect
    task_mappings = {
        "merge_action": {
            "path_key": "parquet",
            "species_map": {
                "zebrafish": "zebrafish-merge-parquet",
                "human": "human-merge-parquet",
                "fly": "fly-merge-parquet",
            }
        },
        "merge_error_identification": {
            "path_key": "species_generalization_parquets",
            "species_map": {
                "fly": "fly_merge_7500nm_parquet",
                "zebrafish": "zebrafish_merge_7500nm_parquet",
                "human": "human_merge_7500nm_parquet",
            }
        },
        "split_action": {
            "path_key": "splits",
            "species_map": {
                "fly": "fly-splits",
                "zebrafish": "zebrafish-splits",
                "human": "human-splits",
            }
        }
    }

    # Iterate through each task
    for task, config in task_mappings.items():
        results[task] = {}

        # Find calibration summary files for this task
        task_dir = base_dir / task
        if not task_dir.exists():
            print(f"Warning: Task directory not found: {task_dir}")
            continue

        # Look for calibration_summary.json files
        for summary_file in task_dir.rglob("calibration_summary.json"):
            # Try to determine species from path
            path_str = str(summary_file)

            # Extract species from the path
            species = None
            for sp, dataset_name in config["species_map"].items():
                if dataset_name in path_str:
                    species = sp
                    break

            if species is None:
                print(f"Warning: Could not determine species for {summary_file}")
                continue

            # Load the summary
            with open(summary_file) as f:
                data = json.load(f)

            # Extract metrics from the first checkpoint
            checkpoint_name = list(data.keys())[0]
            checkpoint_data = data[checkpoint_name]

            metrics = checkpoint_data.get("metrics", {})

            results[task][species] = {
                "ece": metrics.get("ece"),
                "mce": metrics.get("mce"),
                "brier_score": metrics.get("brier_score"),
                "accuracy": checkpoint_data.get("accuracy"),
            }

            print(f"Loaded {task} - {species}: ECE={metrics.get('ece', 'N/A'):.3f}")

    return results

def main():
    base_dir = Path("calibration_results_cross_species")

    results = compile_calibration_results(base_dir)

    # Save compiled results
    output_file = Path("evaluation_results/final_data/calibration_cross_species_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved compiled results to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("Calibration Cross-Species Summary (ECE)")
    print("="*80)

    for task in ["merge_action", "merge_error_identification", "split_action"]:
        if task not in results:
            continue

        task_name = task.replace("_", " ").title()
        print(f"\n{task_name}:")

        for species in ["mouse", "fly", "zebrafish", "human"]:
            if species not in results[task]:
                print(f"  {species.capitalize():12s}: N/A")
            else:
                ece = results[task][species].get("ece")
                if ece is not None:
                    print(f"  {species.capitalize():12s}: ECE={ece:.3f}")
                else:
                    print(f"  {species.capitalize():12s}: N/A")

if __name__ == "__main__":
    main()
