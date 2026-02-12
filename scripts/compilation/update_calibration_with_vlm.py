#!/usr/bin/env python3
"""
Update calibration_cross_species_results.json to include VLM calibration data.
"""

import json
from pathlib import Path


def main():
    # Load VLM calibration results
    vlm_cal_path = Path("evaluation_results/final_data/vlm_calibration_all_results.json")
    with open(vlm_cal_path) as f:
        vlm_data = json.load(f)

    # Load existing cross-species calibration (ResNet)
    cross_species_path = Path("evaluation_results/final_data/calibration_cross_species_results.json")
    with open(cross_species_path) as f:
        cross_species = json.load(f)

    # Add VLM data to cross-species results
    vlm_aggregated = vlm_data['aggregated']

    for task, species_data in vlm_aggregated.items():
        if task not in cross_species:
            cross_species[task] = {}

        for species, metrics in species_data.items():
            if species not in cross_species[task]:
                cross_species[task][species] = {}

            # Add original VLM metrics
            if 'original' in metrics:
                cross_species[task][species]['vlm'] = metrics['original']

            # Add derived VLM metrics
            if 'derived' in metrics:
                cross_species[task][species]['vlm_derived'] = metrics['derived']

    # Save updated results
    with open(cross_species_path, 'w') as f:
        json.dump(cross_species, f, indent=2)

    print(f"Updated {cross_species_path} with VLM calibration data")

    # Print summary
    print("\nVLM Calibration Added:")
    for task, species_data in vlm_aggregated.items():
        print(f"\n{task}:")
        for species in sorted(species_data.keys()):
            has_orig = 'original' in species_data[species]
            has_deriv = 'derived' in species_data[species]
            print(f"  {species}: original={has_orig}, derived={has_deriv}")


if __name__ == "__main__":
    main()
