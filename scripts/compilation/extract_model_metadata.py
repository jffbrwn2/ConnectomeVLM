#!/usr/bin/env python3
"""
Extract model metadata (training samples, parameters) for all trained models.
"""

import json
from pathlib import Path

# Model parameter counts (from torchvision documentation)
MODEL_PARAMS = {
    "siglip2_so400m": 1_152_000_000,  # 1.15B base model (SigLIP-2)
    "resnet50": 23_512_130,  # 23.5M (with 2-class head)
    "resnet18": 11_177_538,
    "resnet34": 21_284_162,
    "resnet101": 42_503_234,
    "resnet152": 58_155_586,
    "wide_resnet101_2": 126_886_784,
    "vit_l_32": 304_326_632,
    "vit_h_14": 632_045_800,
    "convnext_large": 197_767_336,
}

# Training sample counts from test set sizes (0.8/0.1/0.1 split)
TRAIN_SAMPLES_FROM_TEST = {
    "endpoint_error_identification_with_em": 3704,  # test=463 * 8
    "merge_action": 10552,  # test=1319 * 8 (reported as 10402)
    "merge_error_identification": 9600,  # test=1200 * 8
    "split_action": 13296,  # test=1662 * 8
}

# ResNet training types
RESNET_MODELS = {
    "frozen": {
        "train_samples": TRAIN_SAMPLES_FROM_TEST,  # Uses all data, only trains classifier
        "trainable_params": 4098,  # Just the final linear layer (2048*2 + 2)
        "total_params": MODEL_PARAMS["resnet50"],
    },
    "small": {
        "train_samples": {task: 512 for task in TRAIN_SAMPLES_FROM_TEST},
        "trainable_params": MODEL_PARAMS["resnet50"],
        "total_params": MODEL_PARAMS["resnet50"],
    },
    "finetuned": {
        "train_samples": TRAIN_SAMPLES_FROM_TEST,
        "trainable_params": MODEL_PARAMS["resnet50"],
        "total_params": MODEL_PARAMS["resnet50"],
    },
}

# Linear probe
LINEAR_PROBE_PARAMS = 2304  # 1152 features * 2 classes
LINEAR_PROBE_TOTAL_PARAMS = MODEL_PARAMS["siglip2_so400m"] + LINEAR_PROBE_PARAMS  # Base + probe

def save_metadata():
    """Save model metadata for easy access."""
    metadata = {
        "model_parameters": MODEL_PARAMS,
        "train_samples": TRAIN_SAMPLES_FROM_TEST,
        "resnet_configs": RESNET_MODELS,
        "linear_probe_trainable_params": LINEAR_PROBE_PARAMS,
        "linear_probe_total_params": LINEAR_PROBE_TOTAL_PARAMS,
    }

    output_file = Path("paper/model_metadata.json")
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("Model Metadata Summary")
    print("="*60)

    print("\nTraining Samples per Task:")
    for task, samples in TRAIN_SAMPLES_FROM_TEST.items():
        print(f"  {task}: {samples:,}")

    print("\nModel Parameters:")
    for model, params in MODEL_PARAMS.items():
        print(f"  {model}: {params:,} ({params/1e6:.1f}M)")

    print(f"\nLinear Probe: {LINEAR_PROBE_PARAMS:,} ({LINEAR_PROBE_PARAMS/1e3:.1f}K)")

    print("\nResNet Configurations:")
    for config_type, config in RESNET_MODELS.items():
        print(f"  {config_type}:")
        print(f"    Trainable params: {config['trainable_params']:,}")
        sample_range = set(config['train_samples'].values())
        if len(sample_range) == 1:
            print(f"    Train samples: {list(sample_range)[0]:,}")
        else:
            print(f"    Train samples: varies by task")


if __name__ == "__main__":
    save_metadata()
