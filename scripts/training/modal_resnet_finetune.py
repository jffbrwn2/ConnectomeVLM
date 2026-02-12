"""
ResNet-50 baseline fine-tuning script for classification tasks.

This script provides a simple CNN baseline using a pretrained ResNet-50,
fine-tuned on the same tasks as the VLM models for comparison.

For tasks with multiple images per sample, images are arranged into a grid
before being fed to the model.
"""

import modal
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
import time

# Add src/ directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import task_configs (works both locally and in Modal)
try:
    from task_configs import get_task, list_tasks
except ImportError:
    from environment.task_configs import get_task, list_tasks

# Paths for storage (reuse existing volumes where possible)
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/datasets")
CHECKPOINT_DIR = Path("/checkpoints/resnet")
RESULTS_DIR = Path("/results")

# Create volumes
model_volume = modal.Volume.from_name("qwen-finetune-models", create_if_missing=True)
dataset_volume = modal.Volume.from_name("qwen-finetune-datasets", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("resnet-finetune-checkpoints", create_if_missing=True)
results_volume = modal.Volume.from_name("qwen-finetune-results", create_if_missing=True)

# Define the Modal image with PyTorch and torchvision
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "datasets",
        "pandas",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "wandb",
        "scikit-learn",
        "tqdm",
        "numpy",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Add task_configs.py to the container
    .add_local_file("src/environment/task_configs.py", remote_path="/root/task_configs.py")
)

app = modal.App("resnet-finetune-baseline", image=image)


@dataclass
class ResNetTrainingConfig:
    """Configuration for ResNet-50 fine-tuning."""

    # Model settings
    model_name: str = "resnet50"  # resnet18, resnet34, resnet50, resnet101, resnet152
    pretrained: bool = True
    freeze_backbone: bool = False  # If True, only train the classifier head

    # Image settings
    image_size: int = 224  # Standard ResNet input size
    grid_layout: str = "auto"  # "auto", "1x1", "1x2", "2x2", "1x3", "3x1", etc.
    tile_size: int = None  # If set, image_size = tile_size * (rows x cols). E.g., tile_size=512 for 1x3 grid → 512x1536

    # Training hyperparameters
    num_train_epochs: int = 10
    max_steps: int = None  # If set, stop after this many optimizer steps (overrides epochs)
    batch_size: int = 256  # ResNet is small, can use large batches on A10G
    learning_rate: float = 1e-3  # Scale LR with batch size
    weight_decay: float = 1e-4
    warmup_epochs: int = 1

    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    step_lr_gamma: float = 0.1
    step_lr_step_size: int = 7
    plateau_factor: float = 0.1
    plateau_patience: int = 3

    # Dataset settings
    num_samples: int = None  # None = use all data
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    # Absolute sample counts (take precedence over ratios)
    train_samples: int = None  # Absolute number of training samples
    val_samples: int = None    # Absolute number of validation samples
    # Note: test_samples is computed as remainder when train_samples/val_samples are set

    # Other settings
    seed: int = 42
    num_workers: int = 8  # More workers to keep up with large batches

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5

    # Class balancing
    class_balance: bool = False
    class_balance_method: str = "oversample"  # "oversample" or "undersample"

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = "resnet-finetune"
    wandb_run_name: str = None

    # Task settings
    task_name: str = "merge_action"


def create_image_grid_for_resnet(images, target_size=224, layout="auto", tile_size=None):
    """
    Arrange multiple images into a grid suitable for ResNet input.

    Args:
        images: List of PIL images
        target_size: Target size for the final grid (will be square if tile_size not specified)
        layout: Grid layout - "auto", "1x1", "1x2", "2x2", "1x3", "3x1", etc.
        tile_size: If specified, each tile will be tile_size x tile_size, creating a rectangular grid

    Returns:
        Single PIL image (square if tile_size=None, rectangular otherwise)
    """
    from PIL import Image
    import math

    n_images = len(images)

    if n_images == 0:
        raise ValueError("No images provided")

    # Determine grid layout
    if layout == "auto":
        # Choose layout based on number of images
        if n_images == 1:
            rows, cols = 1, 1
        elif n_images == 2:
            rows, cols = 1, 2
        elif n_images == 3:
            rows, cols = 1, 3
        elif n_images == 4:
            rows, cols = 2, 2
        elif n_images <= 6:
            rows, cols = 2, 3
        elif n_images <= 9:
            rows, cols = 3, 3
        elif n_images <= 12:
            rows, cols = 3, 4
        else:
            cols = math.ceil(math.sqrt(n_images))
            rows = math.ceil(n_images / cols)
    else:
        # Parse explicit layout like "2x3"
        parts = layout.lower().split("x")
        rows, cols = int(parts[0]), int(parts[1])

    # Ensure we have enough cells
    if rows * cols < n_images:
        raise ValueError(f"Grid {rows}x{cols} cannot fit {n_images} images")

    # Calculate dimensions
    if tile_size is not None:
        # Use tile_size to create rectangular grid
        cell_width = tile_size
        cell_height = tile_size
        canvas_width = cols * tile_size
        canvas_height = rows * tile_size
    else:
        # Use square grid with target_size
        cell_width = target_size // cols
        cell_height = target_size // rows
        canvas_width = target_size
        canvas_height = target_size

    if n_images == 1:
        # Single image - just resize to canvas size
        img = images[0]
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), (128, 128, 128))

    # Place images
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        # Convert to RGB and resize to cell size
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)

        # Paste onto canvas
        x_offset = col * cell_width
        y_offset = row * cell_height
        canvas.paste(img_resized, (x_offset, y_offset))

    return canvas


def get_class_labels_for_task(task, dataset):
    """
    Determine the unique class labels for a task's dataset.

    Returns:
        Tuple of (label_to_idx dict, idx_to_label dict, num_classes int)
    """
    # Collect all unique ground truth values
    unique_labels = set()
    for sample in dataset:
        gt = task.get_ground_truth(sample)
        # Handle different types of ground truth
        if isinstance(gt, bool):
            unique_labels.add(gt)
        elif isinstance(gt, str):
            unique_labels.add(gt)
        elif isinstance(gt, (int, float)):
            unique_labels.add(gt)
        else:
            # For complex types (lists, tuples), skip - not a classification task
            raise ValueError(
                f"Task {task.name} has non-classification ground truth type: {type(gt)}. "
                f"ResNet baseline only supports classification tasks."
            )

    # Sort labels for consistent ordering
    sorted_labels = sorted(unique_labels, key=lambda x: (str(type(x)), str(x)))

    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(sorted_labels)

    return label_to_idx, idx_to_label, num_classes


class ImageGridDataset:
    """
    Dataset that converts task samples to image grids with class labels.
    """

    def __init__(self, dataset, task, label_to_idx, image_size=224, grid_layout="auto", transform=None, tile_size=None):
        self.dataset = dataset
        self.task = task
        self.label_to_idx = label_to_idx
        self.image_size = image_size
        self.grid_layout = grid_layout
        self.transform = transform
        self.tile_size = tile_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        try:
            # Get images from task
            images = self.task.get_images(sample)

            # Create grid
            grid_image = create_image_grid_for_resnet(
                images, target_size=self.image_size, layout=self.grid_layout, tile_size=self.tile_size
            )

            # Apply transforms
            if self.transform:
                grid_image = self.transform(grid_image)

            # Get label
            gt = self.task.get_ground_truth(sample)
            label = self.label_to_idx[gt]

            return grid_image, label
        except (OSError, IOError) as e:
            # Return None for corrupted images - will be filtered in collate_fn
            print(f"Warning: Skipping corrupted image at index {idx}: {e}")
            return None


def collate_fn_skip_none(batch):
    """
    Custom collate function that filters out None values (corrupted images).
    """
    import torch
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    # Use default collate for the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)


def _merge_overlapping_groups(sample_group_keys):
    """
    Merge group keys that share any element using union-find.

    If sample A has group key (X, Y) and sample B has key (X, Z), they should
    be in the same group because they share element X. This function returns
    a mapping from sample index to merged group ID.

    Args:
        sample_group_keys: List of group keys (tuples or single values) per sample

    Returns:
        List of merged group IDs (integers) per sample
    """
    # Union-find data structure
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build union-find structure from group keys
    for key in sample_group_keys:
        if key is None:
            continue
        # Convert to tuple if not already
        if not isinstance(key, (tuple, list)):
            key = (key,)
        # Union all elements in this key together
        elements = list(key)
        for i in range(1, len(elements)):
            union(elements[0], elements[i])

    # Map each sample to its merged group ID
    merged_groups = []
    for key in sample_group_keys:
        if key is None:
            merged_groups.append(None)
        else:
            if not isinstance(key, (tuple, list)):
                key = (key,)
            # Use the root of the first element as the group ID
            merged_groups.append(find(key[0]))

    return merged_groups


def split_dataset_for_resnet(dataset, config, task, ground_truth_labels):
    """Perform stratified train/val/test split.

    If task.get_split_group_key() returns non-None values, splits are done at the
    group level to prevent data leakage (all samples in a group go to the same split).

    For tasks where group keys contain multiple IDs (e.g., segment identity with
    segment1_id and segment2_id), connected components are used to merge groups
    that share any ID.

    Supports two modes:
    1. Absolute sample counts: If train_samples/val_samples are set, use those exact
       numbers with test being the remainder.
    2. Ratio-based: Use train_split_ratio/val_split_ratio/test_split_ratio.

    Returns:
        Tuple of (train_indices, val_indices, test_indices, split_indices_dict)
    """
    from sklearn.model_selection import train_test_split
    from collections import Counter, defaultdict
    import random

    # Determine if using absolute counts or ratios
    use_absolute_counts = config.train_samples is not None or config.val_samples is not None

    if use_absolute_counts:
        # Absolute sample count mode
        train_samples = config.train_samples or 0
        val_samples = config.val_samples or 0

        total = len(dataset)
        if train_samples + val_samples > total:
            raise ValueError(
                f"train_samples ({train_samples}) + val_samples ({val_samples}) = {train_samples + val_samples} "
                f"exceeds dataset size ({total})"
            )

        print(f"\nUsing absolute sample counts:")
        print(f"  Train: {train_samples}, Val: {val_samples}, Test: {total - train_samples - val_samples} (remainder)")

        # Shuffle indices
        random.seed(config.seed)
        all_indices = list(range(total))
        random.shuffle(all_indices)

        # Split by count
        train_indices = all_indices[:train_samples]
        val_indices = all_indices[train_samples:train_samples + val_samples]
        test_indices = all_indices[train_samples + val_samples:]

        print(f"  Actual split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        # Build split_indices dict
        has_parquet_idx = '_original_parquet_idx' in dataset.column_names if hasattr(dataset, 'column_names') else False
        if has_parquet_idx:
            train_parquet = [dataset[i]['_original_parquet_idx'] for i in train_indices]
            val_parquet = [dataset[i]['_original_parquet_idx'] for i in val_indices]
            test_parquet = [dataset[i]['_original_parquet_idx'] for i in test_indices]
        else:
            train_parquet, val_parquet, test_parquet = train_indices, val_indices, test_indices

        split_indices = {
            'train_indices': train_parquet,
            'val_indices': val_parquet,
            'test_indices': test_parquet,
        }
        return train_indices, val_indices, test_indices, split_indices

    # Ratio-based mode (original logic)
    needs_val = config.val_split_ratio > 0
    needs_test = config.test_split_ratio > 0

    if not (needs_val or needs_test):
        print("\nNo train/val/test split")
        print(f"  Train samples: {len(dataset)}")
        all_indices = list(range(len(dataset)))
        return all_indices, [], [], {'train_indices': all_indices, 'val_indices': [], 'test_indices': []}

    print(f"\nPerforming stratified split...")

    # Check if task uses group-based splitting
    sample_group_keys = [task.get_split_group_key(sample) for sample in dataset]
    use_group_splitting = any(k is not None for k in sample_group_keys)

    if use_group_splitting:
        print(f"  (Using group-based splitting to prevent data leakage)")

        # Check if task requires connected component merging (e.g., segment_identity)
        if task.uses_connected_component_splitting():
            print(f"  (Using connected components to merge overlapping groups)")
            merged_group_ids = _merge_overlapping_groups(sample_group_keys)
            # Use merged group IDs instead of raw keys
            sample_group_keys = merged_group_ids

        # Build mapping from group key to sample indices
        group_to_indices = defaultdict(list)
        for idx, key in enumerate(sample_group_keys):
            group_to_indices[key].append(idx)

        unique_groups = list(group_to_indices.keys())
        print(f"  Unique groups: {len(unique_groups)}")
        print(f"  Samples per group: min={min(len(v) for v in group_to_indices.values())}, "
              f"max={max(len(v) for v in group_to_indices.values())}, "
              f"avg={len(dataset)/len(unique_groups):.1f}")

        # Split groups (not samples)
        # No stratification - groups have mixed labels, and class_balance handles train set after
        avg_samples_per_group = len(dataset) / len(unique_groups)

        test_groups = []
        if config.test_split_ratio > 0:
            train_val_groups, test_groups = train_test_split(
                unique_groups,
                test_size=config.test_split_ratio,
                random_state=config.seed,
                shuffle=True
            )
        else:
            train_val_groups = unique_groups

        val_groups = []
        if config.val_split_ratio > 0:
            val_ratio = config.val_split_ratio / (config.train_split_ratio + config.val_split_ratio)
            train_groups, val_groups = train_test_split(
                train_val_groups,
                test_size=val_ratio,
                random_state=config.seed,
                shuffle=True
            )
        else:
            train_groups = train_val_groups

        # Convert groups back to sample indices
        train_set = set(train_groups)
        val_set = set(val_groups)
        test_set = set(test_groups)

        train_indices = [idx for idx, key in enumerate(sample_group_keys) if key in train_set]
        val_indices = [idx for idx, key in enumerate(sample_group_keys) if key in val_set]
        test_indices = [idx for idx, key in enumerate(sample_group_keys) if key in test_set]

        print(f"  Groups: Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")

    else:
        # Original behavior: split by sample index
        # Check for multi-teacher expanded data
        has_original_idx = '_original_sample_idx' in dataset.column_names if hasattr(dataset, 'column_names') else False

        if has_original_idx:
            print(f"  (Splitting by original samples to avoid data leakage)")
            original_indices = sorted(set(dataset['_original_sample_idx']))
            idx_to_label = {}
            for i, sample in enumerate(dataset):
                orig_idx = sample['_original_sample_idx']
                if orig_idx not in idx_to_label:
                    idx_to_label[orig_idx] = ground_truth_labels[i]
            original_labels = [idx_to_label[idx] for idx in original_indices]
        else:
            original_indices = list(range(len(dataset)))
            original_labels = ground_truth_labels
            idx_to_label = dict(enumerate(ground_truth_labels))

        # Split test set first
        test_orig_indices = []
        if config.test_split_ratio > 0:
            train_val_orig_indices, test_orig_indices = train_test_split(
                original_indices,
                test_size=config.test_split_ratio,
                random_state=config.seed,
                stratify=original_labels,
                shuffle=True
            )
            train_val_labels = [idx_to_label[idx] for idx in train_val_orig_indices]
        else:
            train_val_orig_indices = original_indices
            train_val_labels = original_labels

        # Split validation
        val_orig_indices = []
        if config.val_split_ratio > 0:
            val_ratio = config.val_split_ratio / (config.train_split_ratio + config.val_split_ratio)
            train_orig_indices, val_orig_indices = train_test_split(
                train_val_orig_indices,
                test_size=val_ratio,
                random_state=config.seed,
                stratify=train_val_labels,
                shuffle=True
            )
        else:
            train_orig_indices = train_val_orig_indices

        # Convert to dataset indices
        if has_original_idx:
            train_set, val_set, test_set = set(train_orig_indices), set(val_orig_indices), set(test_orig_indices)
            train_indices = [i for i, s in enumerate(dataset) if s['_original_sample_idx'] in train_set]
            val_indices = [i for i, s in enumerate(dataset) if s['_original_sample_idx'] in val_set]
            test_indices = [i for i, s in enumerate(dataset) if s['_original_sample_idx'] in test_set]
        else:
            train_indices, val_indices, test_indices = train_orig_indices, val_orig_indices, test_orig_indices

    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")

    # Extract original parquet indices for reproducible splits
    has_parquet_idx = '_original_parquet_idx' in dataset.column_names if hasattr(dataset, 'column_names') else False
    if has_parquet_idx:
        train_parquet_indices = [dataset[i]['_original_parquet_idx'] for i in train_indices]
        val_parquet_indices = [dataset[i]['_original_parquet_idx'] for i in val_indices] if val_indices else []
        test_parquet_indices = [dataset[i]['_original_parquet_idx'] for i in test_indices] if test_indices else []
    else:
        train_parquet_indices = train_indices
        val_parquet_indices = val_indices if val_indices else []
        test_parquet_indices = test_indices if test_indices else []

    split_indices = {
        'train_indices': train_parquet_indices,
        'val_indices': val_parquet_indices,
        'test_indices': test_parquet_indices,
    }

    return train_indices, val_indices, test_indices, split_indices


def apply_class_balancing_indices(indices, dataset, task, seed, method="oversample"):
    """
    Apply class balancing to a list of indices.

    Uses task.get_balance_group() to determine how samples are grouped for balancing.
    This allows tasks to define custom grouping logic (e.g., merge_action_multiple_choice
    groups 'none' vs 'not_none' instead of individual answer letters).

    Returns:
        Balanced list of indices (may contain duplicates for oversampling)
    """
    from collections import Counter
    import random

    random.seed(seed)

    # Get labels for these indices using get_balance_group (not get_ground_truth)
    labels = [task.get_balance_group(dataset[i]) for i in indices]
    class_counts = Counter(labels)

    print(f"\n  Original distribution:")
    for label, count in sorted(class_counts.items(), key=lambda x: str(x[0])):
        print(f"    {label}: {count}")

    # Group indices by class
    class_indices = {}
    for idx, label in zip(indices, labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    if method == "oversample":
        max_count = max(class_counts.values())
        balanced = []
        for label, idx_list in class_indices.items():
            if len(idx_list) < max_count:
                additional = random.choices(idx_list, k=max_count - len(idx_list))
                balanced.extend(idx_list + additional)
            else:
                balanced.extend(idx_list)
    elif method == "undersample":
        min_count = min(class_counts.values())
        balanced = []
        for label, idx_list in class_indices.items():
            if len(idx_list) > min_count:
                sampled = random.sample(idx_list, min_count)
                balanced.extend(sampled)
            else:
                balanced.extend(idx_list)
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    random.shuffle(balanced)

    # Print balanced distribution
    balanced_labels = [task.get_ground_truth(dataset[i]) for i in balanced]
    balanced_counts = Counter(balanced_labels)
    print(f"  Balanced distribution:")
    for label, count in sorted(balanced_counts.items(), key=lambda x: str(x[0])):
        print(f"    {label}: {count}")

    return balanced


def create_resnet_model(model_name, num_classes, pretrained=True, freeze_backbone=False, image_size=224):
    """
    Create a vision model for classification.

    Args:
        model_name: One of:
            - ResNet: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
            - Wide ResNet: "wide_resnet50_2", "wide_resnet101_2"
            - ResNeXt: "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"
            - ViT: "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14"
            - ConvNeXt: "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: If True, freeze all layers except the final classifier
        image_size: Input image size (used for ViT models to set position embeddings)

    Returns:
        PyTorch model
    """
    import torch.nn as nn
    from torchvision import models
    from torchvision.models import (
        # ResNet
        ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
        ResNet101_Weights, ResNet152_Weights,
        Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights,
        ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights, ResNeXt101_64X4D_Weights,
        # ViT
        ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights, ViT_H_14_Weights,
        # ConvNeXt
        ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
    )

    # Model and weights mapping
    # Approximate parameter counts:
    #   ResNet: resnet18 ~11M, resnet34 ~21M, resnet50 ~25M, resnet101 ~44M, resnet152 ~60M
    #   Wide ResNet: wide_resnet50_2 ~68M, wide_resnet101_2 ~126M
    #   ResNeXt: resnext50_32x4d ~25M, resnext101_32x8d ~88M, resnext101_64x4d ~83M
    #   ViT: vit_b_16 ~86M, vit_b_32 ~88M, vit_l_16 ~304M, vit_l_32 ~306M, vit_h_14 ~632M
    #   ConvNeXt: convnext_tiny ~29M, convnext_small ~50M, convnext_base ~89M, convnext_large ~198M
    model_configs = {
        # Standard ResNet
        "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, "resnet"),
        "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1, "resnet"),
        "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2, "resnet"),
        "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2, "resnet"),
        "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V2, "resnet"),
        # Wide ResNet (wider channels)
        "wide_resnet50_2": (models.wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V2, "resnet"),
        "wide_resnet101_2": (models.wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2, "resnet"),
        # ResNeXt (grouped convolutions)
        "resnext50_32x4d": (models.resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V2, "resnet"),
        "resnext101_32x8d": (models.resnext101_32x8d, ResNeXt101_32X8D_Weights.IMAGENET1K_V2, "resnet"),
        "resnext101_64x4d": (models.resnext101_64x4d, ResNeXt101_64X4D_Weights.IMAGENET1K_V1, "resnet"),
        # ViT (Vision Transformer)
        "vit_b_16": (models.vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, "vit"),
        "vit_b_32": (models.vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, "vit"),
        "vit_l_16": (models.vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1, "vit"),
        "vit_l_32": (models.vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1, "vit"),
        "vit_h_14": (models.vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1, "vit"),  # Largest ~632M
        # ConvNeXt (modern CNN)
        "convnext_tiny": (models.convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "convnext"),
        "convnext_small": (models.convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1, "convnext"),
        "convnext_base": (models.convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, "convnext"),
        "convnext_large": (models.convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1, "convnext"),
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")

    model_fn, weights, model_type = model_configs[model_name]

    # Load model
    if pretrained and model_type == "vit" and image_size != 224:
        # ViT pretrained weights are tied to image_size=224. Load at 224 first,
        # then create a new model at the target size and transfer weights,
        # interpolating position embeddings.
        import torch
        import torch.nn.functional as F

        print(f"Loading pretrained ViT at 224, then resizing pos embeddings to {image_size}")
        pretrained_model = model_fn(weights=weights)
        model = model_fn(weights=None, image_size=image_size)

        # Transfer all weights except position embedding
        pretrained_state = pretrained_model.state_dict()
        model_state = model.state_dict()

        for key in pretrained_state:
            if "pos_embedding" not in key:
                if key in model_state and pretrained_state[key].shape == model_state[key].shape:
                    model_state[key] = pretrained_state[key]

        # Interpolate position embeddings
        pos_key = "encoder.pos_embedding"
        if pos_key in pretrained_state:
            old_pos = pretrained_state[pos_key]  # (1, old_seq_len, hidden_dim)
            new_seq_len = model_state[pos_key].shape[1]
            old_seq_len = old_pos.shape[1]

            if old_seq_len != new_seq_len:
                # Separate class token and patch embeddings
                cls_token = old_pos[:, :1, :]  # (1, 1, hidden_dim)
                patch_pos = old_pos[:, 1:, :]  # (1, old_patches, hidden_dim)

                # Reshape to 2D grid, interpolate, reshape back
                old_grid_size = int(patch_pos.shape[1] ** 0.5)
                new_grid_size = int((new_seq_len - 1) ** 0.5)
                patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
                patch_pos = F.interpolate(patch_pos, size=(new_grid_size, new_grid_size), mode="bicubic", align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, -1)

                model_state[pos_key] = torch.cat([cls_token, patch_pos], dim=1)
                print(f"  Interpolated pos embeddings: {old_seq_len} -> {new_seq_len} tokens ({old_grid_size}x{old_grid_size} -> {new_grid_size}x{new_grid_size} grid)")
            else:
                model_state[pos_key] = old_pos

        model.load_state_dict(model_state)
        print(f"Loaded {model_name} with interpolated pretrained weights for image_size={image_size}")
        del pretrained_model, pretrained_state
    elif pretrained:
        model = model_fn(weights=weights)
        print(f"Loaded {model_name} with ImageNet pretrained weights")
    else:
        vit_kwargs = {"image_size": image_size} if model_type == "vit" and image_size != 224 else {}
        model = model_fn(weights=None, **vit_kwargs)
        print(f"Loaded {model_name} without pretrained weights")

    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("Froze backbone parameters")

    # Replace classifier head (different for each model type)
    if model_type == "resnet":
        # ResNet, Wide ResNet, ResNeXt: model.fc
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_type == "vit":
        # ViT: model.heads.head
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif model_type == "convnext":
        # ConvNeXt: model.classifier[2] (Sequential: LayerNorm, Flatten, Linear)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Replaced classifier: {in_features} -> {num_classes}")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def get_transforms(image_size, is_training=True):
    """
    Get image transforms for ResNet.

    Args:
        image_size: Target image size
        is_training: If True, include data augmentation

    Returns:
        torchvision transforms
    """
    from torchvision import transforms

    # ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


@app.function(
    gpu="A10G",
    timeout=3600 * 24,  # 24 hours
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def finetune_resnet(config: ResNetTrainingConfig = ResNetTrainingConfig()):
    """
    Fine-tune a pretrained ResNet model on a classification task.

    Args:
        config: ResNetTrainingConfig with all training parameters

    Returns:
        Path to saved model checkpoint
    """
    import json
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    from collections import Counter
    from datetime import datetime
    from tqdm import tqdm
    import os

    os.environ["HF_HOME"] = str(MODEL_DIR)

    # =========================================================================
    # 1. Setup
    # =========================================================================
    print("=" * 60)
    print("ResNet Fine-tuning Baseline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load task
    task = get_task(config.task_name)
    print(f"\nTask: {task.name} - {task.description}")

    # Compute image size from tile_size if specified
    if config.tile_size is not None:
        # Determine grid layout
        grid_layout = config.grid_layout
        if grid_layout == "auto":
            # Auto-determine based on number of images
            n_imgs = task.num_images
            if n_imgs == 1:
                grid_layout = "1x1"
            elif n_imgs == 2:
                grid_layout = "1x2"
            elif n_imgs == 3:
                grid_layout = "1x3"
            elif n_imgs == 4:
                grid_layout = "2x2"
            elif n_imgs == 6:
                grid_layout = "2x3"
            elif n_imgs == 9:
                grid_layout = "3x3"
            elif n_imgs == 12:
                grid_layout = "3x4"
            else:
                # Default to row layout
                grid_layout = f"1x{n_imgs}"
            print(f"  Auto-detected grid layout: {grid_layout}")

        # Parse grid layout (e.g., "2x3" → 2 rows, 3 cols)
        rows, cols = map(int, grid_layout.split('x'))
        computed_height = config.tile_size * rows
        computed_width = config.tile_size * cols

        # Update image_size to match grid
        # For rectangular grids, use the larger dimension (ResNet will handle via AdaptiveAvgPool)
        config.image_size = max(computed_height, computed_width)
        config.grid_layout = grid_layout

        print(f"  Using tile_size={config.tile_size} with {grid_layout} grid")
        print(f"  Computed image size: {computed_height}x{computed_width}")
        print(f"  Set image_size to: {config.image_size}")

    # Initialize W&B
    wandb = None
    if config.use_wandb:
        import wandb as wb
        wandb = wb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"{config.model_name}-{config.task_name}",
            config=asdict(config),
        )

    # =========================================================================
    # 2. Load Dataset
    # =========================================================================
    print(f"\nLoading dataset...")
    dataset = task.load_dataset(cache_dir=str(DATASET_DIR))
    dataset = task.filter_dataset(dataset)
    print(f"  Loaded {len(dataset)} samples after filtering")

    # Limit samples if specified
    if config.num_samples and config.num_samples < len(dataset):
        dataset = dataset.shuffle(seed=config.seed)
        dataset = dataset.select(range(config.num_samples))
        print(f"  Limited to {config.num_samples} samples")

    # Get class labels
    label_to_idx, idx_to_label, num_classes = get_class_labels_for_task(task, dataset)
    print(f"\nClasses ({num_classes}):")
    for idx, label in idx_to_label.items():
        print(f"  {idx}: {label}")

    # =========================================================================
    # 3. Analyze Class Distribution and Split Dataset
    # =========================================================================
    # Use get_balance_group for class distribution (matches Qwen script behavior)
    ground_truth_labels = [task.get_balance_group(sample) for sample in dataset]

    print("\nClass distribution:")
    for label, count in sorted(Counter(ground_truth_labels).items(), key=lambda x: str(x[0])):
        print(f"  {label}: {count} samples")

    train_indices, val_indices, test_indices, split_indices = split_dataset_for_resnet(
        dataset, config, task, ground_truth_labels
    )

    # Apply class balancing to training set if enabled
    if config.class_balance:
        print(f"\nApplying class balancing ({config.class_balance_method})...")
        train_indices = apply_class_balancing_indices(
            train_indices, dataset, task, config.seed, config.class_balance_method
        )

    # =========================================================================
    # 4. Create DataLoaders
    # =========================================================================
    print("\nCreating data loaders...")

    train_transform = get_transforms(config.image_size, is_training=True)
    val_transform = get_transforms(config.image_size, is_training=False)

    # Create datasets with appropriate transforms
    # Note: We wrap the HF dataset with our ImageGridDataset
    class IndexedDataset:
        """Wrapper to select specific indices from a dataset."""
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    train_subset = IndexedDataset(dataset, train_indices)
    val_subset = IndexedDataset(dataset, val_indices) if val_indices else None
    test_subset = IndexedDataset(dataset, test_indices) if test_indices else None

    train_ds = ImageGridDataset(
        train_subset, task, label_to_idx,
        image_size=config.image_size, grid_layout=config.grid_layout,
        transform=train_transform, tile_size=config.tile_size
    )
    val_ds = ImageGridDataset(
        val_subset, task, label_to_idx,
        image_size=config.image_size, grid_layout=config.grid_layout,
        transform=val_transform, tile_size=config.tile_size
    ) if val_subset else None
    test_ds = ImageGridDataset(
        test_subset, task, label_to_idx,
        image_size=config.image_size, grid_layout=config.grid_layout,
        transform=val_transform, tile_size=config.tile_size
    ) if test_subset else None

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
        collate_fn=collate_fn_skip_none
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        collate_fn=collate_fn_skip_none
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        collate_fn=collate_fn_skip_none
    ) if test_ds else None

    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")
    if test_loader:
        print(f"  Test batches: {len(test_loader)}")

    # =========================================================================
    # 5. Create Model
    # =========================================================================
    print(f"\nCreating model...")
    model = create_resnet_model(
        config.model_name, num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        image_size=config.image_size,
    )
    model = model.to(device)

    # =========================================================================
    # 6. Setup Training
    # =========================================================================
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=config.learning_rate,
            momentum=0.9, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # If max_steps is set, compute how many epochs we need
    steps_per_epoch = len(train_loader)
    if config.max_steps is not None:
        import math
        config.num_train_epochs = math.ceil(config.max_steps / steps_per_epoch)
        print(f"  max_steps={config.max_steps}, steps_per_epoch={steps_per_epoch}, "
              f"running {config.num_train_epochs} epochs")

    # Learning rate scheduler
    if config.lr_scheduler == "cosine":
        T_max = config.max_steps if config.max_steps else config.num_train_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max
        )
    elif config.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_lr_step_size, gamma=config.step_lr_gamma
        )
    elif config.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.plateau_factor,
            patience=config.plateau_patience
        )
    else:
        scheduler = None

    # Warmup scheduler
    warmup_scheduler = None
    if config.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=config.warmup_epochs * len(train_loader)
        )

    # =========================================================================
    # 7. Training Loop
    # =========================================================================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    global_step = 0
    max_steps_reached = False

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.wandb_run_name or f"{config.task_name}_{timestamp}"
    checkpoint_dir = CHECKPOINT_DIR / f"{config.model_name}_{run_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = checkpoint_dir / "training_config.json"
    with open(str(config_path), "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Save class mapping
    class_mapping_path = checkpoint_dir / "class_mapping.json"
    with open(str(class_mapping_path), "w") as f:
        # Convert keys to strings for JSON
        mapping = {
            "label_to_idx": {str(k): v for k, v in label_to_idx.items()},
            "idx_to_label": {str(k): str(v) for k, v in idx_to_label.items()},
        }
        json.dump(mapping, f, indent=2)

    # Save test indices (use parquet indices from split_indices for reproducibility)
    test_indices_path = checkpoint_dir / "test_indices.json"
    with open(str(test_indices_path), "w") as f:
        json.dump(split_indices, f, indent=2)

    checkpoint_volume.commit()

    for epoch in range(config.num_train_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_train_epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            # Skip None batches (all images were corrupted)
            if images is None:
                continue
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Warmup scheduler step
            if warmup_scheduler and epoch < config.warmup_epochs:
                warmup_scheduler.step()

            global_step += 1
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * train_correct / train_total:.1f}%"
            })

            # Step-based LR scheduling when using max_steps
            if config.max_steps and scheduler and config.lr_scheduler == "cosine":
                scheduler.step()

            if config.max_steps and global_step >= config.max_steps:
                max_steps_reached = True
                break

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        val_loss = 0.0
        val_acc = 0.0
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    # Skip None batches (all images were corrupted)
                    if images is None:
                        continue
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total

        # Update learning rate scheduler (skip if using per-step cosine with max_steps)
        if scheduler and not (config.max_steps and config.lr_scheduler == "cosine"):
            if config.lr_scheduler == "plateau" and val_loader:
                scheduler.step(val_loss)
            elif config.lr_scheduler != "plateau":
                scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{config.num_train_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        if val_loader:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
        print(f"  LR: {current_lr:.2e}")

        # Log to W&B
        if wandb:
            log_dict = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "learning_rate": current_lr,
            }
            if val_loader:
                log_dict["val_loss"] = val_loss
                log_dict["val_acc"] = val_acc
            wandb.log(log_dict)

        # Save best model
        if val_loader and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0

            # Save checkpoint
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, str(best_path))
            checkpoint_volume.commit()
            print(f"  Saved best model (val_acc: {val_acc:.2%})")
        elif val_loader:
            patience_counter += 1

        # Early stopping
        if config.early_stopping and patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs ({global_step} steps)")
            print(f"Best val_acc: {best_val_acc:.2%} at epoch {best_epoch}")
            break

        if max_steps_reached:
            print(f"\nReached max_steps={config.max_steps} after {epoch + 1} epochs")
            break

    # =========================================================================
    # 8. Final Evaluation
    # =========================================================================
    # Load best model for final evaluation
    if val_loader:
        best_path = checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(str(best_path))
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"\nLoaded best model from epoch {checkpoint['epoch']}")

    # Test set evaluation
    if test_loader:
        print("\n" + "=" * 60)
        print("Test Set Evaluation")
        print("=" * 60)

        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                # Skip None batches (all images were corrupted)
                if images is None:
                    continue
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = test_correct / test_total
        print(f"\nTest Accuracy: {test_acc:.2%} ({test_correct}/{test_total})")

        # Per-class accuracy
        print("\nPer-class accuracy:")
        for idx in sorted(idx_to_label.keys()):
            label = idx_to_label[idx]
            class_mask = [l == idx for l in all_labels]
            if sum(class_mask) > 0:
                class_correct = sum(p == l for p, l, m in zip(all_preds, all_labels, class_mask) if m)
                class_total = sum(class_mask)
                class_acc = class_correct / class_total
                print(f"  {label}: {class_acc:.2%} ({class_correct}/{class_total})")

        if wandb:
            wandb.log({"test_acc": test_acc})

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "num_classes": num_classes,
        "label_to_idx": {str(k): v for k, v in label_to_idx.items()},
        "idx_to_label": {str(k): str(v) for k, v in idx_to_label.items()},
    }, str(final_path))
    checkpoint_volume.commit()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Val Acc: {best_val_acc:.2%} (epoch {best_epoch})")
    if test_loader:
        print(f"Test Acc: {test_acc:.2%}")
    print(f"Checkpoint saved to: {checkpoint_dir}")
    print("=" * 60)

    if wandb:
        wandb.finish()

    return str(checkpoint_dir)


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
)
def evaluate_resnet(
    checkpoint_path: str,
    task_name: str = None,
    num_samples: int = None,
    batch_size: int = 32,
    use_test_set: bool = True,
    dataset_path: str = None,
    max_test_samples: int = 1024,
):
    """
    Evaluate a trained ResNet model.

    Args:
        checkpoint_path: Path to checkpoint directory (relative to CHECKPOINT_DIR)
        task_name: Task to evaluate on (default: same as training)
        num_samples: Number of samples to evaluate (None = all)
        batch_size: Batch size for evaluation
        use_test_set: If True, only evaluate on test set from training split
        dataset_path: Override dataset path for cross-dataset evaluation
    """
    import json
    import torch
    from torch.utils.data import DataLoader
    from collections import Counter
    import os

    os.environ["HF_HOME"] = str(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_dir = CHECKPOINT_DIR / checkpoint_path
    final_model_path = checkpoint_dir / "final_model.pt"
    best_model_path = checkpoint_dir / "best_model.pt"

    model_path = best_model_path if best_model_path.exists() else final_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {checkpoint_dir}")

    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(str(model_path), map_location=device)

    # Load config
    config_path = checkpoint_dir / "training_config.json"
    with open(str(config_path), "r") as f:
        config_dict = json.load(f)
    config = ResNetTrainingConfig(**config_dict)

    # Determine task
    task_name = task_name or config.task_name
    task = get_task(task_name)
    print(f"Task: {task.name}")

    # Load class mapping
    if "label_to_idx" in checkpoint:
        label_to_idx = {eval(k) if k.startswith("(") else (k == "True" if k in ["True", "False"] else k): v
                       for k, v in checkpoint["label_to_idx"].items()}
        idx_to_label = {int(k): eval(v) if v.startswith("(") else (v == "True" if v in ["True", "False"] else v)
                       for k, v in checkpoint["idx_to_label"].items()}
        num_classes = checkpoint["num_classes"]
    else:
        # Load from file
        class_mapping_path = checkpoint_dir / "class_mapping.json"
        with open(str(class_mapping_path), "r") as f:
            mapping = json.load(f)
        label_to_idx = {eval(k) if k.startswith("(") else (k == "True" if k in ["True", "False"] else k): v
                       for k, v in mapping["label_to_idx"].items()}
        idx_to_label = {int(k): eval(v) if v.startswith("(") else (v == "True" if v in ["True", "False"] else v)
                       for k, v in mapping["idx_to_label"].items()}
        num_classes = len(label_to_idx)

    print(f"Classes: {num_classes}")

    # Create model
    model = create_resnet_model(config.model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = task.load_dataset(cache_dir=str(DATASET_DIR), dataset_path=dataset_path)
    dataset = task.filter_dataset(dataset)
    print(f"  Loaded {len(dataset)} samples")

    # Filter to test set if requested
    if use_test_set:
        test_indices_path = checkpoint_dir / "test_indices.json"
        if test_indices_path.exists():
            with open(str(test_indices_path), "r") as f:
                indices_data = json.load(f)
            test_indices = indices_data.get("test_indices", [])
            if test_indices:
                # Filter out indices that are out of bounds
                valid_test_indices = [idx for idx in test_indices if idx < len(dataset)]
                if len(valid_test_indices) < len(test_indices):
                    print(f"  Filtered {len(test_indices) - len(valid_test_indices)} out-of-bounds test indices")

                # Cap test indices if needed (random sample to avoid ordered bias)
                if max_test_samples and len(valid_test_indices) > max_test_samples:
                    import random
                    random.seed(42)  # For reproducibility
                    valid_test_indices = random.sample(valid_test_indices, max_test_samples)
                    print(f"  Randomly sampled {max_test_samples} from test set")

                print(f"  Using {len(valid_test_indices)} test set samples")

                class IndexedDataset:
                    def __init__(self, dataset, indices):
                        self.dataset = dataset
                        self.indices = indices
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        return self.dataset[self.indices[idx]]

                dataset = IndexedDataset(dataset, valid_test_indices)
            else:
                print("  No test indices found, using full dataset")
        else:
            print("  No test_indices.json found, using full dataset")

    # Limit samples (auto-cap at max_test_samples, or use explicit num_samples)
    # Note: If we already capped via test_indices above, this won't apply
    if num_samples is None and max_test_samples and len(dataset) > max_test_samples:
        num_samples = max_test_samples
        print(f"  Auto-capping at {max_test_samples} samples")

    if num_samples and num_samples < len(dataset):
        import random
        random.seed(42)  # For reproducibility
        sampled_indices = random.sample(range(len(dataset)), num_samples)
        class LimitedDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        dataset = LimitedDataset(dataset, sampled_indices)
        print(f"  Randomly sampled {num_samples} samples")

    # Create dataloader
    val_transform = get_transforms(config.image_size, is_training=False)
    eval_ds = ImageGridDataset(
        dataset, task, label_to_idx,
        image_size=config.image_size, grid_layout=config.grid_layout,
        transform=val_transform, tile_size=config.tile_size
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=collate_fn_skip_none
    )

    # Evaluate
    print(f"\nEvaluating on {len(dataset)} samples...")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in eval_loader:
            # Skip None batches (all images were corrupted)
            if images is None:
                continue
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"\n{'=' * 60}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print("=" * 60)

    # Per-class accuracy and balanced accuracy
    print("\nPer-class accuracy:")
    class_accuracies = []
    for idx in sorted(idx_to_label.keys()):
        label = idx_to_label[idx]
        class_mask = [l == idx for l in all_labels]
        if sum(class_mask) > 0:
            class_correct = sum(p == l for p, l, m in zip(all_preds, all_labels, class_mask) if m)
            class_total = sum(class_mask)
            class_acc = class_correct / class_total
            class_accuracies.append(class_acc)
            print(f"  {label}: {class_acc:.2%} ({class_correct}/{class_total})")

    # Balanced accuracy: average of per-class accuracies
    balanced_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0
    print(f"\nBalanced Accuracy: {balanced_accuracy:.2%}")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "correct": correct,
        "total": total
    }


# =============================================================================
# Calibration Analysis Functions
# =============================================================================

def compute_calibration_metrics(confidences, predictions, labels, n_bins=10):
    """
    Compute calibration metrics: ECE, MCE, Brier score.

    Args:
        confidences: Array of confidence scores (max softmax probability)
        predictions: Array of predicted class indices
        labels: Array of ground truth class indices
        n_bins: Number of bins for calibration histogram

    Returns:
        dict with:
            - ece: Expected Calibration Error
            - mce: Maximum Calibration Error
            - brier_score: Brier score
            - bin_accuracies: Accuracy per bin
            - bin_confidences: Average confidence per bin
            - bin_counts: Sample count per bin
            - bin_boundaries: Bin edge values
    """
    import numpy as np

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # ECE: weighted average of |confidence - accuracy|
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            # MCE: maximum |confidence - accuracy|
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)

    # Brier score: mean squared error between confidence and correctness
    brier_score = np.mean((confidences - accuracies) ** 2)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier_score),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_boundaries": bin_boundaries.tolist(),
    }


def plot_reliability_diagram(metrics, title="", save_path=None):
    """
    Create reliability diagram showing calibration.

    Args:
        metrics: Output from compute_calibration_metrics
        title: Plot title
        save_path: If provided, save plot to this path

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    bin_centers = [(metrics['bin_boundaries'][i] + metrics['bin_boundaries'][i+1]) / 2
                   for i in range(len(metrics['bin_accuracies']))]

    # Plot 1: Reliability diagram
    ax = axes[0]
    bar_width = 0.08
    ax.bar(bin_centers, metrics['bin_accuracies'], width=bar_width,
           alpha=0.7, color='steelblue', edgecolor='black', label='Accuracy')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(f'{title}\nECE: {metrics["ece"]:.3f}', fontsize=12)
    else:
        ax.set_title(f'ECE: {metrics["ece"]:.3f}', fontsize=12)

    # Plot 2: Gap between confidence and accuracy
    ax = axes[1]
    gaps = [conf - acc for conf, acc in zip(metrics['bin_confidences'], metrics['bin_accuracies'])]
    colors = ['red' if gap > 0 else 'green' for gap in gaps]
    ax.bar(bin_centers, gaps, width=bar_width, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Confidence - Accuracy', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(f'Calibration Gap\nMCE: {metrics["mce"]:.3f}', fontsize=12)
    ax.grid(alpha=0.3)

    # Plot 3: Sample distribution
    ax = axes[2]
    ax.bar(bin_centers, metrics['bin_counts'], width=bar_width,
           alpha=0.7, color='gray', edgecolor='black')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title('Confidence Distribution', fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved reliability diagram to: {save_path}")

    return fig


def plot_confidence_histogram(confidences, predictions, labels, title="", save_path=None):
    """
    Plot confidence distribution split by correct/incorrect predictions.

    Args:
        confidences: Array of confidence scores
        predictions: Array of predicted class indices
        labels: Array of ground truth class indices
        title: Plot title
        save_path: If provided, save plot to this path

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    correct_mask = predictions == labels

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 21)
    ax.hist(confidences[correct_mask], bins=bins, alpha=0.6,
            label=f'Correct ({correct_mask.sum()})', color='green',
            density=True, edgecolor='black')
    ax.hist(confidences[~correct_mask], bins=bins, alpha=0.6,
            label=f'Incorrect ({(~correct_mask).sum()})', color='red',
            density=True, edgecolor='black')

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    if title:
        ax.set_title(title, fontsize=13)
    else:
        accuracy = correct_mask.mean()
        ax.set_title(f'Confidence Distribution (Accuracy: {accuracy:.2%})', fontsize=13)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved confidence histogram to: {save_path}")

    return fig


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
)
def evaluate_calibration(
    checkpoint_path: str,
    task_name: str = None,
    batch_size: int = 32,
    use_test_set: bool = True,
    dataset_path: str = None,
    n_bins: int = 10,
    max_test_samples: int = None,
):
    """
    Evaluate calibration metrics for a trained model.

    Returns confidences, predictions, labels, and computed metrics.

    Args:
        checkpoint_path: Path to checkpoint directory (relative to CHECKPOINT_DIR)
        task_name: Task to evaluate on (default: same as training)
        batch_size: Batch size for evaluation
        use_test_set: If True, only evaluate on test set from training split
        dataset_path: Override dataset path for cross-dataset evaluation
        n_bins: Number of bins for calibration histogram
        max_test_samples: Maximum number of samples to evaluate (random sampling)

    Returns:
        dict with:
            - metrics: Calibration metrics (ECE, MCE, Brier, bins)
            - confidences: Array of confidence scores
            - predictions: Array of predicted classes
            - labels: Array of ground truth labels
            - accuracy: Overall accuracy
            - checkpoint_path: Path to checkpoint
            - task_name: Task name
            - model_name: Model architecture name
    """
    import json
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    import os

    os.environ["HF_HOME"] = str(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint_dir = CHECKPOINT_DIR / checkpoint_path
    final_model_path = checkpoint_dir / "final_model.pt"
    best_model_path = checkpoint_dir / "best_model.pt"

    model_path = best_model_path if best_model_path.exists() else final_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {checkpoint_dir}")

    print(f"\nEvaluating calibration for: {checkpoint_path}")
    checkpoint = torch.load(str(model_path), map_location=device)

    # Load config
    config_path = checkpoint_dir / "training_config.json"
    with open(str(config_path), "r") as f:
        config_dict = json.load(f)
    config = ResNetTrainingConfig(**config_dict)

    # Determine task
    task_name = task_name or config.task_name
    task = get_task(task_name)

    # Load class mapping
    if "label_to_idx" in checkpoint:
        label_to_idx = {eval(k) if k.startswith("(") else (k == "True" if k in ["True", "False"] else k): v
                       for k, v in checkpoint["label_to_idx"].items()}
        idx_to_label = {int(k): eval(v) if v.startswith("(") else (v == "True" if v in ["True", "False"] else v)
                       for k, v in checkpoint["idx_to_label"].items()}
        num_classes = checkpoint["num_classes"]
    else:
        class_mapping_path = checkpoint_dir / "class_mapping.json"
        with open(str(class_mapping_path), "r") as f:
            mapping = json.load(f)
        label_to_idx = {eval(k) if k.startswith("(") else (k == "True" if k in ["True", "False"] else k): v
                       for k, v in mapping["label_to_idx"].items()}
        idx_to_label = {int(k): eval(v) if v.startswith("(") else (v == "True" if v in ["True", "False"] else v)
                       for k, v in mapping["idx_to_label"].items()}
        num_classes = len(label_to_idx)

    # Create model
    model = create_resnet_model(config.model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset = task.load_dataset(cache_dir=str(DATASET_DIR), dataset_path=dataset_path)

    # Filter to test set if requested
    test_set_loaded = False
    if use_test_set:
        test_indices_path = checkpoint_dir / "test_indices.json"
        if test_indices_path.exists():
            with open(str(test_indices_path), "r") as f:
                indices_data = json.load(f)
            test_indices = indices_data.get("test_indices", [])
            if test_indices:
                print(f"  Loading saved test set: {len(test_indices)} samples")

                # Check if dataset has _original_parquet_idx column
                has_parquet_idx = '_original_parquet_idx' in dataset.column_names

                if has_parquet_idx:
                    # Build mapping from parquet index to dataset index
                    parquet_to_idx = {
                        sample['_original_parquet_idx']: i
                        for i, sample in enumerate(dataset)
                    }
                    # Convert test indices (parquet indices) to dataset indices
                    test_dataset_indices = [
                        parquet_to_idx[idx] for idx in test_indices
                        if idx in parquet_to_idx
                    ]
                    print(f"  Mapped {len(test_dataset_indices)}/{len(test_indices)} test indices")
                else:
                    # Direct indices
                    test_dataset_indices = test_indices

                # Select test samples
                dataset = dataset.select(test_dataset_indices)
                test_set_loaded = True
                print(f"  Test set loaded: {len(dataset)} samples")

    # Apply task filters only if NOT using saved test set
    # (test set was already filtered during training)
    if not test_set_loaded:
        print("  Applying task filters...")
        original_len = len(dataset)
        dataset = task.filter_dataset(dataset)
        if len(dataset) < original_len:
            print(f"    Filtered: {original_len} -> {len(dataset)} samples")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check if test set indices are valid.")

    # Limit samples with random sampling if requested
    if max_test_samples and len(dataset) > max_test_samples:
        import random
        random.seed(42)  # For reproducibility
        sampled_indices = random.sample(range(len(dataset)), max_test_samples)
        dataset = dataset.select(sampled_indices)
        print(f"  Randomly sampled {max_test_samples} samples for calibration")

    # Create dataloader
    val_transform = get_transforms(config.image_size, is_training=False)
    eval_ds = ImageGridDataset(
        dataset, task, label_to_idx,
        image_size=config.image_size, grid_layout=config.grid_layout,
        transform=val_transform, tile_size=config.tile_size
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=collate_fn_skip_none
    )

    # Evaluate and collect predictions with confidences
    print(f"Computing predictions with confidences...")
    all_confidences = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in eval_loader:
            # Skip None batches (all images were corrupted)
            if images is None:
                continue
            images = images.to(device)
            labels = labels.to(device)

            # Get logits and compute softmax probabilities
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            # Confidence = max probability
            confidences, predictions = probs.max(dim=1)

            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    confidences = np.array(all_confidences)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # Compute calibration metrics
    metrics = compute_calibration_metrics(confidences, predictions, labels, n_bins=n_bins)
    accuracy = (predictions == labels).mean()

    print(f"\nCalibration Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")

    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        """Recursively convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj

    return {
        "metrics": convert_to_python_types(metrics),
        "confidences": confidences.tolist(),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
        "accuracy": float(accuracy),
        "checkpoint_path": checkpoint_path,
        "task_name": task_name,
        "model_name": config.model_name,
    }


@app.local_entrypoint()
def analyze_calibration(
    checkpoint: str = None,
    analyze_all: bool = False,
    output_dir: str = "calibration_results",
    dataset: str = None,
    species: str = None,
    use_test_set: bool = True,
):
    """
    Analyze calibration for one or more checkpoints.

    Usage:
        # Single checkpoint
        modal run scripts/model-post-training/modal_resnet_finetune.py::analyze_calibration \\
            --checkpoint "resnet50_merge_action_20240101_120000"

        # Cross-species evaluation
        modal run scripts/model-post-training/modal_resnet_finetune.py::analyze_calibration \\
            --checkpoint "resnet50_merge_action_20240101_120000" \\
            --species "mouse"

        # Custom dataset path
        modal run scripts/model-post-training/modal_resnet_finetune.py::analyze_calibration \\
            --checkpoint "resnet50_merge_action_20240101_120000" \\
            --dataset "merge-parquet-zebrafish"

        # All checkpoints
        modal run scripts/model-post-training/modal_resnet_finetune.py::analyze_calibration \\
            --analyze-all

        # Custom output directory
        modal run scripts/model-post-training/modal_resnet_finetune.py::analyze_calibration \\
            --analyze-all --output-dir "calibration_v2"
    """
    import json
    import numpy as np
    from collections import defaultdict

    print("=" * 60)
    print("ResNet Calibration Analysis")
    print("=" * 60)

    # Determine checkpoints to analyze
    if analyze_all:
        print("\nFinding all checkpoints...")
        # This will be executed on Modal, so we can access checkpoint_volume
        # For now, just prompt user to specify checkpoints manually
        print("ERROR: --analyze-all requires running on Modal with volume access")
        print("Please specify checkpoint(s) with --checkpoint")
        return

    if not checkpoint:
        print("ERROR: Must specify --checkpoint or use --analyze-all")
        return

    checkpoints = [checkpoint] if isinstance(checkpoint, str) else checkpoint

    # Determine dataset path from species or explicit dataset
    dataset_path = dataset
    if species and not dataset:
        # Map species to dataset paths
        species_datasets = {
            "fly": None,  # Default/original dataset
            "mouse": "merge-parquet-mouse",
            "human": "merge-parquet-human",
            "zebrafish": "merge-parquet-zebrafish",
        }
        if species not in species_datasets:
            print(f"ERROR: Unknown species '{species}'. Available: {list(species_datasets.keys())}")
            return
        dataset_path = species_datasets[species]
        print(f"Species: {species} -> Dataset: {dataset_path or 'default'}")

    # Create output directory locally
    output_path = Path(output_dir)
    if dataset_path:
        # Add dataset suffix to output dir for organization
        dataset_suffix = dataset_path.split('-')[-1] if dataset_path else 'default'
        output_path = output_path / dataset_suffix
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nAnalyzing {len(checkpoints)} checkpoint(s)...")
    if dataset_path:
        print(f"Cross-dataset evaluation: {dataset_path}")
    print(f"Output directory: {output_path}\n")

    # Evaluate each checkpoint
    all_results = {}
    for ckpt in checkpoints:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt}")
        if dataset_path:
            print(f"Dataset: {dataset_path}")
        print(f"{'='*60}")

        # Run evaluation on Modal
        # For cross-dataset evaluation, default to full dataset unless explicitly requested
        use_test_for_eval = use_test_set and not dataset_path
        result = evaluate_calibration.remote(
            checkpoint_path=ckpt,
            dataset_path=dataset_path,
            use_test_set=use_test_for_eval,
        )

        # Reconstruct arrays
        confidences = np.array(result['confidences'])
        predictions = np.array(result['predictions'])
        labels = np.array(result['labels'])
        metrics = result['metrics']

        # Create plots
        model_name = result['model_name']
        task_name = result['task_name']
        ckpt_name = Path(ckpt).name

        # Reliability diagram
        reliability_path = output_path / f"{ckpt_name}_reliability.png"
        plot_reliability_diagram(
            metrics,
            title=f"{model_name} on {task_name}",
            save_path=reliability_path
        )

        # Confidence histogram
        histogram_path = output_path / f"{ckpt_name}_confidence_hist.png"
        plot_confidence_histogram(
            confidences, predictions, labels,
            title=f"{model_name} on {task_name}",
            save_path=histogram_path
        )

        all_results[ckpt_name] = result

        print(f"\n  ✓ Generated plots:")
        print(f"    - {reliability_path}")
        print(f"    - {histogram_path}")

    # Save summary JSON (merge with existing results)
    summary_path = output_path / "calibration_summary.json"

    # Load existing results if file exists
    existing_results = {}
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass  # Start fresh if file is corrupted

    # Merge with new results (new results overwrite if same checkpoint)
    existing_results.update(all_results)

    # Save merged results
    with open(summary_path, 'w') as f:
        json.dump(existing_results, f, indent=2)
    print(f"\n✓ Saved summary to: {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<40} {'Acc':>6} {'ECE':>6} {'MCE':>6} {'Brier':>6}")
    print(f"{'-'*60}")
    for ckpt_name, result in all_results.items():
        acc = result['accuracy']
        ece = result['metrics']['ece']
        mce = result['metrics']['mce']
        brier = result['metrics']['brier_score']
        print(f"{ckpt_name:<40} {acc:>6.2%} {ece:>6.3f} {mce:>6.3f} {brier:>6.3f}")
    print(f"{'='*60}\n")


@app.local_entrypoint()
def main(
    task: str = "merge_action",
    model: str = "resnet50",
    num_samples: int = None,
    epochs: int = 10,
    max_steps: int = None,  # If set, stop after this many optimizer steps (overrides epochs)
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    freeze_backbone: bool = False,
    use_wandb: bool = False,
    run_name: str = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    train_samples: int = None,  # Absolute train sample count (overrides ratios)
    val_samples: int = None,    # Absolute val sample count (overrides ratios)
    class_balance: bool = False,
    early_stopping: bool = True,
    early_stopping_patience: int = 5,
    image_size: int = 224,
    grid_layout: str = "auto",
    tile_size: int = None,  # If set, use (tile_size * rows) x (tile_size * cols) for apples-to-apples comparison with VLM
):
    """
    Fine-tune a pretrained ResNet model as a baseline for classification tasks.

    This provides a simple CNN baseline to compare against VLM models.

    Usage:
        # Basic fine-tuning on merge_action task
        modal run scripts/model-post-training/modal_resnet_finetune.py --task merge_action

        # Quick test with small subset (100 train, 50 val, rest for test)
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --task merge_action \\
            --train-samples 100 \\
            --val-samples 50 \\
            --epochs 3

        # Fine-tune on split_action with more epochs
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --task split_action \\
            --epochs 20

        # Use larger models
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --model resnet152 \\
            --task merge_action

        # Use Wide ResNet-101-2 (~126M params)
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --model wide_resnet101_2 \\
            --task merge_action

        # Use ViT-L/16 (~304M params)
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --model vit_l_16 \\
            --task merge_action \\
            --batch-size 64  # ViT needs smaller batch

        # Use ViT-H/14 (largest ViT, ~632M params)
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --model vit_h_14 \\
            --task merge_action \\
            --batch-size 32  # Large model needs smaller batch

        # Use ConvNeXt-Large (~198M params)
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --model convnext_large \\
            --task merge_action

        # Only train classifier head (freeze backbone / linear probe)
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --freeze-backbone \\
            --task merge_action

        # With class balancing
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --class-balance \\
            --task merge_action

        # With W&B tracking
        modal run scripts/model-post-training/modal_resnet_finetune.py \\
            --use-wandb \\
            --run-name "resnet50_merge_v1"

    Available tasks (classification only):
        - merge_action (binary: yes/no)
        - split_action (binary: yes/no)
        - segment_identity (binary: same/different)
        - merge_error_identification (binary: error/control)
        - segment_classification (multiclass: a-g)
        - merge_action_multiple_choice (multiclass: a-d or none)
    """
    # Validate task
    available_tasks = list_tasks()
    if task not in available_tasks:
        print(f"Error: Unknown task '{task}'")
        print(f"Available tasks: {', '.join(available_tasks)}")
        return

    # Validate model
    valid_models = [
        # Standard ResNet
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        # Wide ResNet (wider channels)
        "wide_resnet50_2", "wide_resnet101_2",
        # ResNeXt (grouped convolutions)
        "resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d",
        # ViT (Vision Transformer)
        "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
        # ConvNeXt (modern CNN)
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    ]
    if model not in valid_models:
        print(f"Error: Unknown model '{model}'")
        print(f"Available models:")
        print(f"  ResNet: resnet18, resnet34, resnet50, resnet101, resnet152")
        print(f"  Wide ResNet: wide_resnet50_2, wide_resnet101_2 (~126M)")
        print(f"  ResNeXt: resnext50_32x4d, resnext101_32x8d, resnext101_64x4d")
        print(f"  ViT: vit_b_16, vit_b_32, vit_l_16 (~304M), vit_l_32, vit_h_14 (~632M)")
        print(f"  ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large (~198M)")
        return

    # Auto-adjust image size for vit_h_14 (requires 518x518)
    if model == "vit_h_14" and image_size == 224:
        image_size = 518
        print(f"Note: vit_h_14 requires 518x518 input, auto-adjusted image_size to {image_size}")

    print(f"Starting vision model fine-tuning baseline...")
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Samples: {num_samples if num_samples else 'all'}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Image size: {image_size}")
    print(f"Grid layout: {grid_layout}")
    if train_samples is not None or val_samples is not None:
        print(f"Split: {train_samples or 0} train / {val_samples or 0} val / remainder test (absolute counts)")
    else:
        print(f"Split: {train_split*100:.0f}% train / {val_split*100:.0f}% val / {test_split*100:.0f}% test")
    if class_balance:
        print("Class balancing: enabled (oversample)")
    if early_stopping:
        print(f"Early stopping: patience={early_stopping_patience}")

    config = ResNetTrainingConfig(
        task_name=task,
        model_name=model,
        num_samples=num_samples,
        num_train_epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_backbone=freeze_backbone,
        image_size=image_size,
        grid_layout=grid_layout,
        tile_size=tile_size,
        train_split_ratio=train_split,
        val_split_ratio=val_split,
        test_split_ratio=test_split,
        train_samples=train_samples,
        val_samples=val_samples,
        class_balance=class_balance,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        use_wandb=use_wandb,
        wandb_run_name=run_name,
    )

    checkpoint_path = finetune_resnet.remote(config=config)

    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("=" * 60)


@app.local_entrypoint()
def evaluate(
    checkpoint: str,
    task: str = None,
    num_samples: int = None,
    batch_size: int = 32,
    use_test_set: bool = True,
    dataset_path: str = None,
    max_test_samples: int = 1024,
):
    """
    Evaluate a trained ResNet model.

    Usage:
        # Evaluate on test set
        modal run scripts/model-post-training/modal_resnet_finetune.py::evaluate \\
            --checkpoint "resnet50_merge_action_20240101_120000"

        # Evaluate on full dataset
        modal run scripts/model-post-training/modal_resnet_finetune.py::evaluate \\
            --checkpoint "resnet50_merge_action_20240101_120000" \\
            --use-test-set False

        # Evaluate on a different dataset (cross-dataset evaluation)
        modal run scripts/model-post-training/modal_resnet_finetune.py::evaluate \\
            --checkpoint "resnet50_merge_action_20240101_120000" \\
            --dataset-path "merge-parquet-zebrafish" \\
            --use-test-set False

        # Evaluate on different species dataset
        modal run scripts/model-post-training/modal_resnet_finetune.py::evaluate \\
            --checkpoint "resnet50_merge_action_20240101_120000" \\
            --dataset-path "merge-parquet-human" \\
            --use-test-set False
    """
    print(f"Evaluating checkpoint: {checkpoint}")
    if dataset_path:
        print(f"Using dataset: {dataset_path}")

    result = evaluate_resnet.remote(
        checkpoint_path=checkpoint,
        task_name=task,
        num_samples=num_samples,
        batch_size=batch_size,
        use_test_set=use_test_set,
        dataset_path=dataset_path,
        max_test_samples=max_test_samples,
    )

    print(f"\nFinal accuracy: {result['accuracy']:.2%}")
    print(f"Balanced accuracy: {result['balanced_accuracy']:.2%}")

    # Save results to JSON
    import json
    from pathlib import Path
    from datetime import datetime

    results_dir = Path("evaluation_results/resnet_evaluations")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_suffix = f"_{dataset_path.replace('/', '-')}" if dataset_path else ""
    results_file = results_dir / f"{checkpoint}{dataset_suffix}_{timestamp}.json"

    results_data = {
        "checkpoint": checkpoint,
        "task": task,
        "dataset_path": dataset_path,
        "use_test_set": use_test_set,
        "timestamp": timestamp,
        "accuracy": result['accuracy'],
        "balanced_accuracy": result['balanced_accuracy'],
        "correct": result['correct'],
        "total": result['total'],
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")
