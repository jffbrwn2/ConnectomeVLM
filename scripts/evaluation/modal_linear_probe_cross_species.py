#!/usr/bin/env python3
"""
Train linear probes on mouse data and evaluate on cross-species datasets.

Usage:
    # Single evaluation
    modal run scripts/analysis/modal_linear_probe_cross_species.py::evaluate \
        --task merge_action --species fly

    # Batch evaluation (all tasks x all species)
    modal run scripts/analysis/modal_linear_probe_cross_species.py::batch_evaluate
"""

import modal
import numpy as np
from pathlib import Path

# Volumes
dataset_volume = modal.Volume.from_name("qwen-finetune-datasets", create_if_missing=True)
results_volume = modal.Volume.from_name("qwen-finetune-results", create_if_missing=True)

# Use the same image as modal_qwen_finetune.py for faster startup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth",
        "unsloth_zoo",
    )
    .pip_install(
        "datasets",
        "pandas",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "wandb",
        "scikit-learn",
        "transformers",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file("src/environment/task_configs.py", remote_path="/root/task_configs.py")
)

# Create Modal app
app = modal.App("linear-probe-cross-species")

# Dataset paths
SPECIES_DATASETS = {
    "mouse": None,  # Default
    "fly": {
        "merge_action": "fly-merge-parquet",
        "split_action": "split_generalization_parquets/fly-splits",
        "merge_error_identification": "fly_merge_7500nm_parquet",
    },
    "zebrafish": {
        "merge_action": "zebrafish-merge-parquet",
        "split_action": "split_generalization_parquets/fish-jan24",
        "merge_error_identification": "zebrafish_merge_7500nm_parquet",
        "endpoint_error_identification_with_em": "endpoint_zebrafish_parquet",
    },
    "human": {
        "merge_action": "human-merge-parquet",
        "split_action": "split_generalization_parquets/human-splits",
        "merge_error_identification": "human_merge_7500nm_parquet",
        "endpoint_error_identification_with_em": "endpoint_human_parquet",
    },
}


@app.function(
    image=image,
    volumes={"/datasets": dataset_volume, "/results": results_volume},
    gpu="A10G",
    timeout=3600,
)
def evaluate(task: str, species: str):
    """
    Train linear probe on mouse data and evaluate on cross-species dataset.

    Args:
        task: Task name (merge_action, split_action, etc.)
        species: Species to evaluate on (fly, zebrafish, human)
    """
    import json
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from datetime import datetime

    print(f"="*80)
    print(f"Linear Probe Cross-Species Evaluation")
    print(f"Task: {task}")
    print(f"Species: {species}")
    print(f"="*80)

    # 1. Load mouse features and train probe
    print("\n1. Training linear probe on mouse data...")

    # Look for feature cache in /results/feature_cache
    # Cache file names from linear_probe_sweep results:
    cache_mapping = {
        "merge_action": "google_siglip2-so400m-patch16-512_merge_action_all_mean_seed84_balanced.npz",
        "split_action": "google_siglip2-so400m-patch16-512_split_action_splits-parquet_all_mean_seed84_balanced.npz",
        "merge_error_identification": "google_siglip2-so400m-patch16-512_merge_error_identification_merge-error-identification-parquet_4096_mean_seed84_balanced.npz",
        "endpoint_error_identification_with_em": "google_siglip2-so400m-patch16-512_endpoint_error_identification_with_em_endpoints-with-em-parquet_all_mean_seed86_balanced.npz",
    }

    cache_file_name = cache_mapping.get(task)
    if cache_file_name is None:
        print(f"ERROR: No cache file mapping for task: {task}")
        return None

    cache_dir = Path("/results/feature_cache")
    mouse_cache = cache_dir / cache_file_name

    if not mouse_cache.exists():
        print(f"ERROR: Cache file not found: {mouse_cache}")
        return None

    print(f"   Loading: {mouse_cache.name}")

    data = np.load(mouse_cache)
    X_train = data['X']
    y_train = data['y']

    print(f"   Train samples: {len(X_train)}")
    print(f"   Feature dim: {X_train.shape[1]}")
    print(f"   Class distribution: {np.bincount(y_train)}")

    # Train balanced logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate on mouse train set (sanity check)
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    print(f"   Mouse train accuracy: {train_acc:.4f}")
    print(f"   Mouse train balanced accuracy: {train_bal_acc:.4f}")

    # 2. Extract features for cross-species dataset
    print(f"\n2. Extracting features for {species} dataset...")
    dataset_path = SPECIES_DATASETS.get(species, {}).get(task)
    if dataset_path is None:
        print(f"ERROR: No dataset path for {task} on {species}")
        return None

    # Load cross-species parquet dataset
    import pandas as pd
    from PIL import Image

    cross_dataset_dir = Path("/datasets") / dataset_path
    if not cross_dataset_dir.exists():
        print(f"ERROR: Dataset not found at {cross_dataset_dir}")
        return None

    parquet_file = cross_dataset_dir / "questions.parquet"
    images_dir = cross_dataset_dir / "images"

    if not parquet_file.exists():
        print(f"ERROR: Parquet file not found: {parquet_file}")
        return None

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return None

    df = pd.read_parquet(parquet_file)
    print(f"   Loaded {len(df)} samples from parquet")

    # Limit to 128 samples per class for fast evaluation
    max_samples_per_class = 64  # 64 per class = 128 total

    # Get labels - handle both integer and string labels
    labels_all = df['answer'].values

    # Convert string labels to integers if needed (for merge_error_identification)
    if isinstance(labels_all[0], str):
        # Map 'control' -> 0, 'error' -> 1 (or 'no' -> 0, 'yes' -> 1)
        label_map = {}
        unique_labels = sorted(set(labels_all))
        for i, label in enumerate(unique_labels):
            label_map[label] = i
        print(f"   Label mapping: {label_map}")
        labels_all = np.array([label_map[label] for label in labels_all])

    # Get balanced indices
    labels_array = np.array(labels_all, dtype=int)
    indices_class_0 = np.where(labels_array == 0)[0]
    indices_class_1 = np.where(labels_array == 1)[0]

    # Sample up to max_samples_per_class from each class
    n_class_0 = min(len(indices_class_0), max_samples_per_class)
    n_class_1 = min(len(indices_class_1), max_samples_per_class)

    selected_indices_0 = np.random.choice(indices_class_0, n_class_0, replace=False)
    selected_indices_1 = np.random.choice(indices_class_1, n_class_1, replace=False)

    selected_indices = np.concatenate([selected_indices_0, selected_indices_1])
    np.random.shuffle(selected_indices)

    # Select subset
    df = df.iloc[selected_indices].reset_index(drop=True)
    labels_array_selected = labels_array[selected_indices]
    print(f"   Limited to {len(df)} samples (class-balanced, max {max_samples_per_class} per class)")

    # Extract SigLIP features
    print("   Extracting SigLIP-2 features...")
    from transformers import AutoModel, AutoProcessor
    import torch

    model_name = "google/siglip2-so400m-patch16-512"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    features_list = []
    labels_list = []

    batch_size = 32
    pooling = "mean"  # Must match the pooling used in cached features
    images_per_row = 3  # Grid layout for >3 images

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_labels = labels_array_selected[i:i+batch_size]

        # Load and combine images (same as modal_qwen_finetune.py)
        images_batch = []
        for _, row in batch_df.iterrows():
            # row['images'] is a list of image paths
            image_paths = row['images']

            # Load all images for this sample
            images = []
            for img_path in image_paths:
                full_path = cross_dataset_dir / img_path
                if full_path.exists():
                    images.append(Image.open(full_path).convert('RGB'))

            # Concatenate multiple images into a grid (3 per row) - same as training
            if len(images) > 1:
                # Resize all images to same height first
                max_h = max(img.size[1] for img in images)
                resized = []
                for img in images:
                    if img.size[1] != max_h:
                        ratio = max_h / img.size[1]
                        new_w = int(img.size[0] * ratio)
                        img = img.resize((new_w, max_h), Image.LANCZOS)
                    resized.append(img)

                if len(resized) <= images_per_row:
                    # Single row - horizontal concatenation
                    total_w = sum(img.size[0] for img in resized)
                    combined = Image.new('RGB', (total_w, max_h))
                    x_offset = 0
                    for img in resized:
                        combined.paste(img, (x_offset, 0))
                        x_offset += img.size[0]
                else:
                    # Multiple rows - grid layout
                    num_rows = (len(resized) + images_per_row - 1) // images_per_row
                    rows = []
                    for row_idx in range(num_rows):
                        start_idx = row_idx * images_per_row
                        end_idx = min(start_idx + images_per_row, len(resized))
                        row_images = resized[start_idx:end_idx]

                        # Concatenate images in this row horizontally
                        row_w = sum(img.size[0] for img in row_images)
                        row_img = Image.new('RGB', (row_w, max_h))
                        x_offset = 0
                        for img in row_images:
                            row_img.paste(img, (x_offset, 0))
                            x_offset += img.size[0]
                        rows.append(row_img)

                    # Stack rows vertically
                    max_row_w = max(r.size[0] for r in rows)
                    total_h = sum(r.size[1] for r in rows)
                    combined = Image.new('RGB', (max_row_w, total_h))
                    y_offset = 0
                    for row_img in rows:
                        combined.paste(row_img, (0, y_offset))
                        y_offset += row_img.size[1]

                images_batch.append(combined)
            elif len(images) == 1:
                images_batch.append(images[0])

        # Process images
        inputs = processor(images=images_batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features using same method as cache creation
        with torch.no_grad():
            outputs = model.vision_model(**inputs)
            hidden = outputs.last_hidden_state  # (batch, num_patches, 1152)

            # Apply pooling (must match cache: "mean")
            for b in range(hidden.shape[0]):
                h = hidden[b]
                if pooling == "mean":
                    feat = h.mean(dim=0)
                elif pooling == "max":
                    feat = h.max(dim=0).values
                elif pooling == "cls":
                    feat = h[0]
                else:
                    feat = h.mean(dim=0)
                features_list.append(feat.cpu().float().numpy())

        labels_list.extend(batch_labels.tolist())

        if (i // batch_size) % 10 == 0:
            print(f"   Processed {i}/{len(df)} samples...")

    X_test = np.stack(features_list)
    y_test = np.array(labels_list, dtype=int)

    print(f"   Extracted {len(X_test)} feature vectors")
    print(f"   Class distribution: {np.bincount(y_test)}")

    # 3. Evaluate probe on cross-species features
    print(f"\n3. Evaluating on {species} dataset...")
    y_pred_test = clf.predict(X_test)
    y_proba_test = clf.predict_proba(X_test)

    test_acc = accuracy_score(y_test, y_pred_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)

    correct = (y_pred_test == y_test).sum()
    total = len(y_test)

    print(f"   Accuracy: {test_acc:.4f} ({correct}/{total})")
    print(f"   Balanced Accuracy: {test_bal_acc:.4f}")

    # Per-class accuracy
    for label in [0, 1]:
        mask = y_test == label
        if mask.sum() > 0:
            class_acc = (y_pred_test[mask] == y_test[mask]).mean()
            print(f"   Class {label} accuracy: {class_acc:.4f} ({mask.sum()} samples)")

    # Compute calibration metrics
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    def compute_ece(y_true, y_proba, n_bins=10):
        """Compute Expected Calibration Error."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
        # calibration_curve already filters to non-empty bins
        # Compute bin counts for the returned bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_totals, _ = np.histogram(y_proba, bins=bin_edges)
        non_empty = bin_totals > 0
        # Only use weights for non-empty bins (matching prob_true/prob_pred length)
        bin_weights = bin_totals[non_empty] / len(y_proba)
        ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
        return ece

    def compute_mce(y_true, y_proba, n_bins=10):
        """Compute Maximum Calibration Error."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
        mce = np.max(np.abs(prob_true - prob_pred))
        return mce

    # Get probabilities for positive class
    y_proba_pos = y_proba_test[:, 1]

    ece = compute_ece(y_test, y_proba_pos)
    mce = compute_mce(y_test, y_proba_pos)
    brier = brier_score_loss(y_test, y_proba_pos)

    print(f"   ECE: {ece:.4f}")
    print(f"   MCE: {mce:.4f}")
    print(f"   Brier Score: {brier:.4f}")

    # 4. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "task": task,
        "species": species,
        "dataset_path": dataset_path,
        "model_type": "linear_probe",
        "train_species": "mouse",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": float(train_acc),
        "train_balanced_accuracy": float(train_bal_acc),
        "test_accuracy": float(test_acc),
        "test_balanced_accuracy": float(test_bal_acc),
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "correct": int(correct),
        "total": int(total),
        "timestamp": timestamp,
    }

    # Save to results volume
    results_dir = Path("/results/linear_probe_cross_species")
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = results_dir / f"linear_probe_{task}_{species}_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    results_volume.commit()

    print(f"\n✓ Saved results to {result_file}")

    return result


@app.local_entrypoint()
def batch_evaluate():
    """
    Evaluate linear probes on all cross-species datasets.
    """
    # Define tasks and their target species
    task_species_map = {
        "merge_action": ["fly", "zebrafish", "human"],
        "split_action": ["fly", "zebrafish", "human"],
        "merge_error_identification": ["fly", "zebrafish", "human"],
        "endpoint_error_identification_with_em": ["zebrafish", "human"],  # No fly dataset
    }

    print("="*80)
    print("Batch Linear Probe Cross-Species Evaluation")
    print("="*80)
    print(f"Tasks: {list(task_species_map.keys())}")
    print()

    results = []
    for task, species_list in task_species_map.items():
        for species in species_list:
            print(f"\n{'='*80}")
            print(f"Evaluating: {task} on {species}")
            print(f"{'='*80}")

            try:
                result = evaluate.remote(task=task, species=species)
                if result:
                    results.append(result)
                    print(f"✓ {task} on {species}: Bal Acc={result['test_balanced_accuracy']:.4f}, ECE={result['ece']:.4f}")
                else:
                    print(f"✗ {task} on {species}: Failed")
            except Exception as e:
                print(f"✗ {task} on {species}: Error - {e}")

    # Save summary
    import json
    from pathlib import Path
    from datetime import datetime

    summary_dir = Path("evaluation_results/linear_probe_cross_species")
    summary_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = summary_dir / f"cross_species_summary_{timestamp}.json"

    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Count total expected evaluations
    total_expected = sum(len(species_list) for species_list in task_species_map.values())

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Completed: {len(results)}/{total_expected} evaluations")
    print(f"Saved to: {summary_file}")

    # Print table
    print("\nResults (Balanced Accuracy and ECE):")
    print(f"{'Task':<40s} {'Species':<12s} {'Bal Acc':<10s} {'ECE':<10s}")
    print("-"*72)
    for r in results:
        print(f"{r['task']:<40s} {r['species']:<12s} {r['test_balanced_accuracy']:>8.1%}  {r['ece']:>8.1%}")
