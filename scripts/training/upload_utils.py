"""
Utilities for uploading data to Modal volumes.
"""
import modal
from pathlib import Path

# Create volumes
dataset_volume = modal.Volume.from_name("qwen-finetune-datasets", create_if_missing=True)

DATASET_DIR = Path("/datasets")

app = modal.App("upload-utils")

# Create image with necessary dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas",
    "Pillow",
    "pyarrow",  # For parquet support
)


@app.function(
    image=image,
    volumes={DATASET_DIR: dataset_volume},
    timeout=3600,  # 1 hour for large uploads
)
def _write_file_to_volume(remote_path: str, data: bytes):
    """Write a file to the volume (internal helper)."""
    from pathlib import Path

    remote_file = Path(remote_path)
    remote_file.parent.mkdir(parents=True, exist_ok=True)

    with open(remote_path, 'wb') as f:
        f.write(data)

    dataset_volume.commit()


@app.function(
    image=image,
    volumes={DATASET_DIR: dataset_volume},
    timeout=3600,
)
def _list_volume_contents(path: str = "/datasets"):
    """List contents of a path in the volume."""
    from pathlib import Path
    import os

    target_path = Path(path)
    if not target_path.exists():
        return f"Path does not exist: {path}"

    contents = []
    for root, dirs, files in os.walk(target_path):
        level = root.replace(str(target_path), '').count(os.sep)
        indent = ' ' * 2 * level
        contents.append(f"{indent}{Path(root).name}/")
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = Path(root) / file
            size_mb = file_path.stat().st_size / 1024 / 1024
            contents.append(f"{sub_indent}{file} ({size_mb:.2f} MB)")

    return "\n".join(contents)


@app.local_entrypoint()
def upload_directory(
    local_path: str,
    remote_path: str = None,
    force: bool = False,
):
    """
    Upload an entire directory (including subdirectories) to Modal volume.

    Uses Modal's batch_upload() for fast parallel uploads.

    Usage:
        # Upload splits-parquet directory
        modal run scripts/model-post-training/upload_utils.py::upload_directory \\
            --local-path "data/splits-parquet" \\
            --remote-path "splits-parquet"

        # Upload to root of volume (uses directory name)
        modal run scripts/model-post-training/upload_utils.py::upload_directory \\
            --local-path "data/splits-parquet"

        # Overwrite existing files
        modal run scripts/model-post-training/upload_utils.py::upload_directory \\
            --local-path "data/splits-parquet" --force
    """
    import os

    local_dir = Path(local_path)
    if not local_dir.exists():
        print(f"Error: Directory not found: {local_path}")
        return

    if not local_dir.is_dir():
        print(f"Error: Not a directory: {local_path}")
        return

    # Use directory name if remote path not specified
    if remote_path is None:
        remote_path = local_dir.name

    # Calculate total size for display
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file = Path(root) / file
            total_size += local_file.stat().st_size
            file_count += 1

    print(f"Uploading directory: {local_path}")
    print(f"Remote path (in volume): /{remote_path}")
    print(f"  -> Mounts at: {DATASET_DIR / remote_path}")
    print(f"Files: {file_count}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print()

    # Use batch_upload for fast parallel uploads
    # Note: batch_upload writes to volume root, not the mount path
    # So we use just remote_path, not DATASET_DIR / remote_path
    print("Starting batch upload...")
    with dataset_volume.batch_upload(force=force) as batch:
        batch.put_directory(str(local_dir), f"/{remote_path}")

    print()
    print("Upload complete!")
    print()
    print(f"Files uploaded to volume at: /{remote_path}")
    print(f"  -> Available at: {DATASET_DIR / remote_path}")


@app.local_entrypoint()
def upload_directory_chunked(
    local_path: str,
    remote_path: str = None,
    chunk_size: int = 5000,
    force: bool = False,
):
    """
    Upload a large directory in chunks to avoid timeouts.

    Splits the upload into batches of chunk_size files each.
    Useful for datasets with many images (>10K files).

    Usage:
        # Upload large dataset in chunks of 5000 files
        modal run scripts/model-post-training/upload_utils.py::upload_directory_chunked \\
            --local-path "training_data/merge-multiple-choice-parquet" \\
            --remote-path "merge-multiple-choice-parquet"

        # Use smaller chunks for very large files
        modal run scripts/model-post-training/upload_utils.py::upload_directory_chunked \\
            --local-path "training_data/merge-multiple-choice-parquet" \\
            --chunk-size 2000
    """
    import os
    import tempfile
    import shutil

    local_dir = Path(local_path)
    if not local_dir.exists():
        print(f"Error: Directory not found: {local_path}")
        return

    if not local_dir.is_dir():
        print(f"Error: Not a directory: {local_path}")
        return

    # Use directory name if remote path not specified
    if remote_path is None:
        remote_path = local_dir.name

    # Collect all files
    all_files = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file = Path(root) / file
            rel_path = local_file.relative_to(local_dir)
            all_files.append((local_file, rel_path))

    total_size = sum(f[0].stat().st_size for f in all_files)
    print(f"Uploading directory: {local_path}")
    print(f"Remote path (in volume): /{remote_path}")
    print(f"  -> Mounts at: {DATASET_DIR / remote_path}")
    print(f"Total files: {len(all_files)}")
    print(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Chunk size: {chunk_size} files")
    print()

    # Split files into chunks
    chunks = []
    for i in range(0, len(all_files), chunk_size):
        chunks.append(all_files[i:i + chunk_size])

    print(f"Split into {len(chunks)} chunks")
    print()

    # Upload each chunk
    for chunk_idx, chunk_files in enumerate(chunks):
        chunk_size_bytes = sum(f[0].stat().st_size for f in chunk_files)
        print(f"Uploading chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_files)} files, {chunk_size_bytes / 1024 / 1024:.1f} MB)...")

        # Create a temporary directory with the chunk's files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "chunk"

            # Copy files preserving directory structure
            for local_file, rel_path in chunk_files:
                dest_file = temp_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_file, dest_file)

            # Upload the chunk
            try:
                with dataset_volume.batch_upload(force=force) as batch:
                    batch.put_directory(str(temp_path), f"/{remote_path}")
                print(f"  Chunk {chunk_idx + 1} complete!")
            except Exception as e:
                print(f"  Error uploading chunk {chunk_idx + 1}: {e}")
                print(f"  Retrying with smaller sub-chunks...")

                # Retry with smaller sub-chunks
                sub_chunk_size = max(100, len(chunk_files) // 10)
                for sub_idx in range(0, len(chunk_files), sub_chunk_size):
                    sub_chunk = chunk_files[sub_idx:sub_idx + sub_chunk_size]
                    print(f"    Sub-chunk {sub_idx // sub_chunk_size + 1}: {len(sub_chunk)} files")

                    with tempfile.TemporaryDirectory() as sub_temp:
                        sub_path = Path(sub_temp) / "subchunk"
                        for local_file, rel_path in sub_chunk:
                            dest_file = sub_path / rel_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(local_file, dest_file)

                        with dataset_volume.batch_upload(force=force) as batch:
                            batch.put_directory(str(sub_path), f"/{remote_path}")

    print()
    print("Upload complete!")
    print(f"Files uploaded to volume at: /{remote_path}")
    print(f"  -> Available at: {DATASET_DIR / remote_path}")


@app.local_entrypoint()
def list_volume(path: str = "/datasets"):
    """
    List contents of the volume.

    Usage:
        # List root
        modal run scripts/model-post-training/upload_utils.py::list_volume

        # List specific path
        modal run scripts/model-post-training/upload_utils.py::list_volume \\
            --path "/datasets/splits-parquet"
    """
    print(f"Contents of {path}:")
    print()
    contents = _list_volume_contents.remote(path)
    print(contents)


@app.function(
    image=image,
    volumes={DATASET_DIR: dataset_volume},
    timeout=1800,
)
def _validate_split_images():
    """Validate all images in the splits dataset."""
    import pandas as pd
    from PIL import Image
    from pathlib import Path

    dataset_dir = Path("/datasets/splits-parquet")
    parquet_path = dataset_dir / "questions.parquet"

    if not parquet_path.exists():
        return f"Error: Dataset not found at {parquet_path}"

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples")

    # Check image count distribution
    image_counts = df['images'].apply(lambda x: len(x) if isinstance(x, list) else 1)
    print(f"\nImage count distribution:")
    print(f"  Min: {image_counts.min()}")
    print(f"  Max: {image_counts.max()}")
    print(f"  Mean: {image_counts.mean():.2f}")
    print(f"  Unique counts: {sorted(image_counts.unique())}")

    issues = []
    for idx, row in df.iterrows():
        image_paths = row['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        for img_idx, rel_path in enumerate(image_paths):
            abs_path = dataset_dir / rel_path
            try:
                if not abs_path.exists():
                    issues.append(f"Sample {idx}, Image {img_idx}: File not found: {rel_path}")
                    continue

                img = Image.open(abs_path)
                width, height = img.size

                if width == 0 or height == 0:
                    issues.append(f"Sample {idx}, Image {img_idx}: Zero dimensions {img.size}: {rel_path}")
                elif width < 10 or height < 10:
                    issues.append(f"Sample {idx}, Image {img_idx}: Very small dimensions {img.size}: {rel_path}")
                elif width / height > 20 or height / width > 20:
                    issues.append(f"Sample {idx}, Image {img_idx}: Extreme aspect ratio {width}x{height}: {rel_path}")

                if img.mode not in ['RGB', 'RGBA', 'L']:
                    issues.append(f"Sample {idx}, Image {img_idx}: Unusual mode {img.mode}: {rel_path}")

            except Exception as e:
                issues.append(f"Sample {idx}, Image {img_idx}: Error loading: {rel_path} - {e}")

        if idx % 50 == 0:
            print(f"Checked {idx}/{len(df)} samples...")

    result = f"\n{'='*60}\n"
    result += f"Validation complete: {len(df)} samples checked\n"
    if issues:
        result += f"\nFound {len(issues)} issues:\n"
        for issue in issues[:20]:  # Show first 20
            result += f"  - {issue}\n"
        if len(issues) > 20:
            result += f"  ... and {len(issues) - 20} more issues\n"
    else:
        result += "No issues found! All images valid.\n"
    result += f"{'='*60}\n"

    return result


@app.local_entrypoint()
def validate_splits():
    """
    Validate all images in the splits dataset.

    Usage:
        modal run scripts/model-post-training/upload_utils.py::validate_splits
    """
    print("Validating splits dataset images...")
    print()
    result = _validate_split_images.remote()
    print(result)


@app.function(
    image=image,
    volumes={DATASET_DIR: dataset_volume},
    timeout=600,
)
def _delete_path(path: str):
    """Delete a file or directory from the volume."""
    import shutil
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        return f"Path does not exist: {path}"

    if target.is_file():
        target.unlink()
        dataset_volume.commit()
        return f"Deleted file: {path}"
    elif target.is_dir():
        shutil.rmtree(target)
        dataset_volume.commit()
        return f"Deleted directory: {path}"
    else:
        return f"Unknown path type: {path}"


@app.local_entrypoint()
def delete(
    path: str,
    force: bool = False,
):
    """
    Delete a file or directory from the Modal volume.

    Usage:
        # Delete a specific file
        modal run scripts/model-post-training/upload_utils.py::delete \\
            --path "/datasets/old_file.parquet"

        # Delete a directory (will prompt for confirmation)
        modal run scripts/model-post-training/upload_utils.py::delete \\
            --path "/datasets/splits-parquet"

        # Delete without confirmation
        modal run scripts/model-post-training/upload_utils.py::delete \\
            --path "/datasets/splits-parquet" --force
    """
    print(f"Target path: {path}")

    if not force:
        confirm = input(f"Are you sure you want to delete '{path}'? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    print(f"Deleting {path}...")
    result = _delete_path.remote(path)
    print(result)


