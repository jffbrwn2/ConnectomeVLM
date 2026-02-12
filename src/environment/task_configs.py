"""
Task configuration system for vision-language model fine-tuning.

This module provides a flexible framework for defining different tasks that can be
used with both SFT and RL training scripts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any, Sequence
import numpy as np
import random
from PIL import Image


def create_image_grid(
    images: Sequence[Image.Image],
    rows: int,
    cols: int,
    background_color: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Arrange images in a grid layout (rows x cols).
    
    Inlined here to avoid Modal mount dependency on rendering module.
    """
    expected_count = rows * cols
    if len(images) != expected_count:
        raise ValueError(
            f"Expected {expected_count} images for {rows}x{cols} grid, got {len(images)}"
        )
    
    if not images:
        raise ValueError("no images to arrange in grid")
    
    # Use first image size as reference
    ref_width, ref_height = images[0].size
    
    # Create canvas
    canvas_width = ref_width * cols
    canvas_height = ref_height * rows
    canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
    
    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        # Resize if needed to match reference size
        if img.size != (ref_width, ref_height):
            img = img.resize((ref_width, ref_height), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        x_offset = col * ref_width
        y_offset = row * ref_height
        canvas.paste(img, (x_offset, y_offset))
    
    return canvas


@dataclass
class TaskConfig(ABC):
    """Base class for vision-language task configurations."""

    # Task metadata
    name: str
    description: str

    # Dataset info
    dataset_source: str  # HuggingFace dataset name or local path
    dataset_config: Optional[str] = None  # Config name for HF datasets

    # Image configuration
    num_images: int = 1  # Expected number of images per sample

    # Hash of the dataset file (set by load_dataset)
    _dataset_hash: Optional[str] = None

    @abstractmethod
    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load and return the dataset.

        Args:
            cache_dir: Parent directory containing the dataset folder.
                       The full path will be: cache_dir / self.dataset_source
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
                          Useful for evaluating on different datasets.
        """
        pass

    @abstractmethod
    def filter_dataset(self, dataset):
        """Optional filtering (e.g., remove unwanted classes)."""
        return dataset

    @abstractmethod
    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """
        Create the instruction prompt for a sample.

        Args:
            sample: The sample dict
            answer_only: If True, omit the instruction to include analysis tags
            request_confidence: If True, ask model to report confidence 0-100

        Returns: The text prompt (without images)
        """
        pass

    def get_confidence_request_text(self) -> str:
        """
        Get the confidence request text to append to prompts.

        Returns: Text asking model to report confidence 0-100
        """
        return """\n\nAfter providing your answer, rate your confidence from 0-100, where:
- 0 = completely uncertain (random guess)
- 50 = moderately confident
- 100 = completely certain

Format your confidence as: <confidence>XX</confidence>"""

    def extract_confidence(self, response: str) -> Optional[float]:
        """
        Extract confidence score from model response.

        Args:
            response: The model's full response text

        Returns: Confidence as float 0.0-1.0, or None if not found
        """
        import re
        match = re.search(r'<confidence>(\d+)</confidence>', response)
        if match:
            confidence_int = int(match.group(1))
            # Clamp to 0-100 range and convert to 0.0-1.0
            confidence_int = max(0, min(100, confidence_int))
            return confidence_int / 100.0
        return None

    @abstractmethod
    def get_images(self, sample: Dict) -> List:
        """
        Extract images from a sample.
        Returns: List of PIL images
        """
        pass

    def get_image_paths(self, sample: Dict) -> List[str]:
        """
        Get image paths without loading them.
        Returns: List of absolute path strings

        Override this for tasks with file-based images to enable lazy loading.
        Tasks with embedded images (e.g., HuggingFace datasets) should not override.
        """
        raise NotImplementedError(
            f"Task {self.name} does not support lazy image loading. "
            "Override get_image_paths() to enable it."
        )

    @abstractmethod
    def get_ground_truth(self, sample: Dict) -> Any:
        """Extract ground truth label/answer from sample."""
        pass

    def get_sample_id(self, sample: Dict, index: int = None) -> Dict:
        """
        Get a unique identifier for a sample.
        Used for tracking train/val/test splits.

        Args:
            sample: The sample dict
            index: Optional index in the dataset (fallback if no ID fields found)

        Returns:
            Dict with identifier fields
        """
        # Default implementation: look for common ID fields
        ident = {}
        for field in ['root_id', 'segment_id', 'id', 'neuron_id', 'file_name']:
            if field in sample:
                ident[field] = sample[field]

        if not ident and index is not None:
            ident = {'index': index}

        return ident

    def get_dataset_hash(self) -> Optional[str]:
        """Return the hash of the dataset file, if computed during load_dataset."""
        return self._dataset_hash

    def get_balance_group(self, sample: Dict) -> Any:
        """
        Return the group label for class balancing.

        Override this in subclasses to customize how samples are grouped for
        class balancing. For example, merge_action_multiple_choice groups
        'none' vs 'not_none' instead of individual answer letters.

        Default: returns the same as get_ground_truth().
        """
        return self.get_ground_truth(sample)

    def get_split_group_key(self, sample: Dict) -> Optional[tuple]:
        """
        Get the grouping key for train/val/test splitting.

        Override this in subclasses where multiple samples may represent the
        same underlying instance (e.g., same error location). All samples with
        the same group key will be placed in the same split to prevent data leakage.

        Default: returns None (no grouping, each sample split independently).
        """
        return None

    def uses_connected_component_splitting(self) -> bool:
        """
        Whether to use connected component merging for group-based splitting.

        When True, group keys with multiple elements (e.g., (segment1_id, segment2_id))
        will be merged if they share any element. This is useful for tasks like
        segment_identity where samples involving the same segment should stay together.

        Default: False (group keys are treated atomically).
        """
        return False

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    @abstractmethod
    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """
        Format the expected/target response.
        Can include teacher model responses if available.

        Args:
            sample: The sample dict
            use_teacher_response: Whether to use teacher model responses
            teacher_response_column: Column name containing teacher responses
            rationale_dropout_prob: Probability of dropping the <analysis> section (0.0-1.0).
                                   When dropped, only the <answer> tag is returned.

        Returns: The response string with tags
        """
        pass

    @abstractmethod
    def create_reward_function(self) -> Callable:
        """
        Create reward function for RL training.
        Returns: Function that takes (completions, **sample_fields) -> rewards
        """
        pass

    @abstractmethod
    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """
        Format a complete sample for training (used by SFT).
        Each task implements its own image/content formatting.

        Args:
            sample: Raw sample from dataset
            use_teacher_response: Whether to use teacher model responses if available
            teacher_response_column: Column name containing teacher responses
            rationale_dropout_prob: Probability of dropping the <analysis> section (0.0-1.0)
            lazy_images: If True, store image paths instead of loading PIL images.
                        Enables lazy loading during training for memory efficiency.

        Returns:
            Dict with:
              - 'messages': List of message dicts (ChatML format with images or paths)
              - 'ground_truth': Ground truth for reward/eval
              - '_image_paths': List of image paths (only when lazy_images=True)
        """
        pass

    def _apply_rationale_dropout(self, response: str, answer_text: str,
                                  rationale_dropout_prob: float) -> str:
        """
        Apply rationale dropout to a response.

        If dropout is triggered, returns just the answer without analysis.
        Otherwise returns the full response.

        Args:
            response: The full response with <analysis> and <answer> tags
            answer_text: The answer text (e.g., "a", "yes", "none")
            rationale_dropout_prob: Probability of dropping the analysis (0.0-1.0)

        Returns:
            Either the full response or just the answer tag
        """
        if rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob:
            return f"<answer>{answer_text}</answer>"
        return response


class SegmentClassificationTask(TaskConfig):
    """Neuronal segment classification task."""

    CLASS_MAPPING = {
        "a": "a single soma and process(es)",
        "b": "multiple somas (and processes)",
        "c": "Processes without a soma: These can be axons, dendrites, synapses",
        "d": "Nucleus",
        "e": "Non-neuronal types. These can be glial cells, blood vessels.",
        "f": "None of the above.",
        "g": "Unsure"
    }

    REVERSE_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

    # Classes to exclude from training
    EXCLUDED_CLASSES = ['e', 'f', 'g']

    def __init__(self):
        super().__init__(
            name="segment_classification",
            description="Classify 3D neuronal segments from EM data",
            dataset_source="jeffbbrown2/ConnectomeBench",
            dataset_config="MICrONS, Segment Classification",
            num_images=3,  # front, side, top orthogonal views
        )

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        from datasets import load_dataset
        # Note: SegmentClassificationTask uses HuggingFace datasets, not local parquet
        # dataset_path parameter is not used here but kept for API consistency
        dataset = load_dataset(
            self.dataset_source,
            self.dataset_config,
            split="train",
            cache_dir=cache_dir
        )
        # Add original index before any filtering/shuffling
        dataset = dataset.add_column('_original_parquet_idx', list(range(len(dataset))))
        return dataset

    def filter_dataset(self, dataset):
        """Filter out unwanted classes."""
        excluded_descriptions = [self.CLASS_MAPPING[c] for c in self.EXCLUDED_CLASSES]
        return dataset.filter(
            lambda x: x['ground_truth'] not in excluded_descriptions
        )

    def get_images(self, sample: Dict) -> List:
        """Get the 3 orthogonal views."""
        images = [
            sample['option_1_front_path'],
            sample['option_1_side_path'],
            sample['option_1_top_path']
        ]
        if len(images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(images)}"
            )
        return images

    def get_ground_truth(self, sample: Dict) -> str:
        """Return the ground truth class description."""
        return sample['ground_truth']



    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the segment classification prompt."""
        species = sample['species']
        xmin, ymin, zmin = sample['xmin'], sample['ymin'], sample['zmin']
        xmax, ymax, zmax = sample['xmax'], sample['ymax'], sample['zmax']
        box_size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

        prompt = f"""You are an expert at analyzing neuronal morphology.

We have the electron microscopy data from the {species} brain.

In the images, we have a selected 3D segmentation that is supposed to correspond to a complete neuronal structure. However, it could have split/merge errors as the segmentation algorithm makes mistakes.

The 3D snapshots are three different views of the same segment. The dimensions of the segment's bounding box are {box_size[0]} x {box_size[1]} x {box_size[2]} nm. Describe in detail what you see using the information in the 3D snapshots. Is the segment a neuron (soma and processes)? Multiple neurons merged together (multiple somas)? Processes like axon and dendrites without a cell body? Non-neuronal structures like glia, astrocytes, or blood vessels? Inspect very closely to avoid making errors, using the 3D views and size of the bounding box in your reasoning.

For {species} neurons, the somas tend to be round and generally {'a single process extends' if species == 'fly' else 'multiple processes extend'} from them {'before it branches into many processes' if species == 'fly' else 'outwards'}. Processes can be axons or dendrites, long and often branching. Synapses can also be considered as a part of processes, and these are often small segments (often smaller than a cubic micron). The nucleuses are round and do not have any processes extending from them. Blood vessels are tubular and obviously do not have any processes extending from them. Glial cells lack the branching processes of neurons, and instead appear like jagged masses.

Choose the best answer:
a) A single soma and process(es).
b) Multiple somas (and processes)
c) Processes without a soma. These can be axons, dendrites, synapses.
d) Nucleus.
e) Non-neuronal types. These can be glial cells, blood vessels.
f) None of the above.
g) Unsure

"""
        if answer_only:
            prompt += "Surround your final answer (the letter a, b, c, d, e, f, or g) with <answer> and </answer> tags."
        else:
            prompt += """Surround your analysis with <analysis> and </analysis> tags.
Surround your final answer (the letter a, b, c, d, e, f, or g) with <answer> and </answer> tags."""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        ground_truth = self.get_ground_truth(sample)
        answer_key = self.REVERSE_MAPPING.get(ground_truth, "g")

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_key}</answer>"""
                return self._apply_rationale_dropout(response, answer_key, rationale_dropout_prob)

        # Default synthetic response
        box_size = np.array([
            sample['xmax'] - sample['xmin'],
            sample['ymax'] - sample['ymin'],
            sample['zmax'] - sample['zmin']
        ])

        response = f"""<analysis>
Based on the three orthogonal views of this segment, I can analyze its structure.

The bounding box dimensions of {box_size[0]} x {box_size[1]} x {box_size[2]} nm provide important scale information.

Looking at the morphology across all three views, this segment appears to be: {ground_truth}
</analysis>

<answer>{answer_key}</answer>"""
        return self._apply_rationale_dropout(response, answer_key, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with 3 orthogonal view images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            answer_key = self.REVERSE_MAPPING.get(ground_truth, "g")
            response_text = f"<answer>{answer_key}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)  # Don't double-dropout

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function for GRPO."""
        import re

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract answer from tags
                answer_match = re.search(r'<answer>\s*([a-g])\s*</answer>', completion_text.lower())

                reward = 0.0
                predicted_letter = None
                predicted_description = None

                if answer_match:
                    predicted_letter = answer_match.group(1)
                    predicted_description = self.CLASS_MAPPING.get(predicted_letter)

                    if predicted_description == gt:
                        reward = 1.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging for first few
                if idx < 3:
                    gt_key = self.REVERSE_MAPPING.get(gt, "?")
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: {gt_key} - {gt}")
                    print(f"Predicted: {predicted_letter if predicted_letter else 'NONE'} - {predicted_description if predicted_description else 'NO ANSWER FOUND'}")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class SplitActionTask(TaskConfig):
    """Split action verification task - binary yes/no."""

    def __init__(self):
        super().__init__(
            name="split_action",
            description="Verify if a segmentation split is correct (good/bad)",
            dataset_source="splits-parquet",  # Directory name in dataset volume
            dataset_config=None,
            num_images=3,  # 3 views after filtering out pre_split images
        )
        # Path to dataset directory - set by load_dataset() or manually
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
                       The full path will be: cache_dir / self.dataset_source
                       E.g., "/datasets" -> "/datasets/splits-parquet"
                       Or "training_data" -> "training_data/splits-parquet"
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Upload it first using:\n"
                f"  modal run scripts/model-post-training/upload_utils.py::upload_directory \\\n"
                f"    --local-path 'data/splits-parquet' \\\n"
                f"    --remote-path 'splits-parquet'"
            )
        # Store the dataset directory for use by get_images()
        self._dataset_dir = dataset_dir

        # Compute hash of parquet file for reproducibility tracking
        self._dataset_hash = self._compute_file_hash(parquet_path)

        # Load only essential columns - metadata_full is excluded (too large, causes hangs)
        # metadata now contains only: root_id, center, split_hash, species
        df = pd.read_parquet(parquet_path, columns=['answer', 'images', 'metadata'])

        # Add original parquet index before any filtering/shuffling
        df['_original_parquet_idx'] = range(len(df))

        # Convert the DataFrame to Dataset
        # Images are loaded on-the-fly in get_images() to avoid memory issues
        dataset = Dataset.from_pandas(df)

        return dataset

    def filter_dataset(self, dataset, filter_by_size: bool = False,
                       deduplicate_by_location: bool = True):
        """Filter dataset and apply quality filters.

        Quality filters (from empirical testing to eliminate "unfair" judgments):
        - gt_size > 75: eliminates unfair judgments from tiny proofread roots
        - candidate_size > 20: eliminates unfair judgments from tiny candidates
        - Split precision factor k=20: redefines is_good = (good_part > bad_part * k)
          to make splits more "surgical"

        Args:
            dataset: HuggingFace Dataset to filter
            filter_by_size: Apply size-based quality filters
            deduplicate_by_location: If True, keep at most one good and one bad
                split per unique (center_um, root_id) combination. This reduces
                redundancy from multiple samples at the same physical location.
        """

        # if filter_by_size:
        #     import numpy as np

        #     # Filter parameters
        #     MIN_GT_SIZE = 75
        #     MIN_CANDIDATE_SIZE = 20
        #     PRECISION_FACTOR_K = 20

        #     def is_valid_sample(sample):
        #         # Validate images exist
        #         images = sample.get('images')
        #         if images is None:
        #             raise ValueError("Sample missing 'images' field")
        #         if isinstance(images, (list, np.ndarray)):
        #             if len(images) == 0:
        #                 raise ValueError("Sample has empty 'images' list")
        #         elif isinstance(images, str):
        #             if len(images) == 0:
        #                 raise ValueError("Sample has empty 'images' string")
        #         else:
        #             raise ValueError(f"Unexpected images type: {type(images)}")

        #         # Validate metadata and evaluation_stats exist
        #         metadata = sample.get('metadata')
        #         if metadata is None or not isinstance(metadata, dict):
        #             raise ValueError("Sample missing 'metadata' field or metadata is not a dict")

        #         eval_stats = metadata.get('evaluation_stats')
        #         if eval_stats is None or not isinstance(eval_stats, dict):
        #             raise ValueError("Sample missing 'evaluation_stats' in metadata")

        #         # Apply size filters
        #         gt_size = eval_stats.get('gt_size', 0)
        #         candidate_size = eval_stats.get('candidate_size', 0)

        #         if gt_size <= MIN_GT_SIZE:
        #             return False
        #         if candidate_size <= MIN_CANDIDATE_SIZE:
        #             return False

        #         # Apply precision factor filter
        #         good_part = eval_stats.get('good_part', 0)
        #         bad_part = eval_stats.get('bad_part', 0)

        #         # Keep sample only if it clearly falls into one category:
        #         # - Good split: good_part > bad_part * k (surgical precision)
        #         # - Bad split: bad_part > good_part * k (clearly incorrect)
        #         is_good_by_precision = good_part > bad_part * PRECISION_FACTOR_K
        #         is_bad_by_precision = bad_part > good_part * PRECISION_FACTOR_K

        #         if not (is_good_by_precision or is_bad_by_precision):
        #             return False

        #         return True

        #     original_len = len(dataset)
        #     dataset = dataset.filter(is_valid_sample)
        #     filtered_len = len(dataset)

        #     if filtered_len < original_len:
        #         print(f"Filtered out {original_len - filtered_len} samples "
        #             f"({original_len} -> {filtered_len}, "
        #             f"gt_size>{MIN_GT_SIZE}, candidate_size>{MIN_CANDIDATE_SIZE}, k={PRECISION_FACTOR_K})")

        # if deduplicate_by_location:
        #     import pandas as pd

        #     original_len = len(dataset)

        #     # Convert to pandas for fast vectorized operations
        #     df = dataset.to_pandas()

        #     # Track original dataset index
        #     df['_orig_idx'] = range(len(df))

        #     # Extract root_id and center from metadata column
        #     def extract_group_key(metadata):
        #         if not isinstance(metadata, dict):
        #             return (None, None)
        #         root_id = metadata.get('root_id')
        #         center = metadata.get('center')
        #         if center is not None:
        #             # Normalize to nearest micron (1000 nm)
        #             center_um = tuple(int(round(c / 1000)) for c in center)
        #         else:
        #             center_um = None
        #         return (center_um, root_id)

        #     df['_group_key'] = df['metadata'].apply(extract_group_key)

        #     # Shuffle to randomize which sample is kept per group
        #     df = df.sample(frac=1, random_state=42)

        #     # Group by (center_um, root_id, answer) and keep first of each
        #     # This gives us at most one good and one bad per location
        #     keep_rows = df.groupby(['_group_key', 'answer'], sort=False).head(1)
        #     keep_indices = sorted(keep_rows['_orig_idx'].tolist())

        #     # Select from original dataset
        #     dataset = dataset.select(keep_indices)

        #     deduped_len = len(dataset)
        #     if deduped_len < original_len:
        #         num_groups = df['_group_key'].nunique()
        #         print(f"Deduplicated by location: {original_len} -> {deduped_len} samples "
        #               f"({num_groups} unique (center_um, root_id) groups, "
        #               f"keeping at most 1 good + 1 bad per group)")

        return dataset

    def get_images(self, sample: Dict) -> List:
        """Get the images from the sample, loading from disk.

        Only loads "split" images (post-split visualization), filtering out
        "pre_split" images to reduce memory usage.

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/splits-parquet" (Modal default, for backwards compat)
        """
        from pathlib import Path
        from PIL import Image

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            # Fallback for backwards compatibility
            base_path = Path("/datasets/splits-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Filter to only include split images, not pre_split images
        # Split images contain "split_" but not "pre_split_"
        filtered_paths = [
            p for p in image_paths
            if 'pre_split' not in Path(p).name.lower()
        ]

        loaded_images = []
        for i, rel_path in enumerate(filtered_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                # Validate image properties
                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                # Convert to RGB if needed (handle RGBA, L, etc.)
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    # Grayscale - convert to RGB
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(filtered_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                # Include sample info for debugging
                if 'question_type' in sample:
                    print(f"  Question type: {sample['question_type']}")
                if 'answer' in sample:
                    print(f"  Answer: {sample['answer']}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {filtered_paths}"
            )
        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them.

        Same path resolution and filtering as get_images(), but returns
        paths instead of PIL images for lazy loading support.
        """
        from pathlib import Path

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/splits-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Filter to only include split images, not pre_split images
        filtered_paths = [
            p for p in image_paths
            if 'pre_split' not in Path(p).name.lower()
        ]

        # Return absolute paths as strings
        abs_paths = [str(base_path / rel_path) for rel_path in filtered_paths]

        if len(abs_paths) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {filtered_paths}"
            )
        return abs_paths

    def get_ground_truth(self, sample: Dict) -> bool:
        """Return boolean answer."""
        return sample['answer']

    def get_sample_id(self, sample: Dict, index: int = None) -> Dict:
        """
        Get a unique identifier for a split action sample.

        Uses metadata.split_hash as the primary identifier, with root_id as backup.
        """
        ident = {}
        metadata = sample.get('metadata', {})
        if isinstance(metadata, dict):
            if 'split_hash' in metadata:
                ident['split_hash'] = metadata['split_hash']
            if 'root_id' in metadata:
                ident['root_id'] = metadata['root_id']

        # Fallback to index if no identifiers found
        if not ident and index is not None:
            ident = {'index': index}

        return ident

    def get_split_group_key(self, sample: Dict) -> tuple:
        """
        Get the grouping key for train/val/test splitting.

        Returns (center_um, root_id) tuple to ensure all samples at the same
        physical location (same error instance) stay in the same split.

        This prevents data leakage where the model sees the same error location
        during training and evaluation (just with different image augmentations).
        """
        metadata = sample.get('metadata', {})
        if not isinstance(metadata, dict):
            return (None, None)

        # Get root_id
        root_id = metadata.get('root_id')

        # Get center and normalize to nearest micron (1000 nm)
        center = metadata.get('center')
        if center is not None:
            # Convert to tuple of integers (microns)
            center_um = tuple(int(round(c / 1000)) for c in center)
        else:
            center_um = None

        return (center_um, root_id)

    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the split verification prompt."""
        # prompt = """You are a EM connectomics proofreading expert, and you're deciding if a segment should be split into multiple segments or not.
#You are shown multiple views of 3D segments. The resulting two segments that would result from the proposed split are shown in blue and green.
#Images are presented in groups of 3 (front, side, top).
        prompt = """You are a EM connectomics proofreading expert, and you're deciding if a segment should be split into multiple segments or not.
You are shown multiple views of 3D segments. The resulting two segments that would result from the proposed split are shown in two different colors.
Images are presented in groups of 3 (front, side, top).

**Answer with yes or no**:
- yes = This is a good split (segment should be separated into multiple segments)
- no = This is a bad split (segment should not be separated into multiple segments)
"""

# """
# You are attempting to proofread a 3D segmentation of a neuron.
# You are shown multiple views of a 3D segmentation. This algorithm that initial produced the segment intended to isolate a segment that corresponded to a single neuron. However, that segmentation algorithm may have made a mistake and merge segments together that correspond to distinct neurons. As a result, another algorithm has proposed a split actions to separate this segment into multiple separate segments.

# Your task is to determine: **Is this a good split or a bad split?**
# A good split either a) separates segments that correspond to distinct neuron segments or b) removes portions of a segment from one neuron that likely come from another neuron (for instance, if there a process in the segment that likely come from another neuron due to unnatural connection or morphology). A bad split breaks up a single neuron into multiple segments.

# **Answer with yes or no**:
# - yes = This is a good split (boundaries are correct)
# - no = This is a bad split (boundaries are incorrect)

# """

        if answer_only:
            prompt += "Surround your final answer (yes, no) with <answer> and </answer> tags."
        else:
            prompt += """Surround your analysis and reasoning with <analysis> and </analysis> tags.
Surround your final answer (yes, no) with <answer> and </answer> tags."""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        ground_truth = self.get_ground_truth(sample)
        answer_text = "yes" if ground_truth else "no"

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_text}</answer>"""
                return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

        # Default synthetic response
        split_quality = "good" if ground_truth else "bad"
        response = f"""<analysis>
Examining the segmentation boundaries across the provided views, this appears to be a {split_quality} split.
The boundaries {'properly separate distinct structures' if ground_truth else 'show signs of incorrect segmentation'}.
</analysis>

<answer>{answer_text}</answer>"""
        return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with split action images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            answer_text = "yes" if ground_truth else "no"
            response_text = f"<answer>{answer_text}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function for GRPO."""
        import re

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract answer from tags
                answer_match = re.search(r'<answer>\s*(yes|no)\s*</answer>', completion_text.lower())

                reward = 0.0
                predicted_answer = None

                if answer_match:
                    predicted_answer = answer_match.group(1)
                    predicted_bool = (predicted_answer == "yes")

                    if predicted_bool == gt:
                        reward = 1.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging
                if idx < 3:
                    pred_answer = predicted_answer if predicted_answer else "NONE"
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: {gt} ({'yes' if gt else 'no'})")
                    print(f"Predicted: {pred_answer}")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class MergeActionTask(TaskConfig):
    """Merge action verification task - binary yes/no."""

    def __init__(self):
        super().__init__(
            name="merge_action",
            description="Verify if a segmentation merge is correct (good/bad)",
            dataset_source="merge-parquet",  # Directory name in dataset volume
            dataset_config=None,
            num_images=3,  # 3 views (front, side, top)
        )
        # Path to dataset directory - set by load_dataset() or manually
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
                       The full path will be: cache_dir / self.dataset_source
                       E.g., "/datasets" -> "/datasets/merge-parquet"
                       Or "training_data" -> "training_data/merge-parquet"
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Upload it first using:\n"
                f"  modal run scripts/model-post-training/upload_utils.py::upload_directory \\\n"
                f"    --local-path 'data/merge-parquet' \\\n"
                f"    --remote-path 'merge-parquet'"
            )

        # Store the dataset directory for use by get_images()
        self._dataset_dir = dataset_dir

        # Compute hash of parquet file for reproducibility tracking
        self._dataset_hash = self._compute_file_hash(parquet_path)

        df = pd.read_parquet(parquet_path)

        # Add original parquet index before any filtering/shuffling
        df['_original_parquet_idx'] = range(len(df))

        # Convert the DataFrame to Dataset
        dataset = Dataset.from_pandas(df)

        return dataset

    def filter_dataset(self, dataset):
        """Filter merge action samples.

        Args:
            dataset: HuggingFace Dataset to filter
        """


        # # I don't understand the point of this
        #  if deduplicate_by_location:
        #     original_len = len(dataset)
        #     df = dataset.to_pandas()
        #     df['_orig_idx'] = range(len(df))

        #     def extract_dedup_key(metadata):
        #         """Create dedup key from (center_um, segment_set)."""
        #         if not isinstance(metadata, dict):
        #             return None
        #         # Round center_nm to nearest micron
        #         center_nm = metadata.get('center_nm')
        #         if center_nm is not None:
        #             center_um = tuple(int(round(c / 1000)) for c in center_nm)
        #         else:
        #             return None
        #         # Create frozenset of candidate + partners (order-independent)
        #         candidate = metadata.get('candidate_root_id')
        #         partners = metadata.get('correct_partner_ids', [])
        #         if candidate is None:
        #             return None
        #         segment_set = frozenset([candidate] + list(partners))
        #         return (center_um, segment_set)

        #     df['_dedup_key'] = df['metadata'].apply(extract_dedup_key)

        #     # Drop rows with None keys, then deduplicate
        #     valid_mask = df['_dedup_key'].notna()
        #     df_valid = df[valid_mask]
        #     df_invalid = df[~valid_mask]

        #     # Keep first occurrence of each (center_um, segment_set)
        #     df_deduped = df_valid.drop_duplicates(subset='_dedup_key', keep='first')

        #     # Combine back with invalid rows (keep them)
        #     keep_indices = sorted(df_deduped['_orig_idx'].tolist() + df_invalid['_orig_idx'].tolist())
        #     dataset = dataset.select(keep_indices)

        #     deduped_len = len(dataset)
        #     if deduped_len < original_len:
        #         print(f"Deduplicated by (center_um, segment_set): {original_len} -> {deduped_len} samples")

        return dataset

    def get_images(self, sample: Dict) -> List:
        """Get the images from the sample, loading from disk.

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/merge-parquet" (Modal default, for backwards compat)
        """
        from pathlib import Path
        from PIL import Image

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            # Fallback for backwards compatibility
            base_path = Path("/datasets/merge-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                # Validate image properties
                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                # Convert to RGB if needed (handle RGBA, L, etc.)
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                if 'question_type' in sample:
                    print(f"  Question type: {sample['question_type']}")
                if 'answer' in sample:
                    print(f"  Answer: {sample['answer']}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )
        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them.

        Same path resolution as get_images(), but returns paths instead
        of PIL images for lazy loading support.
        """
        from pathlib import Path

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/merge-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Return absolute paths as strings
        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        if len(abs_paths) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )
        return abs_paths

    def get_ground_truth(self, sample: Dict) -> bool:
        """Return boolean answer."""
        return sample['answer']

    def get_sample_id(self, sample: Dict, index: int = None) -> Dict:
        """
        Get a unique identifier for a merge action sample.

        Uses metadata.sample_dir as the primary identifier.
        """
        ident = {}
        metadata = sample.get('metadata', {})
        if isinstance(metadata, dict):
            if 'sample_dir' in metadata:
                ident['sample_dir'] = metadata['sample_dir']

        # Fallback to index if no identifiers found
        if not ident and index is not None:
            ident = {'index': index}

        return ident

    def get_split_group_key(self, sample: Dict) -> tuple:
        """
        Get the grouping key for train/val/test splitting.

        Returns (segment1_id, interface_um) tuple to ensure all samples at the same
        merge location stay in the same split.

        CONTAMINATION PREVENTION: The merge dataset has a "1 correct + N distractors"
        structure where each (segment1_id, interface_point) location has exactly one
        correct merge candidate (True) and multiple incorrect candidates (False).
        97% of samples share their location with other samples. Without grouping,
        random splits would leak information - the model could memorize which segment
        is correct at each location rather than learning actual merge criteria.
        """
        metadata = sample.get('metadata', {})
        if not isinstance(metadata, dict):
            return (None, None)

        # Get segment1_id directly from metadata
        segment1_id = metadata.get('segment1_id')

        # Get interface_point and round to nearest micron (1000 nm)
        interface_point = metadata.get('interface_point')
        if interface_point is not None:
            interface_um = tuple(int(round(p / 1000)) for p in interface_point)
        else:
            interface_um = None

        return (segment1_id, interface_um)

    def format_prompt(self, sample: Dict, answer_only: bool = False, analysis_only: bool = False, request_confidence: bool = False) -> str:
        """Create the merge verification prompt.

        Args:
            sample: The sample to create prompt for
            answer_only: If True, only ask for answer without analysis
            analysis_only: If True, only ask for analysis without answer
            request_confidence: If True, ask model to report confidence 0-100
        """
#         prompt = """
# You are a EM connectomics proofreading expert, and you're deciding if two segment should be merged together or not.
# You are shown multiple views of two 3D segments. The original segment is blue and a potential merge candidate segment is orange.
# The image is a cropped 3D volume around the center of the volume, so you should pay attention to discontinuities in the center of the image.
# Images are presented in groups of 3 (front, side, top).

        prompt = """
You are proofreading connectomics data, and you're deciding if two segment should be merged together or not.
You are shown multiple views of two 3D segments. The original segment is blue and a potential merge candidate segment is orange.
Images are presented in groups of 3 (front, side, top).

**Answer with yes or no**:
- yes = This is a good merge (segments should be connected)
- no = This is a bad merge (segments should remain separate)
"""
#         if liconn:
# #             prompt += """The segmentations of axons are not perfect and you may see gaps in them. If it seems like if you could draw a smooth line through the fragmented segments to connect them, then you should treat them how you would a continuous axon. However, this is a new dataset and there are some genuine segmentations artifacts that haven't been filtered out. If they do not look like fragmented axons and it's just weird stuff that doesn't look like a typical neuron segmentation you've seen before, then you can output "pass".  
# # """
#             prompt += """This datasets has some weird segmentation artifacts. If what you're seeing doesn't look familiar, please write why and output "pass" as your answer. We really care about calibrated responses for this task"""
#         if truncate_analysis:
#             prompt += """Keep your analysis to 20 words or less"""

#         if guidance:
#             prompt += """

# The following heuristics are gained through training another VLM on this task. These heuristics lead to an increase in accuracy from 0.34 to 0.95. You should follow them. 

# What heuristics are GAINED through training?

# Merge Heuristics:
# - No visible gap; segments meet seamlessly at the junction
# - Consistent diameter/thickness across the color change
# - Continuous curvature with no abrupt kink or direction change
# - Alignment preserved across all orthogonal views (front/side/top)
# - Matching branching pattern; lateral branches continue across the junction
# - Consistent texture/surface features on both sides of the junction
# - No misalignment, offset, or overlap at the interface
# - Uniform shape/size profile along the entire path
# - Smooth, uninterrupted elongated trajectory through both segments\n- Junction proximity consistent with expected continuity at the given scale",
# - Concentric/overlapping spherical structure are generally splits between nucleus and soma and should be merged together.

# Additional instincts you need to reject (these are mistakes you keep making):
# - If the orange segment is taking up a lot of space and is in front of the blue segment in all views, you tend to accept the merge. This is bad heuristics. What really matters is if a) they both have an endpoint at the middle of the volume (you should be integrating knowledge from all three projections) and b) if the endpoints line up in the 3D context, have similar size right at the interface point.
# """


# Reject Heuristics:
# - No continuous surface or volumetric connection between segments in any view
# - Crossings are over/under passes, not joins (no shared boundary)
# - Morphology mismatch: thin smooth filament vs thick, branched/spiny structure
# - Orientation/trajectory mismatch: paths run in different directions or planes
# - Parallel adjacency without convergence or intersection
# - At near-contact points, no tapering or blending\u2014edges stay distinct
# - One segment wraps around/threads through gaps of the other but stays separate
# - Surrounding/occluding network does not integrate the other segment\n- Consistent separation confirmed across front/side/top views\n- Differences in thickness/texture/complexity imply different object types"
#             """

        if answer_only:
            prompt += f"Surround your final answer (yes or no) with <answer> and </answer> tags."
        elif analysis_only:
            prompt += "Surround your analysis with <analysis> and </analysis> tags."
        else:
            prompt += f"""Surround your analysis and reasoning with <analysis> and </analysis> tags.
Surround your final answer (yes or no) with <answer> and </answer> tags."""

        # Add confidence request if requested
        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        ground_truth = self.get_ground_truth(sample)
        answer_text = "yes" if ground_truth else "no"

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_text}</answer>"""
                return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

        # Default synthetic response
        merge_quality = "good" if ground_truth else "bad"
        response = f"""<analysis>
Examining the merged segments across the provided views, this appears to be a {merge_quality} merge.
The connection {'shows structural continuity indicating these segments belong together' if ground_truth else 'appears artificial, suggesting these are distinct structures'}.
</analysis>

<answer>{answer_text}</answer>"""
        return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with merge action images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            answer_text = "yes" if ground_truth else "no"
            response_text = f"<answer>{answer_text}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function for GRPO."""
        import re

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract answer from tags
                answer_match = re.search(r'<answer>\s*(yes|no)\s*</answer>', completion_text.lower())

                reward = 0.0
                predicted_answer = None

                if answer_match:
                    predicted_answer = answer_match.group(1)
                    predicted_bool = (predicted_answer == "yes")

                    if predicted_bool == gt:
                        reward = 1.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging
                if idx < 3:
                    pred_answer = predicted_answer if predicted_answer else "NONE"
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: {gt} ({'yes' if gt else 'no'})")
                    print(f"Predicted: {pred_answer}")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class MergeActionMultipleChoiceTask(TaskConfig):
    """Merge action multiple choice task - pick correct merge partner from up to 4 candidates."""

    # Valid answer choices
    VALID_ANSWERS = ['a', 'b', 'c', 'd', 'none']

    def __init__(self):
        super().__init__(
            name="merge_action_multiple_choice",
            description="Select the correct merge partner from up to 4 candidates, or none",
            dataset_source="merge-multiple-choice-parquet",  # Directory name in dataset volume
            dataset_config=None,
            num_images=12,  # Up to 4 candidates * 3 views each (actual may be less)
        )
        # Path to dataset directory - set by load_dataset() or manually
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
                       The full path will be: cache_dir / self.dataset_source
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Generate it using the proofreading evaluator pipeline."
            )

        # Store the dataset directory for use by get_images()
        self._dataset_dir = dataset_dir

        # Compute hash of parquet file for reproducibility tracking
        self._dataset_hash = self._compute_file_hash(parquet_path)

        df = pd.read_parquet(parquet_path)

        # Add original parquet index before any filtering/shuffling
        df['_original_parquet_idx'] = range(len(df))

        # Convert the DataFrame to Dataset
        dataset = Dataset.from_pandas(df)

        return dataset

    def filter_dataset(self, dataset):
        """No filtering needed for merge action multiple choice."""
        return dataset

    def get_images(self, sample: Dict) -> List:
        """Get the images from the sample, loading from disk.

        Images are stored as a nested structure:
        - sample['images']: List of lists, where each inner list has 3 image paths
          for one candidate (front, side, top views)

        Or flat structure:
        - sample['images']: Flat list of image paths, grouped by candidate
          (first 3 for option A, next 3 for option B, etc.)

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/merge-multiple-choice-parquet" (Modal default)
        """
        from pathlib import Path
        from PIL import Image

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/merge-multiple-choice-parquet")

        image_paths = sample['images']

        # Handle numpy arrays
        if hasattr(image_paths, 'tolist'):
            image_paths = image_paths.tolist()

        # Flatten if nested
        if len(image_paths) > 0 and isinstance(image_paths[0], list):
            flat_paths = []
            for candidate_paths in image_paths:
                flat_paths.extend(candidate_paths)
            image_paths = flat_paths

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                raise

        # Note: actual number of images varies based on number of candidates
        # Each candidate has 3 views, so valid counts are 3, 6, 9, or 12
        if len(loaded_images) % 3 != 0:
            raise ValueError(
                f"Expected multiple of 3 images (3 per candidate), got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )

        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them.

        Same path resolution and flattening as get_images(), but returns
        paths instead of PIL images for lazy loading support.
        """
        from pathlib import Path

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/merge-multiple-choice-parquet")

        image_paths = sample['images']

        # Handle numpy arrays
        if hasattr(image_paths, 'tolist'):
            image_paths = image_paths.tolist()

        # Flatten if nested
        if len(image_paths) > 0 and isinstance(image_paths[0], list):
            flat_paths = []
            for candidate_paths in image_paths:
                flat_paths.extend(candidate_paths)
            image_paths = flat_paths

        # Return absolute paths as strings
        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        # Validate: each candidate has 3 views
        if len(abs_paths) % 3 != 0:
            raise ValueError(
                f"Expected multiple of 3 images (3 per candidate), got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )

        return abs_paths

    def get_num_candidates(self, sample: Dict) -> int:
        """Get number of merge candidates in this sample."""
        # Check for explicit num_candidates (set during inference with sampling)
        if '_num_candidates' in sample:
            return sample['_num_candidates']

        image_paths = sample['images']
        # Handle numpy arrays and lists
        if hasattr(image_paths, 'tolist'):
            image_paths = image_paths.tolist()
        if len(image_paths) > 0 and isinstance(image_paths[0], list):
            return len(image_paths)
        else:
            return len(image_paths) // 3

    def get_ground_truth(self, sample: Dict) -> str:
        """Return the correct answer letter (a, b, c, d) or 'none'."""
        answer = sample['answer']
        if isinstance(answer, str):
            return answer.lower()
        # Handle integer index (0=a, 1=b, etc.)
        if isinstance(answer, int):
            if answer < 0:
                return 'none'
            return chr(ord('a') + answer)
        return 'none'

    def get_balance_group(self, sample: Dict) -> str:
        """
        Return balance group: 'none' vs 'not_none'.

        For class balancing, we want to balance between samples where no merge
        is correct ('none') vs samples where one of the candidates is correct
        ('a', 'b', 'c', 'd').
        """
        gt = self.get_ground_truth(sample)
        return 'none' if gt == 'none' else 'not_none'

    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the multiple choice merge prompt."""
        num_candidates = self.get_num_candidates(sample)
        option_letters = ['A', 'B', 'C', 'D'][:num_candidates]

        options_text = "\n".join([
            f"- {letter} = Merge with candidate {letter}"
            for letter in option_letters
        ])

        prompt = f"""You are a EM connectomics proofreading expert deciding which segment, if any, should be merged with the primary segment.

You are shown the primary segment (blue) paired with up to {num_candidates} candidate segments (orange) that could potentially be merged. Each candidate is shown in 3 views (front, side, top).

The images are presented in order:
{chr(10).join([f'- Option {letter}: Images {i*3+1}-{i*3+3}' for i, letter in enumerate(option_letters)])}

**Choose the best option**:
{options_text}
- none = None of the candidates should be merged with the primary segment

"""
        if answer_only:
            prompt += "Surround your final answer (a, b, c, d, or none) with <answer> and </answer> tags."
        else:
            prompt += """Surround your analysis and reasoning with <analysis> and </analysis> tags.
Surround your final answer (a, b, c, d, or none) with <answer> and </answer> tags."""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        ground_truth = self.get_ground_truth(sample)

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{ground_truth}</answer>"""
                return self._apply_rationale_dropout(response, ground_truth, rationale_dropout_prob)

        # Default synthetic response
        if ground_truth == 'none':
            analysis = "After examining all candidate segments across the provided views, none of them show appropriate structural continuity with the primary segment. The candidates either run parallel to the primary segment, are inappropriately sized, or don't align properly at the interface point."
        else:
            option_upper = ground_truth.upper()
            analysis = f"After examining all candidates, Option {option_upper} shows the best structural continuity with the primary segment. The orange segment in this option continues in the same direction as the blue segment at the interface point, indicating they belong to the same neural process."

        response = f"""<analysis>
{analysis}
</analysis>

<answer>{ground_truth}</answer>"""
        return self._apply_rationale_dropout(response, ground_truth, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with multiple choice merge images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            response_text = f"<answer>{ground_truth}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function for GRPO."""
        import re

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract answer from tags
                answer_match = re.search(r'<answer>\s*(a|b|c|d|none)\s*</answer>', completion_text.lower())

                reward = 0.0
                predicted_answer = None

                if answer_match:
                    predicted_answer = answer_match.group(1)

                    if predicted_answer == gt:
                        reward = 1.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging
                if idx < 3:
                    pred_answer = predicted_answer if predicted_answer else "NONE"
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: {gt}")
                    print(f"Predicted: {pred_answer}")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class EndpointErrorIdentificationTask(TaskConfig):
    """Endpoint error identification task - identify if an endpoint has a split error."""

    def __init__(self):
        super().__init__(
            name="endpoint_error_identification",
            description="Identify if a skeleton endpoint is a split error (needs merge) or natural terminus",
            dataset_source="endpoints-parquet",  # Directory name in dataset volume
            dataset_config=None,
            num_images=3,  # 3 views (front, side, top)
        )
        # Path to dataset directory - set by load_dataset() or manually
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
                       The full path will be: cache_dir / self.dataset_source
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Generate it using split_data_generator.py and convert with question_dataset.py"
            )

        # Store the dataset directory for use by get_images()
        self._dataset_dir = dataset_dir

        # Compute hash of parquet file for reproducibility tracking
        self._dataset_hash = self._compute_file_hash(parquet_path)

        df = pd.read_parquet(parquet_path)

        # Add original parquet index before any filtering/shuffling
        df['_original_parquet_idx'] = range(len(df))

        # Convert the DataFrame to Dataset
        dataset = Dataset.from_pandas(df)

        return dataset

    def filter_dataset(self, dataset, require_em_images: bool = False):
        """Filter endpoint error identification samples.

        Args:
            dataset: HuggingFace Dataset to filter
            require_em_images: If True, only keep samples with 6 images (3 mesh + 3 EM views)
        """
        if require_em_images:
            original_len = len(dataset)

            def has_six_images(sample):
                images = sample.get('images', [])
                if isinstance(images, str):
                    return False  # Single image path
                return len(images) == 6

            dataset = dataset.filter(has_six_images)
            filtered_len = len(dataset)

            if filtered_len < original_len:
                print(f"Filtered by 6 images (EM required): {original_len} -> {filtered_len} samples")

        return dataset

    def get_images(self, sample: Dict) -> List:
        """Get the images from the sample, loading from disk.

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/split-errors-parquet" (Modal default)
        """
        from pathlib import Path
        from PIL import Image

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/endpoint-error-identification")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                if 'question_type' in sample:
                    print(f"  Question type: {sample['question_type']}")
                if 'answer' in sample:
                    print(f"  Answer: {sample['answer']}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )
        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them.

        Same path resolution as get_images(), but returns paths instead
        of PIL images for lazy loading support.
        """
        from pathlib import Path

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/endpoint-error-identification")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Return absolute paths as strings
        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        if len(abs_paths) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )
        return abs_paths

    def get_ground_truth(self, sample: Dict) -> bool:
        """Return boolean answer (True = split error, False = natural terminus)."""
        return sample['answer']

    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the split error identification prompt."""
        prompt = """You are attempting to proofread a 3D segmentation of a neuron.

You are shown multiple views of a 3D neuron segment centered on an endpoint.

Your task is to determine: **Is this endpoint a split error or a natural terminus?**

- **Split error**: The endpoint is where the segmentation was incorrectly split. Signs include:
  - There's a termination of the process without any signs of a synaptic bouton or other terminal structure
  - The endpoint is in the middle of what looks like a continuous structure

- **Natural terminus**: The endpoint is a true biological ending of the neural process. Signs include:
  - It's a synaptic bouton or other terminal structure
  - The ending looks biologically plausible

**Answer with yes or no**:
- yes = This is a split error (needs merge correction)
- no = This is a natural terminus (no correction needed)

"""
        if answer_only:
            prompt += "Surround your final answer (yes or no) with <answer> and </answer> tags."
        else:
            prompt += """Surround your analysis and reasoning with <analysis> and </analysis> tags.
Surround your final answer (yes or no) with <answer> and </answer> tags."""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        ground_truth = self.get_ground_truth(sample)
        answer_text = "yes" if ground_truth else "no"

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_text}</answer>"""
                return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

        # Default synthetic response
        if ground_truth:
            analysis = "Examining the endpoint across the provided views, this appears to be a split error. The process shows signs of continuing beyond this point, suggesting it was incorrectly segmented and should be merged with an adjacent segment."
        else:
            analysis = "Examining the endpoint across the provided views, this appears to be a natural terminus. The process ends in a biologically plausible manner, with no indication of an incorrect split."

        response = f"""<analysis>
{analysis}
</analysis>

<answer>{answer_text}</answer>"""
        return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with endpoint images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            answer_text = "yes" if ground_truth else "no"
            response_text = f"<answer>{answer_text}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function for GRPO."""
        import re

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract answer from tags
                answer_match = re.search(r'<answer>\s*(yes|no)\s*</answer>', completion_text.lower())

                reward = 0.0
                predicted_answer = None

                if answer_match:
                    predicted_answer = answer_match.group(1)
                    predicted_bool = (predicted_answer == "yes")

                    if predicted_bool == gt:
                        reward = 1.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging
                if idx < 3:
                    pred_answer = predicted_answer if predicted_answer else "NONE"
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: {gt} ({'yes (split error)' if gt else 'no (natural terminus)'})")
                    print(f"Predicted: {pred_answer}")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class EndpointErrorIdentificationWithEMTask(EndpointErrorIdentificationTask):
    """Endpoint error identification with EM context - uses both mesh and EM slice views."""

    def __init__(self):
        # Call grandparent __init__ to avoid overwriting our settings
        TaskConfig.__init__(
            self,
            name="endpoint_error_identification_with_em",
            description="Identify if a skeleton endpoint is a split error, using both mesh and EM slice views",
            dataset_source="endpoints-with-em-parquet",  # Directory name in dataset volume
            dataset_config=None,
            num_images=6,  # 3 mesh views + 3 EM views
        )
        self._dataset_dir: Path = None

    def filter_dataset(self, dataset):
        """Filter to only keep samples with exactly 6 images (3 mesh + 3 EM views).

        Args:
            dataset: HuggingFace Dataset to filter
        """
        original_len = len(dataset)

        def has_six_images(sample):
            images = sample.get('images', [])
            if isinstance(images, str):
                return False  # Single image path
            return len(images) == 6

        dataset = dataset.filter(has_six_images)
        filtered_len = len(dataset)

        if filtered_len < original_len:
            print(f"Filtered by 6 images (EM + mesh views): {original_len} -> {filtered_len} samples")

        return dataset

    def get_images(self, sample: Dict) -> List:
        """Get the images from the sample, loading from disk.

        Returns list of 6 images: [front, side, top, em_front, em_side, em_top]

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/endpoints-with-em-parquet"
        """
        from pathlib import Path
        from PIL import Image

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/endpoints-with-em-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )

        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them."""
        from pathlib import Path

        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/endpoints-with-em-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        # This task expects 6 source images (3 mesh + 3 EM)
        expected_source_images = 6
        if len(abs_paths) != expected_source_images:
            raise ValueError(
                f"Expected {expected_source_images} source images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )
        return abs_paths

    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the split error identification prompt with EM context."""
        prompt = """You are attempting to proofread a 3D segmentation of a neuron.

You are shown 6 images of a neuron segment centered on an endpoint:
- **Images 1-3**: 3D mesh renderings (front, side, top views)
- **Images 4-6**: Electron microscopy (EM) slices at the same location (front, side, top views) with the neuron highlighted in yellow

Your task is to determine: **Is this endpoint a split error or a natural terminus?**

**Answer with yes or no**:
- yes = This is a split error (needs merge correction)
- no = This is a natural terminus (no correction needed)

"""
#         prompt = """You are attempting to proofread a 3D segmentation of a neuron.

# You are shown 6 images of a neuron segment centered on an endpoint:
# - **Images 1-3**: 3D mesh renderings (front, side, top views)
# - **Images 4-6**: Electron microscopy (EM) slices at the same location (front, side, top views) with the neuron highlighted in yellow

# Your task is to determine: **Is this endpoint a unnatural termination or a natural terminus?**

# **Answer with yes or no**:
# - yes = This is a unnatural termination (needs merge correction)
# - no = This is a natural terminus (no correction needed)

# """
        if answer_only:
            prompt += "Surround your final answer (yes or no) with <answer> and </answer> tags."
        else:
            prompt += """Surround your analysis and reasoning with <analysis> and </analysis> tags.
Surround your final answer (yes or no) with <answer> and </answer> tags."""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt


class EndpointLocalizationTask(TaskConfig):
    """Endpoint localization task - predict x,y,z coordinates of error location from graph-overlaid images."""

    # Reward thresholds in nanometers
    FULL_REWARD_THRESHOLD_NM = 500.0    # Full reward if prediction within 500nm
    PARTIAL_REWARD_THRESHOLD_NM = 2000.0  # Partial reward up to 2um

    def __init__(self):
        super().__init__(
            name="endpoint_localization",
            description="Predict the x,y,z coordinates of the error location from graph-overlaid neuron images",
            dataset_source="endpoint-localization-parquet",
            dataset_config=None,
            num_images=3,  # front, side, top
        )
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Generate it using endpoint_localization_sampler.py"
            )

        self._dataset_dir = dataset_dir
        self._dataset_hash = self._compute_file_hash(parquet_path)

        df = pd.read_parquet(parquet_path)
        df['_original_parquet_idx'] = range(len(df))

        dataset = Dataset.from_pandas(df)
        return dataset

    def filter_dataset(self, dataset):
        """No filtering needed for endpoint localization."""
        return dataset

    def get_images(self, sample: Dict) -> List:
        """Load the 3 orthogonal view images from disk.

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/endpoint-localization-parquet"
        """
        from pathlib import Path
        from PIL import Image

        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/endpoint-localization-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )
        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them.

        Same path resolution as get_images(), but returns paths instead
        of PIL images for lazy loading support.
        """
        from pathlib import Path

        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/endpoint-localization-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Return absolute paths as strings
        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        if len(abs_paths) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )
        return abs_paths

    def get_ground_truth(self, sample: Dict) -> List[float]:
        """Return the [x, y, z] coordinates in nm."""
        answer = sample['answer']
        if hasattr(answer, 'tolist'):
            answer = answer.tolist()
        return [float(v) for v in answer]

    def get_sample_id(self, sample: Dict, index: int = None) -> Dict:
        """Get a unique identifier for an endpoint localization sample."""
        ident = {}
        metadata = sample.get('metadata', {})
        if isinstance(metadata, dict):
            if 'sample_dir' in metadata:
                ident['sample_dir'] = metadata['sample_dir']
            if 'root_id' in metadata:
                ident['root_id'] = metadata['root_id']

        if not ident and index is not None:
            ident = {'index': index}

        return ident
    def get_balance_group(self, sample: Dict) -> Any:
        return 0
    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the endpoint localization prompt with neighbor reference coordinates."""
        metadata = sample.get('metadata', {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        neighbor_meta = metadata.get('neighbor_meta', [])

        # Handle neighbor_meta as JSON string (from parquet serialization)
        if isinstance(neighbor_meta, str):
            import json
            try:
                neighbor_meta = json.loads(neighbor_meta)
            except:
                neighbor_meta = []

        # Build neighbor reference text
        neighbor_text = ""
        for n in neighbor_meta[:8]:  # Show up to 8 neighbors
            label = n.get('label', '?')
            coord = n.get('coord_nm', [0, 0, 0])
            # Handle both list/tuple and numpy array
            if hasattr(coord, 'tolist'):
                coord = coord.tolist()
            if isinstance(coord, (list, tuple)) and len(coord) == 3:
                neighbor_text += f"  Node {label}: ({coord[0]:.0f}, {coord[1]:.0f}, {coord[2]:.0f}) nm\n"

        no_coords_text = "  (Node coordinates not available)\n"
        reference_text = neighbor_text if neighbor_text else no_coords_text

        prompt = f"""You are analyzing 3D neuron images with numbered graph node overlays.

The images show orthographic views (front, side, top) of a neuron segment. Each numbered circle represents a node on the neuron's mesh.

Reference coordinates for visible nodes:
{reference_text}
Your task: Identify the x, y, z coordinates (in nanometers) at the end of the segment where there is a split error (where the segment was incorrectly broken).


"""
        if answer_only:
            prompt += "Report your answer in the format: <answer>x=[X],y=[Y],z=[Z]</answer>"
        else:
            prompt += """Surround your analysis with <analysis> and </analysis> tags.
Report your final coordinates in the format: <answer>x=[X],y=[Y],z=[Z]</answer>
(coordinates should be integers in nanometers, no units in the answer)"""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        gt = self.get_ground_truth(sample)
        answer_text = f"x={int(gt[0])},y={int(gt[1])},z={int(gt[2])}"

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_text}</answer>"""
                return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

        # Default synthetic response
        response = f"""<analysis>
Analyzing the graph node positions and neuron morphology across the three views,
I can identify a location where the segmentation appears to have an error.
Based on the spatial relationships between the numbered nodes and the visible
discontinuity in the neural process, the error is located at the following coordinates.
</analysis>

<answer>{answer_text}</answer>"""
        return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with endpoint localization images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            answer_text = f"x={int(ground_truth[0])},y={int(ground_truth[1])},z={int(ground_truth[2])}"
            response_text = f"<answer>{answer_text}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function based on L2 distance from ground truth coordinates."""
        import re

        def parse_coordinates(text: str) -> Optional[List[float]]:
            """Extract coordinates from <answer>x=...,y=...,z=...</answer>"""
            match = re.search(
                r'<answer>\s*x\s*=\s*(-?[\d.]+)\s*,\s*y\s*=\s*(-?[\d.]+)\s*,\s*z\s*=\s*(-?[\d.]+)\s*</answer>',
                text, re.IGNORECASE
            )
            if match:
                return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            return None

        full_threshold = self.FULL_REWARD_THRESHOLD_NM
        partial_threshold = self.PARTIAL_REWARD_THRESHOLD_NM

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Parse predicted coordinates
                pred = parse_coordinates(completion_text)

                reward = 0.0
                distance = None

                if pred is not None:
                    gt_arr = np.array(gt)
                    pred_arr = np.array(pred)
                    distance = np.linalg.norm(pred_arr - gt_arr)

                    if distance <= full_threshold:
                        reward = 1.0
                    elif distance <= partial_threshold:
                        # Linear interpolation from 1.0 to 0.2
                        t = (distance - full_threshold) / (partial_threshold - full_threshold)
                        reward = 1.0 - 0.8 * t
                    else:
                        reward = 0.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging for first few
                if idx < 3:
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: ({gt[0]:.0f}, {gt[1]:.0f}, {gt[2]:.0f})")
                    if pred:
                        print(f"Predicted: ({pred[0]:.0f}, {pred[1]:.0f}, {pred[2]:.0f})")
                        print(f"Distance: {distance:.1f} nm")
                    else:
                        print(f"Predicted: COULD NOT PARSE")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class SplitProposalTask(TaskConfig):
    """Split proposal task - predict source/sink coordinates for where to split a merged neuron."""

    # Reward thresholds in nanometers
    FULL_REWARD_THRESHOLD_NM = 500.0
    PARTIAL_REWARD_THRESHOLD_NM = 2000.0

    def __init__(self):
        super().__init__(
            name="split_proposal",
            description="Predict source and sink coordinates for splitting a merged neuron segment",
            dataset_source="split-proposals-parquet",
            dataset_config=None,
            num_images=3,  # front, side, top
        )
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Generate it using from_good_splits_to_split_proposal in question_dataset.py"
            )

        self._dataset_dir = dataset_dir
        self._dataset_hash = self._compute_file_hash(parquet_path)

        df = pd.read_parquet(parquet_path)
        df['_original_parquet_idx'] = range(len(df))

        dataset = Dataset.from_pandas(df)
        return dataset

    def filter_dataset(self, dataset):
        """No filtering needed for split proposal."""
        return dataset

    def get_images(self, sample: Dict) -> List:
        """Load the 3 orthogonal view images from disk."""
        from pathlib import Path
        from PIL import Image

        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/good-split-prop")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if hasattr(image_paths, 'tolist'):
            image_paths = image_paths.tolist()

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )
        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them.

        Same path resolution as get_images(), but returns paths instead
        of PIL images for lazy loading support.
        """
        from pathlib import Path

        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/good-split-prop")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if hasattr(image_paths, 'tolist'):
            image_paths = image_paths.tolist()

        # Return absolute paths as strings
        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        if len(abs_paths) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )
        return abs_paths

    def get_ground_truth(self, sample: Dict) -> tuple:
        """Return (sources, sinks) tuple where each is a list of [x,y,z] coordinates."""
        answer = sample['answer']
        if hasattr(answer, 'tolist'):
            answer = answer.tolist()

        # Ensure we have proper nested structure
        sources, sinks = answer[0], answer[1]
        if hasattr(sources, 'tolist'):
            sources = sources.tolist()
        if hasattr(sinks, 'tolist'):
            sinks = sinks.tolist()

        # Convert inner arrays to lists
        sources = [list(s) if hasattr(s, 'tolist') else list(s) for s in sources]
        sinks = [list(s) if hasattr(s, 'tolist') else list(s) for s in sinks]

        return (sources, sinks)

    def get_sample_id(self, sample: Dict, index: int = None) -> Dict:
        """Get a unique identifier for a split proposal sample."""
        ident = {}
        metadata = sample.get('metadata', {})
        if isinstance(metadata, dict):
            if 'split_hash' in metadata:
                ident['split_hash'] = metadata['split_hash']
            if 'root_id' in metadata:
                ident['root_id'] = metadata['root_id']

        if not ident and index is not None:
            ident = {'index': index}

        return ident

    def get_balance_group(self, sample: Dict) -> Any:
        """All samples are in the same group (all are good splits)."""
        return 0

    def get_split_group_key(self, sample: Dict) -> tuple:
        """
        Get the grouping key for train/val/test splitting.

        Returns (center_um, root_id) tuple to ensure all samples at the same
        physical location stay in the same split.
        """
        metadata = sample.get('metadata', {})
        if not isinstance(metadata, dict):
            return (None, None)

        root_id = metadata.get('root_id')

        center = metadata.get('center')
        if center is not None:
            center_um = tuple(int(round(c / 1000)) for c in center)
        else:
            center_um = None

        return (center_um, root_id)

    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the split proposal prompt with neighbor reference coordinates."""
        metadata = sample.get('metadata', {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        # Get neighbor coordinates for reference
        neighbors = metadata.get('neighbors', [])
        if hasattr(neighbors, 'tolist'):
            neighbors = neighbors.tolist()

        neighbor_text = ""
        for n in neighbors[:15]:  # Show up to 15 neighbors
            if isinstance(n, dict):
                label = n.get('label', '?')
                coord = n.get('coord_nm', [0, 0, 0])
                if hasattr(coord, 'tolist'):
                    coord = coord.tolist()
                if isinstance(coord, (list, tuple)) and len(coord) == 3:
                    neighbor_text += f"  Node {label}: ({coord[0]:.0f}, {coord[1]:.0f}, {coord[2]:.0f}) nm\n"

        reference_text = neighbor_text if neighbor_text else "  (Node coordinates not available)\n"

        prompt = f"""You are analyzing 3D neuron images to identify where a merge error should be split.

The images show orthographic views (front, side, top) of a neuron segment that contains a merge error - two distinct neurons that were incorrectly joined together. Numbered circles represent nodes on the neuron's skeleton graph.

Reference coordinates for visible nodes:
{reference_text}
Your task: Identify the source points (on one side of the split) and sink points (on the other side) that define where the segment can be separated to correct the merge error.

"""
        if answer_only:
            prompt += """Report your answer in the format:
<answer>sources=[(x1,y1,z1),(x2,y2,z2),...];sinks=[(x1,y1,z1),(x2,y2,z2),...]</answer>"""
        else:
            prompt += """Surround your analysis with <analysis> and </analysis> tags.
Report your final coordinates in the format:
<answer>sources=[(x1,y1,z1),(x2,y2,z2),...];sinks=[(x1,y1,z1),(x2,y2,z2),...]</answer>
(coordinates should be integers in nanometers)"""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def _format_coords_list(self, coords: List) -> str:
        """Format a list of coordinates as [(x,y,z),(x,y,z),...]"""
        formatted = []
        for c in coords:
            if hasattr(c, 'tolist'):
                c = c.tolist()
            formatted.append(f"({int(c[0])},{int(c[1])},{int(c[2])})")
        return "[" + ",".join(formatted) + "]"

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        sources, sinks = self.get_ground_truth(sample)
        sources_str = self._format_coords_list(sources)
        sinks_str = self._format_coords_list(sinks)
        answer_text = f"sources={sources_str};sinks={sinks_str}"

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_text}</answer>"""
                return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

        # Default synthetic response
        response = f"""<analysis>
Analyzing the neuron morphology across the three orthographic views, I can identify
where the merge error occurs. The segment shows two distinct processes that were
incorrectly joined together. By examining the graph node positions and the
discontinuity in neural structure, I can determine the appropriate source and sink
points for the split operation.
</analysis>

<answer>{answer_text}</answer>"""
        return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with split proposal images."""
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        if answer_only:
            sources, sinks = ground_truth
            sources_str = self._format_coords_list(sources)
            sinks_str = self._format_coords_list(sinks)
            response_text = f"<answer>sources={sources_str};sinks={sinks_str}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function based on distance from ground truth coordinates."""
        import re

        def parse_split_points(text: str) -> Optional[tuple]:
            """Extract sources and sinks from answer format."""
            match = re.search(
                r'<answer>\s*sources\s*=\s*\[(.*?)\]\s*;\s*sinks\s*=\s*\[(.*?)\]\s*</answer>',
                text, re.IGNORECASE | re.DOTALL
            )
            if not match:
                return None

            def parse_coord_list(s: str) -> List[List[float]]:
                coords = []
                # Find all (x,y,z) tuples
                for m in re.finditer(r'\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)', s):
                    coords.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                return coords

            sources = parse_coord_list(match.group(1))
            sinks = parse_coord_list(match.group(2))

            if not sources or not sinks:
                return None

            return (sources, sinks)

        full_threshold = self.FULL_REWARD_THRESHOLD_NM
        partial_threshold = self.PARTIAL_REWARD_THRESHOLD_NM

        def min_distance_to_set(point: np.ndarray, point_set: List) -> float:
            """Compute minimum distance from point to any point in the set."""
            if not point_set:
                return float('inf')
            distances = [np.linalg.norm(point - np.array(p)) for p in point_set]
            return min(distances)

        def compute_set_distance(pred_set: List, gt_set: List) -> float:
            """Compute average minimum distance from predicted points to ground truth."""
            if not pred_set:
                return float('inf')
            distances = []
            for p in pred_set:
                d = min_distance_to_set(np.array(p), gt_set)
                distances.append(d)
            return np.mean(distances)

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                pred = parse_split_points(completion_text)

                reward = 0.0
                avg_distance = None

                if pred is not None:
                    gt_sources, gt_sinks = gt
                    pred_sources, pred_sinks = pred

                    # Compute distances for sources and sinks separately
                    source_dist = compute_set_distance(pred_sources, gt_sources)
                    sink_dist = compute_set_distance(pred_sinks, gt_sinks)
                    avg_distance = (source_dist + sink_dist) / 2

                    if avg_distance <= full_threshold:
                        reward = 1.0
                    elif avg_distance <= partial_threshold:
                        t = (avg_distance - full_threshold) / (partial_threshold - full_threshold)
                        reward = 1.0 - 0.8 * t
                    else:
                        reward = 0.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging
                if idx < 3:
                    gt_sources, gt_sinks = gt
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth Sources: {len(gt_sources)} points")
                    print(f"Ground Truth Sinks: {len(gt_sinks)} points")
                    if pred:
                        print(f"Predicted Sources: {len(pred[0])} points")
                        print(f"Predicted Sinks: {len(pred[1])} points")
                        print(f"Avg Distance: {avg_distance:.1f} nm")
                    else:
                        print(f"Predicted: COULD NOT PARSE")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class SegmentIdentityTask(TaskConfig):
    """Segment identity task - determine if two images show the same segment."""

    def __init__(self):
        super().__init__(
            name="segment_identity",
            description="Determine if two images show the same segment (different views/zoom) or different segments",
            dataset_source="segment-identity-parquet",
            dataset_config=None,
            num_images=2,  # Two images to compare
        )
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Generate it using segment_identity_sampler.py"
            )

        self._dataset_dir = dataset_dir
        self._dataset_hash = self._compute_file_hash(parquet_path)

        df = pd.read_parquet(parquet_path)
        df['_original_parquet_idx'] = range(len(df))

        dataset = Dataset.from_pandas(df)
        return dataset

    def filter_dataset(self, dataset):
        """No filtering needed for segment identity."""
        return dataset

    def get_images(self, sample: Dict) -> List:
        """Load the 2 comparison images from disk.

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/segment-identity-parquet"
        """
        from pathlib import Path
        from PIL import Image

        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/segment-identity-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if hasattr(image_paths, 'tolist'):
            image_paths = image_paths.tolist()

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )
        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them."""
        from pathlib import Path

        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/segment-identity-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if hasattr(image_paths, 'tolist'):
            image_paths = image_paths.tolist()

        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        if len(abs_paths) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )
        return abs_paths

    def get_ground_truth(self, sample: Dict) -> bool:
        """Return boolean answer (True = same segment, False = different segments)."""
        return sample['answer']

    def get_sample_id(self, sample: Dict, index: int = None) -> Dict:
        """Get a unique identifier for a segment identity sample."""
        ident = {}
        metadata = sample.get('metadata', {})
        if isinstance(metadata, dict):
            if 'sample_id' in metadata:
                ident['sample_id'] = metadata['sample_id']
            if 'segment1_id' in metadata:
                ident['segment1_id'] = metadata['segment1_id']
            if 'segment2_id' in metadata:
                ident['segment2_id'] = metadata['segment2_id']

        if not ident and index is not None:
            ident = {'index': index}

        return ident

    def get_split_group_key(self, sample: Dict) -> tuple:
        """
        Get the grouping key for train/val/test splitting.

        Returns a tuple of (segment1_id, segment2_id) representing ALL segments
        involved in this sample. The splitting code should use connected components
        to ensure that any samples sharing a segment ID end up in the same split.

        For example, if sample A uses segments {X, Y} and sample B uses {X, Z},
        both should be in the same split because they share segment X.

        This prevents data leakage where the model sees a segment during training
        and then sees it again (possibly from a different angle) during evaluation.
        """
        metadata = sample.get('metadata', {})
        if isinstance(metadata, str):
            import json
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}

        segment1_id = metadata.get('segment1_id')
        segment2_id = metadata.get('segment2_id')

        # Return both segment IDs - splitting code should use connected components
        # to group samples that share ANY segment ID
        if segment1_id is not None and segment2_id is not None:
            return (segment1_id, segment2_id)
        elif segment1_id is not None:
            return (segment1_id,)
        elif segment2_id is not None:
            return (segment2_id,)
        else:
            return None

    def uses_connected_component_splitting(self) -> bool:
        """Enable connected component merging for segment identity task.

        Samples sharing any segment ID should be in the same split to prevent
        data leakage (e.g., seeing segment X in training and evaluation).
        """
        return True

    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the segment identity prompt."""
        # Note: We intentionally don't include view angle/extent metadata in the prompt
        # to avoid leaking information that could make the task trivially easy.
        # The model should determine segment identity from visual features alone.

        prompt = """You are analyzing two 3D neuron segment images to determine if they show the same segment or different segments.

The two images may show:
- The SAME segment from different viewing angles and/or zoom levels
- Two DIFFERENT segments

Your task: Determine whether these two images show the **same neuronal segment** or **different segments**.

**Answer with yes or no**:
- yes = These are images of the SAME segment (different views/zoom levels)
- no = These are images of DIFFERENT segments

"""
        if answer_only:
            prompt += "Surround your final answer (yes or no) with <answer> and </answer> tags."
        else:
            prompt += """Surround your analysis and reasoning with <analysis> and </analysis> tags.
Surround your final answer (yes or no) with <answer> and </answer> tags."""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        ground_truth = self.get_ground_truth(sample)
        answer_text = "yes" if ground_truth else "no"

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_text}</answer>"""
                return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

        # Default synthetic response
        if ground_truth:
            analysis = "Comparing the two images, I can see that despite differences in viewing angle or zoom level, the overall morphology and structure of the segment are consistent. Key features like branching patterns, process orientations, and relative proportions match between the views, indicating these are images of the same neuronal segment."
        else:
            analysis = "Comparing the two images, I observe significant differences in the morphological characteristics of the segments. The branching patterns, process orientations, and overall structure differ between the two images, indicating these are images of two distinct neuronal segments."

        response = f"""<analysis>
{analysis}
</analysis>

<answer>{answer_text}</answer>"""
        return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with segment identity images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            answer_text = "yes" if ground_truth else "no"
            response_text = f"<answer>{answer_text}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function for GRPO."""
        import re

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract answer from tags
                answer_match = re.search(r'<answer>\s*(yes|no)\s*</answer>', completion_text.lower())

                reward = 0.0
                predicted_answer = None

                if answer_match:
                    predicted_answer = answer_match.group(1)
                    predicted_bool = (predicted_answer == "yes")

                    if predicted_bool == gt:
                        reward = 1.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging
                if idx < 3:
                    pred_answer = predicted_answer if predicted_answer else "NONE"
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: {gt} ({'yes (same segment)' if gt else 'no (different segments)'})")
                    print(f"Predicted: {pred_answer}")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


class MergeErrorIdentificationTask(TaskConfig):
    """Merge error identification task - identify if a junction has a merge error."""

    def __init__(self):
        super().__init__(
            name="merge_error_identification",
            description="Identify if a junction point is a merge error (incorrectly joined segments) or a valid connection",
            dataset_source="merge-error-identification-parquet",  # Directory name in dataset volume
            dataset_config=None,
            num_images=3,  # 3 views (front, side, top)
        )
        # Path to dataset directory - set by load_dataset() or manually
        self._dataset_dir: Path = None

    def load_dataset(self, cache_dir: str, dataset_path: str = None):
        """Load dataset from the specified directory.

        Args:
            cache_dir: Parent directory containing the dataset folder.
                       The full path will be: cache_dir / self.dataset_source
            dataset_path: Optional explicit path to the dataset directory.
                          If provided, overrides cache_dir / dataset_source.
        """
        import pandas as pd
        from datasets import Dataset
        from pathlib import Path

        # Use explicit dataset_path if provided, otherwise construct from cache_dir
        if dataset_path:
            dataset_dir = Path(dataset_path)
            # If relative path, make it relative to cache_dir
            if not dataset_dir.is_absolute():
                dataset_dir = Path(cache_dir) / dataset_dir
        else:
            dataset_dir = Path(cache_dir) / self.dataset_source
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {parquet_path}. "
                f"Generate it using the junction error data generation scripts."
            )

        # Store the dataset directory for use by get_images()
        self._dataset_dir = dataset_dir

        # Compute hash of parquet file for reproducibility tracking
        self._dataset_hash = self._compute_file_hash(parquet_path)

        df = pd.read_parquet(parquet_path)

        # Add original parquet index before any filtering/shuffling
        df['_original_parquet_idx'] = range(len(df))

        # Convert the DataFrame to Dataset
        dataset = Dataset.from_pandas(df)

        return dataset

    def filter_dataset(self, dataset):
        """Filter to unique (root_id, interface_point) locations.

        The same junction location may have been sampled multiple times with
        different junction_hash values. Keep only the first occurrence to
        avoid data duplication.
        """
        # seen_locations = set()
        # keep_indices = []

        # for idx, sample in enumerate(dataset):
        #     meta = sample.get('metadata', {})
        #     root_id = meta.get('root_id')
        #     interface_point = meta.get('interface_point')
        #     if interface_point is not None:
        #         interface_point = tuple(interface_point)
        #     print(f"root_id: {root_id}, interface_point: {interface_point}")
        #     location_key = (root_id, interface_point)
        #     if location_key not in seen_locations:
        #         seen_locations.add(location_key)
        #         keep_indices.append(idx)

        # filtered = dataset.select(keep_indices)
        # print(f"  Filtered {len(dataset)} -> {len(filtered)} samples (removed {len(dataset) - len(filtered)} duplicate locations)")
        return dataset #filtered

    def get_images(self, sample: Dict) -> List:
        """Get the images from the sample, loading from disk.

        Path resolution order:
        1. sample['_base_path'] - if set explicitly on the sample
        2. self._dataset_dir - set by load_dataset()
        3. Fallback to "/datasets/merge-error-identification-parquet" (Modal default)
        """
        from pathlib import Path
        from PIL import Image

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/merge-error-identification-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        loaded_images = []
        for i, rel_path in enumerate(image_paths):
            abs_path = base_path / rel_path
            try:
                img = Image.open(abs_path)

                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError(f"Image has zero dimensions: {img.size}")

                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')

                loaded_images.append(img)
            except Exception as e:
                print(f"\nError loading image {i+1}/{len(image_paths)}: {rel_path}")
                print(f"  Path: {abs_path}")
                print(f"  Base path: {base_path}")
                print(f"  Error: {e}")
                if 'question_type' in sample:
                    print(f"  Question type: {sample['question_type']}")
                if 'answer' in sample:
                    print(f"  Answer: {sample['answer']}")
                raise

        if len(loaded_images) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(loaded_images)}. "
                f"Paths: {image_paths}"
            )
        return loaded_images

    def get_image_paths(self, sample: Dict) -> List[str]:
        """Get absolute image paths without loading them.

        Same path resolution as get_images(), but returns paths instead
        of PIL images for lazy loading support.
        """
        from pathlib import Path

        # Resolve base path with priority: sample > task > default
        if '_base_path' in sample:
            base_path = Path(sample['_base_path'])
        elif self._dataset_dir is not None:
            base_path = self._dataset_dir
        else:
            base_path = Path("/datasets/merge-error-identification-parquet")

        image_paths = sample['images']
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Return absolute paths as strings
        abs_paths = [str(base_path / rel_path) for rel_path in image_paths]

        if len(abs_paths) != self.num_images:
            raise ValueError(
                f"Expected {self.num_images} images for {self.name}, got {len(abs_paths)}. "
                f"Paths: {image_paths}"
            )
        return abs_paths

    def get_ground_truth(self, sample: Dict) -> bool:
        """Return boolean answer (True = merge error, False = valid connection)."""
        # answer is 'error' or 'control'
        return sample['answer'] == 'error'

    def get_split_group_key(self, sample: Dict) -> tuple:
        """Group by (root_id, interface_point) to prevent data leakage.

        The same junction location (root_id + center) may have been sampled multiple
        times with different junction_hash values. These represent the same biological
        location and should always be in the same train/val/test split.
        """
        meta = sample.get('metadata', {})
        root_id = meta.get('root_id')
        interface_point = meta.get('interface_point')
        # Convert interface_point to tuple for hashability
        if interface_point is not None:
            interface_point = tuple(interface_point)
        return (root_id, interface_point)

    def format_prompt(self, sample: Dict, answer_only: bool = False, request_confidence: bool = False) -> str:
        """Create the merge error identification prompt."""
#         prompt = """You are attempting to proofread a 3D segmentation of a neuron.

# You are shown multiple views of a 3D neuron segment centered on a junction point where multiple branches meet.

# Your task is to determine: **Is this junction a merge error or a valid connection?**

# - **Merge error**: The junction incorrectly joins two separate neurons that should not be connected. Signs include:
#   - The junction appears to connect branches from different neurons
#   - There's an unnatural or forced-looking connection point
#   - The morphology suggests the branches belong to different cells

# - **Valid connection**: The junction is a true biological branching point of a single neuron. Signs include:
#   - The branches appear to belong to the same neuron
#   - The junction looks like natural neuronal branching
#   - The morphology is consistent with a single cell's structure

# **Answer with yes or no**:
# - yes = This is a merge error (needs split correction)
# - no = This is a valid connection (no correction needed)

# """
        prompt = """
You are a EM connectomics proofreading expert, and you're deciding if a segment has a merge error or not.
You are shown multiple views of a 3D segment.
Images are presented in groups of 3 (front, side, top).

**Answer with yes or no**:
- yes = This is a merge error (needs split correction)
- no = This is a valid connection (no correction needed)
"""
        if answer_only:
            prompt += "Surround your final answer (yes or no) with <answer> and </answer> tags."
        else:
            prompt += """Surround your analysis and reasoning with <analysis> and </analysis> tags.
Surround your final answer (yes or no) with <answer> and </answer> tags."""

        if request_confidence:
            prompt += self.get_confidence_request_text()

        return prompt

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None,
                       rationale_dropout_prob: float = 0.0) -> str:
        """Format the expected response."""
        ground_truth = self.get_ground_truth(sample)
        answer_text = "yes" if ground_truth else "no"

        # Use teacher response if available
        if use_teacher_response and teacher_response_column in sample:
            import pandas as pd
            if sample[teacher_response_column] is not None and pd.notna(sample[teacher_response_column]):
                teacher_analysis = sample[teacher_response_column]
                response = f"""<analysis>
{teacher_analysis}
</analysis>

<answer>{answer_text}</answer>"""
                return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

        # Default synthetic response
        if ground_truth:
            analysis = "Examining the junction across the provided views, this appears to be a merge error. The connected branches show characteristics suggesting they belong to different neurons, and the junction point looks unnatural."
        else:
            analysis = "Examining the junction across the provided views, this appears to be a valid connection. The branches appear to belong to the same neuron with natural branching morphology consistent with a single cell's structure."

        response = f"""<analysis>
{analysis}
</analysis>

<answer>{answer_text}</answer>"""
        return self._apply_rationale_dropout(response, answer_text, rationale_dropout_prob)

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None,
                                   rationale_dropout_prob: float = 0.0,
                                   lazy_images: bool = False) -> Dict:
        """Format sample for SFT training with junction images."""
        # Decide dropout upfront so prompt and response are consistent
        answer_only = rationale_dropout_prob > 0.0 and random.random() < rationale_dropout_prob

        prompt_text = self.format_prompt(sample, answer_only=answer_only)
        ground_truth = self.get_ground_truth(sample)

        # If answer_only, skip the analysis in response too
        if answer_only:
            answer_text = "yes" if ground_truth else "no"
            response_text = f"<answer>{answer_text}</answer>"
        else:
            response_text = self.format_response(sample, use_teacher_response, teacher_response_column,
                                                 rationale_dropout_prob=0.0)

        # Build user message with images (or paths for lazy loading)
        user_content = []
        if lazy_images:
            image_paths = self.get_image_paths(sample)
            for path in image_paths:
                user_content.append({"type": "image", "path": path})
        else:
            images = self.get_images(sample)
            for img in images:
                user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        }]

        result = {
            "messages": messages,
            "ground_truth": ground_truth,
        }

        # Include image paths for lazy loading
        if lazy_images:
            result["_image_paths"] = image_paths

        return result

    def create_reward_function(self) -> Callable:
        """Create reward function for GRPO."""
        import re

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                print("Warning: No ground_truth provided to reward function")
                return [0.0] * len(completions)

            rewards = []
            for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
                # Handle different completion formats
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract answer from tags
                answer_match = re.search(r'<answer>\s*(yes|no)\s*</answer>', completion_text.lower())

                reward = 0.0
                predicted_answer = None

                if answer_match:
                    predicted_answer = answer_match.group(1)
                    predicted_bool = (predicted_answer == "yes")

                    if predicted_bool == gt:
                        reward = 1.0

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                reward = min(1.0, max(0.0, reward))
                rewards.append(reward)

                # Debug logging
                if idx < 3:
                    pred_answer = predicted_answer if predicted_answer else "NONE"
                    print(f"\n{'='*60}")
                    print(f"Sample {idx + 1}:")
                    print(f"Ground Truth: {gt} ({'yes (merge error)' if gt else 'no (valid connection)'})")
                    print(f"Predicted: {pred_answer}")
                    print(f"Reward: {reward:.2f}")
                    print(f"\nCompletion (first 500 chars):")
                    print(completion_text[:500])
                    print(f"{'='*60}\n")

            return rewards

        return reward_fn


# Task registry for easy access
TASK_REGISTRY = {
    "segment_classification": SegmentClassificationTask,
    "split_action": SplitActionTask,
    "merge_action": MergeActionTask,
    "merge_action_multiple_choice": MergeActionMultipleChoiceTask,
    "endpoint_error_identification": EndpointErrorIdentificationTask,
    "endpoint_error_identification_with_em": EndpointErrorIdentificationWithEMTask,
    "endpoint_localization": EndpointLocalizationTask,
    "split_proposal": SplitProposalTask,
    "segment_identity": SegmentIdentityTask,
    "merge_error_identification": MergeErrorIdentificationTask,
}


def get_task(task_name: str) -> TaskConfig:
    """Get a task configuration by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name]()


def list_tasks() -> List[str]:
    """List all available task names."""
    return list(TASK_REGISTRY.keys())
