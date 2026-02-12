#!/usr/bin/env python3
"""
End-to-end proofreading evaluation pipeline.

Evaluates VLM performance on the full proofreading task:
1. Candidate Generation: Find potential error locations (skeleton endpoints with EM slices)
2. Error Identification: VLM predicts which candidates are actual errors (split or merge)
3. Correction Image Generation: Render candidate merge partners for identified split errors
4. Error Correction: VLM evaluates each candidate with binary yes/no merge decision

Key features:
- Uses skeleton direction to find mesh tip at endpoints (ground-truth agnostic)
- Includes EM slice views (XY, XZ, YZ) alongside mesh projections for identification
- Binary merge_action task for correction (not multiple choice)
- Reports ties when multiple candidates are predicted as correct merges
- Supports both split errors (need merge) and merge errors (need split)

Each VLM stage uses a task-specific fine-tuned adapter:
- identification_adapter: For endpoint_error_identification_with_em task
- correction_adapter: For merge_action (binary) task

Usage:
    # Single root evaluation (for testing)
    python scripts/analysis/proofreading_evaluator.py \
        --root-id 864691135572735469 \
        --output-dir evaluation_results/proofreading

    # Batch evaluation with Modal (recommended for production)
    # See modal_proofreading_inference.py for Modal-based inference
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from caveclient import CAVEclient


@dataclass
class LLMConfig:
    """Configuration for VLM inference.

    Supports two backends:
    - API-based (OpenAI, Anthropic): Set model to "gpt-4o", "claude-3-opus", etc.
    - Modal-based (vLLM with adapters): Use modal_proofreading_inference.py

    For Modal inference with fine-tuned adapters, see InferenceConfig in
    modal_proofreading_inference.py instead.
    """
    # Model name (API model like "gpt-4o" or "claude-3-5-sonnet")
    model: str = "gpt-4o"

    # Inference settings
    max_tokens: int = 1024
    max_concurrent: int = 10  # Max concurrent API requests


@dataclass
class CandidateLocation:
    """A candidate error location (skeleton endpoint)."""
    endpoint_idx: int
    coord_nm: np.ndarray
    root_id: int
    is_ground_truth_error: bool = False
    distance_to_nearest_error_nm: Optional[float] = None


@dataclass
class IdentificationResult:
    """Result of error identification stage."""
    candidate: CandidateLocation
    predicted_is_error: bool
    confidence: Optional[float] = None
    response: Optional[str] = None


@dataclass
class CorrectionCandidate:
    """A candidate merge partner for correction."""
    segment_id: int
    is_correct: bool = False  # True if this is the ground truth merge partner


@dataclass
class BinaryCorrectionResult:
    """Result of binary merge decision for a single candidate."""
    segment_id: int
    predicted_merge: bool  # True if model said "yes" to merge
    is_correct_partner: bool  # True if this is the ground truth merge partner
    response: Optional[str] = None


@dataclass
class CorrectionResult:
    """Result of error correction stage using binary merge decisions."""
    candidate: CandidateLocation
    binary_results: List[BinaryCorrectionResult]  # One per candidate segment
    predicted_merge_ids: List[int]  # Segments predicted as correct merges (yes)
    correct_segment_id: Optional[int]  # Ground truth correct segment (None if false positive)
    has_tie: bool = False  # True if multiple candidates predicted as correct
    response: Optional[str] = None  # Combined responses for logging


# =============================================================================
# Merge Error (Junction) Data Structures
# =============================================================================

@dataclass
class JunctionLocation:
    """A candidate merge error location (skeleton junction/branchpoint)."""
    junction_idx: int
    coord_nm: np.ndarray
    root_id: int
    degree: int = 3  # Number of branches at this junction
    is_ground_truth_error: bool = False
    distance_to_nearest_error_nm: Optional[float] = None


@dataclass
class MergeErrorIdentificationResult:
    """Result of merge error identification stage."""
    junction: JunctionLocation
    predicted_is_error: bool
    confidence: Optional[float] = None
    response: Optional[str] = None


@dataclass
class ProofreadingResults:
    """Complete evaluation results."""
    root_id: int

    # Ground truth
    num_ground_truth_errors: int
    ground_truth_error_coords: List[np.ndarray]

    # Candidate localization
    num_candidates: int
    num_true_positive_candidates: int  # Candidates that are actual errors

    # Identification stage
    identification_results: List[IdentificationResult] = field(default_factory=list)
    identification_true_positives: int = 0  # Correctly identified errors
    identification_false_positives: int = 0  # Non-errors identified as errors
    identification_false_negatives: int = 0  # Missed errors
    identification_precision: float = 0.0
    identification_recall: float = 0.0

    # Correction stage (binary merge decisions)
    correction_results: List[CorrectionResult] = field(default_factory=list)
    correction_correct_unique: int = 0  # Exactly one correct merge partner predicted
    correction_correct_with_tie: int = 0  # Correct partner predicted but with ties
    correction_wrong_only: int = 0  # Only wrong partners predicted (introduces error)
    correction_none_when_should: int = 0  # No merges predicted when there was a correct option
    correction_none_correct: int = 0  # No merges predicted and none were correct (FP from identification)
    correction_ties_total: int = 0  # Total number of cases with multiple "yes" predictions

    # End-to-end
    end_to_end_correct: int = 0  # Error identified AND corrected properly (unique or with tie)
    end_to_end_false_corrections: int = 0  # False ID + wrong correction (new errors introduced)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "root_id": self.root_id,
            "ground_truth": {
                "num_errors": self.num_ground_truth_errors,
                "error_coords": [c.tolist() for c in self.ground_truth_error_coords],
            },
            "candidate_localization": {
                "num_candidates": self.num_candidates,
                "num_true_positive_candidates": self.num_true_positive_candidates,
            },
            "identification": {
                "true_positives": self.identification_true_positives,
                "false_positives": self.identification_false_positives,
                "false_negatives": self.identification_false_negatives,
                "precision": self.identification_precision,
                "recall": self.identification_recall,
            },
            "correction": {
                "correct_unique": self.correction_correct_unique,
                "correct_with_tie": self.correction_correct_with_tie,
                "wrong_only": self.correction_wrong_only,
                "none_when_should": self.correction_none_when_should,
                "none_correct": self.correction_none_correct,
                "ties_total": self.correction_ties_total,
            },
            "end_to_end": {
                "correct": self.end_to_end_correct,
                "false_corrections": self.end_to_end_false_corrections,
            },
        }


@dataclass
class BatchProgress:
    """Track progress for batch evaluation with resumability.

    Saves after each root_id completes, allowing resume from any point.
    """
    stage: str  # "candidates", "identification", "localization", "correction", "evaluation", "done"
    root_ids_requested: List[int]
    root_ids_completed: Dict[str, List[int]] = field(default_factory=dict)  # stage -> completed
    root_ids_failed: Dict[str, Dict[int, str]] = field(default_factory=dict)  # stage -> {root_id: error}
    started_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()

    def save(self, path: Path):
        """Save progress to JSON file."""
        self.last_updated = datetime.now().isoformat()
        # Convert int keys to strings for JSON serialization
        failed_serializable = {
            stage: {str(k): v for k, v in errors.items()}
            for stage, errors in self.root_ids_failed.items()
        }
        data = {
            "stage": self.stage,
            "root_ids_requested": self.root_ids_requested,
            "root_ids_completed": self.root_ids_completed,
            "root_ids_failed": failed_serializable,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BatchProgress":
        """Load progress from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Convert string keys back to ints
        data['root_ids_failed'] = {
            stage: {int(k): v for k, v in errors.items()}
            for stage, errors in data.get('root_ids_failed', {}).items()
        }
        return cls(**data)

    def mark_completed(self, stage: str, root_id: int):
        """Mark a root_id as completed for a stage."""
        if stage not in self.root_ids_completed:
            self.root_ids_completed[stage] = []
        if root_id not in self.root_ids_completed[stage]:
            self.root_ids_completed[stage].append(root_id)

    def mark_failed(self, stage: str, root_id: int, error: str):
        """Mark a root_id as failed for a stage."""
        if stage not in self.root_ids_failed:
            self.root_ids_failed[stage] = {}
        self.root_ids_failed[stage][root_id] = error

    def get_remaining(self, stage: str) -> List[int]:
        """Get root_ids that haven't been completed or failed for a stage."""
        completed = set(self.root_ids_completed.get(stage, []))
        failed = set(self.root_ids_failed.get(stage, {}).keys())
        return [rid for rid in self.root_ids_requested if rid not in completed and rid not in failed]

    def get_successful(self, stage: str) -> List[int]:
        """Get root_ids that completed successfully for a stage."""
        return self.root_ids_completed.get(stage, [])

    def summary(self) -> str:
        """Return a summary string of current progress."""
        lines = [f"Stage: {self.stage}"]
        lines.append(f"Requested: {len(self.root_ids_requested)} root_ids")
        for s in ["candidates", "identification", "correction"]:
            completed = len(self.root_ids_completed.get(s, []))
            failed = len(self.root_ids_failed.get(s, {}))
            if completed or failed:
                lines.append(f"  {s}: {completed} completed, {failed} failed")
        return "\n".join(lines)


def find_tip_along_skeleton_direction(
    mesh,
    skeleton,
    endpoint_coord: np.ndarray,
    radius_nm: float = 5000.0,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Find mesh vertex at the 'tip' by extending in the skeleton direction.

    This method uses the skeleton structure to determine which direction
    the terminus is pointing, then finds the mesh vertex furthest in that
    direction. This is ground-truth agnostic (doesn't need interface location).

    Algorithm:
    1. Find the skeleton endpoint node and its parent node
    2. Compute direction from parent -> endpoint (outward direction)
    3. Find mesh vertices within radius of skeleton endpoint
    4. Select the vertex furthest along that outward direction

    Args:
        mesh: Mesh object with vertices attribute
        skeleton: Skeleton object with vertices and edges attributes
        endpoint_coord: Coordinate of the skeleton endpoint
        radius_nm: Radius of local region to consider

    Returns:
        Tuple of (coord_nm, vertex_idx) for the tip vertex,
        or None if insufficient vertices or no parent found
    """
    from scipy.spatial import cKDTree

    vertices = np.array(mesh.vertices)
    skel_vertices = skeleton.vertices
    edges = skeleton.edges

    # Find the skeleton node closest to endpoint_coord
    skel_tree = cKDTree(skel_vertices)
    _, endpoint_idx = skel_tree.query(endpoint_coord)

    # Find parent node (connected node that's not the endpoint)
    parent_idx = None
    for edge in edges:
        if edge[0] == endpoint_idx:
            parent_idx = edge[1]
            break
        elif edge[1] == endpoint_idx:
            parent_idx = edge[0]
            break

    if parent_idx is None:
        return None

    parent_coord = skel_vertices[parent_idx]
    ep_coord = skel_vertices[endpoint_idx]

    # Direction from parent to endpoint (outward direction)
    direction = ep_coord - parent_coord
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return None
    direction = direction / norm

    # Find mesh vertices within radius of endpoint
    tree = cKDTree(vertices)
    indices = tree.query_ball_point(ep_coord, radius_nm)

    if len(indices) < 10:
        return None

    local_vertices = vertices[indices]

    # Project onto direction and find furthest in outward direction
    projections = np.dot(local_vertices - ep_coord, direction)
    best_local_idx = np.argmax(projections)

    return (local_vertices[best_local_idx], indices[best_local_idx])


# =============================================================================
# Inference Backend Abstraction
# =============================================================================

from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    """Abstract base class for inference backends.

    Supports two implementations:
    - APIBackend: Direct API calls to OpenAI, Anthropic, etc.
    - ModalBackend: Modal-based inference with vLLM and LoRA adapters

    The backend handles:
    - Loading/preparing data for inference
    - Running the VLM inference
    - Parsing responses into structured results
    """

    @abstractmethod
    async def run_identification(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run endpoint error identification inference.

        Args:
            parquet_dir: Directory containing questions.parquet and images/
            samples: List of sample dicts from the parquet

        Returns:
            List of dicts with keys:
                - sample_idx: Index in the samples list
                - predicted_is_error: bool
                - response: Raw model response string
        """
        pass

    @abstractmethod
    async def run_correction(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run binary merge action correction inference.

        For each error location, evaluates each candidate segment with a binary
        yes/no merge decision. Reports all candidates that get "yes" predictions.

        Args:
            parquet_dir: Directory containing questions.parquet and images/
            samples: List of sample dicts from the parquet

        Returns:
            List of dicts with keys:
                - sample_idx: Index in the samples list
                - binary_results: List of {segment_id, predicted_merge, response} for each candidate
                - predicted_merge_ids: List of segment IDs that got "yes" predictions
                - correct_segment_id: Ground truth correct segment (or None)
                - has_tie: True if multiple candidates predicted as correct
        """
        pass

    @abstractmethod
    async def run_merge_error_identification(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run merge error identification inference on junction candidates.

        Args:
            parquet_dir: Directory containing questions.parquet and images/
            samples: List of sample dicts from the parquet

        Returns:
            List of dicts with keys:
                - sample_idx: Index in the samples list
                - predicted_is_error: bool (True = merge error, False = valid junction)
                - response: Raw model response string
        """
        pass


class APIBackend(InferenceBackend):
    """Inference backend using direct API calls (OpenAI, Anthropic, etc.)."""

    def __init__(self, config: LLMConfig):
        self.config = config

    async def run_identification(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run identification using API."""
        from src.environment.task_configs import get_task
        from scripts.util import LLMProcessor
        import base64
        import io

        task = get_task("endpoint_error_identification")
        task._dataset_dir = parquet_dir

        llm = LLMProcessor(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            max_concurrent=self.config.max_concurrent
        )

        # Prepare messages
        batch_messages = []
        for sample in samples:
            prompt = task.format_prompt(sample)
            sample['_base_path'] = parquet_dir
            images = task.get_images(sample)

            content = []
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
            content.append({"type": "text", "text": prompt})

            batch_messages.append([{"role": "user", "content": content}])

        # Process
        responses = await llm.process_batch_conversations(batch_messages)

        # Parse responses
        results = []
        for idx, response in enumerate(responses):
            predicted_is_error = False
            if response:
                match = re.search(r'<answer>\s*(yes|no)\s*</answer>', response.lower())
                if match:
                    predicted_is_error = (match.group(1) == "yes")

            results.append({
                "sample_idx": idx,
                "predicted_is_error": predicted_is_error,
                "response": response,
            })

        return results

    async def run_correction(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run binary merge correction using API.

        For each error location, evaluates each candidate segment individually
        with a binary yes/no merge decision using the merge_action task.
        """
        from src.environment.task_configs import get_task
        from scripts.util import LLMProcessor
        from PIL import Image
        import base64
        import io

        task = get_task("merge_action")  # Binary merge task
        task._dataset_dir = parquet_dir

        llm = LLMProcessor(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            max_concurrent=self.config.max_concurrent
        )

        # Build a flat list of all (sample_idx, candidate) pairs for batch processing
        all_inference_items = []  # List of (sample_idx, candidate_info, images)

        for sample_idx, sample in enumerate(samples):
            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            all_candidates = metadata.get('all_candidates', [])
            if hasattr(all_candidates, 'tolist'):
                all_candidates = all_candidates.tolist()
            elif not isinstance(all_candidates, list):
                all_candidates = list(all_candidates)

            correct_segment_id = metadata.get('correct_segment_id')
            images_per_candidate = metadata.get('images_per_candidate', 3)

            # Reconstruct image paths per candidate from flat list
            all_image_paths = sample.get('images', [])
            if hasattr(all_image_paths, 'tolist'):
                all_image_paths = all_image_paths.tolist()

            # Reconstruct candidates with image paths
            for cand_idx, c in enumerate(all_candidates):
                if hasattr(c, 'tolist'):
                    c = c.tolist()
                if not isinstance(c, dict):
                    continue

                start_idx = cand_idx * images_per_candidate
                end_idx = start_idx + images_per_candidate
                image_paths = all_image_paths[start_idx:end_idx]

                candidate_info = {
                    'segment_id': c.get('segment_id'),
                    'is_correct': c.get('is_correct', False) or (c.get('segment_id') == correct_segment_id),
                    'image_paths': image_paths,
                }

                all_inference_items.append((sample_idx, candidate_info, correct_segment_id))

        # Prepare batch messages for all candidates
        batch_messages = []
        for sample_idx, candidate_info, _ in all_inference_items:
            # Load images for this candidate
            images = []
            for img_path in candidate_info['image_paths']:
                if not Path(img_path).is_absolute():
                    img_path = parquet_dir / img_path
                img = Image.open(img_path)
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                elif img.mode == 'L':
                    img = img.convert('RGB')
                images.append(img)

            # Build prompt using binary merge_action task
            prompt = task.format_prompt({})

            content = []
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
            content.append({"type": "text", "text": prompt})

            batch_messages.append([{"role": "user", "content": content}])

        # Process all candidates in batch
        responses = await llm.process_batch_conversations(batch_messages)

        # Group results by sample_idx
        sample_results = {}  # sample_idx -> list of binary results

        for (sample_idx, candidate_info, correct_segment_id), response in zip(all_inference_items, responses):
            predicted_merge = False
            if response:
                match = re.search(r'<answer>\s*(yes|no)\s*</answer>', response.lower())
                if match:
                    predicted_merge = (match.group(1) == "yes")

            binary_result = {
                'segment_id': candidate_info['segment_id'],
                'predicted_merge': predicted_merge,
                'is_correct_partner': candidate_info['is_correct'],
                'response': response,
            }

            if sample_idx not in sample_results:
                sample_results[sample_idx] = {
                    'binary_results': [],
                    'correct_segment_id': correct_segment_id,
                }
            sample_results[sample_idx]['binary_results'].append(binary_result)

        # Convert to final results format
        results = []
        for sample_idx in sorted(sample_results.keys()):
            sample_data = sample_results[sample_idx]
            binary_results = sample_data['binary_results']
            correct_segment_id = sample_data['correct_segment_id']

            # Find all segments predicted as correct merges
            predicted_merge_ids = [
                r['segment_id'] for r in binary_results if r['predicted_merge']
            ]

            has_tie = len(predicted_merge_ids) > 1

            results.append({
                "sample_idx": sample_idx,
                "binary_results": binary_results,
                "predicted_merge_ids": predicted_merge_ids,
                "correct_segment_id": correct_segment_id,
                "has_tie": has_tie,
            })

        return results

    async def run_merge_error_identification(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run merge error identification using API.

        Uses the merge_error_identification task to classify junctions
        as merge errors or valid connections.
        """
        from src.environment.task_configs import get_task
        from scripts.util import LLMProcessor
        import base64
        import io

        task = get_task("merge_error_identification")
        task._dataset_dir = parquet_dir

        llm = LLMProcessor(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            max_concurrent=self.config.max_concurrent
        )

        # Prepare messages
        batch_messages = []
        for sample in samples:
            prompt = task.format_prompt(sample)
            sample['_base_path'] = parquet_dir
            images = task.get_images(sample)

            content = []
            for img in images:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
            content.append({"type": "text", "text": prompt})

            batch_messages.append([{"role": "user", "content": content}])

        # Process
        responses = await llm.process_batch_conversations(batch_messages)

        # Parse responses - merge_error_identification returns 'yes' or 'no'
        results = []
        for idx, response in enumerate(responses):
            predicted_is_error = False
            if response:
                # Check for <answer>yes</answer> or <answer>no</answer>
                match = re.search(r'<answer>\s*(yes|no)\s*</answer>', response.lower())
                if match:
                    predicted_is_error = (match.group(1) == "yes")

            results.append({
                "sample_idx": idx,
                "predicted_is_error": predicted_is_error,
                "response": response,
            })

        return results


class ModalBackend(InferenceBackend):
    """Inference backend using Modal with merged models.

    Handles:
    - Uploading parquet datasets to Modal volume
    - Running inference with merged models (configured in configs/proofreading_models.json)
    - Returning results to local orchestrator

    Model configuration is read from configs/proofreading_models.json which specifies:
    - model_path: Path to merged model on /checkpoints volume
    - answer_only: Whether to use answer-only mode
    - num_samples: Number of samples for ensemble voting
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        identification_task: Optional[str] = None,
        correction_task: Optional[str] = None,
        config_group: Optional[str] = None,  # NEW: Group name like "vlm_all"
        include_em_slices: bool = True,  # NEW: Whether EM slices are included in data
        # Legacy parameters (ignored, kept for backward compatibility)
        identification_adapter: Optional[str] = None,
        correction_adapter: Optional[str] = None,
    ):
        """Initialize Modal backend.

        Args:
            config_path: Path to model config JSON (default: configs/proofreading_models.json)
            identification_task: Task name for identification (auto-detected if None)
            correction_task: Task name for correction (default: merge_action)
            config_group: Group name (e.g., "vlm_all") to use all tasks from that group
            include_em_slices: Whether EM slices are included (affects task selection)
        """
        self.config_path = config_path
        self.config_group = config_group

        # If group specified, resolve tasks from group
        if config_group:
            identification_task, correction_task = self._resolve_group_tasks(
                config_group, config_path, include_em_slices
            )

        self.identification_task = identification_task
        self.correction_task = correction_task or "merge_action"

        # Import Modal utilities lazily
        self._upload_fn = None
        self._run_task_fn = None
        self._modal_app = None

    def _resolve_group_tasks(self, group_name: str, config_path: Optional[str] = None, include_em_slices: bool = True) -> tuple:
        """Resolve identification and correction tasks from a group name.

        Args:
            group_name: Group name like "vlm_all", "siglip_all", "resnet_all"
            config_path: Path to config JSON
            include_em_slices: Whether to prefer EM-enabled tasks

        Returns:
            (identification_task, correction_task) tuple
        """
        import json
        from pathlib import Path

        if config_path is None:
            config_path = Path(__file__).parent / "proofreading_models.json"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        if group_name not in config:
            raise ValueError(f"Group '{group_name}' not found in config. Available groups: {list(config.keys())}")

        group_config = config[group_name]

        # Check if this is actually a group (nested dict) or a single task
        if not isinstance(group_config, dict):
            raise ValueError(f"'{group_name}' is not a valid group configuration")

        # Look for identification task (prefer EM version if include_em_slices=True)
        identification_task = None
        if include_em_slices:
            # Prefer EM-enabled task if available
            search_order = ["endpoint_error_identification_with_em", "endpoint_error_identification"]
        else:
            # Prefer non-EM task
            search_order = ["endpoint_error_identification", "endpoint_error_identification_with_em"]

        for task_key in search_order:
            if task_key in group_config:
                identification_task = f"{group_name}.{task_key}"
                break

        # Look for correction task
        correction_task = None
        if "merge_action" in group_config:
            correction_task = f"{group_name}.merge_action"

        print(f"Resolved group '{group_name}':")
        print(f"  Identification: {identification_task or 'auto-detect'}")
        print(f"  Correction: {correction_task or 'not found'}")

        return identification_task, correction_task

    def _get_modal_functions(self):
        """Lazily import Modal functions and app."""
        if self._upload_fn is None:
            from scripts.proofreading.modal_proofreading_inference import (
                upload_directory_to_volume,
                run_task_inference,
                app,
            )
            self._upload_fn = upload_directory_to_volume
            self._run_task_fn = run_task_inference
            self._modal_app = app
        return self._upload_fn, self._run_task_fn

    async def run_identification(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run split error identification on Modal using endpoint_error_identification task.

        Dynamically chooses between endpoint_error_identification (3 images)
        and endpoint_error_identification_with_em (6 images) based on the
        number of images in the parquet file.
        """
        import uuid

        upload_directory_to_volume, run_task_inference = self._get_modal_functions()

        # Determine task name
        if self.identification_task:
            # User specified exact task name
            # Strip group prefix if present (e.g., "vlm_all.endpoint_error_identification_with_em" -> "endpoint_error_identification_with_em")
            task_name = self.identification_task.split(".")[-1] if "." in self.identification_task else self.identification_task
            print(f"Using specified identification task: {task_name}")
        else:
            # Auto-detect based on number of images per sample
            # endpoint_error_identification_with_em expects 6 images (3 mesh + 3 EM)
            # endpoint_error_identification expects 3 images (mesh only)
            if samples and len(samples) > 0:
                first_images = samples[0].get('images', [])
                if hasattr(first_images, 'tolist'):
                    first_images = first_images.tolist()
                num_images = len(first_images)

                if num_images == 6:
                    task_name = "endpoint_error_identification_with_em"
                    print(f"Auto-detected task with EM views (6 images per sample)")
                else:
                    task_name = "endpoint_error_identification"
                    print(f"Auto-detected task without EM views ({num_images} images per sample)")
            else:
                task_name = "endpoint_error_identification"
                print("No samples, defaulting to endpoint_error_identification")

        # Generate unique volume path for this batch
        batch_id = str(uuid.uuid4())[:8]
        volume_path = f"/datasets/eval_{batch_id}/candidates"

        print(f"Uploading candidates to Modal volume: {volume_path}")
        upload_directory_to_volume(parquet_dir, volume_path)

        print(f"Running identification inference on Modal...")
        print(f"→ View logs in real-time at the Modal dashboard link shown above")
        print(f"→ Or run: modal app logs proofreading-inference")
        print(f"{'='*60}\n")

        with self._modal_app.run():
            results = run_task_inference.remote(
                task_name=task_name,
                dataset_path=volume_path,
            )

        print(f"\n{'='*60}")
        print(f"Received {len(results)} predictions from Modal")

        # Convert to standard format
        converted_results = []
        for r in results:
            converted_results.append({
                "sample_idx": r["sample_idx"],
                "predicted_is_error": r["prediction"],
                "response": r["response"],
            })

        return converted_results

    async def run_correction(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run binary correction on Modal.

        Note: The parquet has grouped format (one row per error with multiple
        candidates). We expand this into a flat parquet (one row per candidate)
        for Modal's binary merge_action task.
        """
        import tempfile
        import shutil
        import uuid
        from pathlib import Path as PathlibPath

        upload_directory_to_volume, run_task_inference = self._get_modal_functions()

        # Expand grouped samples into flat list (one per candidate)
        # Track mapping for result grouping
        flat_samples = []  # Each item: (error_idx, segment_id, is_correct, correct_segment_id, image_paths)

        for sample_idx, sample in enumerate(samples):
            metadata = sample.get('metadata', {})
            if hasattr(metadata, 'tolist'):
                metadata = dict(metadata)

            all_candidates = metadata.get('all_candidates', [])
            if hasattr(all_candidates, 'tolist'):
                all_candidates = all_candidates.tolist()

            correct_segment_id = metadata.get('correct_segment_id')
            error_idx = metadata.get('error_idx', sample_idx)
            images_per_candidate = metadata.get('images_per_candidate', 3)

            all_image_paths = sample.get('images', [])
            if hasattr(all_image_paths, 'tolist'):
                all_image_paths = all_image_paths.tolist()

            for cand_idx, c in enumerate(all_candidates):
                if hasattr(c, 'tolist'):
                    c = c.tolist()
                if not isinstance(c, dict):
                    continue

                segment_id = c.get('segment_id')
                is_correct = c.get('is_correct', False) or (segment_id == correct_segment_id)

                start_idx = cand_idx * images_per_candidate
                end_idx = start_idx + images_per_candidate
                image_paths = all_image_paths[start_idx:end_idx]

                flat_samples.append({
                    'error_idx': error_idx,
                    'segment_id': segment_id,
                    'is_correct': is_correct,
                    'correct_segment_id': correct_segment_id,
                    'image_paths': image_paths,
                })

        if not flat_samples:
            print("No candidates to process")
            return []

        print(f"Expanded {len(samples)} errors into {len(flat_samples)} candidate samples")

        # Create temporary flat parquet
        from src.training.question_dataset import DatasetQuestion, QuestionDataset, QuestionType, AnswerSpace

        flat_questions = []
        for i, fs in enumerate(flat_samples):
            # Resolve image paths to absolute
            abs_paths = []
            for img_path in fs['image_paths']:
                if not PathlibPath(img_path).is_absolute():
                    img_path = str(parquet_dir / img_path)
                abs_paths.append(img_path)

            question = DatasetQuestion(
                question_type=QuestionType.MERGE_VERIFICATION,
                answer_space=AnswerSpace.YES_OR_NO,
                answer=False,
                images=abs_paths,
                metadata={
                    "error_idx": fs['error_idx'],
                    "segment_id": fs['segment_id'],
                    "is_correct": fs['is_correct'],
                    "correct_segment_id": fs['correct_segment_id'],
                }
            )
            flat_questions.append(question)

        # Save to temp directory
        temp_dir = PathlibPath(tempfile.mkdtemp(prefix="modal_correction_"))
        try:
            flat_dataset = QuestionDataset(flat_questions)
            flat_dataset.to_parquet(str(temp_dir / "questions.parquet"), move_images=False)

            # Generate unique volume path
            batch_id = str(uuid.uuid4())[:8]
            volume_path = f"/datasets/eval_{batch_id}/correction"

            print(f"Uploading {len(flat_questions)} candidate samples to Modal volume: {volume_path}")
            upload_directory_to_volume(temp_dir, volume_path)

            # Strip group prefix if present (e.g., "vlm_all.merge_action" -> "merge_action")
            task_name = self.correction_task.split(".")[-1] if "." in self.correction_task else self.correction_task
            print(f"Running binary correction inference on Modal (task: {task_name})...")
            print(f"→ View logs in real-time at the Modal dashboard link shown above")
            print(f"{'='*60}\n")

            with self._modal_app.run():
                results = run_task_inference.remote(
                    task_name=task_name,
                    dataset_path=volume_path,
                )

            print(f"\n{'='*60}")
            print(f"Received {len(results)} predictions from Modal")

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Group results by error_idx using our flat_samples mapping
        error_results = {}

        for r in results:
            sample_idx = r.get("sample_idx", 0)
            predicted_merge = r.get("prediction", False)

            if sample_idx >= len(flat_samples):
                continue

            fs = flat_samples[sample_idx]
            error_idx = fs['error_idx']

            binary_result = {
                'segment_id': fs['segment_id'],
                'predicted_merge': predicted_merge,
                'is_correct_partner': fs['is_correct'],
                'response': r.get("response"),
            }

            if error_idx not in error_results:
                error_results[error_idx] = {
                    'binary_results': [],
                    'correct_segment_id': fs['correct_segment_id'],
                }
            error_results[error_idx]['binary_results'].append(binary_result)

        # Convert to final results format
        converted_results = []
        for error_idx in sorted(error_results.keys()):
            error_data = error_results[error_idx]
            binary_results = error_data['binary_results']
            correct_segment_id = error_data['correct_segment_id']

            predicted_merge_ids = [
                r['segment_id'] for r in binary_results if r['predicted_merge']
            ]
            has_tie = len(predicted_merge_ids) > 1

            converted_results.append({
                "sample_idx": error_idx,
                "binary_results": binary_results,
                "predicted_merge_ids": predicted_merge_ids,
                "correct_segment_id": correct_segment_id,
                "has_tie": has_tie,
            })

        return converted_results

    async def run_merge_error_identification(
        self,
        parquet_dir: Path,
        samples: List[Dict],
    ) -> List[Dict]:
        """Run merge error identification on Modal using merge_error_identification task."""
        import uuid

        upload_directory_to_volume, run_task_inference = self._get_modal_functions()

        # Generate unique volume path for this batch
        batch_id = str(uuid.uuid4())[:8]
        volume_path = f"/datasets/eval_{batch_id}/junctions"

        print(f"Uploading junction candidates to Modal volume: {volume_path}")
        upload_directory_to_volume(parquet_dir, volume_path)

        print(f"Running merge error identification inference on Modal...")
        print(f"→ View logs in real-time at the Modal dashboard link shown above")
        print(f"{'='*60}\n")

        results = run_task_inference.remote(
            task_name="merge_error_identification",
            dataset_path=volume_path,
        )

        print(f"\n{'='*60}")
        print(f"Received {len(results)} predictions from Modal")

        # Convert to standard format
        converted_results = []
        for r in results:
            converted_results.append({
                "sample_idx": r["sample_idx"],
                "predicted_is_error": r["prediction"],
                "response": r["response"],
            })

        return converted_results

    async def run_identification_batch(
        self,
        batch: List[Tuple[int, Path]],  # [(root_id, parquet_dir), ...]
    ) -> Dict[int, List[Dict]]:
        """Run identification on multiple roots in a SINGLE Modal call.

        Merges all parquets into one dataset with root_id labels, uploads once,
        runs inference once, then separates results by root_id.

        Args:
            batch: List of (root_id, parquet_dir) tuples

        Returns:
            Dict mapping root_id to list of prediction dicts
        """
        import uuid
        import tempfile
        import shutil
        import os
        import pandas as pd
        from src.training.question_dataset import QuestionDataset

        upload_directory_to_volume, run_task_inference = self._get_modal_functions()

        # Generate unique batch ID
        batch_id = str(uuid.uuid4())[:8]

        # Merge all parquets into a single dataset with root_id column
        print(f"Merging {len(batch)} candidate datasets...")
        temp_dir = tempfile.mkdtemp(prefix="batch_candidates_")

        try:
            all_questions = []
            root_id_mapping = []  # Track which samples belong to which root

            for root_id, parquet_dir in batch:
                # Load this root's questions
                dataset = QuestionDataset.from_parquet(parquet_dir)
                questions = dataset.questions

                # Resolve relative image paths to absolute
                for q in questions:
                    absolute_images = []
                    for img_path in q.images:
                        if not os.path.isabs(img_path):
                            # Relative path - resolve relative to parquet directory
                            absolute_path = (parquet_dir / img_path).resolve()
                            absolute_images.append(str(absolute_path))
                        else:
                            absolute_images.append(img_path)
                    q.images = absolute_images

                    # Add root_id to metadata
                    if not hasattr(q.metadata, 'get'):
                        q.metadata = dict(q.metadata) if q.metadata else {}
                    q.metadata['batch_root_id'] = root_id
                    all_questions.append(q)

                root_id_mapping.extend([root_id] * len(questions))

                print(f"  Root {root_id}: {len(questions)} samples")

            print(f"  Total: {len(all_questions)} samples across {len(batch)} roots")

            # Create merged dataset
            merged_dataset = QuestionDataset(all_questions)
            parquet_file = os.path.join(temp_dir, "questions.parquet")
            merged_dataset.to_parquet(parquet_file)

            # Determine task name
            first_parquet = batch[0][1] / "questions.parquet"
            df = pd.read_parquet(first_parquet)
            num_images = len(df.iloc[0]['images']) if len(df) > 0 else 3

            if self.identification_task:
                task_name = self.identification_task.split(".")[-1] if "." in self.identification_task else self.identification_task
            else:
                task_name = "endpoint_error_identification_with_em" if num_images == 6 else "endpoint_error_identification"

            # Upload merged dataset
            volume_path = f"/datasets/batch_{batch_id}/merged_candidates"
            print(f"\nUploading merged dataset to Modal: {volume_path}")
            upload_directory_to_volume(temp_dir, volume_path)

            # Single Modal inference call for ALL roots
            print(f"\nRunning SINGLE batch identification inference on Modal (task: {task_name})...")
            print(f"  Processing {len(all_questions)} samples from {len(batch)} roots")
            print(f"→ View logs at Modal dashboard")
            print(f"{'='*60}\n")

            with self._modal_app.run():
                results = run_task_inference.remote(
                    task_name=task_name,
                    dataset_path=volume_path,
                )

            print(f"\n{'='*60}")
            print(f"Received {len(results)} predictions from Modal")
            print(f"{'='*60}\n")

            # Separate results by root_id
            print("Separating results by root...")
            results_by_root = {}

            for idx, r in enumerate(results):
                root_id = root_id_mapping[idx]

                if root_id not in results_by_root:
                    results_by_root[root_id] = []

                results_by_root[root_id].append({
                    "sample_idx": len(results_by_root[root_id]),  # Re-index per root
                    "predicted_is_error": r["prediction"],
                    "response": r["response"],
                })

            for root_id, res_list in results_by_root.items():
                print(f"  Root {root_id}: {len(res_list)} results")

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        return results_by_root

    async def run_correction_batch(
        self,
        batch: List[Tuple[int, Path]],  # [(root_id, parquet_dir), ...]
    ) -> Dict[int, List[Dict]]:
        """Run correction on multiple roots in a SINGLE Modal call.

        Merges all parquets into one dataset with root_id labels, uploads once,
        runs inference once, then separates and groups results by root_id.

        Args:
            batch: List of (root_id, parquet_dir) tuples

        Returns:
            Dict mapping root_id to list of correction result dicts
        """
        import uuid
        import tempfile
        import shutil
        import os
        import pandas as pd
        from src.training.question_dataset import QuestionDataset

        upload_directory_to_volume, run_task_inference = self._get_modal_functions()

        # Generate unique batch ID
        batch_id = str(uuid.uuid4())[:8]

        # Merge all parquets into a single dataset
        print(f"Merging {len(batch)} correction datasets...")
        temp_dir = tempfile.mkdtemp(prefix="batch_corrections_")

        try:
            all_questions = []
            root_id_mapping = []  # (root_id, error_idx, segment_id, is_correct, correct_segment_id)

            for root_id, parquet_dir in batch:
                # Load this root's questions
                dataset = QuestionDataset.from_parquet(parquet_dir)
                questions = dataset.questions

                # Track metadata for result grouping
                parquet_path = parquet_dir / "questions.parquet"
                df = pd.read_parquet(parquet_path)

                # Expand grouped format (one row per error) to flat format (one row per candidate)
                num_candidates_for_root = 0
                for idx, q in enumerate(questions):
                    row = df.iloc[idx]
                    metadata = row.get('metadata', {})
                    if hasattr(metadata, 'tolist'):
                        metadata = dict(metadata)

                    all_candidates = metadata.get('all_candidates', [])
                    if hasattr(all_candidates, 'tolist'):
                        all_candidates = all_candidates.tolist()

                    correct_segment_id = metadata.get('correct_segment_id')
                    error_idx = metadata.get('error_idx', idx)
                    images_per_candidate = metadata.get('images_per_candidate', 3)

                    all_image_paths = q.images
                    if hasattr(all_image_paths, 'tolist'):
                        all_image_paths = all_image_paths.tolist()

                    # Create one question per candidate
                    for cand_idx, c in enumerate(all_candidates):
                        if hasattr(c, 'tolist'):
                            c = c.tolist()
                        if not isinstance(c, dict):
                            continue

                        segment_id = c.get('segment_id')
                        is_correct = c.get('is_correct', False) or (segment_id == correct_segment_id)

                        # Extract this candidate's images
                        start_idx = cand_idx * images_per_candidate
                        end_idx = start_idx + images_per_candidate
                        image_paths = all_image_paths[start_idx:end_idx]

                        # Resolve relative image paths to absolute
                        absolute_images = []
                        for img_path in image_paths:
                            if not os.path.isabs(img_path):
                                absolute_path = (parquet_dir / img_path).resolve()
                                absolute_images.append(str(absolute_path))
                            else:
                                absolute_images.append(img_path)

                        # Create binary question for this candidate
                        from src.training.question_dataset import DatasetQuestion, QuestionType, AnswerSpace
                        candidate_question = DatasetQuestion(
                            question_type=QuestionType.MERGE_VERIFICATION,
                            answer_space=AnswerSpace.YES_OR_NO,
                            answer=is_correct,
                            images=absolute_images,
                            metadata={
                                "batch_root_id": root_id,
                                "error_idx": error_idx,
                                "segment_id": segment_id,
                                "is_correct": is_correct,
                                "correct_segment_id": correct_segment_id,
                            }
                        )
                        all_questions.append(candidate_question)
                        num_candidates_for_root += 1

                        # Store metadata for result grouping
                        root_id_mapping.append({
                            'root_id': root_id,
                            'error_idx': error_idx,
                            'segment_id': segment_id,
                            'is_correct': is_correct,
                            'correct_segment_id': correct_segment_id,
                        })

                print(f"  Root {root_id}: {len(questions)} errors -> {num_candidates_for_root} candidates")

            print(f"  Total: {len(all_questions)} samples across {len(batch)} roots")

            # Create merged dataset
            merged_dataset = QuestionDataset(all_questions)
            parquet_file = os.path.join(temp_dir, "questions.parquet")
            merged_dataset.to_parquet(parquet_file)

            # Determine task name
            task_name = self.correction_task.split(".")[-1] if "." in self.correction_task else self.correction_task

            # Upload merged dataset
            volume_path = f"/datasets/batch_{batch_id}/merged_corrections"
            print(f"\nUploading merged dataset to Modal: {volume_path}")
            upload_directory_to_volume(temp_dir, volume_path)

            # Single Modal inference call for ALL roots
            print(f"\nRunning SINGLE batch correction inference on Modal (task: {task_name})...")
            print(f"  Processing {len(all_questions)} samples from {len(batch)} roots")
            print(f"→ View logs at Modal dashboard")
            print(f"{'='*60}\n")

            with self._modal_app.run():
                results = run_task_inference.remote(
                    task_name=task_name,
                    dataset_path=volume_path,
                )

            print(f"\n{'='*60}")
            print(f"Received {len(results)} predictions from Modal")
            print(f"{'='*60}\n")

            # Separate and group results by root_id and error_idx
            print("Separating and grouping results by root...")
            error_results_by_root = {}  # root_id -> error_idx -> error_data

            for idx, r in enumerate(results):
                mapping = root_id_mapping[idx]
                root_id = mapping['root_id']
                error_idx = mapping['error_idx']
                segment_id = mapping['segment_id']
                is_correct = mapping['is_correct']
                correct_segment_id = mapping['correct_segment_id']

                predicted_merge = r.get("prediction", False)

                binary_result = {
                    'segment_id': segment_id,
                    'predicted_merge': predicted_merge,
                    'is_correct_partner': is_correct,
                    'response': r.get("response"),
                }

                if root_id not in error_results_by_root:
                    error_results_by_root[root_id] = {}

                if error_idx not in error_results_by_root[root_id]:
                    error_results_by_root[root_id][error_idx] = {
                        'binary_results': [],
                        'correct_segment_id': correct_segment_id,
                    }

                error_results_by_root[root_id][error_idx]['binary_results'].append(binary_result)

            # Convert to final results format
            results_by_root = {}

            for root_id, error_results in error_results_by_root.items():
                converted_results = []

                for error_idx in sorted(error_results.keys()):
                    error_data = error_results[error_idx]
                    binary_results = error_data['binary_results']
                    correct_segment_id = error_data['correct_segment_id']

                    predicted_merge_ids = [
                        r['segment_id'] for r in binary_results if r['predicted_merge']
                    ]
                    has_tie = len(predicted_merge_ids) > 1

                    converted_results.append({
                        "sample_idx": error_idx,
                        "binary_results": binary_results,
                        "predicted_merge_ids": predicted_merge_ids,
                        "correct_segment_id": correct_segment_id,
                        "has_tie": has_tie,
                    })

                results_by_root[root_id] = converted_results
                print(f"  Root {root_id}: {len(converted_results)} error groups")

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

        return results_by_root


class ProofreadingEvaluator:
    """
    End-to-end proofreading evaluation pipeline.

    Stages:
    1. Load ground truth errors for the root_id
    2. Generate candidate error locations (skeleton endpoints)
    3. Run VLM to identify which candidates are errors
    4. Generate correction images for identified errors
    5. Run VLM to select correct merge partners
    6. Evaluate against ground truth
    """

    def __init__(
        self,
        root_id: int,
        output_dir: Path,
        backend: InferenceBackend,
        species: str = "mouse",
        match_threshold_nm: float = 4000.0,
        view_extent_nm: float = 5000.0,
        max_endpoint_candidates: int = None,
        max_junction_candidates: int = None,
        include_em_slices: bool = True,
        em_window_size_nm: float = None,
        skeleton_direction_radius_nm: float = 5000.0,
        error_types: List[str] = None,
        include_all_gt_errors_in_correction: bool = False,
        root_size_threshold: int = 1000,
    ):
        """
        Initialize the proofreading evaluator.

        Args:
            root_id: Neuron root ID to evaluate
            output_dir: Directory for outputs (parquets, images, results)
            backend: Inference backend (APIBackend or ModalBackend)
            species: Species name (mouse, fly, human)
            match_threshold_nm: Distance threshold for matching candidates to ground truth
            view_extent_nm: View extent for rendering images
            max_endpoint_candidates: Maximum skeleton endpoints to evaluate for split errors (None=all)
            max_junction_candidates: Maximum skeleton junctions to evaluate for merge errors (None=all)
            include_em_slices: If True, include EM slice views (XY, XZ, YZ) for identification
            em_window_size_nm: Size of EM volume to fetch for slices (default: auto)
            skeleton_direction_radius_nm: Radius for skeleton direction tip search (default: 5000nm)
            error_types: Types of errors to evaluate: ["split", "merge"] (default: ["split"])
            include_all_gt_errors_in_correction: If True, include all GT errors in correction step
                even if the model didn't identify them (tracks was_identified_by_model flag)
            root_size_threshold: Minimum supervoxel count for error retrieval (default: 1000)
        """
        self.root_id = root_id
        self.backend = backend
        self.species = species
        self.match_threshold_nm = match_threshold_nm
        self.view_extent_nm = view_extent_nm
        self.root_size_threshold = root_size_threshold
        self.max_endpoint_candidates = max_endpoint_candidates
        self.max_junction_candidates = max_junction_candidates
        self.include_em_slices = include_em_slices
        self.em_window_size_nm = em_window_size_nm
        self.skeleton_direction_radius_nm = skeleton_direction_radius_nm
        self.error_types = error_types or ["split"]  # Default to split errors only
        self.include_all_gt_errors_in_correction = include_all_gt_errors_in_correction

        # Determine configuration name for organizing results
        config_name = self._get_config_name()

        # Create output directories organized by configuration
        self.output_dir = Path(output_dir) / config_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir = self.output_dir / "candidates"  # For split error endpoints
        self.junctions_dir = self.output_dir / "junctions"    # For merge error junctions
        self.correction_dir = self.output_dir / "correction"
        self.candidates_dir.mkdir(exist_ok=True)
        self.junctions_dir.mkdir(exist_ok=True)
        self.correction_dir.mkdir(exist_ok=True)

        # Initialize client
        from connectome.utils import get_client_for_species
        self.client = get_client_for_species(species)

        # Load ground truth errors
        self.ground_truth_errors = self._load_ground_truth()

    def _get_config_name(self) -> str:
        """Determine configuration name for organizing results.

        Returns config key from proofreading_models.json (e.g., 'merge_action',
        'merge_action_siglip', 'endpoint_error_identification_resnet') or the
        model name for API-based backends (e.g., 'gpt-4o', 'claude-3-5-sonnet').
        """
        if isinstance(self.backend, APIBackend):
            # Use the model name, sanitized for filesystem
            model_name = self.backend.config.model
            # Replace slashes and other problematic characters
            model_name = model_name.replace("/", "-").replace(":", "-")
            return model_name
        elif isinstance(self.backend, ModalBackend):
            # If using a config group, use the group name
            if self.backend.config_group:
                return self.backend.config_group
            # Otherwise use correction_task as the primary config identifier
            # This is the task that will be used for the correction stage
            return self.backend.correction_task
        else:
            return "unknown"

    def _load_ground_truth(self) -> List[Dict]:
        """Load ground truth errors for the root_id based on configured error_types."""
        from connectome.errors import get_errors_for_root

        errors = list(get_errors_for_root(
            self.client,
            self.root_id,
            use_cache=True,
            root_size_threshold=self.root_size_threshold
        ))

        # Filter to configured error types
        filtered_errors = [e for e in errors if e.get('error_type') in self.error_types]

        # Count by type
        split_count = sum(1 for e in filtered_errors if e.get('error_type') == 'split')
        merge_count = sum(1 for e in filtered_errors if e.get('error_type') == 'merge')

        print(f"Loaded {len(filtered_errors)} ground truth errors for root {self.root_id}")
        print(f"  Split errors (need merge): {split_count}")
        print(f"  Merge errors (need split): {merge_count}")

        return filtered_errors

    def _get_ground_truth_coords(self) -> List[np.ndarray]:
        """Get coordinates of ground truth errors."""
        coords = []
        for error in self.ground_truth_errors:
            if 'interface_point' in error:
                coord = error['interface_point']
                if not isinstance(coord, np.ndarray):
                    coord = np.array(coord)
                coords.append(coord)
        return coords

    # =========================================================================
    # Stage 1: Candidate Localization
    # =========================================================================

    def generate_candidates_parquet(self, max_workers: int = 8) -> Path:
        """
        Generate candidate error locations with images.

        Uses skeleton endpoints as candidates and renders mesh projections plus
        optional EM slice views (XY, XZ, YZ) for each endpoint.

        Uses skeleton direction to find the mesh tip at each endpoint (ground-truth
        agnostic approach that doesn't require knowing the interface location).

        Args:
            max_workers: Maximum number of parallel workers for data preparation.
                         Rendering is done serially (GPU isn't thread-safe).

        Returns:
            Path to the candidates parquet directory
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from connectome.skeleton import get_endpoints_for_split_detection, render_endpoint_views, _get_cloudvolume_mesh, EndpointInfo
        from scripts.endpoint_data_generator import label_endpoints, LabeledEndpoint
        from src.training.question_dataset import QuestionDataset, DatasetQuestion, QuestionType, AnswerSpace

        print(f"\n{'='*60}")
        print("Stage 1: Candidate Generation")
        print(f"{'='*60}")

        # Get skeleton endpoints with skeleton structure for direction finding
        print("Getting skeleton endpoints with skeleton structure...")
        result = get_endpoints_for_split_detection(
            self.root_id, client=self.client, return_skeleton=True, species=self.species
        )
        endpoints, skeleton = result
        print(f"Found {len(endpoints)} skeleton endpoints")

        # Get ground truth error coordinates for labeling
        gt_coords = self._get_ground_truth_coords()

        # Label endpoints as split errors or natural termini
        if gt_coords:
            labeled_endpoints = label_endpoints(
                endpoints,
                gt_coords,
            )
        else:
            # No ground truth - all endpoints are "unknown"
            labeled_endpoints = [
                LabeledEndpoint(
                    endpoint=ep,
                    is_split_error=False,
                    distance_to_error_nm=None,
                    error_coord_nm=None
                )
                for ep in endpoints
            ]

        # Limit number of endpoints if specified
        if self.max_endpoint_candidates is not None and len(labeled_endpoints) > self.max_endpoint_candidates:
            # Prioritize: include all ground truth errors first, then sample the rest
            gt_endpoints = [lbl for lbl in labeled_endpoints if lbl.is_split_error]
            non_gt_endpoints = [lbl for lbl in labeled_endpoints if not lbl.is_split_error]

            if len(gt_endpoints) >= self.max_endpoint_candidates:
                # More GT errors than limit - just take first N
                labeled_endpoints = gt_endpoints[:self.max_endpoint_candidates]
            else:
                # Include all GT errors + sample from non-GT
                import random
                remaining_slots = self.max_endpoint_candidates - len(gt_endpoints)
                sampled_non_gt = random.sample(non_gt_endpoints, min(remaining_slots, len(non_gt_endpoints)))
                labeled_endpoints = gt_endpoints + sampled_non_gt

            print(f"Limited to {len(labeled_endpoints)} endpoints (max_endpoint_candidates={self.max_endpoint_candidates})")

        images_dir = self.candidates_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Load the neuron mesh once
        mesh = _get_cloudvolume_mesh(self.root_id, species=self.species)
        if mesh is None:
            raise RuntimeError(f"Failed to load mesh for root {self.root_id}")

        # Step 1: Prepare data in parallel (skeleton direction finding)
        print(f"Preparing endpoint data in parallel (max_workers={max_workers})...")

        def prepare_endpoint(args):
            """Prepare endpoint data (can be parallelized - no GPU)."""
            i, lbl = args
            ep = lbl.endpoint

            # Find the mesh tip using skeleton direction (ground-truth agnostic)
            tip_result = None
            if skeleton is not None and ep.skeleton_coord_nm is not None:
                tip_result = find_tip_along_skeleton_direction(
                    mesh=mesh,
                    skeleton=skeleton,
                    endpoint_coord=ep.skeleton_coord_nm,
                    radius_nm=self.skeleton_direction_radius_nm,
                )

            if tip_result is not None:
                tip_coord, tip_vertex_idx = tip_result
                endpoint_to_render = EndpointInfo(
                    skeleton_idx=ep.skeleton_idx,
                    skeleton_coord_nm=tip_coord,
                    mesh_coord_nm=tip_coord,
                    distance_to_mesh_nm=0.0,
                    root_id=self.root_id,
                )
            else:
                endpoint_to_render = ep
                tip_coord = ep.mesh_coord_nm

            return {
                'index': i,
                'lbl': lbl,
                'endpoint_to_render': endpoint_to_render,
                'tip_coord': tip_coord,
                'used_skeleton_direction': tip_result is not None,
            }

        # Parallel data preparation
        prepared_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(prepare_endpoint, (i, lbl)): i
                for i, lbl in enumerate(labeled_endpoints)
            }
            for future in as_completed(futures):
                try:
                    prepared_data.append(future.result())
                except Exception as e:
                    i = futures[future]
                    print(f"  Warning: Failed to prepare endpoint {i}: {e}")

        # Sort by index
        prepared_data.sort(key=lambda x: x['index'])
        print(f"  Prepared {len(prepared_data)} endpoints")

        # Step 2: Render serially (GPU isn't thread-safe)
        print(f"Rendering images for {len(prepared_data)} endpoints...")
        if self.include_em_slices:
            print("  Including EM slice views (XY, XZ, YZ)")

        questions = []
        for data in prepared_data:
            i = data['index']
            lbl = data['lbl']
            endpoint_to_render = data['endpoint_to_render']
            tip_coord = data['tip_coord']

            endpoint_dir = images_dir / f"endpoint_{i}"
            endpoint_dir.mkdir(exist_ok=True)

            try:
                render_result = render_endpoint_views(
                    endpoint=endpoint_to_render,
                    mesh=mesh,
                    output_dir=endpoint_dir,
                    extent_nm=self.view_extent_nm,
                    canvas_size_px=(512, 512),
                    show_projection_legend=True,
                    show_endpoint_marker=False,
                    include_em_slices=self.include_em_slices,
                    em_window_size_nm=self.em_window_size_nm,
                    species=self.species,
                )

                image_paths = [
                    render_result.image_paths.get('front'),
                    render_result.image_paths.get('side'),
                    render_result.image_paths.get('top'),
                ]

                if self.include_em_slices:
                    # EM slices use keys: em_front, em_side, em_top (not em_xy, em_xz, em_yz)
                    em_paths = [
                        render_result.image_paths.get('em_front'),
                        render_result.image_paths.get('em_side'),
                        render_result.image_paths.get('em_top'),
                    ]
                    image_paths.extend([p for p in em_paths if p is not None])

                # Keep absolute paths - to_parquet will make them relative
                image_paths = [str(p) for p in image_paths if p is not None]
                coord_for_correction = tip_coord.tolist() if hasattr(tip_coord, 'tolist') else list(tip_coord)

                question = DatasetQuestion(
                    question_type=QuestionType.ENDPOINT_ERROR_IDENTIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=lbl.is_split_error,
                    images=image_paths,
                    metadata={
                        "root_id": self.root_id,
                        "endpoint_idx": i,
                        "coord_nm": coord_for_correction,
                        "skeleton_coord_nm": lbl.endpoint.skeleton_coord_nm.tolist() if lbl.endpoint.skeleton_coord_nm is not None else None,
                        "is_ground_truth_error": lbl.is_split_error,
                        "distance_to_error_nm": lbl.distance_to_error_nm,
                        "used_skeleton_direction": data['used_skeleton_direction'],
                    },
                )
                questions.append(question)

                if (len(questions)) % 10 == 0:
                    print(f"  Rendered {len(questions)}/{len(prepared_data)} endpoints")

            except Exception as e:
                print(f"  Warning: Failed to render endpoint {i}: {e}")
                continue

        # Save as parquet
        dataset = QuestionDataset(questions)
        parquet_path = self.candidates_dir / "questions.parquet"
        dataset.to_parquet(str(parquet_path), move_images=False)

        print(f"Saved {len(questions)} candidates to {parquet_path}")

        # Summary
        num_true_errors = sum(1 for q in questions if q.answer)
        print(f"  True errors (ground truth): {num_true_errors}")
        print(f"  Natural termini: {len(questions) - num_true_errors}")

        return self.candidates_dir

    # =========================================================================
    # Stage 2: Error Identification
    # =========================================================================

    async def run_identification(
        self,
        candidates_parquet: Path = None
    ) -> Tuple[List[IdentificationResult], List[CandidateLocation]]:
        """
        Run VLM to identify which candidates are errors.

        Args:
            candidates_parquet: Path to candidates parquet (default: self.candidates_dir)

        Returns:
            Tuple of (identification_results, identified_error_locations)
        """
        if candidates_parquet is None:
            candidates_parquet = self.candidates_dir

        print(f"\n{'='*60}")
        print("Stage 2: Error Identification")
        print(f"{'='*60}")

        # Load candidates
        df = pd.read_parquet(candidates_parquet / "questions.parquet")
        samples = df.to_dict('records')

        print(f"Loaded {len(samples)} candidates for identification")

        # Run inference via backend (API or Modal)
        backend_results = await self.backend.run_identification(
            parquet_dir=candidates_parquet,
            samples=samples,
        )

        # Convert backend results to IdentificationResult objects
        results = []
        identified_errors = []

        for backend_result in backend_results:
            idx = backend_result["sample_idx"]
            sample = samples[idx]

            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            predicted_is_error = backend_result["predicted_is_error"]
            response = backend_result["response"]

            # Create candidate location
            candidate = CandidateLocation(
                endpoint_idx=metadata.get('endpoint_idx', 0),
                coord_nm=np.array(metadata.get('coord_nm', [0, 0, 0])),
                root_id=self.root_id,
                is_ground_truth_error=metadata.get('is_ground_truth_error', False),
                distance_to_nearest_error_nm=metadata.get('distance_to_error_nm'),
            )

            result = IdentificationResult(
                candidate=candidate,
                predicted_is_error=predicted_is_error,
                response=response,
            )
            results.append(result)

            if predicted_is_error:
                identified_errors.append(candidate)

        # Save identification responses
        identification_responses = []
        for i, (sample, result) in enumerate(zip(samples, results)):
            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            coord = metadata.get('coord_nm', [0, 0, 0])
            if hasattr(coord, 'tolist'):
                coord = coord.tolist()

            identification_responses.append({
                "endpoint_idx": i,
                "coord_nm": coord,
                "is_ground_truth_error": metadata.get('is_ground_truth_error', False),
                "predicted_is_error": result.predicted_is_error,
                "model_response": result.response,
                "correct": result.predicted_is_error == metadata.get('is_ground_truth_error', False),
            })

        responses_file = self.candidates_dir / "identification_responses.json"
        with open(responses_file, 'w') as f:
            json.dump(identification_responses, f, indent=2)
        print(f"Saved identification responses to {responses_file}")

        # Print summary
        num_predicted_errors = sum(1 for r in results if r.predicted_is_error)
        num_actual_errors = sum(1 for r in results if r.candidate.is_ground_truth_error)

        print(f"Identification complete:")
        print(f"  Predicted errors: {num_predicted_errors}")
        print(f"  Actual errors (ground truth): {num_actual_errors}")

        return results, identified_errors

    # =========================================================================
    # Merge Error Pipeline: Junction Candidate Generation & Identification
    # =========================================================================

    def _get_merge_error_coords(self) -> List[np.ndarray]:
        """Get coordinates of ground truth merge errors only."""
        coords = []
        for error in self.ground_truth_errors:
            if error.get('error_type') == 'merge' and 'interface_point' in error:
                coord = error['interface_point']
                if not isinstance(coord, np.ndarray):
                    coord = np.array(coord)
                coords.append(coord)
        return coords

    def generate_junction_candidates_parquet(self, max_workers: int = 8) -> Path:
        """
        Generate candidate merge error locations (junctions) with images.

        Uses skeleton branchpoints (degree >= 3 nodes) as candidates for merge
        error detection. Renders mesh projections and optional EM slice views
        at each junction.

        Args:
            max_workers: Maximum number of parallel workers for data preparation.
                         Rendering is done serially (GPU isn't thread-safe).

        Returns:
            Path to the junctions parquet directory
        """
        from connectome.skeleton import get_junctions_for_merge_detection, render_junction_views, _get_cloudvolume_mesh

        print(f"\n{'='*60}")
        print("Merge Error Pipeline - Stage 1: Junction Candidate Generation")
        print(f"{'='*60}")

        # Get skeleton junctions
        print("Getting skeleton junctions...")
        junctions, skeleton = get_junctions_for_merge_detection(
            self.root_id, client=self.client, return_skeleton=True, species=self.species
        )
        print(f"Found {len(junctions)} skeleton junctions (branchpoints)")

        # Get ground truth merge error coordinates for labeling
        gt_merge_coords = self._get_merge_error_coords()
        print(f"Ground truth merge errors: {len(gt_merge_coords)}")

        # Label junctions as merge errors or valid connections
        labeled_junctions = []
        for jnc in junctions:
            is_merge_error = False
            distance_to_error = None

            if gt_merge_coords:
                # Find nearest ground truth merge error
                distances = [np.linalg.norm(jnc.skeleton_coord_nm - gt) for gt in gt_merge_coords]
                if distances:
                    min_dist = min(distances)
                    distance_to_error = min_dist
                    if min_dist < self.match_threshold_nm:
                        is_merge_error = True

            labeled_junctions.append({
                'junction': jnc,
                'is_merge_error': is_merge_error,
                'distance_to_error_nm': distance_to_error,
            })

        # Limit number of junctions if specified
        if self.max_junction_candidates is not None and len(labeled_junctions) > self.max_junction_candidates:
            # Prioritize: include all ground truth errors first, then sample the rest
            gt_junctions = [lbl for lbl in labeled_junctions if lbl['is_merge_error']]
            non_gt_junctions = [lbl for lbl in labeled_junctions if not lbl['is_merge_error']]

            if len(gt_junctions) >= self.max_junction_candidates:
                labeled_junctions = gt_junctions[:self.max_junction_candidates]
            else:
                import random
                remaining_slots = self.max_junction_candidates - len(gt_junctions)
                sampled_non_gt = random.sample(non_gt_junctions, min(remaining_slots, len(non_gt_junctions)))
                labeled_junctions = gt_junctions + sampled_non_gt

            print(f"Limited to {len(labeled_junctions)} junctions (max_junction_candidates={self.max_junction_candidates})")

        images_dir = self.junctions_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Load the neuron mesh once
        mesh = _get_cloudvolume_mesh(self.root_id, species=self.species)

        # For junctions, data prep is minimal (no skeleton direction finding needed)
        # So we just render serially to avoid GPU threading issues
        print(f"Rendering images for {len(labeled_junctions)} junctions...")
        if self.include_em_slices:
            print("  Including EM slice views (XY, XZ, YZ)")

        samples = []
        for i, lbl in enumerate(labeled_junctions):
            jnc = lbl['junction']

            junction_dir = images_dir / f"junction_{i}"
            junction_dir.mkdir(exist_ok=True)

            try:
                render_result = render_junction_views(
                    junction=jnc,
                    mesh=mesh,
                    output_dir=junction_dir,
                    extent_nm=self.view_extent_nm,
                    canvas_size_px=(512, 512),
                    show_projection_legend=True,
                    show_junction_marker=False,
                    include_em_slices=self.include_em_slices,
                    em_window_size_nm=self.em_window_size_nm,
                    species=self.species,
                )

                image_paths = [
                    render_result.image_paths.get('front'),
                    render_result.image_paths.get('side'),
                    render_result.image_paths.get('top'),
                ]

                if self.include_em_slices:
                    em_paths = [
                        render_result.image_paths.get('em_top'),
                        render_result.image_paths.get('em_side'),
                        render_result.image_paths.get('em_front'),
                    ]
                    image_paths.extend([p for p in em_paths if p is not None])

                # Convert to relative paths from junctions_dir
                rel_image_paths = []
                for p in image_paths:
                    if p is not None:
                        rel_path = Path(p).relative_to(self.junctions_dir)
                        rel_image_paths.append(str(rel_path))

                answer = 'error' if lbl['is_merge_error'] else 'control'

                samples.append({
                    'answer': answer,
                    'images': rel_image_paths,
                    'metadata': {
                        "root_id": self.root_id,
                        "junction_idx": i,
                        "interface_point": jnc.mesh_coord_nm.tolist(),
                        "skeleton_coord_nm": jnc.skeleton_coord_nm.tolist(),
                        "degree": jnc.degree,
                        "is_ground_truth_error": lbl['is_merge_error'],
                        "distance_to_error_nm": lbl['distance_to_error_nm'],
                    },
                })

                if (len(samples)) % 10 == 0:
                    print(f"  Rendered {len(samples)}/{len(labeled_junctions)} junctions")

            except Exception as e:
                print(f"  Warning: Failed to render junction {i}: {e}")
                continue

        # Save as parquet directly
        df = pd.DataFrame(samples)
        parquet_path = self.junctions_dir / "questions.parquet"
        df.to_parquet(parquet_path)

        print(f"Saved {len(samples)} junction candidates to {parquet_path}")

        # Summary
        num_true_errors = sum(1 for s in samples if s['answer'] == 'error')
        print(f"  Merge errors (ground truth): {num_true_errors}")
        print(f"  Valid junctions: {len(samples) - num_true_errors}")

        return self.junctions_dir

    async def run_merge_error_identification(
        self,
        junctions_parquet: Path = None
    ) -> Tuple[List[MergeErrorIdentificationResult], List[JunctionLocation]]:
        """
        Run VLM to identify which junction candidates are merge errors.

        Args:
            junctions_parquet: Path to junctions parquet (default: self.junctions_dir)

        Returns:
            Tuple of (identification_results, identified_merge_error_locations)
        """
        if junctions_parquet is None:
            junctions_parquet = self.junctions_dir

        print(f"\n{'='*60}")
        print("Merge Error Pipeline - Stage 2: Merge Error Identification")
        print(f"{'='*60}")

        # Load junction candidates
        df = pd.read_parquet(junctions_parquet / "questions.parquet")
        samples = df.to_dict('records')

        print(f"Loaded {len(samples)} junction candidates for identification")

        # Run inference via backend
        backend_results = await self.backend.run_merge_error_identification(
            parquet_dir=junctions_parquet,
            samples=samples,
        )

        # Convert backend results to MergeErrorIdentificationResult objects
        results = []
        identified_merge_errors = []

        for backend_result in backend_results:
            idx = backend_result["sample_idx"]
            sample = samples[idx]

            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            predicted_is_error = backend_result["predicted_is_error"]
            response = backend_result["response"]

            # Create junction location
            junction = JunctionLocation(
                junction_idx=metadata.get('junction_idx', 0),
                coord_nm=np.array(metadata.get('interface_point', [0, 0, 0])),
                root_id=self.root_id,
                degree=metadata.get('degree', 3),
                is_ground_truth_error=metadata.get('is_ground_truth_error', False),
                distance_to_nearest_error_nm=metadata.get('distance_to_error_nm'),
            )

            result = MergeErrorIdentificationResult(
                junction=junction,
                predicted_is_error=predicted_is_error,
                response=response,
            )
            results.append(result)

            if predicted_is_error:
                identified_merge_errors.append(junction)

        # Save identification responses
        identification_responses = []
        for i, (sample, result) in enumerate(zip(samples, results)):
            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            coord = metadata.get('interface_point', [0, 0, 0])
            if hasattr(coord, 'tolist'):
                coord = coord.tolist()

            identification_responses.append({
                "junction_idx": i,
                "coord_nm": coord,
                "degree": metadata.get('degree', 3),
                "is_ground_truth_error": metadata.get('is_ground_truth_error', False),
                "predicted_is_error": result.predicted_is_error,
                "model_response": result.response,
                "correct": result.predicted_is_error == metadata.get('is_ground_truth_error', False),
            })

        responses_file = self.junctions_dir / "merge_error_identification_responses.json"
        with open(responses_file, 'w') as f:
            json.dump(identification_responses, f, indent=2)
        print(f"Saved merge error identification responses to {responses_file}")

        # Print summary
        num_predicted_errors = sum(1 for r in results if r.predicted_is_error)
        num_actual_errors = sum(1 for r in results if r.junction.is_ground_truth_error)

        print(f"Merge error identification complete:")
        print(f"  Predicted merge errors: {num_predicted_errors}")
        print(f"  Actual merge errors (ground truth): {num_actual_errors}")

        return results, identified_merge_errors

    # =========================================================================
    # Stage 2: Correction Image Generation (for split errors)
    # =========================================================================

    def _setup_visualizer(self) -> "ConnectomeVisualizer":
        """
        Setup visualizer with correct timestamp for correction generation.

        Returns:
            Configured ConnectomeVisualizer instance
        """
        from datetime import datetime, timezone
        from src.rendering.connectome_visualizer import ConnectomeVisualizer

        root_timestamps = self.client.chunkedgraph.get_root_timestamps([self.root_id])
        root_timestamp = root_timestamps[0] if root_timestamps else None
        if root_timestamp is not None:
            if not hasattr(root_timestamp, 'tzinfo') or root_timestamp.tzinfo is None:
                root_timestamp = root_timestamp.replace(tzinfo=timezone.utc)
            print(f"Using timestamp of root {self.root_id}: {root_timestamp}")
        else:
            print(f"Warning: Could not get timestamp for root {self.root_id}, using current time")

        visualizer = ConnectomeVisualizer(
            output_dir=str(self.correction_dir),
            species=self.species,
            use_mesh_cache=True,
        )
        visualizer.timestamp_dt = root_timestamp
        visualizer.load_neurons([self.root_id])

        return visualizer

    def _build_errors_to_process(
        self,
        identified_errors: List[CandidateLocation],
        identification_results: List["IdentificationResult"],
    ) -> List[Tuple[CandidateLocation, bool]]:
        """
        Build list of errors to process with identification status.

        Args:
            identified_errors: Legacy list of identified error locations
            identification_results: Full identification results

        Returns:
            List of (CandidateLocation, was_identified_by_model) tuples
        """
        errors_to_process = []

        if self.include_all_gt_errors_in_correction and identification_results is not None:
            # Include all GT errors, tracking which ones were identified
            for result in identification_results:
                if result.candidate.is_ground_truth_error:
                    errors_to_process.append((result.candidate, result.predicted_is_error))

            if errors_to_process:
                num_identified = sum(1 for _, identified in errors_to_process if identified)
                num_missed = len(errors_to_process) - num_identified
                print(f"Including all {len(errors_to_process)} GT errors in correction:")
                print(f"  Identified by model: {num_identified}")
                print(f"  Missed by model (forced): {num_missed}")
        elif identified_errors:
            # Legacy behavior: only include errors that were identified
            errors_to_process = [(e, True) for e in identified_errors]
            print(f"Generating correction images for {len(errors_to_process)} identified errors")

        return errors_to_process

    def _find_correct_segment(
        self,
        candidate: CandidateLocation,
        nearby_ids: List[int],
        supervoxels_by_root: Dict[int, List[int]],
    ) -> Optional[int]:
        """
        Find which nearby segment is the ground truth merge partner.

        For a candidate error location that is a GT error, determines which of the
        nearby candidate segments was actually merged in the proofread segmentation.

        Args:
            candidate: The error location
            nearby_ids: List of nearby segment IDs to check
            supervoxels_by_root: Mapping from segment ID to supervoxels at error location

        Returns:
            segment_id if found, None otherwise
        """
        if not candidate.is_ground_truth_error:
            return None

        # Find the matching ground truth error
        for error in self.ground_truth_errors:
            error_coord = np.array(error.get('interface_point', []))
            if np.linalg.norm(error_coord - candidate.coord_nm) < self.match_threshold_nm:
                latest_root_id = error.get('latest_root_id')
                if latest_root_id is None:
                    print(f"  ⚠️  No latest_root_id in error dict")
                    return None

                # Get timestamp when the proofread segment exists
                try:
                    corrected_timestamps = self.client.chunkedgraph.get_root_timestamps([latest_root_id])
                    corrected_ts = corrected_timestamps[0] if corrected_timestamps else None
                except Exception as e:
                    print(f"  ⚠️  Failed to get timestamp for latest_root_id {latest_root_id}: {e}")
                    return None

                if corrected_ts is None:
                    print(f"  ⚠️  Could not get timestamp for latest_root_id {latest_root_id}")
                    return None

                print(f"  Checking candidates against latest_root_id {latest_root_id} at timestamp {corrected_ts}")

                # Check each candidate to see if it was merged into latest_root_id
                for seg_id in nearby_ids:
                    svs = supervoxels_by_root.get(seg_id, [])
                    if not svs:
                        continue
                    try:
                        root_at_corrected = self.client.chunkedgraph.get_roots(
                            [svs[0]], timestamp=corrected_ts
                        )[0]
                        if int(root_at_corrected) == int(latest_root_id):
                            print(f"  ✓ Correct merge partner {seg_id} FOUND (supervoxel {svs[0]} -> {latest_root_id})")
                            return seg_id
                    except Exception as e:
                        print(f"  Warning: Failed to check segment {seg_id}: {e}")
                        continue

                print(f"  ⚠️  No candidate was merged into latest_root_id {latest_root_id}")
                return None

        return None

    def _fetch_correction_data_for_error(
        self,
        candidate: CandidateLocation,
        visualizer: "ConnectomeVisualizer",
    ) -> Dict:
        """
        Fetch nearby segments and determine correct segment for a single error.

        This is the PARALLELIZABLE part - pure network I/O, no GPU operations.

        Args:
            candidate: The error location to process
            visualizer: Pre-configured ConnectomeVisualizer with timestamp set

        Returns:
            {
                'candidate': CandidateLocation,
                'nearby_ids': List[int],
                'supervoxels_by_root': Dict[int, List[int]],
                'correct_segment_id': Optional[int],
                'error': Optional[str],
            }
        """
        from src.training.merge_sampler import find_nearby_segments

        try:
            # Find nearby segments as merge candidates
            nearby_ids, supervoxels_by_root = find_nearby_segments(
                visualizer,
                candidate.coord_nm,
                exclude_ids={self.root_id},
                window_size_nm=256.0,
                view_extent_nm=self.view_extent_nm,
                vertices_threshold=100,
            )

            if not nearby_ids:
                return {
                    'candidate': candidate,
                    'nearby_ids': [],
                    'supervoxels_by_root': {},
                    'correct_segment_id': None,
                    'error': 'No nearby segments found',
                }

            print(f"  Found {len(nearby_ids)} candidate segments: {nearby_ids}")
            print(f"  is_ground_truth_error: {candidate.is_ground_truth_error}")

            # Determine correct segment if GT error
            correct_segment_id = self._find_correct_segment(
                candidate,
                nearby_ids,
                supervoxels_by_root
            )

            return {
                'candidate': candidate,
                'nearby_ids': nearby_ids,
                'supervoxels_by_root': supervoxels_by_root,
                'correct_segment_id': correct_segment_id,
                'error': None,
            }
        except Exception as e:
            return {
                'candidate': candidate,
                'nearby_ids': [],
                'supervoxels_by_root': {},
                'correct_segment_id': None,
                'error': str(e),
            }

    def _render_correction_images(
        self,
        error_idx: int,
        fetch_result: Dict,
        visualizer: "ConnectomeVisualizer",
        images_dir: Path,
    ) -> Dict:
        """
        Render correction images for a single error using pre-fetched data.

        This is the SERIAL part - GPU rendering operations.

        Args:
            error_idx: Index of this error (for naming)
            fetch_result: Output from _fetch_correction_data_for_error()
            visualizer: Pre-configured ConnectomeVisualizer
            images_dir: Base directory for images

        Returns:
            {
                'candidate': CandidateLocation,
                'image_paths': List[str],
                'candidates_info': List[Dict],
                'correct_segment_id': Optional[int],
            }
        """
        from src.training.merge_sampler import render_segment_pair, MergeEditInfo

        candidate = fetch_result['candidate']
        nearby_ids = fetch_result['nearby_ids']
        correct_segment_id = fetch_result['correct_segment_id']

        error_dir = images_dir / f"error_{error_idx}"
        error_dir.mkdir(exist_ok=True)

        all_image_paths = []
        candidates_info = []

        for j, seg_id in enumerate(nearby_ids):
            option_dir = error_dir / f"candidate_{j}"
            option_dir.mkdir(exist_ok=True)

            try:
                # Create a minimal edit info for rendering
                edit_info = MergeEditInfo(
                    neuron_id=self.root_id,
                    timestamp=0,
                    operation_id=error_idx,
                    before_root_ids=[self.root_id, seg_id],
                    after_root_ids=[],
                    interface_point=candidate.coord_nm,
                    vertex_counts={},
                    species=self.species,
                )

                result = render_segment_pair(
                    edit_info=edit_info,
                    client=self.client,
                    output_dir=option_dir,
                    segment1_id=self.root_id,
                    segment2_id=seg_id,
                    is_positive=(seg_id == correct_segment_id),
                    view_extent_nm=self.view_extent_nm,
                )

                if result:
                    # Get absolute paths
                    abs_paths = [str(p) for p in result.image_paths]
                    all_image_paths.append(abs_paths)

                    candidates_info.append({
                        "candidate_idx": j,
                        "segment_id": seg_id,
                        "is_correct": (seg_id == correct_segment_id),
                        "image_paths": abs_paths,
                    })

            except Exception as e:
                print(f"    Warning: Failed to render candidate {j}: {e}")
                continue

        # Flatten image paths for parquet storage
        flat_image_paths = []
        for c in candidates_info:
            flat_image_paths.extend(c["image_paths"])

        # Remove image_paths from candidates_info to avoid duplication
        candidates_meta = [
            {k: v for k, v in c.items() if k != "image_paths"}
            for c in candidates_info
        ]

        return {
            'candidate': candidate,
            'image_paths': flat_image_paths,
            'candidates_info': candidates_meta,
            'correct_segment_id': correct_segment_id,
        }

    def _build_correction_question(
        self,
        error_idx: int,
        render_result: Dict,
        was_identified_by_model: bool,
    ) -> "DatasetQuestion":
        """
        Build a DatasetQuestion from render result.

        Args:
            error_idx: Index of this error
            render_result: Output from _render_correction_images()
            was_identified_by_model: Whether model identified this error

        Returns:
            DatasetQuestion for this error
        """
        from src.training.question_dataset import DatasetQuestion, QuestionType, AnswerSpace

        candidate = render_result['candidate']
        correct_segment_id = render_result['correct_segment_id']

        question = DatasetQuestion(
            question_type=QuestionType.MERGE_VERIFICATION,
            answer_space=AnswerSpace.MULTIPLE_CHOICE,
            answer="none",  # Placeholder - real answer computed at inference
            images=render_result['image_paths'],
            metadata={
                "root_id": self.root_id,
                "error_idx": error_idx,
                "coord_nm": candidate.coord_nm.tolist(),
                "all_candidates": render_result['candidates_info'],
                "correct_segment_id": correct_segment_id,
                "is_ground_truth_error": candidate.is_ground_truth_error,
                "was_identified_by_model": was_identified_by_model,
                "images_per_candidate": 3,
            }
        )

        return question

    def generate_correction_parquet(
        self,
        identified_errors: List[CandidateLocation] = None,
        identification_results: List[IdentificationResult] = None,
    ) -> Path:
        """
        Generate correction images for identified errors.

        For each identified error location, finds nearby segments and renders
        them as merge candidates.

        Refactored to separate data fetching (parallelizable I/O) from rendering (serial GPU).

        Args:
            identified_errors: List of candidate locations identified as errors (legacy)
            identification_results: Full identification results (used when include_all_gt_errors_in_correction=True)

        Returns:
            Path to the correction parquet directory
        """
        from src.training.question_dataset import QuestionDataset

        print(f"\n{'='*60}")
        print("Stage 3: Correction Image Generation")
        print(f"{'='*60}")

        # Build list of errors to process
        errors_to_process = self._build_errors_to_process(identified_errors, identification_results)

        if not errors_to_process:
            print("No errors to process - skipping correction stage")
            return None

        # Setup visualizer once
        visualizer = self._setup_visualizer()

        # Phase 1: Fetch all data (network I/O)
        print(f"\nFetching nearby segments data for {len(errors_to_process)} errors...")
        fetch_results = []
        for i, (candidate, was_identified) in enumerate(errors_to_process):
            id_status = "identified" if was_identified else "missed (forced)"
            print(f"\nProcessing error {i + 1}/{len(errors_to_process)} at {candidate.coord_nm} [{id_status}]")

            result = self._fetch_correction_data_for_error(candidate, visualizer)
            fetch_results.append((candidate, was_identified, result))

        # Phase 2: Render all images serially (GPU operations)
        print(f"\nRendering correction images for {len(fetch_results)} errors...")
        questions = []
        images_dir = self.correction_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for error_idx, (candidate, was_identified, fetch_result) in enumerate(fetch_results):
            if fetch_result['error']:
                print(f"  Skipping error {error_idx}: {fetch_result['error']}")
                continue

            if not fetch_result['nearby_ids']:
                print(f"  No nearby segments found for error {error_idx}")
                continue

            render_result = self._render_correction_images(
                error_idx,
                fetch_result,
                visualizer,
                images_dir,
            )

            if not render_result['candidates_info']:
                print(f"  No valid images rendered for error {error_idx}")
                continue

            # Build question
            question = self._build_correction_question(
                error_idx,
                render_result,
                was_identified,
            )
            questions.append(question)

        if not questions:
            print("No correction questions generated")
            return None

        # Save as parquet
        dataset = QuestionDataset(questions)
        dataset.to_parquet(str(self.correction_dir / "questions.parquet"), move_images=False)

        print(f"\nSaved {len(questions)} correction questions to {self.correction_dir}")

        return self.correction_dir

    # =========================================================================
    # Stage 3: Error Correction (Binary Merge Decisions)
    # =========================================================================

    async def run_correction(
        self,
        correction_parquet: Path = None
    ) -> List[CorrectionResult]:
        """
        Run VLM to evaluate each candidate with binary yes/no merge decision.

        For each identified error, evaluates all nearby segments individually
        and reports which ones the model predicts should be merged.

        Args:
            correction_parquet: Path to correction parquet (default: self.correction_dir)

        Returns:
            List of CorrectionResult with binary decisions for each candidate
        """
        if correction_parquet is None:
            correction_parquet = self.correction_dir

        if correction_parquet is None or not (correction_parquet / "questions.parquet").exists():
            print("No correction parquet found - skipping correction stage")
            return []

        print(f"\n{'='*60}")
        print("Stage 3: Error Correction (Binary Merge Decisions)")
        print(f"{'='*60}")

        # Load correction questions
        df = pd.read_parquet(correction_parquet / "questions.parquet")
        samples = df.to_dict('records')

        print(f"Loaded {len(samples)} error locations for correction")

        # Run inference via backend (API or Modal)
        backend_results = await self.backend.run_correction(
            parquet_dir=correction_parquet,
            samples=samples,
        )

        # Convert backend results to CorrectionResult objects
        results = []

        for backend_result in backend_results:
            idx = backend_result["sample_idx"]
            sample = samples[idx]

            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            binary_results = backend_result.get("binary_results", [])
            predicted_merge_ids = backend_result.get("predicted_merge_ids", [])
            correct_segment_id = backend_result.get("correct_segment_id")
            has_tie = backend_result.get("has_tie", False)

            # Create candidate location
            candidate = CandidateLocation(
                endpoint_idx=metadata.get('error_idx', 0),
                coord_nm=np.array(metadata.get('coord_nm', [0, 0, 0])),
                root_id=self.root_id,
                is_ground_truth_error=metadata.get('is_ground_truth_error', False),
            )

            # Create binary correction results
            binary_correction_results = [
                BinaryCorrectionResult(
                    segment_id=br['segment_id'],
                    predicted_merge=br['predicted_merge'],
                    is_correct_partner=br.get('is_correct_partner', False),
                    response=br.get('response'),
                )
                for br in binary_results
            ]

            # Combine responses for logging
            combined_response = "\n---\n".join([
                f"Segment {br['segment_id']}: {'YES' if br['predicted_merge'] else 'NO'}"
                for br in binary_results
            ])

            result = CorrectionResult(
                candidate=candidate,
                binary_results=binary_correction_results,
                predicted_merge_ids=predicted_merge_ids,
                correct_segment_id=correct_segment_id,
                has_tie=has_tie,
                response=combined_response,
            )
            results.append(result)

        # Save correction responses
        correction_responses = []
        for i, (sample, result, backend_result) in enumerate(zip(samples, results, backend_results)):
            metadata = sample.get('metadata', {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            coord = metadata.get('coord_nm', [0, 0, 0])
            if hasattr(coord, 'tolist'):
                coord = coord.tolist()

            # Determine correctness
            correct_in_predicted = result.correct_segment_id in result.predicted_merge_ids if result.correct_segment_id else False

            correction_responses.append({
                "error_idx": i,
                "coord_nm": coord,
                "is_ground_truth_error": metadata.get('is_ground_truth_error', False),
                "total_candidates": len(backend_result.get('binary_results', [])),
                "predicted_merge_ids": result.predicted_merge_ids,
                "correct_segment_id": result.correct_segment_id,
                "has_tie": result.has_tie,
                "correct_in_predicted": correct_in_predicted,
                "binary_results": [
                    {
                        "segment_id": br.segment_id,
                        "predicted_merge": br.predicted_merge,
                        "is_correct_partner": br.is_correct_partner,
                    }
                    for br in result.binary_results
                ],
            })

        responses_file = self.correction_dir / "correction_responses.json"
        with open(responses_file, 'w') as f:
            json.dump(correction_responses, f, indent=2)
        print(f"Saved correction responses to {responses_file}")

        # Print summary
        num_with_correct = sum(1 for r in results if r.correct_segment_id in r.predicted_merge_ids)
        num_with_ties = sum(1 for r in results if r.has_tie)
        num_with_no_prediction = sum(1 for r in results if len(r.predicted_merge_ids) == 0)

        print(f"Correction complete:")
        print(f"  Correct segment in predictions: {num_with_correct}/{len(results)}")
        print(f"  Cases with ties (multiple yes): {num_with_ties}")
        print(f"  Cases with no predictions: {num_with_no_prediction}")

        return results

    # =========================================================================
    # Evaluation
    # =========================================================================

    def evaluate(
        self,
        identification_results: List[IdentificationResult],
        correction_results: List[CorrectionResult],
    ) -> ProofreadingResults:
        """
        Compute evaluation metrics.

        Args:
            identification_results: Results from error identification stage
            correction_results: Results from error correction stage

        Returns:
            ProofreadingResults with all metrics
        """
        print(f"\n{'='*60}")
        print("Evaluation")
        print(f"{'='*60}")

        # Initialize results
        results = ProofreadingResults(
            root_id=self.root_id,
            num_ground_truth_errors=len(self.ground_truth_errors),
            ground_truth_error_coords=self._get_ground_truth_coords(),
            num_candidates=len(identification_results),
            num_true_positive_candidates=sum(
                1 for r in identification_results if r.candidate.is_ground_truth_error
            ),
            identification_results=identification_results,
            correction_results=correction_results,
        )

        # Identification metrics
        for r in identification_results:
            if r.predicted_is_error:
                if r.candidate.is_ground_truth_error:
                    results.identification_true_positives += 1
                else:
                    results.identification_false_positives += 1
            else:
                if r.candidate.is_ground_truth_error:
                    results.identification_false_negatives += 1

        # Precision and recall
        if results.identification_true_positives + results.identification_false_positives > 0:
            results.identification_precision = (
                results.identification_true_positives /
                (results.identification_true_positives + results.identification_false_positives)
            )

        if results.identification_true_positives + results.identification_false_negatives > 0:
            results.identification_recall = (
                results.identification_true_positives /
                (results.identification_true_positives + results.identification_false_negatives)
            )

        # Correction metrics (binary merge decisions)
        for r in correction_results:
            has_predictions = len(r.predicted_merge_ids) > 0
            correct_in_predicted = r.correct_segment_id in r.predicted_merge_ids if r.correct_segment_id else False

            if r.has_tie:
                results.correction_ties_total += 1

            if r.correct_segment_id is not None:
                # There was a correct answer
                if correct_in_predicted:
                    if r.has_tie:
                        results.correction_correct_with_tie += 1
                    else:
                        results.correction_correct_unique += 1
                elif has_predictions:
                    # Predicted wrong segments only
                    results.correction_wrong_only += 1
                else:
                    # No predictions when there should have been
                    results.correction_none_when_should += 1
            else:
                # No correct answer (false positive from identification)
                if not has_predictions:
                    results.correction_none_correct += 1
                else:
                    # Predicted merges on a non-error = introduces new errors
                    results.correction_wrong_only += 1

        # End-to-end metrics
        # A correction is "end-to-end correct" if:
        # 1. It was a real error (true positive identification)
        # 2. The correct merge partner was in the predictions
        for r in correction_results:
            if r.candidate.is_ground_truth_error:
                correct_in_predicted = r.correct_segment_id in r.predicted_merge_ids if r.correct_segment_id else False
                if correct_in_predicted:
                    results.end_to_end_correct += 1
            else:
                # False positive identification
                if len(r.predicted_merge_ids) > 0:
                    # Model made a correction on a non-error = introduces new error
                    results.end_to_end_false_corrections += 1

        # Print summary
        print(f"\nIdentification:")
        print(f"  True Positives: {results.identification_true_positives}")
        print(f"  False Positives: {results.identification_false_positives}")
        print(f"  False Negatives: {results.identification_false_negatives}")
        print(f"  Precision: {results.identification_precision:.2%}")
        print(f"  Recall: {results.identification_recall:.2%}")

        print(f"\nCorrection (Binary Merge Decisions):")
        print(f"  Correct (unique): {results.correction_correct_unique}")
        print(f"  Correct (with tie): {results.correction_correct_with_tie}")
        print(f"  Wrong only: {results.correction_wrong_only}")
        print(f"  None (correct - FP from ID): {results.correction_none_correct}")
        print(f"  None (wrong - missed): {results.correction_none_when_should}")
        print(f"  Total ties: {results.correction_ties_total}")

        print(f"\nEnd-to-End:")
        print(f"  Correctly proofread: {results.end_to_end_correct}")
        print(f"  False corrections (new errors): {results.end_to_end_false_corrections}")

        return results

    # =========================================================================
    # Visualization: Before/After Proofreading
    # =========================================================================

    def get_proofread_segment_ids(self) -> Dict[str, Any]:
        """
        Get the segment IDs for before/after proofreading visualization.

        Returns:
            Dict with:
                - original_root_id: The unproofread segment
                - latest_root_ids: List of latest root IDs (result of proofreading)
                - merged_segment_ids: Set of segment IDs that were merged into the original
                - split_segment_ids: Set of segment IDs that were split off
        """
        from connectome.errors import get_errors_between

        print(f"Getting proofread segment IDs for root {self.root_id}...")

        # Get latest roots (descendants after proofreading)
        latest_roots = self.client.chunkedgraph.get_latest_roots(self.root_id)
        latest_roots = [int(r) for r in latest_roots if r != self.root_id]

        print(f"Found {len(latest_roots)} latest root descendants")

        merged_segment_ids = set()
        split_segment_ids = set()

        for latest_root in latest_roots:
            try:
                error_dict, supervoxel_dict = get_errors_between(
                    self.client,
                    self.root_id,
                    latest_root,
                    use_cache=True,
                )

                # corrected_only = supervoxels that were merged in (split errors fixed)
                # original_only = supervoxels that were split off (merge errors fixed)
                corrected_only = supervoxel_dict.get('corrected_only', set())
                original_only = supervoxel_dict.get('original_only', set())

                # Get root IDs for these supervoxels at the corrected timestamp
                # For corrected_only, these are segments that should be merged with original
                if corrected_only:
                    # These supervoxels are now part of latest_root, we need their original roots
                    for sv in list(corrected_only)[:10]:  # Sample to avoid too many lookups
                        try:
                            # Get what root this supervoxel belonged to before the merge
                            original_root_for_sv = self.client.chunkedgraph.get_roots(
                                [sv], timestamp=None  # Current timestamp
                            )
                            if original_root_for_sv:
                                merged_segment_ids.add(int(original_root_for_sv[0]))
                        except Exception:
                            pass

                print(f"  Latest root {latest_root}: {len(corrected_only)} merged SVs, {len(original_only)} split SVs")

            except Exception as e:
                print(f"  Error processing latest root {latest_root}: {e}")
                continue

        # Remove the original from merged set if present
        merged_segment_ids.discard(self.root_id)
        for lr in latest_roots:
            merged_segment_ids.discard(lr)

        return {
            'original_root_id': self.root_id,
            'latest_root_ids': latest_roots,
            'merged_segment_ids': merged_segment_ids,
            'split_segment_ids': split_segment_ids,
        }

    def render_before_after(
        self,
        output_dir: Path = None,
        correction_results: List[CorrectionResult] = None,
    ) -> Dict[str, Path]:
        """
        Render before/after proofreading comparison images.

        Renders three versions showing the whole segment:
        - BEFORE: Original unproofread segment (blue)
        - GROUND TRUTH: Human proofread result for the first error location (green)
        - MODEL PREDICTION: Original + merged segments predicted by model (multi-color)

        Args:
            output_dir: Directory to save images (default: self.output_dir / "before_after")
            correction_results: Results from correction stage to get predicted merges.
                               If None, falls back to get_proofread_segment_ids().

        Returns:
            Dict with paths to rendered images:
                - before_*: Original unproofread segment
                - ground_truth_*: Human proofread result
                - prediction_*: Model prediction (original + merged segments)
        """
        from rendering.connectome_visualizer import ConnectomeVisualizer
        from rendering.render_pipeline import render_neuron_views
        from rendering.render_utils import MeshSpec

        if output_dir is None:
            output_dir = self.output_dir / "before_after"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get ground truth info (latest roots from human proofreading)
        segment_info = self.get_proofread_segment_ids()

        # Get predicted merge IDs from correction results (using binary decisions)
        predicted_merge_ids = []
        if correction_results:
            for result in correction_results:
                # With binary decisions, predicted_merge_ids contains all "yes" predictions
                predicted_merge_ids.extend(result.predicted_merge_ids)
            # Deduplicate
            predicted_merge_ids = list(set(predicted_merge_ids))
        else:
            # Fall back to proofread segment IDs
            predicted_merge_ids = list(segment_info['merged_segment_ids'])

        # Get the ground truth segment for the first error location
        gt_root_for_center = None
        if self.ground_truth_errors:
            first_error = self.ground_truth_errors[0]
            gt_root_for_center = first_error.get('latest_root_id')
            print(f"  Ground truth segment for first error: {gt_root_for_center}")

        print(f"\nRendering before/after comparison:")
        print(f"  Original (before): {self.root_id}")
        print(f"  Ground truth: {gt_root_for_center}")
        print(f"  Model prediction (merged segments): {predicted_merge_ids}")

        result_paths = {}

        def compute_mesh_extent(mesh):
            """Compute extent needed to show entire mesh."""
            vertices = mesh.vertices
            bbox_min = vertices.min(axis=0)
            bbox_max = vertices.max(axis=0)
            bbox_size = bbox_max - bbox_min
            # Use the largest dimension with some padding
            return float(np.max(bbox_size) * 1.2)

        # =====================================================================
        # Render BEFORE (original unproofread segment)
        # =====================================================================
        print("\nRendering BEFORE (original unproofread segment)...")

        visualizer = ConnectomeVisualizer(
            output_dir=str(output_dir),
            species=self.species,
            use_mesh_cache=True,
        )
        visualizer.load_neurons([self.root_id])

        if not visualizer.neurons:
            print("  Failed to load original segment mesh")
            return result_paths

        original_mesh = visualizer.neurons[0]

        # Center on mesh centroid and compute extent to show whole segment
        center = np.mean(original_mesh.vertices, axis=0)
        extent = compute_mesh_extent(original_mesh)
        print(f"  Mesh centroid: {center}")
        print(f"  Mesh extent: {extent:.0f} nm")

        mesh_specs = [
            MeshSpec(root_id=self.root_id, mesh=original_mesh, opacity=1.0, color="#1f77b4"),
        ]

        before_paths, _ = render_neuron_views(
            root_id=self.root_id,
            meshes=mesh_specs,
            neuron_graph=None,
            center_coord=center,
            center_node=None,
            extent_nm=extent,
            output_dir=output_dir,
            base_name_prefix="before",
            show_projection_legend=True,
            include_graph=False,
            mesh_crop_enabled=False,  # Show whole mesh
        )

        for view, path in before_paths.items():
            result_paths[f'before_{view}'] = Path(path)

        print(f"  Saved: {list(before_paths.keys())}")

        # =====================================================================
        # Render GROUND TRUTH (human proofread result)
        # =====================================================================
        if gt_root_for_center:
            print(f"\nRendering GROUND TRUTH (human proofread - {gt_root_for_center})...")

            visualizer_gt = ConnectomeVisualizer(
                output_dir=str(output_dir),
                species=self.species,
                use_mesh_cache=True,
            )
            visualizer_gt.load_neurons([gt_root_for_center])

            if visualizer_gt.neurons:
                gt_mesh = visualizer_gt.neurons[0]

                # Center and extent for ground truth mesh
                gt_center = np.mean(gt_mesh.vertices, axis=0)
                gt_extent = compute_mesh_extent(gt_mesh)
                print(f"  Mesh centroid: {gt_center}")
                print(f"  Mesh extent: {gt_extent:.0f} nm")

                mesh_specs_gt = [
                    MeshSpec(root_id=gt_root_for_center, mesh=gt_mesh, opacity=1.0, color="#2ca02c"),
                ]

                gt_paths, _ = render_neuron_views(
                    root_id=gt_root_for_center,
                    meshes=mesh_specs_gt,
                    neuron_graph=None,
                    center_coord=gt_center,
                    center_node=None,
                    extent_nm=gt_extent,
                    output_dir=output_dir,
                    base_name_prefix="ground_truth",
                    show_projection_legend=True,
                    include_graph=False,
                    mesh_crop_enabled=False,  # Show whole mesh
                )

                for view, path in gt_paths.items():
                    result_paths[f'ground_truth_{view}'] = Path(path)

                print(f"  Saved: {list(gt_paths.keys())}")
            else:
                print("  Failed to load ground truth mesh")
        else:
            print("\nNo ground truth segment found - skipping ground truth render")

        # =====================================================================
        # Render PREDICTION (original + merged segments from model)
        # =====================================================================
        if predicted_merge_ids:
            print(f"\nRendering PREDICTION (original + {len(predicted_merge_ids)} merged segments)...")

            all_ids = [self.root_id] + predicted_merge_ids
            visualizer_pred = ConnectomeVisualizer(
                output_dir=str(output_dir),
                species=self.species,
                use_mesh_cache=True,
            )
            visualizer_pred.load_neurons(all_ids)

            # Compute combined bounding box for all meshes
            all_vertices = []
            for neuron in visualizer_pred.neurons:
                all_vertices.append(neuron.vertices)
            if all_vertices:
                combined_vertices = np.vstack(all_vertices)
                pred_center = np.mean(combined_vertices, axis=0)
                bbox_size = combined_vertices.max(axis=0) - combined_vertices.min(axis=0)
                pred_extent = float(np.max(bbox_size) * 1.2)
                print(f"  Combined centroid: {pred_center}")
                print(f"  Combined extent: {pred_extent:.0f} nm")
            else:
                pred_center = center
                pred_extent = extent

            # Original in blue, merged segments in orange/green/red/purple
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            mesh_specs_pred = []
            for i, (neuron, nid) in enumerate(zip(visualizer_pred.neurons, all_ids)):
                color = colors[i % len(colors)]
                mesh_specs_pred.append(
                    MeshSpec(root_id=nid, mesh=neuron, opacity=0.9, color=color)
                )

            pred_paths, _ = render_neuron_views(
                root_id=self.root_id,
                meshes=mesh_specs_pred,
                neuron_graph=None,
                center_coord=pred_center,
                center_node=None,
                extent_nm=pred_extent,
                output_dir=output_dir,
                base_name_prefix="prediction",
                show_projection_legend=True,
                include_graph=False,
                mesh_crop_enabled=False,  # Show whole mesh
            )

            for view, path in pred_paths.items():
                result_paths[f'prediction_{view}'] = Path(path)

            print(f"  Saved: {list(pred_paths.keys())}")
        else:
            print("\nNo predicted merges - skipping prediction render")

        print(f"\nBefore/after images saved to: {output_dir}")
        return result_paths

    def render_error_locations(
        self,
        output_dir: Path = None,
        view_extent_nm: float = None,
    ) -> List[Dict[str, Path]]:
        """
        Render zoomed views at each ground truth error location.

        Args:
            output_dir: Directory to save images (default: self.output_dir / "error_locations")
            view_extent_nm: View extent in nm (default: self.view_extent_nm)

        Returns:
            List of dicts, one per error, with paths to rendered images
        """
        from rendering.connectome_visualizer import ConnectomeVisualizer
        from rendering.render_pipeline import render_neuron_views
        from rendering.render_utils import MeshSpec

        if output_dir is None:
            output_dir = self.output_dir / "error_locations"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if view_extent_nm is None:
            view_extent_nm = self.view_extent_nm

        if not self.ground_truth_errors:
            print("No ground truth errors to render")
            return []

        print(f"\nRendering {len(self.ground_truth_errors)} error locations...")

        # Create visualizer and load mesh once
        visualizer = ConnectomeVisualizer(
            output_dir=str(output_dir),
            species=self.species,
            use_mesh_cache=True,
        )
        visualizer.load_neurons([self.root_id])

        if not visualizer.neurons:
            print("  Failed to load segment mesh")
            return []

        mesh = visualizer.neurons[0]

        all_results = []

        for i, error in enumerate(self.ground_truth_errors):
            error_coord = np.array(error.get('interface_point', [0, 0, 0]))
            error_type = error.get('error_type', 'unknown')

            print(f"  Error {i+1}: {error_type} at {error_coord}")

            mesh_specs = [
                MeshSpec(root_id=self.root_id, mesh=mesh, opacity=1.0, color="#1f77b4"),
            ]

            error_dir = output_dir / f"error_{i}"
            error_dir.mkdir(exist_ok=True)

            paths, _ = render_neuron_views(
                root_id=self.root_id,
                meshes=mesh_specs,
                neuron_graph=None,
                center_coord=error_coord,
                center_node=None,
                extent_nm=view_extent_nm,
                output_dir=error_dir,
                base_name_prefix=f"error_{i}_{error_type}",
                show_projection_legend=True,
                include_graph=False,
                mesh_crop_enabled=True,
            )

            result = {
                'error_idx': i,
                'error_type': error_type,
                'coord': error_coord.tolist(),
            }
            for view, path in paths.items():
                result[view] = Path(path)

            all_results.append(result)

        print(f"\nError location images saved to: {output_dir}")
        return all_results

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    async def run_full_pipeline(self) -> ProofreadingResults:
        """
        Run the complete proofreading evaluation pipeline.

        Pipeline stages:
        1. Candidate Generation: Find skeleton endpoints, use skeleton direction for tip
        2. Error Identification: VLM predicts which candidates are errors (with EM slices)
        3. Correction Image Generation: Render nearby segments for identified errors
        4. Error Correction: Binary merge decisions for each candidate segment
        5. Evaluation: Compute metrics

        Returns:
            ProofreadingResults with all metrics
        """
        print(f"\n{'='*60}")
        print(f"PROOFREADING EVALUATION PIPELINE")
        print(f"Root ID: {self.root_id}")
        print(f"Error types: {self.error_types}")
        print(f"Include EM slices: {self.include_em_slices}")
        print(f"{'='*60}")

        # Stage 1: Candidate Generation
        candidates_path = self.generate_candidates_parquet()

        # Stage 2: Error Identification
        identification_results, identified_errors = await self.run_identification(candidates_path)

        # Stage 3: Correction Image Generation (for split errors only)
        correction_path = self.generate_correction_parquet(
            identified_errors=identified_errors,
            identification_results=identification_results,
        )

        # Stage 4: Error Correction (binary decisions)
        correction_results = []
        if correction_path:
            correction_results = await self.run_correction(correction_path)

        # Evaluation
        results = self.evaluate(identification_results, correction_results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"proofreading_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        print(f"\nResults saved to: {results_file}")

        # Render before/after comparison images
        print(f"\n{'='*60}")
        print("Rendering Before/After Comparison Images")
        print(f"{'='*60}")
        self.render_before_after(correction_results=correction_results)

        return results


# =============================================================================
# CLI
# =============================================================================

async def run_parallel_evaluation(
    root_ids: List[int],
    backend: InferenceBackend,
    species: str,
    output_dir: Path,
    match_threshold_nm: float,
    view_extent_nm: float,
    max_endpoint_candidates: Optional[int],
    include_em_slices: bool,
    em_window_size_nm: Optional[float],
    skeleton_direction_radius_nm: float,
    error_types: List[str],
    root_size_threshold: int,
    force_regenerate: bool = False,
    inference_only: bool = False,
    correction_only: bool = False,
) -> List[Dict]:
    """Run proofreading evaluation on multiple root IDs with batched Modal inference.

    Strategy:
    1. Generate ALL candidate parquets locally (serial)
    2. Batch upload to Modal → Single identification call for ALL roots
    3. Generate ALL correction parquets locally (serial)
    4. Batch upload to Modal → Single correction call for ALL roots

    This minimizes Modal overhead (2 calls total instead of 2*N).
    """
    import time

    print(f"\n{'='*60}")
    print("BATCHED EVALUATION MODE")
    print(f"{'='*60}")
    print(f"Total root IDs: {len(root_ids)}")
    print(f"Strategy: Batch all data generation → 2 Modal calls for all roots")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Phase 1: Generate ALL candidate parquets locally
    print(f"{'='*60}")
    print("PHASE 1: Candidate Generation (All Roots)")
    print(f"{'='*60}\n")

    evaluators = []
    candidate_data = []  # (root_id, evaluator, candidates_path)

    for idx, root_id in enumerate(root_ids, 1):
        try:
            evaluator = ProofreadingEvaluator(
                root_id=root_id,
                output_dir=output_dir / str(root_id),
                backend=backend,
                species=species,
                match_threshold_nm=match_threshold_nm,
                view_extent_nm=view_extent_nm,
                max_endpoint_candidates=max_endpoint_candidates,
                include_em_slices=include_em_slices,
                em_window_size_nm=em_window_size_nm,
                skeleton_direction_radius_nm=skeleton_direction_radius_nm,
                error_types=error_types,
                root_size_threshold=root_size_threshold,
            )

            # Check if candidates already exist
            candidates_path = evaluator.candidates_dir
            parquet_file = candidates_path / "questions.parquet"

            # Skip generation in inference-only or correction-only modes
            if inference_only or correction_only:
                if not parquet_file.exists():
                    print(f"[{idx}/{len(root_ids)}] ERROR: Candidates parquet not found for root {root_id} (required for --inference-only/--correction-only)")
                    candidate_data.append((root_id, None, None))
                    continue
                print(f"[{idx}/{len(root_ids)}] Using existing candidates for root {root_id} (inference-only mode)")
            elif parquet_file.exists() and not force_regenerate:
                print(f"[{idx}/{len(root_ids)}] Candidates already exist for root {root_id}, skipping generation")
            else:
                if parquet_file.exists():
                    print(f"[{idx}/{len(root_ids)}] Force regenerating candidates for root {root_id}...")
                else:
                    print(f"[{idx}/{len(root_ids)}] Generating candidates for root {root_id}...")
                candidates_path = evaluator.generate_candidates_parquet()

            candidate_data.append((root_id, evaluator, candidates_path))
            evaluators.append(evaluator)

        except Exception as e:
            print(f"  ERROR: Failed to generate candidates for {root_id}: {e}")
            candidate_data.append((root_id, None, None))

    # Phase 2: Batch identification inference (or load from file if correction-only)
    print(f"\n{'='*60}")
    if correction_only:
        print("PHASE 2: Load Identification Results (correction-only mode)")
    else:
        print("PHASE 2: Batch Identification Inference")
    print(f"{'='*60}\n")

    identification_results_by_root = {}

    # Skip inference and load from file in correction-only mode
    if correction_only:
        import json
        for root_id, evaluator, candidates_path in candidate_data:
            if evaluator is None or candidates_path is None:
                continue

            # Load identification responses from saved file
            responses_file = evaluator.candidates_dir / "identification_responses.json"
            if not responses_file.exists():
                print(f"ERROR: identification_responses.json not found for root {root_id} (required for --correction-only)")
                continue

            with open(responses_file, 'r') as f:
                id_responses = json.load(f)

            # Convert to backend result format
            backend_results = []
            for resp in id_responses:
                backend_results.append({
                    "sample_idx": resp["endpoint_idx"],
                    "predicted_is_error": resp["predicted_is_error"],
                    "response": resp["model_response"],
                })

            identification_results_by_root[root_id] = backend_results
            print(f"Loaded {len(backend_results)} identification results for root {root_id}")

    # Check if backend supports batch operations
    elif isinstance(backend, ModalBackend) and hasattr(backend, 'run_identification_batch'):
        # Use batch API
        batch_candidates = [(rid, path) for rid, ev, path in candidate_data if path is not None]
        identification_results_by_root = await backend.run_identification_batch(batch_candidates)

        # Save identification responses for each root
        import pandas as pd
        import json
        for root_id, evaluator, candidates_path in candidate_data:
            if evaluator is None or candidates_path is None:
                continue

            id_results = identification_results_by_root.get(root_id, [])
            if not id_results:
                continue

            # Load parquet to get metadata
            parquet_path = candidates_path / "questions.parquet"
            df = pd.read_parquet(parquet_path)

            # Build identification responses
            identification_responses = []
            for idx, backend_result in enumerate(id_results):
                if idx >= len(df):
                    continue

                row = df.iloc[idx]
                metadata = row.get('metadata', {})
                coord_nm = metadata.get('coord_nm', [0, 0, 0])
                if hasattr(coord_nm, 'tolist'):
                    coord_nm = coord_nm.tolist()

                identification_responses.append({
                    "endpoint_idx": idx,
                    "coord_nm": coord_nm,
                    "is_ground_truth_error": metadata.get('is_ground_truth_error', False),
                    "predicted_is_error": backend_result["predicted_is_error"],
                    "model_response": backend_result["response"],
                    "correct": backend_result["predicted_is_error"] == metadata.get('is_ground_truth_error', False),
                })

            # Save to file
            responses_file = evaluator.candidates_dir / "identification_responses.json"
            with open(responses_file, 'w') as f:
                json.dump(identification_responses, f, indent=2)
            print(f"Saved identification responses for root {root_id} to {responses_file}")
    else:
        # Fall back to individual calls (these already save the files)
        print("Backend doesn't support batching, running individual calls...")
        identification_results_by_root = {}
        for root_id, evaluator, candidates_path in candidate_data:
            if evaluator is not None and candidates_path is not None:
                results, _ = await evaluator.run_identification(candidates_path)
                identification_results_by_root[root_id] = results

    # Phase 3: Generate ALL correction parquets locally (or skip if inference-only)
    print(f"\n{'='*60}")
    if inference_only or correction_only:
        print("PHASE 3: Load Existing Correction Data (inference-only mode)")
    else:
        print("PHASE 3: Correction Image Generation")
    print(f"{'='*60}\n")

    correction_data = []  # (root_id, evaluator, correction_path, identification_results)

    for root_id, evaluator, candidates_path in candidate_data:
        if evaluator is None or candidates_path is None:
            continue

        try:
            # Check if corrections already exist
            correction_path = evaluator.correction_dir
            correction_parquet = correction_path / "questions.parquet"

            # In inference-only or correction-only mode, require existing parquets
            if inference_only or correction_only:
                if not correction_parquet.exists():
                    print(f"ERROR: Correction parquet not found for root {root_id} (required for --inference-only/--correction-only)")
                    continue
                print(f"Using existing correction data for root {root_id} (inference-only mode)")

                # Load identification results for evaluation
                from pathlib import Path as PathlibPath
                import pandas as pd

                parquet_path = candidates_path / "questions.parquet"
                df = pd.read_parquet(parquet_path)

                id_results = identification_results_by_root.get(root_id, [])
                identification_results = []

                for idx, backend_result in enumerate(id_results):
                    if idx >= len(df):
                        continue

                    row = df.iloc[idx]
                    metadata = row.get('metadata', {})
                    coord_nm = metadata.get('coord_nm', [0, 0, 0])

                    candidate = CandidateLocation(
                        endpoint_idx=metadata.get('endpoint_idx', idx),
                        coord_nm=np.array(coord_nm),
                        root_id=root_id,
                        is_ground_truth_error=metadata.get('is_ground_truth_error', False),
                        distance_to_nearest_error_nm=metadata.get('distance_to_error_nm', None),
                    )

                    result = IdentificationResult(
                        candidate=candidate,
                        predicted_is_error=backend_result["predicted_is_error"],
                        response=backend_result["response"],
                    )
                    identification_results.append(result)

                correction_data.append((root_id, evaluator, correction_path, identification_results))

            elif correction_parquet.exists() and not force_regenerate:
                print(f"Corrections already exist for root {root_id}, skipping generation")

                # Still need to load identification results for evaluation
                from pathlib import Path as PathlibPath
                import pandas as pd

                parquet_path = candidates_path / "questions.parquet"
                df = pd.read_parquet(parquet_path)

                id_results = identification_results_by_root.get(root_id, [])
                identification_results = []

                for idx, backend_result in enumerate(id_results):
                    if idx >= len(df):
                        continue

                    row = df.iloc[idx]
                    metadata = row.get('metadata', {})
                    coord_nm = metadata.get('coord_nm', [0, 0, 0])

                    candidate = CandidateLocation(
                        endpoint_idx=metadata.get('endpoint_idx', idx),
                        coord_nm=np.array(coord_nm),
                        root_id=root_id,
                        is_ground_truth_error=metadata.get('is_ground_truth_error', False),
                        distance_to_nearest_error_nm=metadata.get('distance_to_error_nm', None),
                    )

                    result = IdentificationResult(
                        candidate=candidate,
                        predicted_is_error=backend_result["predicted_is_error"],
                        response=backend_result["response"],
                    )
                    identification_results.append(result)

                correction_data.append((root_id, evaluator, correction_path, identification_results))

            else:
                if correction_parquet.exists():
                    print(f"Force regenerating corrections for root {root_id}...")
                else:
                    print(f"Generating corrections for root {root_id}...")

                # Get identification results for this root
                id_results = identification_results_by_root.get(root_id, [])

                # Convert to IdentificationResult objects
                from pathlib import Path as PathlibPath
                import pandas as pd

                parquet_path = candidates_path / "questions.parquet"
                df = pd.read_parquet(parquet_path)

                identification_results = []
                identified_errors = []

                for idx, backend_result in enumerate(id_results):
                    if idx >= len(df):
                        continue

                    row = df.iloc[idx]
                    metadata = row.get('metadata', {})
                    coord_nm = metadata.get('coord_nm', [0, 0, 0])

                    candidate = CandidateLocation(
                        endpoint_idx=metadata.get('endpoint_idx', idx),
                        coord_nm=np.array(coord_nm),
                        root_id=root_id,
                        is_ground_truth_error=metadata.get('is_ground_truth_error', False),
                        distance_to_nearest_error_nm=metadata.get('distance_to_error_nm', None),
                    )

                    result = IdentificationResult(
                        candidate=candidate,
                        predicted_is_error=backend_result["predicted_is_error"],
                        response=backend_result["response"],
                    )
                    identification_results.append(result)

                    if result.predicted_is_error:
                        identified_errors.append(candidate)

                # Generate correction parquet (internally does fetch then render)
                correction_path = evaluator.generate_correction_parquet(
                    identified_errors=identified_errors,
                    identification_results=identification_results,
                )

                if correction_path:
                    correction_data.append((root_id, evaluator, correction_path, identification_results))

        except Exception as e:
            print(f"  ERROR: Failed to generate corrections for {root_id}: {e}")

    # Phase 4: Batch correction inference
    print(f"\n{'='*60}")
    print("PHASE 4: Batch Correction Inference")
    print(f"{'='*60}\n")

    correction_results_by_root = {}

    if isinstance(backend, ModalBackend) and hasattr(backend, 'run_correction_batch'):
        # Use batch API
        batch_corrections = [(rid, path) for rid, ev, path, _ in correction_data if path is not None]
        correction_results_by_root = await backend.run_correction_batch(batch_corrections)

        # Save correction responses for each root
        import json
        import pandas as pd
        for root_id, evaluator, correction_path, _ in correction_data:
            if correction_path is None:
                continue

            corr_results_dicts = correction_results_by_root.get(root_id, [])
            if not corr_results_dicts:
                continue

            # Load correction parquet to get metadata
            parquet_path = correction_path / "questions.parquet"
            df = pd.read_parquet(parquet_path)

            # Build correction responses
            correction_responses = []
            for corr_dict in corr_results_dicts:
                error_idx = corr_dict["sample_idx"]

                # Get metadata from parquet
                candidate_coord = [0, 0, 0]
                is_ground_truth = False
                if error_idx < len(df):
                    row = df.iloc[error_idx]
                    metadata = row.get('metadata', {})
                    candidate_coord = metadata.get('coord_nm', [0, 0, 0])
                    if hasattr(candidate_coord, 'tolist'):
                        candidate_coord = candidate_coord.tolist()
                    is_ground_truth = metadata.get('is_ground_truth_error', False)

                # Get predicted merge IDs
                predicted_merge_ids = corr_dict.get("predicted_merge_ids", [])
                correct_segment_id = corr_dict.get("correct_segment_id")
                has_tie = corr_dict.get("has_tie", False)

                # Count correct in predicted
                correct_in_predicted = correct_segment_id in predicted_merge_ids if correct_segment_id else False

                correction_responses.append({
                    "error_idx": error_idx,
                    "coord_nm": candidate_coord,
                    "is_ground_truth_error": is_ground_truth,
                    "total_candidates": len(corr_dict["binary_results"]),
                    "predicted_merge_ids": predicted_merge_ids,
                    "correct_segment_id": correct_segment_id,
                    "has_tie": has_tie,
                    "correct_in_predicted": correct_in_predicted,
                    "binary_results": [
                        {
                            "segment_id": br["segment_id"],
                            "predicted_merge": br["predicted_merge"],
                            "is_correct_partner": br["is_correct_partner"],
                        }
                        for br in corr_dict["binary_results"]
                    ],
                })

            # Save to file
            responses_file = evaluator.correction_dir / "correction_responses.json"
            with open(responses_file, 'w') as f:
                json.dump(correction_responses, f, indent=2)
            print(f"Saved correction responses for root {root_id} to {responses_file}")
    else:
        # Fall back to individual calls (these already save the files)
        print("Backend doesn't support batching, running individual calls...")
        for root_id, evaluator, correction_path, _ in correction_data:
            if correction_path is not None:
                results = await evaluator.run_correction(correction_path)
                correction_results_by_root[root_id] = results

    # Phase 5: Evaluate all roots
    print(f"\n{'='*60}")
    print("PHASE 5: Evaluation")
    print(f"{'='*60}\n")

    all_results = []

    for root_id, evaluator, candidates_path in candidate_data:
        if evaluator is None:
            all_results.append({"root_id": root_id, "error": "Failed during candidate generation", "status": "failed"})
            continue

        try:
            # Get results for this root
            id_results = identification_results_by_root.get(root_id, [])

            # Find identification_results for this root
            identification_results = []
            for _, eval_obj, _, id_res in correction_data:
                if eval_obj == evaluator:
                    identification_results = id_res
                    break

            # Get correction results and convert dicts to CorrectionResult objects
            corr_results_dicts = correction_results_by_root.get(root_id, [])
            corr_results = []

            for corr_dict in corr_results_dicts:
                # Find the corresponding candidate from identification_results
                sample_idx = corr_dict["sample_idx"]
                candidate = identification_results[sample_idx].candidate if sample_idx < len(identification_results) else None

                if candidate:
                    # Convert binary_results dicts to BinaryCorrectionResult objects
                    binary_results_objs = []
                    for br in corr_dict["binary_results"]:
                        binary_results_objs.append(BinaryCorrectionResult(
                            segment_id=br["segment_id"],
                            predicted_merge=br["predicted_merge"],
                            is_correct_partner=br["is_correct_partner"],
                            response=br.get("response"),
                        ))

                    corr_result = CorrectionResult(
                        candidate=candidate,
                        binary_results=binary_results_objs,
                        predicted_merge_ids=corr_dict["predicted_merge_ids"],
                        correct_segment_id=corr_dict.get("correct_segment_id"),
                        has_tie=corr_dict.get("has_tie", False),
                    )
                    corr_results.append(corr_result)

            # Evaluate
            if identification_results:
                results = evaluator.evaluate(identification_results, corr_results)

                # Save per-root results file
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = evaluator.output_dir / f"proofreading_results_{timestamp}.json"
                with open(results_file, 'w') as f:
                    import json
                    json.dump(results.to_dict(), f, indent=2)
                print(f"Results saved to: {results_file}")

                # Render before/after comparison images
                print(f"\nRendering before/after comparison for root {root_id}...")
                try:
                    evaluator.render_before_after(correction_results=corr_results)
                except Exception as e:
                    print(f"  Warning: Failed to render before/after for {root_id}: {e}")

                all_results.append({
                    "root_id": root_id,
                    "results": results.to_dict(),
                    "status": "success"
                })
            else:
                all_results.append({
                    "root_id": root_id,
                    "error": "No identification results",
                    "status": "failed"
                })

        except Exception as e:
            print(f"  ERROR: Evaluation failed for {root_id}: {e}")
            all_results.append({
                "root_id": root_id,
                "error": str(e),
                "status": "failed"
            })

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Batch processing complete: {len(root_ids)} roots in {total_time:.1f}s")
    print(f"{'='*60}\n")

    # Save batch results to JSON
    import json
    from datetime import datetime

    # Determine config name from backend
    if isinstance(backend, ModalBackend):
        config_name = backend.config_group if backend.config_group else "modal"
    else:
        config_name = "api"

    batch_output_dir = output_dir / config_name
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    batch_results_file = batch_output_dir / "batch_results.json"

    batch_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_roots": len(root_ids),
        "successful": sum(1 for r in all_results if r['status'] == 'success'),
        "failed": sum(1 for r in all_results if r['status'] == 'failed'),
        "total_time_seconds": total_time,
        "config": config_name,
        "species": species,
        "parameters": {
            "match_threshold_nm": match_threshold_nm,
            "view_extent_nm": view_extent_nm,
            "max_endpoint_candidates": max_endpoint_candidates,
            "include_em_slices": include_em_slices,
            "em_window_size_nm": em_window_size_nm,
            "skeleton_direction_radius_nm": skeleton_direction_radius_nm,
            "error_types": error_types,
            "root_size_threshold": root_size_threshold,
        },
        "results": all_results,
    }

    with open(batch_results_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)

    print(f"Saved batch results to: {batch_results_file}")

    return all_results


def main():
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="End-to-end proofreading evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # API backend (OpenAI, Anthropic, etc.)
  python proofreading_evaluator.py --root-id 123456 --backend api --model gpt-4o

  # Modal backend with adapters
  python proofreading_evaluator.py --root-id 123456 --backend modal \\
      --identification-adapter "endpoint_id_20260115" \\
      --correction-adapter "merge_action_20260115"
""",
    )

    # Root ID specification (mutually exclusive: single ID, multiple IDs, or file)
    root_group = parser.add_mutually_exclusive_group(required=True)
    root_group.add_argument("--root-id", type=int,
                           help="Single neuron root ID to evaluate")
    root_group.add_argument("--root-ids", type=int, nargs="+",
                           help="Multiple root IDs to evaluate (space-separated)")
    root_group.add_argument("--root-ids-file", type=str,
                           help="Path to file containing root IDs (one per line)")

    parser.add_argument("--output-dir", default="evaluation_results/proofreading",
                        help="Output directory")
    parser.add_argument("--parallel", action="store_true",
                        help="Process multiple root IDs in parallel (faster for batches)")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Force regeneration of all data even if parquets already exist")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip image generation, only run inference on existing parquets (must have been run before)")
    parser.add_argument("--correction-only", action="store_true",
                        help="Skip identification, only run correction inference on existing data (implies --inference-only)")
    parser.add_argument("--species", default="mouse",
                        choices=["mouse", "fly", "human", "zebrafish"],
                        help="Species (default: mouse)")
    parser.add_argument("--match-threshold-nm", type=float, default=2000.0,
                        help="Distance threshold for matching candidates to ground truth (nm)")
    parser.add_argument("--view-extent-nm", type=float, default=5000.0,
                        help="View extent for rendering images (nm)")
    parser.add_argument("--max-endpoint-candidates", type=int, default=None,
                        help="Maximum skeleton endpoints to evaluate (identification stage, default: all)")

    # EM and skeleton direction options
    parser.add_argument("--include-em-slices", action="store_true", default=True,
                        help="Include EM slice views (XY, XZ, YZ) for identification (default: True)")
    parser.add_argument("--no-em-slices", action="store_true",
                        help="Disable EM slice views")
    parser.add_argument("--em-window-nm", type=float, default=None,
                        help="Size of EM volume to fetch for slices (default: auto)")
    parser.add_argument("--skeleton-direction-radius-nm", type=float, default=5000.0,
                        help="Radius for skeleton direction tip search (default: 5000nm)")

    # Error type options
    parser.add_argument("--error-types", nargs="+", default=["split"],
                        choices=["split", "merge"],
                        help="Types of errors to evaluate (default: split)")
    parser.add_argument("--root-size-threshold", type=int, default=1000,
                        help="Minimum supervoxel count for error retrieval (default: 1000, use 0 to disable)")

    # Backend selection
    parser.add_argument("--backend", default="api", choices=["api", "modal"],
                        help="Inference backend: 'api' for direct API calls, 'modal' for Modal with adapters")

    # API backend options
    api_group = parser.add_argument_group("API backend options")
    api_group.add_argument("--model", default="gpt-4o",
                           help="LLM model name for API backend (default: gpt-4o)")
    api_group.add_argument("--max-tokens", type=int, default=1024,
                           help="Max tokens for model response")
    api_group.add_argument("--max-concurrent", type=int, default=10,
                           help="Max concurrent API requests")

    # Modal backend options
    modal_group = parser.add_argument_group("Modal backend options")
    modal_group.add_argument("--config-group", default=None,
                             help="Configuration group name from proofreading_models.json "
                                  "(e.g., 'vlm_all', 'siglip_all', 'resnet_all'). "
                                  "Overrides --identification-task and --correction-task.")
    modal_group.add_argument("--identification-task", default=None,
                             help="Task name for identification stage (e.g., endpoint_error_identification_siglip). "
                                  "If not specified, auto-detects based on image count.")
    modal_group.add_argument("--correction-task", default=None,
                             help="Task name for correction stage (e.g., merge_action_resnet). "
                                  "Default: merge_action")
    modal_group.add_argument("--identification-adapter", default=None,
                             help="[Deprecated] LoRA adapter path for identification stage")
    modal_group.add_argument("--correction-adapter", default=None,
                             help="[Deprecated] LoRA adapter path for correction stage")

    # Preview options
    preview_group = parser.add_argument_group("3D Preview options")
    preview_group.add_argument("--preview", action="store_true",
                               help="Show interactive 3D preview of neuron before running pipeline")
    preview_group.add_argument("--preview-errors", action="store_true",
                               help="Show ground truth errors in preview")
    preview_group.add_argument("--save-preview", type=str, default=None,
                               help="Save preview to HTML file (e.g., 'preview.html')")
    preview_group.add_argument("--preview-only", action="store_true",
                               help="Only show preview, don't run pipeline")

    args = parser.parse_args()

    # Parse root IDs from different sources
    root_ids = []
    if args.root_id:
        root_ids = [args.root_id]
    elif args.root_ids:
        root_ids = args.root_ids
    elif args.root_ids_file:
        with open(args.root_ids_file, 'r') as f:
            root_ids = [int(line.strip()) for line in f if line.strip() and not line.strip().startswith('#')]

    print(f"\n{'='*60}")
    print(f"Batch Proofreading Evaluation")
    print(f"{'='*60}")
    print(f"Root IDs to process: {len(root_ids)}")
    print(f"  {root_ids}")
    print(f"{'='*60}\n")

    # Create backend based on selection
    # Handle --no-em-slices flag
    include_em_slices = args.include_em_slices and not args.no_em_slices

    if args.backend == "api":
        llm_config = LLMConfig(
            model=args.model,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
        )
        backend = APIBackend(llm_config)
        print(f"Using API backend with model: {args.model}")
    else:
        backend = ModalBackend(
            identification_task=args.identification_task,
            correction_task=args.correction_task,
            config_group=args.config_group,
            include_em_slices=include_em_slices,
        )
        print(f"Using Modal backend")
        if args.config_group:
            print(f"  Configuration group: {args.config_group}")
        if args.identification_task:
            print(f"  Identification task: {args.identification_task}")
        if args.correction_task:
            print(f"  Correction task: {args.correction_task}")


    # Create species-specific output directory
    species_output_dir = Path(args.output_dir) / args.species

    # Choose parallel or sequential processing
    if args.parallel and len(root_ids) > 1:
        # Parallel mode for multiple root IDs
        all_results = asyncio.run(run_parallel_evaluation(
            root_ids=root_ids,
            backend=backend,
            species=args.species,
            output_dir=species_output_dir,
            match_threshold_nm=args.match_threshold_nm,
            view_extent_nm=args.view_extent_nm,
            max_endpoint_candidates=args.max_endpoint_candidates,
            include_em_slices=include_em_slices,
            em_window_size_nm=args.em_window_nm,
            skeleton_direction_radius_nm=args.skeleton_direction_radius_nm,
            error_types=args.error_types,
            root_size_threshold=args.root_size_threshold,
            force_regenerate=args.force_regenerate,
            inference_only=args.inference_only,
            correction_only=args.correction_only,
        ))
    else:
        # Sequential mode (original behavior)
        if args.parallel and len(root_ids) == 1:
            print("Note: --parallel specified but only 1 root ID provided, using sequential mode\n")

        all_results = []
        for idx, root_id in enumerate(root_ids, 1):
            print(f"\n{'='*60}")
            print(f"Processing Root ID {idx}/{len(root_ids)}: {root_id}")
            print(f"{'='*60}\n")

            # Show preview if requested (only for single root ID)
            if (args.preview or args.preview_only) and len(root_ids) == 1:
                try:
                    from src.visualization.mesh_viewer import preview_proofreading

                    print(f"\n{'='*60}")
                    print("3D PREVIEW")
                    print(f"{'='*60}")

                    preview_proofreading(
                        root_id=root_id,
                        species=args.species,
                        show_errors=args.preview_errors,
                        show=True,
                        save_path=args.save_preview,
                    )

                    if args.preview_only:
                        print("\n--preview-only specified, exiting without running pipeline")
                        return

                except ImportError as e:
                    print(f"Warning: Could not load preview module: {e}")
                    print("Install plotly with: pip install plotly")
                    if args.preview_only:
                        return

            # Create evaluator for this root ID
            try:
                evaluator = ProofreadingEvaluator(
                    root_id=root_id,
                    output_dir=species_output_dir / str(root_id),
                    backend=backend,
                    species=args.species,
                    match_threshold_nm=args.match_threshold_nm,
                    view_extent_nm=args.view_extent_nm,
                    max_endpoint_candidates=args.max_endpoint_candidates,
                    include_em_slices=include_em_slices,
                    em_window_size_nm=args.em_window_nm,
                    skeleton_direction_radius_nm=args.skeleton_direction_radius_nm,
                    error_types=args.error_types,
                    root_size_threshold=args.root_size_threshold,
                )

                # Run pipeline
                results = asyncio.run(evaluator.run_full_pipeline())

                print(f"\n{'='*60}")
                print(f"RESULTS FOR ROOT ID: {root_id}")
                print(f"{'='*60}")
                print(json.dumps(results.to_dict(), indent=2))

                all_results.append({
                    "root_id": root_id,
                    "results": results.to_dict(),
                    "status": "success"
                })

            except Exception as e:
                print(f"\n{'='*60}")
                print(f"ERROR PROCESSING ROOT ID: {root_id}")
                print(f"{'='*60}")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

                all_results.append({
                    "root_id": root_id,
                    "error": str(e),
                    "status": "failed"
                })

                # Continue with next root ID instead of crashing
                continue

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total root IDs: {len(root_ids)}")
    print(f"Successful: {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in all_results if r['status'] == 'failed')}")

    if any(r['status'] == 'failed' for r in all_results):
        print("\nFailed root IDs:")
        for r in all_results:
            if r['status'] == 'failed':
                print(f"  {r['root_id']}: {r['error']}")

    print(f"{'='*60}\n")

    # Save batch results to JSON (sequential mode)
    if len(root_ids) > 1:
        from datetime import datetime

        # Determine config name from backend
        if args.backend == "modal":
            config_name = args.config_group if args.config_group else "modal"
        else:
            config_name = args.model.replace("/", "_") if args.model else "api"

        batch_output_dir = species_output_dir / config_name
        batch_output_dir.mkdir(parents=True, exist_ok=True)

        batch_results_file = batch_output_dir / "batch_results.json"

        batch_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_roots": len(root_ids),
            "successful": sum(1 for r in all_results if r['status'] == 'success'),
            "failed": sum(1 for r in all_results if r['status'] == 'failed'),
            "config": config_name,
            "species": args.species,
            "parameters": {
                "match_threshold_nm": args.match_threshold_nm,
                "view_extent_nm": args.view_extent_nm,
                "max_endpoint_candidates": args.max_endpoint_candidates,
                "include_em_slices": include_em_slices,
                "em_window_size_nm": args.em_window_nm,
                "skeleton_direction_radius_nm": args.skeleton_direction_radius_nm,
                "error_types": args.error_types,
                "root_size_threshold": args.root_size_threshold,
            },
            "results": all_results,
        }

        with open(batch_results_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)

        print(f"Saved batch results to: {batch_results_file}\n")


if __name__ == "__main__":
    main()
