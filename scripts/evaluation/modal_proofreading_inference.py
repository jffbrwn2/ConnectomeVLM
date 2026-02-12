#!/usr/bin/env python3
"""
Modal inference backend for proofreading evaluation.

This module provides the Modal-based compute backend for VLM inference using
merged models. It is designed to be called from proofreading_evaluator.py
which handles the orchestration of the multi-stage pipeline.

Architecture:
    proofreading_evaluator.py (orchestrator)
    ├── generate_candidates_parquet()     [LOCAL - image generation]
    ├── ModalBackend.run_identification() [calls this module]
    │   ├── upload_directory_to_volume()
    │   └── run_task_inference.remote()
    ├── generate_correction_parquet()     [LOCAL - uses ID results]
    └── ModalBackend.run_correction()     [calls this module]

Tasks supported (configured in scripts/proofreading/proofreading_models.json):
- merge_action: Binary yes/no for merge partner selection
- split_action: Binary yes/no for split action evaluation
- split_proposal: Propose split points for merge errors
- endpoint_error_identification_with_em: Identify split errors with EM views
- merge_error_identification: Identify merge errors at junctions

Usage:
    # Recommended: Use proofreading_evaluator.py with Modal backend
    python scripts/analysis/proofreading_evaluator.py \\
        --root-id 864691135572735469 \\
        --backend modal

    # Direct CLI usage (for debugging/testing)
    modal run scripts/analysis/modal_proofreading_inference.py::upload_dataset \\
        --local-dir "evaluation_results/candidates" \\
        --volume-path "/datasets/eval_123/candidates"

    modal run scripts/analysis/modal_proofreading_inference.py::run_inference \\
        --dataset-path "/datasets/eval_123/candidates" \\
        --task-name "merge_action"
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

# =============================================================================
# Modal Configuration
# =============================================================================

MODEL_DIR = Path("/models")
CHECKPOINT_DIR = Path("/checkpoints")
DATASET_DIR = Path("/datasets")
RESULTS_DIR = Path("/results")

model_volume = modal.Volume.from_name("qwen-finetune-models", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("qwen-finetune-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("qwen-finetune-datasets", create_if_missing=True)
results_volume = modal.Volume.from_name("qwen-finetune-results", create_if_missing=True)

# ResNet checkpoints volume (separate from VLM checkpoints)
resnet_checkpoint_volume = modal.Volume.from_name("resnet-finetune-checkpoints", create_if_missing=True)
RESNET_CHECKPOINT_DIR = Path("/resnet_checkpoints")

# vLLM compile cache volume (for torch.compile cache persistence)
compile_cache_volume = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
COMPILE_CACHE_DIR = Path("/root/.cache/vllm")

# Modal image with vLLM and dependencies
modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "vllm>=0.11.0",
        "qwen-vl-utils",
    )
    .pip_install(
        "accelerate",
        "datasets",
        "pandas",
        "pyarrow",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "torchvision",  # For ResNet
        "scikit-learn",  # For SigLIP probe (LogisticRegression)
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Add local task_configs for prompt formatting
    .add_local_file(
        "src/environment/task_configs.py",
        remote_path="/root/task_configs.py"
    )
    # Add model config for task settings
    .add_local_file(
        "scripts/proofreading/proofreading_models.json",
        remote_path="/root/proofreading_models.json"
    )
    .add_local_file(
        "src/inference/model_config.py",
        remote_path="/root/model_config.py"
    )
)

app = modal.App("proofreading-inference", image=modal_image)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TaskConfig:
    """Configuration for a single task's inference settings."""
    model_type: str = "vlm"  # "vlm", "siglip_probe", or "resnet"
    model_path: Optional[str] = None  # Path to model (location depends on model_type)
    answer_only: bool = False  # Whether to use answer-only mode (VLM only)
    num_samples: int = 1  # Number of samples/votes for ensemble
    description: str = ""
    # SigLIP probe specific
    vision_model: str = "google/siglip2-so400m-patch16-512"  # Vision encoder for feature extraction
    pooling: str = "mean"  # Feature pooling method: "mean", "max", "cls"

    def is_configured(self) -> bool:
        return self.model_path is not None

    @classmethod
    def from_dict(cls, d: Dict) -> "TaskConfig":
        return cls(
            model_type=d.get("model_type", "vlm"),
            model_path=d.get("model_path"),
            answer_only=d.get("answer_only", False),
            num_samples=d.get("num_samples", 1),
            description=d.get("description", ""),
            vision_model=d.get("vision_model", "google/siglip2-so400m-patch16-512"),
            pooling=d.get("pooling", "mean"),
        )


# Valid proofreading tasks (base names - variants like _siglip, _resnet are also valid)
PROOFREADING_TASKS_BASE = [
    "merge_action",
    "split_action",
    "split_proposal",
    "endpoint_error_identification",
    "endpoint_error_identification_with_em",
    "merge_error_identification",
]


def load_proofreading_config(config_path: str = "/root/proofreading_models.json") -> Dict[str, TaskConfig]:
    """Load proofreading model configuration.

    Supports both flat task names and group-prefixed names (e.g., "vlm_all.merge_action").

    Args:
        config_path: Path to config JSON file

    Returns:
        Dict mapping task names to TaskConfig
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config not found at {config_path}, using empty config")
        return {}

    with open(config_path) as f:
        data = json.load(f)

    configs = {}
    for top_level_key, top_level_data in data.items():
        if top_level_key.startswith("_"):
            continue  # Skip comments

        # Check if this is a group (nested dict with task names) or a single task
        if isinstance(top_level_data, dict) and "model_type" not in top_level_data:
            # This is a group - flatten it with group prefix
            for task_name, task_data in top_level_data.items():
                if task_name.startswith("_"):
                    continue
                # Add both the prefixed name (for group access) and unprefixed (for backward compat)
                prefixed_name = f"{top_level_key}.{task_name}"
                configs[prefixed_name] = TaskConfig.from_dict(task_data)
                # Also add without prefix if not already present (backward compatibility)
                if task_name not in configs:
                    configs[task_name] = TaskConfig.from_dict(task_data)
        else:
            # This is a single task config (legacy format)
            configs[top_level_key] = TaskConfig.from_dict(top_level_data)

    return configs


def get_task_model_path(task_name: str, config: Dict[str, TaskConfig]) -> Optional[str]:
    """Get the model path for a task from config."""
    if task_name not in config:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(config.keys())}")
    return config[task_name].model_path


@app.function(
    volumes={
        str(CHECKPOINT_DIR): checkpoint_volume,
    },
)
def list_checkpoints(model_path: str = None, filter_str: str = None) -> List[str]:
    """Debug function to list checkpoint directory contents."""
    result = []
    result.append(f"Listing contents of {CHECKPOINT_DIR}:")

    if not CHECKPOINT_DIR.exists():
        result.append(f"  ERROR: {CHECKPOINT_DIR} does not exist!")
        return result

    items = sorted(CHECKPOINT_DIR.iterdir())

    # Filter if requested
    if filter_str:
        items = [item for item in items if filter_str.lower() in item.name.lower()]
        result.append(f"  (filtered by '{filter_str}')")

    for item in items[:50]:
        result.append(f"  {item.name}")
    if len(items) > 50:
        result.append(f"  ... and {len(items) - 50} more")

    if model_path:
        full_path = CHECKPOINT_DIR / model_path
        result.append(f"\nChecking model path: {full_path}")
        result.append(f"  .exists() = {full_path.exists()}")
        if full_path.exists():
            result.append(f"  Contents:")
            for item in sorted(full_path.iterdir())[:10]:
                result.append(f"    {item.name}")

    return result


# =============================================================================
# Inference Class (keeps model warm)
# =============================================================================

@app.cls(
    gpu="H100:2",
    timeout=7200,  # 2 hours for large batches
    min_containers=1,  # Keep 1 container warm to avoid cold start between stages
    volumes={
        MODEL_DIR: model_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
        COMPILE_CACHE_DIR: compile_cache_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ProofreadingInference:
    """Modal class for VLM inference with adapter hot-swapping.

    Model loads once on container start via @modal.enter().
    Stays warm for multiple method calls - adapters swap via LoRARequest.
    """

    base_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    tensor_parallel_size: int = 2
    max_model_len: int = 32768+4096
    max_lora_rank: int = 64
    max_tokens: int = 1024

    @modal.enter()
    def load_model(self):
        """Load vLLM model once when container starts."""
        from vllm import LLM, SamplingParams

        print(f"\n{'='*60}")
        print("Loading vLLM model")
        print(f"{'='*60}")
        print(f"  Model: {self.base_model}")
        print(f"  Tensor parallel: {self.tensor_parallel_size}")
        print(f"  Max model len: {self.max_model_len}")
        print(f"  LoRA enabled: True (max_rank={self.max_lora_rank})")

        self.llm = LLM(
            model=self.base_model,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            # LoRA settings
            enable_lora=True,
            max_lora_rank=self.max_lora_rank,
            # Multimodal settings for Qwen3-VL
            limit_mm_per_prompt={"image": 12},  # Up to 12 images per prompt
            mm_processor_kwargs={
                "min_pixels": 4 * 28 * 28,
                "max_pixels": 16384 * 28 * 28,
            },
        )

        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=0,  # Deterministic for evaluation
            repetition_penalty=1.15,
        )

        print("Model loaded successfully\n")

    @modal.method()
    def run_identification(
        self,
        dataset_path: str,
        adapter_path: Optional[str] = None,
    ) -> List[Dict]:
        """Run endpoint error identification inference.

        Args:
            dataset_path: Path to parquet directory on /datasets volume
            adapter_path: LoRA adapter path on /checkpoints volume (optional)

        Returns:
            List of dicts with 'sample_idx', 'prediction' (bool), 'response'
        """
        return self._run_inference(
            dataset_path=dataset_path,
            task_name="endpoint_error_identification",
            adapter_path=adapter_path,
        )

    @modal.method()
    def run_localization(
        self,
        dataset_path: str,
        adapter_path: Optional[str] = None,
    ) -> List[Dict]:
        """Run endpoint localization inference.

        Args:
            dataset_path: Path to parquet directory on /datasets volume
            adapter_path: LoRA adapter path on /checkpoints volume (optional)

        Returns:
            List of dicts with 'sample_idx', 'prediction' ([x,y,z] or None), 'response'
        """
        return self._run_inference(
            dataset_path=dataset_path,
            task_name="endpoint_localization",
            adapter_path=adapter_path,
        )

    @modal.method()
    def run_merge_action(
        self,
        dataset_path: str,
        adapter_path: Optional[str] = None,
    ) -> List[Dict]:
        """Run binary merge action inference (yes/no for each candidate).

        Args:
            dataset_path: Path to parquet directory on /datasets volume
            adapter_path: LoRA adapter path on /checkpoints volume (optional)

        Returns:
            List of dicts with 'sample_idx', 'prediction' (bool), 'response'
        """
        return self._run_inference(
            dataset_path=dataset_path,
            task_name="merge_action",
            adapter_path=adapter_path,
        )

    @modal.method()
    def run_merge_error_identification(
        self,
        dataset_path: str,
        adapter_path: Optional[str] = None,
    ) -> List[Dict]:
        """Run merge error identification inference.

        Identifies if a junction point is a merge error (incorrectly joined segments)
        or a valid connection.

        Args:
            dataset_path: Path to parquet directory on /datasets volume
            adapter_path: LoRA adapter path on /checkpoints volume (optional)

        Returns:
            List of dicts with 'sample_idx', 'prediction' (bool), 'response'
        """
        return self._run_inference(
            dataset_path=dataset_path,
            task_name="merge_error_identification",
            adapter_path=adapter_path,
        )

    @modal.method()
    def run_correction(
        self,
        dataset_path: str,
        adapter_path: Optional[str] = None,
    ) -> List[Dict]:
        """Run merge action correction inference.

        Args:
            dataset_path: Path to parquet directory on /datasets volume
            adapter_path: LoRA adapter path on /checkpoints volume (optional)

        Returns:
            List of dicts with 'sample_idx', 'prediction' (a/b/c/d/none), 'response'
        """
        return self._run_inference(
            dataset_path=dataset_path,
            task_name="merge_action_multiple_choice",
            adapter_path=adapter_path,
        )

    def _run_inference(
        self,
        dataset_path: str,
        task_name: str,
        adapter_path: Optional[str] = None,
    ) -> List[Dict]:
        """Internal method to run batch inference."""
        import pandas as pd
        from PIL import Image
        from vllm.lora.request import LoRARequest

        print(f"\n{'='*60}")
        print(f"Running inference: {task_name}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Adapter: {adapter_path or 'base model'}")
        print(f"{'='*60}")

        # Setup LoRA request if adapter provided
        lora_request = None
        if adapter_path:
            full_adapter_path = str(CHECKPOINT_DIR / adapter_path)
            print(f"Loading LoRA adapter: {full_adapter_path}")
            lora_request = LoRARequest("adapter", 1, full_adapter_path)

        # Load dataset
        dataset_dir = Path(dataset_path)
        parquet_path = dataset_dir / "questions.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Dataset not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        samples = df.to_dict('records')
        print(f"Loaded {len(samples)} samples")

        # Get task config for prompt formatting
        # Import from the copied task_configs.py
        sys.path.insert(0, "/root")
        from task_configs import get_task

        # Strip group prefix if present (e.g., "vlm_all.merge_action" -> "merge_action")
        base_task_name = task_name.split(".")[-1] if "." in task_name else task_name
        task = get_task(base_task_name)
        task._dataset_dir = dataset_dir

        # Prepare batch inputs
        batch_inputs = []
        valid_indices = []

        for i, sample in enumerate(samples):
            try:
                # Format prompt
                prompt = task.format_prompt(sample)

                # Load images
                sample['_base_path'] = dataset_dir
                images = task.get_images(sample)

                batch_inputs.append({
                    "prompt": self._build_chat_prompt(prompt),
                    "multi_modal_data": {"image": images if len(images) > 1 else images[0]},
                })
                valid_indices.append(i)

            except Exception as e:
                print(f"  Warning: Failed to load sample {i}: {e}")
                continue

        if not batch_inputs:
            print("No valid samples to process")
            return []

        print(f"Processing {len(batch_inputs)} valid samples...")

        # Run inference
        if lora_request:
            outputs = self.llm.generate(
                batch_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_request,
            )
        else:
            outputs = self.llm.generate(
                batch_inputs,
                sampling_params=self.sampling_params,
            )

        # Parse outputs
        results = []
        for idx, output in zip(valid_indices, outputs):
            response = output.outputs[0].text
            prediction = self._parse_response(response, task_name)

            results.append({
                "sample_idx": idx,
                "prediction": prediction,
                "response": response,
            })

        print(f"Inference complete: {len(results)} predictions")
        return results

    def _build_chat_prompt(self, user_prompt: str) -> str:
        """Build chat-formatted prompt for Qwen3-VL."""
        return f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _parse_response(self, response: str, task_name: str) -> Any:
        """Parse model response based on task type."""
        response_lower = response.lower()

        if task_name in ("endpoint_error_identification", "merge_action", "merge_error_identification"):
            # These tasks use yes/no format
            match = re.search(r'<answer>\s*(yes|no)\s*</answer>', response_lower)
            if match:
                return match.group(1) == "yes"
            # Fallback: look for yes/no anywhere
            if "yes" in response_lower and "no" not in response_lower:
                return True
            return False

        elif task_name == "endpoint_localization":
            match = re.search(
                r'<answer>\s*x\s*=\s*([0-9.-]+)\s*,\s*y\s*=\s*([0-9.-]+)\s*,\s*z\s*=\s*([0-9.-]+)\s*</answer>',
                response_lower
            )
            if match:
                return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            return None

        elif task_name == "merge_action_multiple_choice":
            match = re.search(r'<answer>\s*(a|b|c|d|none)\s*</answer>', response_lower)
            if match:
                return match.group(1)
            return "none"

        return response


# =============================================================================
# Merged Model Inference (for vision encoder fine-tuned models)
# =============================================================================

@app.function(
    gpu="H100:2",
    timeout=7200,
    volumes={
        MODEL_DIR: model_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
        COMPILE_CACHE_DIR: compile_cache_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_merged_model_inference(
    model_path: str,
    dataset_path: str,
    task_name: str,
) -> List[Dict]:
    """Run inference with a fully merged model (no LoRA).

    Use this for models where the vision encoder was fine-tuned,
    since vLLM LoRA doesn't support vision encoder adapters.

    Args:
        model_path: Path to merged model on /checkpoints volume
        dataset_path: Path to parquet directory on /datasets volume
        task_name: Task name for prompt formatting

    Returns:
        List of dicts with 'sample_idx', 'prediction', 'response'
    """
    import os
    import re
    import torch
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    # Set cache directories
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    num_gpus = torch.cuda.device_count()

    print(f"\n{'='*60}")
    print(f"Running merged model inference: {task_name}")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Available GPUs: {num_gpus}")
    print(f"{'='*60}")

    # Resolve model path - support both checkpoint paths and HF model IDs
    if model_path.startswith("/") or (CHECKPOINT_DIR / model_path).exists():
        if model_path.startswith("/"):
            model_to_load = model_path
        else:
            model_to_load = str(CHECKPOINT_DIR / model_path)
        print(f"Loading merged model from checkpoint: {model_to_load}")
    else:
        # HuggingFace model ID
        model_to_load = model_path
        print(f"Loading HuggingFace model: {model_to_load}")

    # Get task config for num_images
    sys.path.insert(0, "/root")
    from task_configs import get_task

    # Strip group prefix if present (e.g., "vlm_all.merge_action" -> "merge_action")
    base_task_name = task_name.split(".")[-1] if "." in task_name else task_name
    task = get_task(base_task_name)

    # Load model with vLLM (matching evaluate_classification.py config)
    llm_kwargs = {
        "model": model_to_load,
        "tensor_parallel_size": num_gpus,
        "dtype": "bfloat16",
        "max_model_len": 55000,
        "limit_mm_per_prompt": {"image": getattr(task, 'num_images', 12)},
        "mm_processor_kwargs": {
            "min_pixels": 512 * 512,
            "max_pixels": 512 * 512,
            "size": {"shortest_edge": 512, "longest_edge": 512}
        },
    }

    print(f"Loading vLLM with tensor_parallel_size={num_gpus}...")
    llm = LLM(**llm_kwargs)
    print("Model loaded successfully!")

    # Load processor for chat template formatting
    processor = AutoProcessor.from_pretrained(model_to_load)

    # Apply mm_processor_kwargs to processor for consistency
    mm_kwargs = llm_kwargs.get("mm_processor_kwargs", {})
    if mm_kwargs and hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        if "min_pixels" in mm_kwargs:
            ip.min_pixels = mm_kwargs["min_pixels"]
        if "max_pixels" in mm_kwargs:
            ip.max_pixels = mm_kwargs["max_pixels"]
        if "size" in mm_kwargs:
            ip.size = mm_kwargs["size"]

    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0,  # Deterministic
        repetition_penalty=1.15,
    )

    # Load dataset
    dataset_dir = Path(dataset_path)
    parquet_path = dataset_dir / "questions.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    samples = df.to_dict('records')
    print(f"Loaded {len(samples)} samples")

    # Debug: Print first sample's metadata structure
    if samples:
        first_sample = samples[0]
        print(f"\n{'='*60}")
        print("DEBUG: First sample structure")
        print(f"{'='*60}")
        print(f"Keys: {list(first_sample.keys())}")
        metadata = first_sample.get('metadata', {})
        print(f"Metadata type: {type(metadata)}")
        print(f"Metadata: {metadata}")
        if isinstance(metadata, dict):
            neighbor_meta = metadata.get('neighbor_meta', [])
            print(f"neighbor_meta type: {type(neighbor_meta)}")
            print(f"neighbor_meta: {neighbor_meta}")
        print(f"{'='*60}\n")

    # Set dataset dir for image loading
    task._dataset_dir = dataset_dir

    # Prepare batch inputs
    print(f"Preparing {len(samples)} samples for batch inference...")
    batch_inputs = []
    valid_indices = []

    for i, sample in enumerate(tqdm(samples, desc="Loading samples")):
        try:
            prompt_text = task.format_prompt(sample)
            sample['_base_path'] = dataset_dir
            images = task.get_images(sample)

            # Build messages for chat template (matching evaluate_classification.py)
            user_content = []
            for _ in images:
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": user_content}]

            # Apply chat template to get formatted prompt
            formatted_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            batch_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": images if len(images) > 1 else images[0]},
            })
            valid_indices.append(i)

        except Exception as e:
            print(f"  Warning: Failed to load sample {i}: {e}")
            continue

    if not batch_inputs:
        print("No valid samples to process")
        return []

    # Debug: Print first prompt
    print(f"\n{'='*60}")
    print("DEBUG: First formatted prompt")
    print(f"{'='*60}")
    print(batch_inputs[0]["prompt"])
    print(f"{'='*60}")
    print(f"Number of images: {len(batch_inputs[0]['multi_modal_data']['image']) if isinstance(batch_inputs[0]['multi_modal_data']['image'], list) else 1}")
    print(f"{'='*60}\n")

    # Run batch inference
    print(f"\nRunning vLLM batch inference on {len(batch_inputs)} samples...")
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    # Parse results
    print(f"\nProcessing {len(outputs)} outputs...")
    results = []

    for idx, (sample_idx, output) in enumerate(zip(valid_indices, outputs)):
        response = output.outputs[0].text.strip() if output.outputs else ""
        prediction = _parse_response_static(response, task_name)

        results.append({
            "sample_idx": sample_idx,
            "prediction": prediction,
            "response": response,
        })

        # Print first few samples for debugging
        if idx < 3:
            print(f"\nSample {sample_idx}:")
            print(f"  Response: {response[:300]}...")
            print(f"  Prediction: {prediction}")

    print(f"\n{'='*60}")
    print(f"Completed {len(results)} samples")
    print(f"{'='*60}")

    return results


def _parse_response_static(response: str, task_name: str) -> Any:
    """Static version of response parsing for use in function context."""
    import re

    response_lower = response.lower()

    # Binary yes/no tasks
    binary_tasks = (
        "endpoint_error_identification",
        "endpoint_error_identification_with_em",
        "merge_action",
        "merge_error_identification",
        "split_action",
    )

    if task_name in binary_tasks:
        match = re.search(r'<answer>\s*(yes|no)\s*</answer>', response_lower)
        if match:
            return match.group(1) == "yes"
        # Fallback: check for yes/no without tags
        if "yes" in response_lower and "no" not in response_lower:
            return True
        if "no" in response_lower and "yes" not in response_lower:
            return False
        return None

    elif task_name == "endpoint_localization":
        # Coordinate prediction - handles multiple formats:
        # 1. [x, y, z] - list format
        # 2. x=X,y=Y,z=Z - named format
        # 3. x, y, z - comma-separated

        # Try [x, y, z] format first
        match = re.search(r'<answer>\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?\s*</answer>', response)
        if match:
            return [float(match.group(1)), float(match.group(2)), float(match.group(3))]

        # Try x=X,y=Y,z=Z format
        match = re.search(r'<answer>\s*x\s*=\s*([\d.]+)\s*,?\s*y\s*=\s*([\d.]+)\s*,?\s*z\s*=\s*([\d.]+)\s*</answer>', response, re.IGNORECASE)
        if match:
            return [float(match.group(1)), float(match.group(2)), float(match.group(3))]

        return None

    elif task_name == "merge_action_multiple_choice":
        # Multiple choice a/b/c/d/none
        match = re.search(r'<answer>\s*([a-d]|none)\s*</answer>', response_lower)
        if match:
            return match.group(1)
        return None

    elif task_name == "split_proposal":
        # Split points: sources=[(x,y,z),...];sinks=[(x,y,z),...]
        match = re.search(
            r'<answer>\s*sources\s*=\s*\[(.*?)\]\s*;\s*sinks\s*=\s*\[(.*?)\]\s*</answer>',
            response, re.IGNORECASE | re.DOTALL
        )
        if not match:
            return None

        def parse_coord_list(text: str) -> list:
            coords = []
            for coord_match in re.finditer(r'\(\s*([\d.-]+)\s*,\s*([\d.-]+)\s*,\s*([\d.-]+)\s*\)', text):
                coords.append([
                    float(coord_match.group(1)),
                    float(coord_match.group(2)),
                    float(coord_match.group(3))
                ])
            return coords

        sources = parse_coord_list(match.group(1))
        sinks = parse_coord_list(match.group(2))

        if not sources or not sinks:
            return None

        return (sources, sinks)

    return response


# =============================================================================
# SigLIP Probe Inference
# =============================================================================

@app.function(
    gpu="A10G",  # Small GPU - SigLIP-2 is only ~400M params
    timeout=3600,
    volumes={
        MODEL_DIR: model_volume,
        RESULTS_DIR: results_volume,
        DATASET_DIR: dataset_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_siglip_probe_inference(
    task_name: str,
    dataset_path: str,
    probe_weights_path: str,
    vision_model: str = "google/siglip2-so400m-patch16-512",
    pooling: str = "mean",
    batch_size: int = 32,
) -> List[Dict]:
    """Run inference using a SigLIP linear probe.

    Extracts features from the frozen SigLIP-2 vision encoder and applies
    a trained logistic regression probe for classification.

    Args:
        task_name: Task name for loading task config (for get_images, etc.)
        dataset_path: Path to parquet directory on /datasets volume
        probe_weights_path: Path to probe weights (.npz) relative to RESULTS_DIR
        vision_model: HuggingFace model ID for vision encoder
        pooling: Feature pooling method ("mean", "max", "cls")
        batch_size: Batch size for feature extraction

    Returns:
        List of dicts with 'sample_idx', 'prediction', 'response' (empty string)
    """
    import os
    import torch
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoModel, AutoProcessor

    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    print(f"\n{'='*60}")
    print(f"Running SigLIP probe inference: {task_name}")
    print(f"  Vision model: {vision_model}")
    print(f"  Probe weights: {probe_weights_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Pooling: {pooling}")
    print(f"{'='*60}")

    # Load probe weights
    probe_path = RESULTS_DIR / probe_weights_path
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe weights not found: {probe_path}")

    print(f"\nLoading probe weights from: {probe_path}")
    saved = np.load(str(probe_path), allow_pickle=True)
    coef = saved["coef"]
    intercept = saved["intercept"]
    classes = saved["classes"]
    print(f"  Coef shape: {coef.shape}")
    print(f"  Intercept: {intercept}")
    print(f"  Classes: {classes}")

    # Load vision encoder
    print(f"\nLoading vision encoder: {vision_model}...")
    model = AutoModel.from_pretrained(
        vision_model,
        torch_dtype=torch.float16,
        cache_dir=str(MODEL_DIR),
    ).cuda()
    processor = AutoProcessor.from_pretrained(vision_model, cache_dir=str(MODEL_DIR))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {num_params:.0f}M params")

    # Load task config for image handling
    try:
        from task_configs import get_task
    except ImportError:
        from environment.task_configs import get_task

    # Strip group prefix if present (e.g., "siglip_all.merge_action_siglip" -> "merge_action_siglip")
    base_task_name = task_name.split(".")[-1] if "." in task_name else task_name
    # Strip _siglip suffix if present (e.g., "merge_action_siglip" -> "merge_action")
    base_task_name = base_task_name.split("_siglip")[0]
    task = get_task(base_task_name)

    # Load dataset
    dataset_dir = Path(dataset_path)
    parquet_path = dataset_dir / "questions.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    samples = df.to_dict('records')
    print(f"Loaded {len(samples)} samples")

    task._dataset_dir = dataset_dir

    # Extract features and predict
    results = []
    for i in range(0, len(samples), batch_size):
        batch_end = min(i + batch_size, len(samples))
        print(f"  Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}...")

        batch_features = []
        batch_indices = []

        for j in range(i, batch_end):
            sample = samples[j]
            sample['_base_path'] = dataset_dir

            try:
                images = task.get_images(sample)
                # Create grid if multiple images
                if len(images) > 1:
                    from task_configs import create_image_grid
                    # Determine grid layout
                    n = len(images)
                    if n == 2:
                        rows, cols = 1, 2
                    elif n == 3:
                        rows, cols = 1, 3
                    elif n == 4:
                        rows, cols = 2, 2
                    elif n == 6:
                        rows, cols = 2, 3
                    else:
                        import math
                        cols = math.ceil(math.sqrt(n))
                        rows = math.ceil(n / cols)
                    grid_image = create_image_grid(images, rows, cols)
                else:
                    grid_image = images[0]

                # Process image
                inputs = processor(images=grid_image, return_tensors="pt")
                inputs = {k: v.cuda().half() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.vision_model(**inputs)
                    hidden_states = outputs.last_hidden_state  # [1, num_patches, hidden_dim]

                    # Pool features
                    if pooling == "mean":
                        features = hidden_states.mean(dim=1)  # [1, hidden_dim]
                    elif pooling == "max":
                        features = hidden_states.max(dim=1).values
                    elif pooling == "cls":
                        features = hidden_states[:, 0, :]  # CLS token
                    else:
                        features = hidden_states.mean(dim=1)

                    batch_features.append(features.cpu().numpy().flatten())
                    batch_indices.append(j)

            except Exception as e:
                print(f"    Warning: Failed to process sample {j}: {e}")
                continue

        # Apply probe to batch
        if batch_features:
            X_batch = np.array(batch_features)
            # Logistic regression: logit = X @ coef.T + intercept
            logits = X_batch @ coef.T + intercept
            # For binary classification, predict class 1 if logit > 0
            if len(classes) == 2:
                predictions = (logits.flatten() > 0).astype(int)
                probs = 1 / (1 + np.exp(-logits.flatten()))  # Sigmoid
            else:
                predictions = np.argmax(logits, axis=1)
                probs = None

            for idx, pred, sample_idx in zip(range(len(batch_indices)), predictions, batch_indices):
                # Map back to original class labels
                pred_label = classes[pred]
                # Convert to boolean if classes are True/False
                if isinstance(pred_label, (np.bool_, bool)):
                    pred_label = bool(pred_label)
                elif pred_label in ["True", "False"]:
                    pred_label = pred_label == "True"

                results.append({
                    "sample_idx": sample_idx,
                    "prediction": pred_label,
                    "response": "",  # No text response for probe
                    "probability": float(probs[idx]) if probs is not None else None,
                })

    print(f"\nCompleted {len(results)} predictions")
    return results


# =============================================================================
# ResNet Inference
# =============================================================================

@app.function(
    gpu="A10G",  # Small GPU sufficient for ResNet
    timeout=3600,
    volumes={
        MODEL_DIR: model_volume,
        RESNET_CHECKPOINT_DIR: resnet_checkpoint_volume,
        DATASET_DIR: dataset_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_resnet_inference(
    task_name: str,
    dataset_path: str,
    checkpoint_path: str,
    batch_size: int = 32,
) -> List[Dict]:
    """Run inference using a fine-tuned ResNet model.

    Args:
        task_name: Task name for loading task config
        dataset_path: Path to parquet directory on /datasets volume
        checkpoint_path: Path to ResNet checkpoint directory relative to RESNET_CHECKPOINT_DIR
        batch_size: Batch size for inference

    Returns:
        List of dicts with 'sample_idx', 'prediction', 'response' (empty string)
    """
    import os
    import json
    import torch
    import torch.nn as nn
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from PIL import Image
    from tqdm import tqdm
    from torchvision import models, transforms

    os.environ["HF_HOME"] = str(MODEL_DIR)

    print(f"\n{'='*60}")
    print(f"Running ResNet inference: {task_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint_dir = RESNET_CHECKPOINT_DIR / checkpoint_path
    best_model_path = checkpoint_dir / "best_model.pt"
    final_model_path = checkpoint_dir / "final_model.pt"

    model_path = best_model_path if best_model_path.exists() else final_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {checkpoint_dir}")

    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(str(model_path), map_location=device)

    # Load config
    config_path = checkpoint_dir / "training_config.json"
    if config_path.exists():
        with open(str(config_path), "r") as f:
            config_dict = json.load(f)
        image_size = config_dict.get("image_size", 224)
        model_name = config_dict.get("model_name", "resnet50")
        grid_layout = config_dict.get("grid_layout", "auto")
    else:
        image_size = 224
        model_name = "resnet50"
        grid_layout = "auto"

    # Load class mapping
    if "label_to_idx" in checkpoint:
        label_to_idx = checkpoint["label_to_idx"]
        idx_to_label = checkpoint["idx_to_label"]
        num_classes = checkpoint["num_classes"]
    else:
        class_mapping_path = checkpoint_dir / "class_mapping.json"
        with open(str(class_mapping_path), "r") as f:
            mapping = json.load(f)
        label_to_idx = mapping["label_to_idx"]
        idx_to_label = mapping["idx_to_label"]
        num_classes = len(label_to_idx)

    # Parse idx_to_label (stored as strings)
    idx_to_label_parsed = {}
    for k, v in idx_to_label.items():
        if v in ["True", "False"]:
            idx_to_label_parsed[int(k)] = v == "True"
        else:
            idx_to_label_parsed[int(k)] = v

    print(f"  Classes: {num_classes}")
    print(f"  Image size: {image_size}")

    # Create model
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    elif model_name == "resnet101":
        model = models.resnet101(weights=None)
    else:
        model = models.resnet50(weights=None)

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load task config
    try:
        from task_configs import get_task
    except ImportError:
        from environment.task_configs import get_task

    # Strip group prefix if present (e.g., "resnet_all.merge_action_resnet" -> "merge_action_resnet")
    base_task_name = task_name.split(".")[-1] if "." in task_name else task_name
    # Strip _resnet suffix if present (e.g., "merge_action_resnet" -> "merge_action")
    base_task_name = base_task_name.split("_resnet")[0]
    task = get_task(base_task_name)

    # Load dataset
    dataset_dir = Path(dataset_path)
    parquet_path = dataset_dir / "questions.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    samples = df.to_dict('records')
    print(f"Loaded {len(samples)} samples")

    task._dataset_dir = dataset_dir

    # Helper to create image grid
    def create_grid(images, target_size=224, layout="auto"):
        import math
        n = len(images)
        if n == 1:
            img = images[0]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img.resize((target_size, target_size), Image.Resampling.LANCZOS)

        if layout == "auto":
            if n == 2:
                rows, cols = 1, 2
            elif n == 3:
                rows, cols = 1, 3
            elif n == 4:
                rows, cols = 2, 2
            elif n <= 6:
                rows, cols = 2, 3
            else:
                cols = math.ceil(math.sqrt(n))
                rows = math.ceil(n / cols)
        else:
            parts = layout.lower().split("x")
            rows, cols = int(parts[0]), int(parts[1])

        cell_w = target_size // cols
        cell_h = target_size // rows
        canvas = Image.new("RGB", (target_size, target_size), (128, 128, 128))

        for idx, img in enumerate(images):
            if idx >= rows * cols:
                break
            row, col = idx // cols, idx % cols
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(img, (col * cell_w, row * cell_h))

        return canvas

    # Run inference
    results = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch_end = min(i + batch_size, len(samples))
            print(f"  Processing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}...")

            batch_tensors = []
            batch_indices = []

            for j in range(i, batch_end):
                sample = samples[j]
                sample['_base_path'] = dataset_dir

                try:
                    images = task.get_images(sample)
                    grid_image = create_grid(images, image_size, grid_layout)
                    tensor = transform(grid_image)
                    batch_tensors.append(tensor)
                    batch_indices.append(j)
                except Exception as e:
                    print(f"    Warning: Failed to process sample {j}: {e}")
                    continue

            if batch_tensors:
                batch = torch.stack(batch_tensors).to(device)
                outputs = model(batch)
                _, predicted = outputs.max(1)

                # Softmax for probabilities
                probs = torch.softmax(outputs, dim=1)

                for idx, (pred, sample_idx) in enumerate(zip(predicted.cpu().numpy(), batch_indices)):
                    pred_label = idx_to_label_parsed[int(pred)]
                    prob = probs[idx, pred].item()

                    results.append({
                        "sample_idx": sample_idx,
                        "prediction": pred_label,
                        "response": "",
                        "probability": prob,
                    })

    print(f"\nCompleted {len(results)} predictions")
    return results


# =============================================================================
# Config-Based Merged Model Inference
# =============================================================================

@app.function(
    gpu="H100:2",
    timeout=7200,
    volumes={
        MODEL_DIR: model_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
        COMPILE_CACHE_DIR: compile_cache_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_task_inference(
    task_name: str,
    dataset_path: str,
    model_path: Optional[str] = None,
    answer_only: bool = False,
    num_samples: int = 1,
    temperature: float = 0.0,
) -> List[Dict]:
    """Run inference for a proofreading task using merged model.

    This is the main entry point for running inference on proofreading tasks.
    It loads the model configuration and runs inference with the specified settings.

    Args:
        task_name: One of the proofreading tasks (merge_action, split_action, etc.)
        dataset_path: Path to parquet directory on /datasets volume
        model_path: Override model path (if None, uses config)
        answer_only: Whether to use answer-only mode (no analysis)
        num_samples: Number of samples/votes for ensemble (temperature > 0 if > 1)
        temperature: Sampling temperature (0 = deterministic)

    Returns:
        List of dicts with 'sample_idx', 'prediction', 'response'
        If num_samples > 1, also includes 'all_predictions' and 'all_responses'
    """
    import os
    import torch
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    # Set cache directories
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    num_gpus = torch.cuda.device_count()

    # Load config if model_path not provided
    config = load_proofreading_config()
    task_config = None
    model_type = "vlm"  # Default

    if model_path is None:
        if task_name not in config:
            raise ValueError(f"Unknown task: {task_name}. Available: {list(config.keys())}")
        task_config = config[task_name]
        model_path = task_config.model_path
        model_type = task_config.model_type
        if model_path is None:
            raise ValueError(f"No model configured for task: {task_name}")
        # Use config values if not overridden
        if not answer_only:
            answer_only = task_config.answer_only
        if num_samples == 1:
            num_samples = task_config.num_samples
    else:
        # model_path provided directly - check if task exists in config to get model_type
        if task_name in config:
            task_config = config[task_name]
            model_type = task_config.model_type

    # Dispatch to appropriate inference function based on model_type
    if model_type == "siglip_probe":
        print(f"Dispatching to SigLIP probe inference for task: {task_name}")
        return run_siglip_probe_inference.remote(
            task_name=task_name,
            dataset_path=dataset_path,
            probe_weights_path=model_path,
            vision_model=task_config.vision_model if task_config else "google/siglip2-so400m-patch16-512",
            pooling=task_config.pooling if task_config else "mean",
        )

    if model_type == "resnet":
        print(f"Dispatching to ResNet inference for task: {task_name}")
        return run_resnet_inference.remote(
            task_name=task_name,
            dataset_path=dataset_path,
            checkpoint_path=model_path,
        )

    # Continue with VLM inference
    print(f"\n{'='*60}")
    print(f"Running task inference: {task_name}")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Answer only: {answer_only}")
    print(f"  Num samples: {num_samples}")
    print(f"  Temperature: {temperature}")
    print(f"  Available GPUs: {num_gpus}")
    print(f"{'='*60}")

    # Debug: List checkpoint directory contents
    print(f"\n[DEBUG] Listing contents of {CHECKPOINT_DIR}:")
    if CHECKPOINT_DIR.exists():
        for item in sorted(CHECKPOINT_DIR.iterdir())[:20]:
            print(f"  {item.name}")
        total = len(list(CHECKPOINT_DIR.iterdir()))
        if total > 20:
            print(f"  ... and {total - 20} more")
    else:
        print(f"  ERROR: {CHECKPOINT_DIR} does not exist!")

    # Check specific model path
    full_model_path = CHECKPOINT_DIR / model_path
    print(f"\n[DEBUG] Checking model path: {full_model_path}")
    print(f"  .exists() = {full_model_path.exists()}")
    if full_model_path.parent.exists():
        print(f"  Parent exists, contents:")
        for item in sorted(full_model_path.parent.iterdir())[:10]:
            print(f"    {item.name}")

    # Resolve model path
    if model_path.startswith("/") or (CHECKPOINT_DIR / model_path).exists():
        if model_path.startswith("/"):
            model_to_load = model_path
        else:
            model_to_load = str(CHECKPOINT_DIR / model_path)
        print(f"Loading merged model from checkpoint: {model_to_load}")
    else:
        model_to_load = model_path
        print(f"Loading HuggingFace model: {model_to_load}")

    # Get task config for prompt/image handling
    sys.path.insert(0, "/root")
    from task_configs import get_task

    # Strip group prefix if present (e.g., "vlm_all.merge_action" -> "merge_action")
    base_task_name = task_name.split(".")[-1] if "." in task_name else task_name
    task = get_task(base_task_name)

    # Load vLLM model
    llm_kwargs = {
        "model": model_to_load,
        "tensor_parallel_size": num_gpus,
        "dtype": "bfloat16",
        "max_model_len": 55000,
        "limit_mm_per_prompt": {"image": getattr(task, 'num_images', 12)},
        "mm_processor_kwargs": {
            "min_pixels": 512 * 512,
            "max_pixels": 512 * 512,
            "size": {"shortest_edge": 512, "longest_edge": 512}
        },
    }

    print(f"Loading vLLM with tensor_parallel_size={num_gpus}...")
    llm = LLM(**llm_kwargs)
    print("Model loaded successfully!")

    # Load processor for chat template
    processor = AutoProcessor.from_pretrained(model_to_load)

    # Apply mm_processor_kwargs
    mm_kwargs = llm_kwargs.get("mm_processor_kwargs", {})
    if mm_kwargs and hasattr(processor, 'image_processor'):
        ip = processor.image_processor
        if "min_pixels" in mm_kwargs:
            ip.min_pixels = mm_kwargs["min_pixels"]
        if "max_pixels" in mm_kwargs:
            ip.max_pixels = mm_kwargs["max_pixels"]
        if "size" in mm_kwargs:
            ip.size = mm_kwargs["size"]

    # Sampling parameters
    # Use temperature > 0 if num_samples > 1 for diversity
    actual_temp = temperature if num_samples == 1 else max(temperature, 0.7)
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=actual_temp,
        repetition_penalty=1.15,
    )

    # Load dataset
    dataset_dir = Path(dataset_path)
    parquet_path = dataset_dir / "questions.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    samples = df.to_dict('records')
    print(f"Loaded {len(samples)} samples")

    task._dataset_dir = dataset_dir

    # Prepare batch inputs
    print(f"Preparing samples for batch inference...")
    batch_inputs = []
    valid_indices = []

    for i, sample in enumerate(tqdm(samples, desc="Loading samples")):
        try:
            prompt_text = task.format_prompt(sample, answer_only=answer_only)
            sample['_base_path'] = dataset_dir
            images = task.get_images(sample)

            # Build messages for chat template
            user_content = []
            for _ in images:
                user_content.append({"type": "image"})
            user_content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": user_content}]

            formatted_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            batch_inputs.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": images if len(images) > 1 else images[0]},
            })
            valid_indices.append(i)

        except Exception as e:
            print(f"  Warning: Failed to load sample {i}: {e}")
            continue

    if not batch_inputs:
        print("No valid samples to process")
        return []

    # Run inference (multiple times if num_samples > 1)
    all_results = []

    for sample_run in range(num_samples):
        if num_samples > 1:
            print(f"\nRunning sample {sample_run + 1}/{num_samples}...")

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        run_results = []
        for idx, (sample_idx, output) in enumerate(zip(valid_indices, outputs)):
            response = output.outputs[0].text.strip() if output.outputs else ""
            prediction = _parse_response_static(response, task_name)
            run_results.append({
                "sample_idx": sample_idx,
                "prediction": prediction,
                "response": response,
            })

        all_results.append(run_results)

    # Aggregate results
    if num_samples == 1:
        results = all_results[0]
    else:
        # Majority voting for ensemble
        results = []
        for i in range(len(all_results[0])):
            sample_idx = all_results[0][i]["sample_idx"]
            all_preds = [run[i]["prediction"] for run in all_results]
            all_resps = [run[i]["response"] for run in all_results]

            # Majority vote (for boolean predictions)
            if all(isinstance(p, bool) for p in all_preds):
                final_pred = sum(all_preds) > len(all_preds) / 2
            else:
                # For non-boolean, take most common
                from collections import Counter
                final_pred = Counter(all_preds).most_common(1)[0][0]

            results.append({
                "sample_idx": sample_idx,
                "prediction": final_pred,
                "response": all_resps[0],  # First response as representative
                "all_predictions": all_preds,
                "all_responses": all_resps,
                "vote_count": sum(1 for p in all_preds if p == final_pred),
            })

    print(f"\n{'='*60}")
    print(f"Completed {len(results)} samples")
    print(f"{'='*60}")

    return results


# =============================================================================
# Volume Upload Utilities
# =============================================================================

@app.function(volumes={DATASET_DIR: dataset_volume}, timeout=1800)
def _write_files_to_volume(file_data: List[Tuple[str, bytes]]) -> int:
    """Write files to the dataset volume.

    Args:
        file_data: List of (remote_path, file_bytes) tuples

    Returns:
        Number of files written
    """
    from pathlib import Path

    count = 0
    for remote_path, data in file_data:
        path = Path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)
        count += 1

    dataset_volume.commit()
    return count


def _do_upload_batches(files_to_upload, volume_path, batch_size, max_batch_bytes):
    """Internal function to upload batches - must be called within app.run() context."""
    total_uploaded = 0
    batch = []
    batch_bytes = 0

    for local_path, remote_path, file_size in files_to_upload:
        # Check if adding this file would exceed limits
        if batch and (len(batch) >= batch_size or batch_bytes + file_size > max_batch_bytes):
            # Upload current batch
            count = _write_files_to_volume.remote(batch)
            total_uploaded += count
            print(f"  Uploaded {total_uploaded}/{len(files_to_upload)} files")
            batch = []
            batch_bytes = 0

        # Read file and add to batch
        with open(local_path, 'rb') as f:
            file_bytes = f.read()
        batch.append((remote_path, file_bytes))
        batch_bytes += file_size

    # Upload final batch
    if batch:
        count = _write_files_to_volume.remote(batch)
        total_uploaded += count
        print(f"  Uploaded {total_uploaded}/{len(files_to_upload)} files")

    return total_uploaded


def upload_directory_to_volume(
    local_dir: Path,
    volume_path: str,
    batch_size: int = 50,
    max_batch_bytes: int = 50 * 1024 * 1024,  # 50MB per batch
) -> int:
    """Upload a local directory to the Modal dataset volume.

    Args:
        local_dir: Local directory to upload
        volume_path: Destination path on volume (e.g., "/datasets/eval_123")
        batch_size: Max files per upload batch
        max_batch_bytes: Max bytes per upload batch

    Returns:
        Total number of files uploaded
    """
    import os

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise ValueError(f"Local directory does not exist: {local_dir}")

    # Collect all files with their data
    files_to_upload = []
    total_bytes = 0

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = Path(root) / file
            rel_path = local_path.relative_to(local_dir)
            remote_path = f"{volume_path}/{rel_path}"

            file_size = local_path.stat().st_size
            total_bytes += file_size
            files_to_upload.append((local_path, remote_path, file_size))

    print(f"Uploading {len(files_to_upload)} files ({total_bytes / 1024 / 1024:.1f} MB) to {volume_path}")

    # Upload batches (already in Modal app context via local_entrypoint)
    total_uploaded = _do_upload_batches(files_to_upload, volume_path, batch_size, max_batch_bytes)

    print(f"Upload complete: {total_uploaded} files")
    return total_uploaded


@app.function(volumes={DATASET_DIR: dataset_volume}, timeout=300)
def list_volume_directory(volume_path: str) -> List[str]:
    """List contents of a directory on the dataset volume."""
    from pathlib import Path

    path = Path(volume_path)
    if not path.exists():
        return []

    files = []
    for item in path.rglob("*"):
        if item.is_file():
            files.append(str(item.relative_to(path)))

    return files


@app.function(volumes={RESULTS_DIR: results_volume}, timeout=300)
def save_results_to_volume(results: Dict, volume_path: str) -> str:
    """Save results JSON to the results volume."""
    from pathlib import Path

    path = Path(volume_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    results_volume.commit()
    return str(path)


@app.function(volumes={RESULTS_DIR: results_volume}, timeout=300)
def load_results_from_volume(volume_path: str) -> Optional[Dict]:
    """Load results JSON from the results volume."""
    from pathlib import Path

    path = Path(volume_path)
    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)


# =============================================================================
# CLI Entrypoints
# =============================================================================

@app.local_entrypoint()
def upload_dataset(
    local_dir: str,
    volume_path: str,
):
    """Upload a local dataset directory to the Modal volume.

    Usage:
        modal run scripts/analysis/modal_proofreading_inference.py::upload_dataset \
            --local-dir "evaluation_results/batch_123/candidates" \
            --volume-path "/datasets/eval_batch_123/candidates"
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"Error: Directory not found: {local_dir}")
        sys.exit(1)

    count = upload_directory_to_volume(local_path, volume_path)
    print(f"\nUploaded {count} files to {volume_path}")


@app.local_entrypoint()
def run_inference(
    dataset_path: str,
    task_name: str,
    adapter_path: str = None,
    output_path: str = None,
):
    """Run inference on a dataset.

    Usage:
        modal run scripts/analysis/modal_proofreading_inference.py::run_inference \
            --dataset-path "/datasets/eval_batch_123/candidates" \
            --task-name "endpoint_error_identification" \
            --adapter-path "my_adapter_checkpoint" \
            --output-path "results/identification_results.json"
    """
    # Create inference instance and run
    inference = ProofreadingInference()

    if task_name == "endpoint_error_identification":
        results = inference.run_identification.remote(dataset_path, adapter_path)
    elif task_name == "merge_error_identification":
        results = inference.run_merge_error_identification.remote(dataset_path, adapter_path)
    elif task_name == "endpoint_localization":
        results = inference.run_localization.remote(dataset_path, adapter_path)
    elif task_name == "merge_action_multiple_choice":
        results = inference.run_correction.remote(dataset_path, adapter_path)
    else:
        print(f"Error: Unknown task name: {task_name}")
        sys.exit(1)

    print(f"\nReceived {len(results)} predictions")

    # Save results locally if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    # Print summary
    if task_name in ("endpoint_error_identification", "merge_error_identification"):
        positives = sum(1 for r in results if r['prediction'])
        print(f"Predicted errors: {positives}/{len(results)}")
    elif task_name == "endpoint_localization":
        valid = sum(1 for r in results if r['prediction'] is not None)
        print(f"Valid localizations: {valid}/{len(results)}")
    elif task_name == "merge_action_multiple_choice":
        choices = {}
        for r in results:
            p = r['prediction']
            choices[p] = choices.get(p, 0) + 1
        print(f"Choice distribution: {choices}")


@app.local_entrypoint()
def run_merged_inference(
    model_path: str,
    dataset_path: str,
    task_name: str,
    output_path: str = None,
):
    """Run inference with a fully merged model (no LoRA).

    Use this for models where the vision encoder was fine-tuned.

    Usage:
        modal run scripts/analysis/modal_proofreading_inference.py::run_merged_inference \\
            --model-path "endpoint_localization_merged_model" \\
            --dataset-path "/datasets/test_localization" \\
            --task-name "endpoint_localization" \\
            --output-path "results/localization_results.json"
    """
    results = run_merged_model_inference.remote(
        model_path=model_path,
        dataset_path=dataset_path,
        task_name=task_name,
    )

    print(f"\nReceived {len(results)} predictions")

    # Save results locally if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    # Print summary based on task
    if task_name == "endpoint_error_identification" or task_name == "merge_action":
        positives = sum(1 for r in results if r['prediction'])
        print(f"Predicted yes: {positives}/{len(results)}")
    elif task_name == "endpoint_localization":
        valid = sum(1 for r in results if r['prediction'] is not None)
        print(f"Valid localizations: {valid}/{len(results)}")


@app.local_entrypoint()
def run_proofreading_task(
    task_name: str,
    dataset_path: str,
    model_path: str = None,
    answer_only: bool = False,
    num_samples: int = 1,
    output_path: str = None,
):
    """Run proofreading task inference using config-based merged models.

    This is the recommended entrypoint for running proofreading inference.
    Uses configs/proofreading_models.json for model paths and settings.

    Tasks:
        - merge_action: Binary yes/no for merge partner selection
        - split_action: Binary yes/no for split action evaluation
        - split_proposal: Propose split points for merge errors
        - endpoint_error_identification_with_em: Identify split errors
        - merge_error_identification: Identify merge errors

    Usage:
        modal run scripts/analysis/modal_proofreading_inference.py::run_proofreading_task \\
            --task-name "merge_action" \\
            --dataset-path "/datasets/eval_123/correction"

        # With overrides:
        modal run scripts/analysis/modal_proofreading_inference.py::run_proofreading_task \\
            --task-name "merge_action" \\
            --dataset-path "/datasets/eval_123/correction" \\
            --model-path "my_merged_model" \\
            --answer-only \\
            --num-samples 5
    """
    results = run_task_inference.remote(
        task_name=task_name,
        dataset_path=dataset_path,
        model_path=model_path,
        answer_only=answer_only,
        num_samples=num_samples,
    )

    print(f"\nReceived {len(results)} predictions")

    # Save results locally if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    # Print summary
    binary_tasks = ("merge_action", "split_action", "endpoint_error_identification_with_em", "merge_error_identification")
    if task_name in binary_tasks:
        positives = sum(1 for r in results if r['prediction'])
        print(f"Predicted yes: {positives}/{len(results)}")
        if num_samples > 1:
            avg_votes = sum(r.get('vote_count', 1) for r in results) / len(results)
            print(f"Average vote count: {avg_votes:.1f}/{num_samples}")
    elif task_name == "split_proposal":
        valid = sum(1 for r in results if r['prediction'] is not None)
        print(f"Valid proposals: {valid}/{len(results)}")


@app.local_entrypoint()
def show_config():
    """Show the current proofreading model configuration.

    Usage:
        modal run scripts/analysis/modal_proofreading_inference.py::show_config
    """
    from src.inference.model_config import load_model_config, DEFAULT_CONFIG_PATH

    print(f"Loading config from: {DEFAULT_CONFIG_PATH}")
    config = load_model_config()
    print(config.summary())
    print(f"\nConfigured tasks: {config.configured_tasks()}")


@app.local_entrypoint()
def debug_checkpoints(model_path: str = None, filter_str: str = None):
    """List checkpoint volume contents for debugging.

    Usage:
        modal run scripts/proofreading/modal_proofreading_inference.py::debug_checkpoints
        modal run scripts/proofreading/modal_proofreading_inference.py::debug_checkpoints --model-path "some_model_name"
        modal run scripts/proofreading/modal_proofreading_inference.py::debug_checkpoints --filter-str "merged"
    """
    results = list_checkpoints.remote(model_path, filter_str)
    for line in results:
        print(line)


# =============================================================================
# Note: Orchestration has moved to proofreading_evaluator.py
# =============================================================================
# The run_all_stages and run_batch functions have been removed.
# Orchestration (generate data -> upload -> infer -> generate next stage data)
# now happens in proofreading_evaluator.py using the ModalBackend class.
#
# This file now focuses on:
# - Config-based merged model inference (run_task_inference, run_proofreading_task)
# - Volume upload utilities (upload_directory_to_volume)
# - Legacy LoRA-based inference (ProofreadingInference class)
# =============================================================================


if __name__ == "__main__":
    print("This script should be run via Modal CLI:")
    print()
    print("Recommended (config-based merged models):")
    print("  modal run scripts/analysis/modal_proofreading_inference.py::run_proofreading_task --help")
    print("  modal run scripts/analysis/modal_proofreading_inference.py::show_config")
    print()
    print("Utilities:")
    print("  modal run scripts/analysis/modal_proofreading_inference.py::upload_dataset --help")
    print()
    print("Legacy (LoRA adapters):")
    print("  modal run scripts/analysis/modal_proofreading_inference.py::run_inference --help")
    print()
    print("For full proofreading evaluation with orchestration, use:")
    print("  python scripts/analysis/proofreading_evaluator.py --backend modal --help")
