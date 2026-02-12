import modal
import sys
from pathlib import Path
from dataclasses import dataclass
import time

# Add src/ directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import task_configs (works both locally via environment.task_configs and in Modal via /root/task_configs.py)
try:
    from task_configs import get_task, list_tasks
except ImportError:
    from environment.task_configs import get_task, list_tasks

# Paths for storage
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/datasets")
CHECKPOINT_DIR = Path("/checkpoints")
RESULTS_DIR = Path("/results")

# Create volumes
model_volume = modal.Volume.from_name("qwen-finetune-models", create_if_missing=True)
dataset_volume = modal.Volume.from_name("qwen-finetune-datasets", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("qwen-finetune-checkpoints", create_if_missing=True)
results_volume = modal.Volume.from_name("qwen-finetune-results", create_if_missing=True)

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # Required for installing unsloth from GitHub
    # Install unsloth - simple installation as recommended by unsloth docs
    .pip_install(
        "unsloth",
        "unsloth_zoo",
    )
    # Then install additional dependencies
    .pip_install(
        "datasets",
        "pandas",
        "Pillow",
        "huggingface_hub[hf_transfer]",
        "wandb",
        "scikit-learn",  # For stratified train/val/test splitting
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Add task_configs.py to the container
    .add_local_file("src/environment/task_configs.py", remote_path="/root/task_configs.py")
)

app = modal.App("qwen-segment-finetune", image=image)

# Also make the app available for the evaluation script to import
__all__ = ["app", "image"]


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct"
    max_seq_length: int = 2048

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # None = auto-detect

    # LoRA component selection - controls which parts of the model get LoRA adapters
    # These map to Unsloth's FastVisionModel.get_peft_model() parameters
    finetune_vision_layers: bool = False   # Apply LoRA to vision encoder (ViT blocks)
    finetune_language_layers: bool = True  # Apply LoRA to language model layers
    finetune_attention_modules: bool = True  # Apply LoRA to attention layers (q/k/v/o proj)
    finetune_mlp_modules: bool = True      # Apply LoRA to MLP/FFN layers

    # Merger/projector settings - the bridge between vision encoder and language model
    # IMPORTANT: Without training the merger, gradients can't flow to vision encoder!
    finetune_merger: bool = True  # Apply LoRA to vision-language merger/projector
    merger_modules_to_save: bool = False  # If True, fully train merger (not LoRA) via modules_to_save

    # Quantization
    load_in_4bit: bool = False  # Use QLoRA for memory efficiency (default: False for full precision)

    # Backend selection
    use_unsloth: bool = True  # Use Unsloth for optimized training (default: True)

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    gpu_count: int = 2  # Number of GPUs to use
    warmup_steps: int = 5
    max_steps: int = -1  # -1 means use num_train_epochs

    # Optimization
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"  # Options: linear, cosine, reduce_lr_on_plateau, etc.

    # Reduce LR on plateau settings (only used when lr_scheduler_type="reduce_lr_on_plateau")
    reduce_lr_factor: float = 0.1  # Factor to multiply LR by when reducing (default: 0.1)
    reduce_lr_patience: int = 3  # Eval steps with no improvement before reducing LR (default: 10)
    reduce_lr_threshold: float = 0.0001  # Minimum change to qualify as improvement (default: 0.0001)
    reduce_lr_min: float = 0.0  # Minimum LR, won't reduce below this (default: 0)

    # Other settings
    logging_steps: int = 1
    save_steps: int = 100
    save_total_limit: int = None  # Keep all checkpoints to ensure best model is never deleted
    fp16: bool = False
    bf16: bool = True
    seed: int = 42

    # Dataset settings
    num_samples: int = None  # None = use all data
    train_split_ratio: float = 0.9  # Fraction for training
    val_split_ratio: float = 0.1   # Fraction for validation (ignored if val_samples is set)
    val_samples: int = None  # Absolute number of validation samples (takes precedence over val_split_ratio)
    test_split_ratio: float = 0.0  # Fraction for test (ignored if test_samples is set)
    test_samples: int = None  # Absolute number of test samples (takes precedence over test_split_ratio)

    # Evaluation settings
    eval_strategy: str = "steps"  # "steps", "epoch", or "no"
    eval_steps: int = 25  # How often to run evaluation (if strategy is "steps")
    load_best_model_at_end: bool = True  # Load best checkpoint at end
    metric_for_best_model: str = "eval_loss"  # Metric to use for best model selection

    # Early stopping settings
    early_stopping: bool = False  # Enable early stopping based on validation loss
    early_stopping_patience: int = 3  # Number of eval steps with no improvement before stopping
    early_stopping_threshold: float = 0.0  # Minimum improvement to count as improvement

    # W&B settings
    use_wandb: bool = False
    wandb_project: str = "qwen-finetune"  # Generic project name
    wandb_run_name: str = None

    # Resume settings
    resume_from_checkpoint: str = None  # Path to checkpoint folder to resume from
    resume_adapter_only: bool = False  # If True, load adapter weights but ignore trainer state (fresh training from step 0)
    wandb_resume_id: str = None  # W&B run ID to resume (takes precedence over auto-detection)

    # Task settings
    task_name: str = "segment_classification"  # Default task

    # Teacher distillation settings
    min_correct_responses: int = 1  # Minimum correct responses required per sample (filters samples with fewer)

    # Class balancing settings
    class_balance: bool = False  # Enable class balancing
    class_balance_method: str = "oversample"  # "oversample" (duplicate minority) or "undersample" (subsample majority)

    # Positive sample filtering (applied after split to preserve group-based splitting)
    positive_train_samples: int = None  # Exact number of positive samples for training (None = use all)
    positive_val_samples: int = None  # Exact number of positive samples for validation (None = use all)

    # Fine-grained class sampling (takes precedence over class_balance if set)
    positive_train_samples: int = None  # Exact number of positive training samples
    negative_train_samples: int = None  # Exact number of negative training samples
    positive_val_samples: int = None  # Exact number of positive validation samples
    negative_val_samples: int = None  # Exact number of negative validation samples

    # Rationale dropout settings
    rationale_dropout_prob: float = 0.0  # Probability of dropping <analysis> section (0.0-1.0, dynamic per-sample)


# =============================================================================
# Lazy Image Loading Dataset
# =============================================================================

class LazyImageDataset:
    """
    Dataset wrapper that loads images on-the-fly when items are accessed.

    This enables memory-efficient training with large datasets by only loading
    images for the current batch, rather than all images upfront.

    Works with any collator since images are loaded BEFORE collation.
    """

    def __init__(self, data: list, debug: bool = True):
        """
        Args:
            data: List of samples with image paths in messages
            debug: If True, log info about first few image loads
        """
        self.data = data
        self.debug = debug
        self._load_count = 0
        self._debug_limit = 5  # Only log first N loads

    def __len__(self):
        return len(self.data)

    def _load_image(self, path: str):
        """Load a single image from path."""
        from PIL import Image
        img = Image.open(path)
        # Convert to RGB if needed
        if img.mode not in ['RGB']:
            img = img.convert('RGB')
        return img

    def _replace_paths_with_images(self, sample: dict) -> dict:
        """Replace image paths with loaded PIL images in a sample."""
        if "messages" not in sample:
            return sample

        messages = sample["messages"]
        new_messages = []

        for msg in messages:
            new_msg = {"role": msg["role"]}

            if isinstance(msg["content"], list):
                new_content = []
                for item in msg["content"]:
                    if item.get("type") == "image" and "path" in item:
                        # Load image from path
                        img = self._load_image(item["path"])
                        new_content.append({"type": "image", "image": img})
                    elif item.get("type") == "image" and "image" in item:
                        # Already has PIL image
                        new_content.append(item)
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content
            else:
                new_msg["content"] = msg["content"]

            new_messages.append(new_msg)

        return {"messages": new_messages}

    def __getitem__(self, idx):
        """Load images when item is accessed."""
        sample = self.data[idx]
        result = self._replace_paths_with_images(sample)

        # Debug logging for first few loads
        if self.debug and self._load_count < self._debug_limit:
            self._load_count += 1
            # Count images in result
            num_images = 0
            image_sizes = []
            for msg in result.get("messages", []):
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image" and "image" in item:
                            img = item["image"]
                            num_images += 1
                            image_sizes.append(f"{img.size[0]}x{img.size[1]}")
            print(f"[LazyImageDataset] Sample {idx}: loaded {num_images} images, sizes: {image_sizes}", flush=True)

        return result


# =============================================================================
# VLM Fine-Tuner Classes
# =============================================================================

class VLMFineTuner:
    """Base class for VLM fine-tuning. Subclasses implement backend-specific logic."""

    def __init__(self, config: TrainingConfig, task, model_dir: Path):
        self.config = config
        self.task = task
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None  # Could be tokenizer or processor depending on backend

    def load_model(self, load_existing_adapter: Path = None):
        """Load base model and add LoRA adapters. Must be implemented by subclass."""
        raise NotImplementedError

    def create_collate_fn(self, use_teacher_responses: bool, teacher_response_column: str,
                          rationale_dropout_prob: float):
        """Create data collator for training. Must be implemented by subclass."""
        raise NotImplementedError

    def preprocess_dataset(self, dataset, use_teacher_responses: bool,
                           teacher_response_column: str, rationale_dropout_prob: float):
        """Optionally preprocess dataset before training. Default: no preprocessing."""
        return dataset

    def get_sft_config_kwargs(self, output_dir: Path, eval_dataset_exists: bool) -> dict:
        """Build SFTConfig kwargs. Shared logic with backend-specific overrides."""
        import torch

        config = self.config
        kwargs = {
            "output_dir": str(output_dir),

            # Training hyperparameters
            "num_train_epochs": config.num_train_epochs,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,

            # Optimization
            "optim": config.optim,
            "weight_decay": config.weight_decay,
            "lr_scheduler_type": config.lr_scheduler_type,

            # Precision
            "fp16": config.fp16 and not torch.cuda.is_bf16_supported(),
            "bf16": config.bf16 and torch.cuda.is_bf16_supported(),

            # Logging and saving
            "logging_steps": config.logging_steps,
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,

            # Evaluation
            "eval_strategy": config.eval_strategy if eval_dataset_exists else "no",
            "eval_steps": config.eval_steps if config.eval_strategy == "steps" else None,
            "per_device_eval_batch_size": config.per_device_train_batch_size,

            # Best model checkpointing
            "load_best_model_at_end": config.load_best_model_at_end if eval_dataset_exists else False,
            "metric_for_best_model": config.metric_for_best_model if eval_dataset_exists else None,
            "greater_is_better": False if config.metric_for_best_model == "eval_loss" else True,
            "save_strategy": config.eval_strategy if eval_dataset_exists else "steps",

            # Other
            "seed": config.seed,
            "report_to": "wandb" if config.use_wandb else "none",
        }

        # Add lr_scheduler_kwargs for reduce_lr_on_plateau
        if config.lr_scheduler_type == "reduce_lr_on_plateau":
            kwargs["lr_scheduler_kwargs"] = {
                "factor": config.reduce_lr_factor,
                "patience": config.reduce_lr_patience,
                "threshold": config.reduce_lr_threshold,
                "min_lr": config.reduce_lr_min,
            }

        return kwargs

    def save_model(self, output_path: Path, training_config_dict: dict, test_metadata: dict = None):
        """Save LoRA adapters and training config."""
        import json

        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

        # Save training config
        with open(str(output_path / "training_config.json"), "w") as f:
            json.dump(training_config_dict, f, indent=2)

        # Save test indices if provided
        if test_metadata is not None:
            with open(str(output_path / "test_indices.json"), "w") as f:
                json.dump(test_metadata, f, indent=2)
            print(f"  Test indices saved to: {output_path / 'test_indices.json'}")


class UnslothFineTuner(VLMFineTuner):
    """Unsloth-based fine-tuning with FastVisionModel."""

    def load_model(self, load_existing_adapter: Path = None):
        from unsloth import FastVisionModel

        print(f"\nLoading model: {self.config.model_name}")
        print(f"  Backend: Unsloth")
        print(f"  4-bit quantization: {self.config.load_in_4bit}")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=self.config.model_name,
            load_in_4bit=self.config.load_in_4bit,
            max_seq_length=self.config.max_seq_length,
            cache_dir=str(self.model_dir),
        )

        if load_existing_adapter:
            from peft import PeftModel
            print(f"\nLoading existing LoRA adapters from: {load_existing_adapter}")
            self.model = PeftModel.from_pretrained(
                self.model, str(load_existing_adapter), is_trainable=True
            )
            print(f"  Loaded adapter weights - continuing training from step 0")
        else:
            print("\nAdding fresh LoRA adapters (Unsloth)...")
            print(f"  LoRA components:")
            print(f"    finetune_vision_layers: {self.config.finetune_vision_layers}")
            print(f"    finetune_language_layers: {self.config.finetune_language_layers}")
            print(f"    finetune_attention_modules: {self.config.finetune_attention_modules}")
            print(f"    finetune_mlp_modules: {self.config.finetune_mlp_modules}")
            print(f"    finetune_merger: {self.config.finetune_merger}")
            print(f"    merger_modules_to_save: {self.config.merger_modules_to_save}")

            # Build target_modules list if we need to include merger
            # Unsloth's default regex doesn't match merger, so we need custom target_modules
            target_modules = self.config.lora_target_modules
            modules_to_save = None

            if self.config.finetune_merger:
                if self.config.merger_modules_to_save:
                    # Full fine-tune merger (not LoRA) - train all merger weights
                    modules_to_save = ["visual.merger"]
                    print(f"  Merger: full fine-tuning via modules_to_save")
                elif not self.config.finetune_vision_layers:
                    # Vision is OFF but merger is ON - use regex to target ONLY merger
                    # Can't use "linear_fc1/linear_fc2" as they'd match vision encoder too
                    # PEFT uses re.fullmatch() so regex must match the FULL module path
                    if target_modules is None:
                        # Use regex pattern to match full paths - merger and language model only
                        target_modules = r".*(merger\.linear_fc|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
                        print(f"  Merger: LoRA via regex (merger only, no vision)")
                else:
                    # Both vision and merger ON - use custom target_modules
                    # linear_fc1/linear_fc2 will match both vision MLP and merger
                    # NOTE: Do NOT add "merger" directly - PEFT can't LoRA compound modules
                    if target_modules is None:
                        # Build explicit target_modules that cover vision, language, AND merger
                        target_modules = [
                            # Vision encoder attention (Qwen3-VL uses combined qkv)
                            "qkv", "proj",
                            # Vision encoder MLP AND merger (both use these layer names)
                            "linear_fc1", "linear_fc2",
                            # Language model attention
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            # Language model MLP
                            "gate_proj", "up_proj", "down_proj",
                        ]
                        print(f"  Merger: LoRA via linear_fc1/linear_fc2 in target_modules")

            # Log final target_modules
            if target_modules:
                print(f"  target_modules: {target_modules}")

            # Build get_peft_model kwargs
            peft_kwargs = {
                "finetune_vision_layers": self.config.finetune_vision_layers,
                "finetune_language_layers": self.config.finetune_language_layers,
                "finetune_attention_modules": self.config.finetune_attention_modules,
                "finetune_mlp_modules": self.config.finetune_mlp_modules,
                "r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "bias": "none",
                "random_state": self.config.seed,
                "use_rslora": False,
                "loftq_config": None,
            }

            # Add optional parameters
            if target_modules is not None:
                peft_kwargs["target_modules"] = target_modules
            if modules_to_save is not None:
                peft_kwargs["modules_to_save"] = modules_to_save

            self.model = FastVisionModel.get_peft_model(self.model, **peft_kwargs)

    def preprocess_dataset(self, dataset, use_teacher_responses: bool,
                           teacher_response_column: str, rationale_dropout_prob: float):
        """Convert dataset to Unsloth VLM format with 'messages' column.

        Uses lazy image loading - stores paths instead of PIL images.
        Images are loaded on-the-fly by the collator during training.
        """
        from tqdm import tqdm
        task = self.task

        # Check if task supports lazy loading
        try:
            # Test if get_image_paths is implemented
            task.get_image_paths(dataset[0])
            use_lazy_images = True
            print("\nUsing lazy image loading (memory efficient)", flush=True)
        except NotImplementedError:
            use_lazy_images = False
            print("\nTask does not support lazy loading, loading all images upfront", flush=True)

        def convert_to_messages(example):
            """Convert a single example to Unsloth VLM messages format."""
            formatted = task.format_sample_for_training(
                example,
                use_teacher_response=use_teacher_responses,
                teacher_response_column=teacher_response_column,
                rationale_dropout_prob=rationale_dropout_prob,
                lazy_images=use_lazy_images
            )
            # Only include 'messages' - paths are embedded in the message content
            # Extra keys can confuse the trainer's collator selection
            return {"messages": formatted["messages"]}

        print(f"\nConverting dataset to Unsloth VLM format...", flush=True)

        # Convert to list first
        dataset_list = list(dataset)
        print(f"  Dataset size: {len(dataset_list)} samples", flush=True)

        if use_lazy_images:
            # No threading needed - just formatting text, no I/O
            converted_list = [convert_to_messages(ex) for ex in tqdm(
                dataset_list, desc="Preprocessing (lazy)"
            )]
            # Wrap in LazyImageDataset for on-the-fly image loading
            return LazyImageDataset(converted_list)
        else:
            # Use threading for image loading (I/O-bound)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=64) as executor:
                converted_list = list(tqdm(
                    executor.map(convert_to_messages, dataset_list),
                    total=len(dataset_list),
                    desc="Preprocessing (loading images)"
                ))
            return converted_list

    def create_collate_fn(self, use_teacher_responses: bool, teacher_response_column: str,
                          rationale_dropout_prob: float):
        """Return UnslothVisionDataCollator for proper VLM training.

        Images are loaded by LazyImageDataset before reaching the collator,
        so no wrapper is needed here.
        """
        from unsloth.trainer import UnslothVisionDataCollator
        from unsloth import FastVisionModel

        # Enable model for training
        FastVisionModel.for_training(self.model)

        # Qwen3-VL chat template delimiters for train_on_responses_only
        return UnslothVisionDataCollator(
            self.model,
            self.tokenizer,
            # Train only on assistant responses, not user prompts
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    def get_sft_config_kwargs(self, output_dir: Path, eval_dataset_exists: bool) -> dict:
        kwargs = super().get_sft_config_kwargs(output_dir, eval_dataset_exists)
        kwargs["max_length"] = self.config.max_seq_length

        # Required settings for Unsloth VLM training
        # These tell SFTTrainer to skip dataset preparation and let
        # UnslothVisionDataCollator handle everything
        kwargs["remove_unused_columns"] = False
        kwargs["dataset_text_field"] = ""
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

        return kwargs


class PEFTFineTuner(VLMFineTuner):
    """Standard transformers + PEFT fine-tuning."""

    def load_model(self, load_existing_adapter: Path = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"\nLoading model: {self.config.model_name}")
        print(f"  Backend: transformers + PEFT")
        print(f"  4-bit quantization: {self.config.load_in_4bit}")

        # Load processor
        self.tokenizer = AutoProcessor.from_pretrained(
            self.config.model_name,
            cache_dir=str(self.model_dir),
        )
        self.tokenizer.tokenizer.padding_side = 'left'
        if self.tokenizer.tokenizer.pad_token is None:
            self.tokenizer.tokenizer.pad_token = self.tokenizer.tokenizer.eos_token

        # Configure quantization
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto" if self.config.load_in_4bit else None,
            cache_dir=str(self.model_dir),
        )

        if not self.config.load_in_4bit:
            self.model = self.model.to("cuda")

        self.model.gradient_checkpointing_enable()

        # Add LoRA adapters
        if load_existing_adapter:
            from peft import PeftModel
            print(f"\nLoading existing LoRA adapters from: {load_existing_adapter}")
            self.model = PeftModel.from_pretrained(
                self.model, str(load_existing_adapter), is_trainable=True
            )
            print(f"  Loaded adapter weights - continuing training from step 0")
        else:
            print("\nAdding fresh LoRA adapters (PEFT)...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()

    def preprocess_dataset(self, dataset, use_teacher_responses: bool,
                           teacher_response_column: str, rationale_dropout_prob: float):
        """Preprocess dataset to tokenized format for PEFT backend."""
        task = self.task
        tokenizer = self.tokenizer

        def preprocess_example(example):
            formatted = task.format_sample_for_training(
                example,
                use_teacher_response=use_teacher_responses,
                teacher_response_column=teacher_response_column,
                rationale_dropout_prob=rationale_dropout_prob
            )

            text = tokenizer.apply_chat_template(
                formatted["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            images = task.get_images(example)
            images_rgb = [img.convert("RGB") for img in images]

            encoded = tokenizer(
                images=[images_rgb],
                text=[text],
                return_tensors="pt",
                padding=False,
            )

            result = {k: v[0].tolist() for k, v in encoded.items()}
            result["labels"] = result["input_ids"].copy()
            return result

        print("\nPreprocessing dataset for PEFT backend...")
        preprocessed = dataset.map(
            preprocess_example,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )
        print(f"  Preprocessed {len(preprocessed)} samples")
        return preprocessed

    def create_collate_fn(self, use_teacher_responses: bool, teacher_response_column: str,
                          rationale_dropout_prob: float):
        """Create collator for pre-tokenized PEFT data."""
        import torch
        tokenizer = self.tokenizer

        def collate_fn(examples):
            pad_token_id = tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, 'tokenizer') else tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = 0

            max_len = max(len(ex["input_ids"]) for ex in examples)

            batch = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

            for ex in examples:
                seq_len = len(ex["input_ids"])
                padding_len = max_len - seq_len

                batch["input_ids"].append(ex["input_ids"] + [pad_token_id] * padding_len)

                if "attention_mask" in ex:
                    batch["attention_mask"].append(ex["attention_mask"] + [0] * padding_len)
                else:
                    batch["attention_mask"].append([1] * seq_len + [0] * padding_len)

                batch["labels"].append(ex["labels"] + [-100] * padding_len)

            batch = {k: torch.tensor(v) for k, v in batch.items()}

            if "pixel_values" in examples[0]:
                batch["pixel_values"] = torch.stack([torch.tensor(ex["pixel_values"]) for ex in examples])

            if "image_grid_thw" in examples[0]:
                batch["image_grid_thw"] = torch.stack([torch.tensor(ex["image_grid_thw"]) for ex in examples])

            return batch

        return collate_fn

    def get_sft_config_kwargs(self, output_dir: Path, eval_dataset_exists: bool) -> dict:
        kwargs = super().get_sft_config_kwargs(output_dir, eval_dataset_exists)
        kwargs["max_length"] = None  # PEFT uses pre-tokenized data
        return kwargs


# =============================================================================
# Dataset Loading and Preparation Helpers
# =============================================================================

def load_teacher_dataset(
    augmented_dataset_path: str,
    teacher_model: str,
    min_correct_responses: int,
    dataset_dir: Path,
) -> "Dataset":
    """Load and process dataset with teacher responses."""
    import pandas as pd
    import re
    from datasets import Dataset

    # Resolve path
    if not augmented_dataset_path.startswith('/'):
        full_path = dataset_dir / augmented_dataset_path
    else:
        full_path = Path(augmented_dataset_path)

    print(f"\nLoading augmented dataset from: {full_path}")
    df = pd.read_parquet(full_path)
    print(f"  Loaded {len(df)} samples")

    # Detect teacher models from columns
    teacher_response_cols = [c for c in df.columns if c.startswith('teacher_response_')]
    model_gen_pattern = re.compile(r'^teacher_response_(.+?)(?:_gen(\d+))?$')

    model_generations = {}
    for col in teacher_response_cols:
        match = model_gen_pattern.match(col)
        if match:
            base_model = match.group(1)
            gen_idx = int(match.group(2)) if match.group(2) is not None else None
            full_suffix = col.replace('teacher_response_', '')

            if gen_idx is not None:
                actual_model = base_model
            else:
                actual_model = full_suffix

            if actual_model not in model_generations:
                model_generations[actual_model] = []
            model_generations[actual_model].append((full_suffix, gen_idx))

    available_models = list(model_generations.keys())

    if available_models:
        print(f"\n  Available teacher models:")
        for model_suffix in available_models:
            gen_list = model_generations[model_suffix]
            total_correct = 0
            for full_suffix, _ in gen_list:
                correct_col = f'teacher_correct_{full_suffix}'
                if correct_col in df.columns:
                    total_correct += (df[correct_col] == True).sum()
            num_gens = len(gen_list)
            gen_info = f" ({num_gens} generations)" if num_gens > 1 else ""
            print(f"    {model_suffix}{gen_info}: {total_correct} correct")

        # Select models to use
        if teacher_model:
            if teacher_model not in available_models:
                raise ValueError(f"Teacher model '{teacher_model}' not found. Available: {available_models}")
            models_to_use = [teacher_model]
            print(f"\n  Using single-teacher mode: {teacher_model}")
        else:
            models_to_use = available_models
            print(f"\n  Using multi-teacher mode")

        # Filter by minimum correct responses
        if min_correct_responses > 1:
            print(f"\n  Filtering samples with >= {min_correct_responses} correct responses...")
            correct_counts = []
            for _, row in df.iterrows():
                count = sum(
                    1 for model_suffix in models_to_use
                    for full_suffix, _ in model_generations[model_suffix]
                    if f'teacher_correct_{full_suffix}' in df.columns
                    and row.get(f'teacher_correct_{full_suffix}') == True
                )
                correct_counts.append(count)

            df['_correct_count'] = correct_counts
            before_count = len(df)
            df = df[df['_correct_count'] >= min_correct_responses].copy()
            df = df.drop(columns=['_correct_count'])
            print(f"    Filtered from {before_count} to {len(df)} samples")

        # Expand to individual correct responses
        expanded_rows = []
        for orig_idx, (df_idx, row) in enumerate(df.iterrows()):
            for model_suffix in models_to_use:
                for full_suffix, gen_idx in model_generations[model_suffix]:
                    correct_col = f'teacher_correct_{full_suffix}'
                    analysis_col = f'teacher_analysis_{full_suffix}'

                    if correct_col in df.columns and row.get(correct_col) == True:
                        if analysis_col in df.columns and pd.notna(row.get(analysis_col)):
                            new_row = row.copy()
                            new_row['teacher_analysis'] = row[analysis_col]
                            new_row['teacher_model_used'] = model_suffix
                            new_row['teacher_generation'] = gen_idx
                            new_row['_original_sample_idx'] = orig_idx
                            expanded_rows.append(new_row)

        df = pd.DataFrame(expanded_rows)
        print(f"  Expanded to {len(df)} training examples")
        print(f"  (from {df['_original_sample_idx'].nunique()} unique samples)")

    else:
        # Legacy format
        print(f"\n  Using legacy single-model format")
        if 'teacher_correct' in df.columns:
            df = df[df['teacher_correct'] == True].copy()
            print(f"  Filtered to {len(df)} correct samples")

    return Dataset.from_pandas(df)


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


def split_dataset(dataset, config: TrainingConfig, task, ground_truth_labels: list):
    """Perform stratified train/val/test split.

    If task.get_split_group_key() returns non-None values, splits are done at the
    group level to prevent data leakage (all samples in a group go to the same split).

    For tasks where group keys contain multiple IDs (e.g., segment identity with
    segment1_id and segment2_id), connected components are used to merge groups
    that share any ID.
    """
    from sklearn.model_selection import train_test_split
    from collections import Counter, defaultdict

    needs_val = config.val_samples is not None or config.val_split_ratio > 0
    needs_test = config.test_samples is not None or config.test_split_ratio > 0

    if not (needs_val or needs_test):
        print("\nNo train/val/test split")
        print(f"  Train samples: {len(dataset)}")
        all_indices = list(range(len(dataset)))
        return dataset, None, None, {'train_indices': all_indices, 'val_indices': [], 'test_indices': []}

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
        if config.test_samples is not None:
            # test_samples is in samples, convert to approximate group count
            test_group_count = max(1, int(config.test_samples / avg_samples_per_group))
            train_val_groups, test_groups = train_test_split(
                unique_groups,
                test_size=test_group_count,
                random_state=config.seed,
                shuffle=True
            )
        elif config.test_split_ratio > 0:
            train_val_groups, test_groups = train_test_split(
                unique_groups,
                test_size=config.test_split_ratio,
                random_state=config.seed,
                shuffle=True
            )
        else:
            train_val_groups = unique_groups

        val_groups = []
        if config.val_samples is not None:
            # val_samples is in samples, convert to approximate group count
            val_group_count = max(1, int(config.val_samples / avg_samples_per_group))
            train_groups, val_groups = train_test_split(
                train_val_groups,
                test_size=val_group_count,
                random_state=config.seed,
                shuffle=True
            )
        elif config.val_split_ratio > 0:
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
        has_original_idx = '_original_sample_idx' in dataset.column_names

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
        if config.test_samples is not None:
            train_val_orig_indices, test_orig_indices = train_test_split(
                original_indices,
                test_size=config.test_samples,
                random_state=config.seed,
                stratify=original_labels,
                shuffle=True
            )
            train_val_labels = [idx_to_label[idx] for idx in train_val_orig_indices]
        elif config.test_split_ratio > 0:
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
        if config.val_samples is not None:
            train_orig_indices, val_orig_indices = train_test_split(
                train_val_orig_indices,
                test_size=config.val_samples,
                random_state=config.seed,
                stratify=train_val_labels,
                shuffle=True
            )
        elif config.val_split_ratio > 0:
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

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(val_indices) if val_indices else None
    test_dataset = dataset.select(test_indices) if test_indices else None

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(eval_dataset) if eval_dataset else 0} samples")
    print(f"  Test: {len(test_dataset) if test_dataset else 0} samples")

    # Extract original parquet indices for reproducible splits
    has_parquet_idx = '_original_parquet_idx' in dataset.column_names
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

    return train_dataset, eval_dataset, test_dataset, split_indices


def apply_class_balancing(dataset, task, seed: int, method: str = "oversample"):
    """Apply class balancing via oversampling or undersampling.

    Uses task.get_balance_group() to determine how samples are grouped for balancing.
    This allows tasks to define custom grouping logic (e.g., merge_action_multiple_choice
    groups 'none' vs 'not_none' instead of individual answer letters).

    Args:
        dataset: The dataset to balance
        task: Task config with get_balance_group method
        seed: Random seed for reproducibility
        method: "oversample" (duplicate minority class) or "undersample" (subsample majority class)
    """
    from collections import Counter
    import random

    print(f"\nApplying class balancing via {method}...")

    # Use get_balance_group for grouping (defaults to get_ground_truth)
    labels = [task.get_balance_group(sample) for sample in dataset]
    class_counts = Counter(labels)

    print(f"  Original distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"    {label}: {count}")

    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    random.seed(seed)

    if method == "oversample":
        # Duplicate minority class samples to match majority
        max_count = max(class_counts.values())
        balanced_indices = []
        for label, indices in class_indices.items():
            if len(indices) < max_count:
                additional = random.choices(indices, k=max_count - len(indices))
                balanced_indices.extend(indices + additional)
            else:
                balanced_indices.extend(indices)
    elif method == "undersample":
        # Subsample majority class to match minority
        min_count = min(class_counts.values())
        balanced_indices = []
        for label, indices in class_indices.items():
            if len(indices) > min_count:
                sampled = random.sample(indices, min_count)
                balanced_indices.extend(sampled)
            else:
                balanced_indices.extend(indices)
    else:
        raise ValueError(f"Unknown class_balance_method: {method}. Use 'oversample' or 'undersample'.")

    random.shuffle(balanced_indices)
    balanced_dataset = dataset.select(balanced_indices)

    # Print balanced distribution
    balanced_labels = [task.get_balance_group(sample) for sample in balanced_dataset]
    balanced_counts = Counter(balanced_labels)
    print(f"  Balanced distribution:")
    for label, count in sorted(balanced_counts.items()):
        print(f"    {label}: {count}")
    print(f"  Total: {len(balanced_dataset)} samples")

    return balanced_dataset


def apply_positive_sample_filtering(dataset, task, target_positive_count: int, seed: int, split_name: str = "train"):
    """Filter dataset to have exactly target_positive_count positive samples.

    If task uses group-based splitting, only keeps negative samples that share a group
    with at least one selected positive sample. This prevents contamination from groups
    that have no positive representation.

    This is applied AFTER group-based splitting to preserve data leakage prevention.

    Args:
        dataset: The dataset to filter
        task: Task config with get_balance_group method
        target_positive_count: Desired number of positive samples
        seed: Random seed for reproducibility
        split_name: Name of split (for logging)

    Returns:
        Filtered dataset with target_positive_count positives and negatives from same groups
    """
    from collections import Counter
    import random

    print(f"\nFiltering {split_name} set to {target_positive_count} positive samples...")

    # Use get_balance_group to determine positive vs negative
    labels = [task.get_balance_group(sample) for sample in dataset]
    class_counts = Counter(labels)

    print(f"  Original distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"    {label}: {count}")

    # Get group keys for all samples (if task uses group-based splitting)
    sample_group_keys = [task.get_split_group_key(sample) for sample in dataset]
    use_group_filtering = any(k is not None for k in sample_group_keys)

    # Separate indices by class
    positive_indices = []
    negative_indices = []

    for idx, label in enumerate(labels):
        # Assume the "positive" class - this may need to be task-specific
        # For binary classification, typically True/1/"yes" is positive
        if label in [True, 1, "yes", "merge", "split"]:
            positive_indices.append(idx)
        else:
            negative_indices.append(idx)

    print(f"  Identified: {len(positive_indices)} positives, {len(negative_indices)} negatives")

    # Check if we have enough positives
    if len(positive_indices) < target_positive_count:
        print(f"  WARNING: Only {len(positive_indices)} positive samples available, requested {target_positive_count}")
        print(f"  Using all {len(positive_indices)} available positives")
        sampled_positive_indices = positive_indices
    else:
        # Sample target number of positives
        random.seed(seed)
        sampled_positive_indices = random.sample(positive_indices, target_positive_count)
        print(f"  Sampled {len(sampled_positive_indices)} positive samples")

    # Filter negatives to only those in groups with selected positives
    if use_group_filtering:
        # Get groups represented by selected positives
        positive_groups = set(sample_group_keys[idx] for idx in sampled_positive_indices)
        print(f"  Positive samples span {len(positive_groups)} unique groups")

        # Only keep negatives from these groups
        filtered_negative_indices = [
            idx for idx in negative_indices
            if sample_group_keys[idx] in positive_groups
        ]
        print(f"  Filtered negatives: {len(negative_indices)} -> {len(filtered_negative_indices)} (same groups as positives)")
        negative_indices = filtered_negative_indices
    else:
        print(f"  No group filtering (task doesn't use group-based splitting)")

    # Combine sampled positives with filtered negatives
    filtered_indices = sampled_positive_indices + negative_indices
    random.shuffle(filtered_indices)

    filtered_dataset = dataset.select(filtered_indices)

    # Print final distribution
    filtered_labels = [task.get_balance_group(sample) for sample in filtered_dataset]
    filtered_counts = Counter(filtered_labels)
    print(f"  Filtered distribution:")
    for label, count in sorted(filtered_counts.items()):
        print(f"    {label}: {count}")
    print(f"  Total: {len(filtered_dataset)} samples")

    return filtered_dataset


def prepare_test_metadata(test_dataset, task, config: TrainingConfig, split_indices: dict = None) -> dict:
    """Prepare metadata for test set including actual indices for reproducibility."""
    if split_indices is None:
        split_indices = {}

    # Base metadata
    metadata = {
        'num_test_samples': len(test_dataset) if test_dataset else 0,
        'test_split_ratio': config.test_split_ratio,
        'test_samples': config.test_samples,
        'seed': config.seed,
    }

    # Add dataset hash for integrity verification
    dataset_hash = task.get_dataset_hash()
    if dataset_hash:
        metadata['dataset_hash'] = dataset_hash

    # Add split indices (for use with evaluate_classification.py)
    if split_indices:
        metadata['train_indices'] = split_indices.get('train_indices', [])
        metadata['val_indices'] = split_indices.get('val_indices', [])
        metadata['test_indices'] = split_indices.get('test_indices', [])

    # Also include sample identifiers for backwards compatibility and debugging
    if test_dataset is not None and len(test_dataset) > 0:
        test_identifiers = []
        for idx, sample in enumerate(test_dataset):
            # Use task's get_sample_id() method to get unique identifier
            test_id = task.get_sample_id(sample, index=idx)
            # Add ground truth for validation on resume
            test_id['ground_truth'] = task.get_ground_truth(sample)
            # Add original parquet index if available
            if '_original_parquet_idx' in sample:
                test_id['_original_parquet_idx'] = sample['_original_parquet_idx']
            # Convert any numpy arrays to lists for JSON serialization
            for key, value in test_id.items():
                if hasattr(value, 'tolist'):
                    test_id[key] = value.tolist()
            test_identifiers.append(test_id)
        metadata['test_identifiers'] = test_identifiers

    return metadata


# =============================================================================
# Main Fine-tuning Function
# =============================================================================

@app.function(
    gpu="H100:2",  # ⚠️ MUST match config.gpu_count parameter
    timeout=3600*24,  # 2 hours
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
def finetune_qwen(
    config: TrainingConfig = TrainingConfig(),
    augmented_dataset_path: str = None,
    use_teacher_responses: bool = False,
    teacher_response_column: str = 'teacher_analysis',
    teacher_model_name: str = None,
    teacher_model: str = None,
):
    """
    Fine-tune Qwen-VL model on a configured task.

    Args:
        config: TrainingConfig object with all training parameters
        augmented_dataset_path: Optional path to parquet file with teacher responses
        use_teacher_responses: If True, use teacher model responses from augmented dataset
        teacher_response_column: Column name containing teacher model's responses
        teacher_model_name: Optional name of teacher model for tracking
        teacher_model: Optional model suffix to filter to (e.g., 'o4_mini', 'claude_37_sonnet').
                      If None, uses all available teacher models (multi-teacher mode).
    """
    import json
    import torch
    import os
    from collections import Counter
    from datetime import datetime

    from unsloth import FastVisionModel  # noqa: F401

    from trl import SFTTrainer, SFTConfig

    # =========================================================================
    # 1. Setup and Validation
    # =========================================================================
    num_gpus = torch.cuda.device_count()
    assert num_gpus == config.gpu_count, (
        f"GPU mismatch! Found {num_gpus} but config expects {config.gpu_count}. "
        f"Update @app.function decorator to match."
    )

    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Load task
    task = get_task(config.task_name)

    # Print training info
    print("=" * 60)
    print(f"Task: {task.name} - {task.description}")
    print(f"Backend: {'Unsloth' if config.use_unsloth else 'transformers + PEFT'}")
    if use_teacher_responses:
        print(f"Mode: Teacher Distillation" + (f" ({teacher_model_name})" if teacher_model_name else ""))
    else:
        print(f"Mode: Standard Fine-tuning")
    effective_batch = config.per_device_train_batch_size * config.gpu_count * config.gradient_accumulation_steps
    print(f"GPUs: {config.gpu_count}x H100, Effective batch size: {effective_batch}")
    print("=" * 60)

    # =========================================================================
    # 2. Initialize W&B (if enabled)
    # =========================================================================
    wandb = None
    if config.use_wandb:
        import wandb as wb
        wandb = wb

        wandb_run_id = config.wandb_resume_id
        # Only auto-detect W&B run ID for full checkpoint resume, not adapter-only
        if not wandb_run_id and config.resume_from_checkpoint and not config.resume_adapter_only:
            for search_path in [CHECKPOINT_DIR / config.resume_from_checkpoint,
                               (CHECKPOINT_DIR / config.resume_from_checkpoint).parent]:
                wandb_id_file = search_path / "wandb_run_id.txt"
                if wandb_id_file.exists():
                    wandb_run_id = wandb_id_file.read_text().strip()
                    print(f"Found W&B run ID to resume: {wandb_run_id}")
                    break

        if wandb_run_id:
            wandb.init(project=config.wandb_project, id=wandb_run_id, resume="must", config=config.__dict__)
        else:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"{config.model_name.split('/')[-1]}-finetune",
                config=config.__dict__
            )

        # Save run ID for future resumption
        model_short = config.model_name.split('/')[-1]
        subdir = f"{model_short}_{config.wandb_run_name}" if config.wandb_run_name else config.model_name.replace("/", "_")
        wandb_dir = CHECKPOINT_DIR / subdir
        wandb_dir.mkdir(parents=True, exist_ok=True)
        (wandb_dir / "wandb_run_id.txt").write_text(wandb.run.id)

    # =========================================================================
    # 3. Check Resume Path
    # =========================================================================
    load_existing_adapter = None
    resume_checkpoint_path = None

    if config.resume_from_checkpoint:
        resume_path = CHECKPOINT_DIR / config.resume_from_checkpoint
        has_trainer_state = (resume_path / "trainer_state.json").exists()
        has_adapter_config = (resume_path / "adapter_config.json").exists()

        if has_trainer_state and not config.resume_adapter_only:
            # Full checkpoint resume (optimizer, scheduler, step count)
            print(f"\nResuming from checkpoint: {resume_path}")
            resume_checkpoint_path = resume_path
        elif has_adapter_config:
            # Adapter-only load (fresh training from step 0)
            if config.resume_adapter_only and has_trainer_state:
                print(f"\nLoading adapter weights only (ignoring trainer state): {resume_path}")
            else:
                print(f"\nLoading adapter from completed run: {resume_path}")
            load_existing_adapter = resume_path
        else:
            raise FileNotFoundError(f"Invalid resume path: {resume_path}")

    # =========================================================================
    # 4. Create Fine-tuner (backend-specific)
    # =========================================================================
    if config.use_unsloth:
        finetuner = UnslothFineTuner(config, task, MODEL_DIR)
    else:
        finetuner = PEFTFineTuner(config, task, MODEL_DIR)

    finetuner.load_model(load_existing_adapter)

    # =========================================================================
    # 5. Load and Prepare Dataset
    # =========================================================================
    if augmented_dataset_path and use_teacher_responses:
        dataset = load_teacher_dataset(
            augmented_dataset_path, teacher_model, config.min_correct_responses, DATASET_DIR
        )
        teacher_response_column = 'teacher_analysis'
    else:
        print(f"\nLoading dataset for task: {task.name}...")
        dataset = task.load_dataset(cache_dir=str(DATASET_DIR))
        dataset = task.filter_dataset(dataset)
        print(f"  Loaded {len(dataset)} samples after filtering")

    # Limit samples if specified (shuffle first to avoid class imbalance from ordered datasets)
    if config.num_samples and config.num_samples < len(dataset):
        dataset = dataset.shuffle(seed=config.seed)
        dataset = dataset.select(range(config.num_samples))
        print(f"Limited to {config.num_samples} samples (shuffled)")

    # Analyze class distribution
    print("\nClass distribution:")
    ground_truth_labels = [task.get_balance_group(sample) for sample in dataset]
    for label, count in sorted(Counter(ground_truth_labels).items()):
        print(f"  {label}: {count} samples")

    # =========================================================================
    # 6. Split Dataset (or load saved split when resuming)
    # =========================================================================
    saved_split_loaded = False

    # When resuming, try to load saved split indices for reproducibility
    if config.resume_from_checkpoint:
        resume_base = CHECKPOINT_DIR / config.resume_from_checkpoint
        # Check both the checkpoint dir and its parent (run dir)
        for search_path in [resume_base, resume_base.parent]:
            saved_indices_path = search_path / "test_indices.json"
            if saved_indices_path.exists():
                print(f"\nLoading saved split indices from: {saved_indices_path}")
                with open(str(saved_indices_path), 'r') as f:
                    saved_metadata = json.load(f)

                # Validate dataset hash if available
                if 'dataset_hash' in saved_metadata:
                    saved_hash = saved_metadata['dataset_hash']
                    current_hash = task.get_dataset_hash()
                    if current_hash and saved_hash != current_hash:
                        print(f"  WARNING: Dataset hash mismatch!")
                        print(f"    Saved hash:   {saved_hash[:16]}...")
                        print(f"    Current hash: {current_hash[:16]}...")
                        print(f"  The parquet file has changed since training.")
                        print(f"  Falling back to recomputing split...")
                        break
                    elif current_hash:
                        print(f"  Dataset hash verified: {current_hash[:16]}...")

                # Check if we have the actual indices (new format)
                if 'train_indices' in saved_metadata and 'val_indices' in saved_metadata:
                    saved_train_indices = saved_metadata['train_indices']
                    saved_val_indices = saved_metadata['val_indices']
                    saved_test_indices = saved_metadata.get('test_indices', [])

                    # Check if dataset has _original_parquet_idx column
                    has_parquet_idx = '_original_parquet_idx' in dataset.column_names

                    if has_parquet_idx:
                        # Build mapping from parquet index to current dataset index
                        parquet_to_current = {
                            sample['_original_parquet_idx']: i
                            for i, sample in enumerate(dataset)
                        }

                        # Convert saved parquet indices to current dataset indices
                        train_indices = [parquet_to_current[idx] for idx in saved_train_indices if idx in parquet_to_current]
                        val_indices = [parquet_to_current[idx] for idx in saved_val_indices if idx in parquet_to_current]
                        test_indices = [parquet_to_current[idx] for idx in saved_test_indices if idx in parquet_to_current]

                        # Check if any indices were lost (filtered out)
                        if len(train_indices) != len(saved_train_indices):
                            print(f"  WARNING: {len(saved_train_indices) - len(train_indices)} train samples were filtered out")
                        if len(val_indices) != len(saved_val_indices):
                            print(f"  WARNING: {len(saved_val_indices) - len(val_indices)} val samples were filtered out")
                        if len(test_indices) != len(saved_test_indices):
                            print(f"  WARNING: {len(saved_test_indices) - len(test_indices)} test samples were filtered out")
                    else:
                        # Legacy mode: indices are direct dataset indices
                        train_indices = saved_train_indices
                        val_indices = saved_val_indices
                        test_indices = saved_test_indices

                        # Validate indices are within bounds
                        max_idx = max(
                            max(train_indices) if train_indices else -1,
                            max(val_indices) if val_indices else -1,
                            max(test_indices) if test_indices else -1
                        )
                        if max_idx >= len(dataset):
                            print(f"  WARNING: Saved indices out of bounds (max={max_idx}, dataset size={len(dataset)})")
                            print(f"  Falling back to recomputing split...")
                            break

                    # Validate test samples have expected ground truth (catches dataset changes)
                    if 'test_identifiers' in saved_metadata and test_indices:
                        test_identifiers = saved_metadata['test_identifiers']
                        if len(test_identifiers) == len(test_indices):
                            mismatches = 0
                            for idx, expected in zip(test_indices, test_identifiers):
                                actual_gt = task.get_ground_truth(dataset[idx])
                                expected_gt = expected.get('ground_truth')
                                if actual_gt != expected_gt:
                                    mismatches += 1
                            if mismatches > 0:
                                print(f"  WARNING: {mismatches}/{len(test_indices)} test samples have mismatched ground truth!")
                                print(f"  This suggests the dataset has changed since training.")
                                print(f"  Falling back to recomputing split...")
                                break
                            else:
                                print(f"  Validated: all {len(test_indices)} test samples have correct ground truth")

                    # Use saved indices (converted to current dataset indices)
                    train_dataset = dataset.select(train_indices)
                    eval_dataset = dataset.select(val_indices) if val_indices else None
                    test_dataset = dataset.select(test_indices) if test_indices else None

                    # Keep the original parquet indices for saving
                    split_indices = {
                        'train_indices': saved_train_indices,
                        'val_indices': saved_val_indices,
                        'test_indices': saved_test_indices,
                    }

                    print(f"  Loaded split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
                    saved_split_loaded = True
                    break
                else:
                    print(f"  WARNING: Saved file missing train_indices/val_indices (old format)")
                    print(f"  Falling back to recomputing split...")

    if not saved_split_loaded:
        train_dataset, eval_dataset, test_dataset, split_indices = split_dataset(
            dataset, config, task, ground_truth_labels
        )

    test_metadata = prepare_test_metadata(test_dataset, task, config, split_indices)

    # =========================================================================
    # 7. Apply Positive Sample Filtering (if specified)
    # =========================================================================
    # Apply BEFORE class balancing so we filter first, then optionally balance
    if config.positive_train_samples is not None:
        train_dataset = apply_positive_sample_filtering(
            train_dataset, task, config.positive_train_samples, config.seed, split_name="train"
        )

    if config.positive_val_samples is not None and eval_dataset is not None:
        eval_dataset = apply_positive_sample_filtering(
            eval_dataset, task, config.positive_val_samples, config.seed, split_name="val"
        )

    # =========================================================================
    # 8. Apply Class Balancing (if enabled)
    # =========================================================================
    if config.class_balance:
        train_dataset = apply_class_balancing(train_dataset, task, config.seed, config.class_balance_method)

    # =========================================================================
    # 9. Preprocess Dataset (PEFT only) and Create Collate Function
    # =========================================================================
    rationale_dropout_prob = config.rationale_dropout_prob

    # Print rationale dropout examples if enabled
    if rationale_dropout_prob > 0:
        print(f"\n{'='*60}")
        print(f"Rationale Dropout Examples (prob={rationale_dropout_prob:.1%})")
        print(f"{'='*60}")
        num_examples = min(5, len(train_dataset))
        dropped_count = 0
        kept_count = 0
        for i in range(num_examples):
            sample = train_dataset[i]
            formatted = task.format_sample_for_training(
                sample,
                use_teacher_response=use_teacher_responses,
                teacher_response_column=teacher_response_column,
                rationale_dropout_prob=rationale_dropout_prob
            )
            user_content = formatted["messages"][0]["content"]
            prompt_text = [c["text"] for c in user_content if c.get("type") == "text"][0]
            response_content = formatted["messages"][-1]["content"]
            # Extract text from list format [{"type": "text", "text": "..."}]
            response = response_content[0]["text"] if isinstance(response_content, list) else response_content
            has_analysis = "<analysis>" in response
            if has_analysis:
                kept_count += 1
            else:
                dropped_count += 1
            status = "KEPT" if has_analysis else "DROPPED"
            print(f"\nExample {i+1} - Analysis {status}:")
            print(f"  Prompt ending: ...{prompt_text}")
            print(f"  Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        print(f"\nSummary: {dropped_count}/{num_examples} dropped, {kept_count}/{num_examples} kept")
        print(f"{'='*60}\n")

    # Preprocess dataset (both backends now have preprocessing)
    # - Unsloth: converts to messages format with PIL images
    # - PEFT: tokenizes the data
    train_dataset = finetuner.preprocess_dataset(
        train_dataset, use_teacher_responses, teacher_response_column, rationale_dropout_prob
    )
    if eval_dataset is not None:
        eval_dataset = finetuner.preprocess_dataset(
            eval_dataset, use_teacher_responses, teacher_response_column, rationale_dropout_prob
        )

    # Create collate function
    collate_fn = finetuner.create_collate_fn(
        use_teacher_responses, teacher_response_column, rationale_dropout_prob
    )

    # =========================================================================
    # 9. Setup Training
    # =========================================================================
    print("\nSetting up training...")

    # Create checkpoint output directory
    model_short_name = config.model_name.split('/')[-1]
    if config.wandb_run_name:
        checkpoint_subdir = f"{model_short_name}_{config.wandb_run_name}"
    else:
        checkpoint_subdir = config.model_name.replace("/", "_")
    checkpoint_output_dir = CHECKPOINT_DIR / checkpoint_subdir

    # Save test_indices.json to run directory immediately (for resumption if training crashes)
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    test_indices_path = checkpoint_output_dir / "test_indices.json"
    with open(str(test_indices_path), "w") as f:
        json.dump(test_metadata, f, indent=2)
    print(f"  Test indices saved to: {test_indices_path}")

    # Build training config dict early - serialize entire dataclass for reproducibility
    # This is saved to each checkpoint so we can always trace what settings were used
    from dataclasses import asdict
    training_config_dict = asdict(config)
    training_config_dict["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_config_dict["use_teacher_responses"] = use_teacher_responses
    if augmented_dataset_path:
        training_config_dict["augmented_dataset_path"] = augmented_dataset_path
        training_config_dict["teacher_response_column"] = teacher_response_column
    if teacher_model_name:
        training_config_dict["teacher_model_name"] = teacher_model_name

    # Save training_config.json to run directory immediately
    training_config_path = checkpoint_output_dir / "training_config.json"
    with open(str(training_config_path), "w") as f:
        json.dump(training_config_dict, f, indent=2)
    print(f"  Training config saved to: {training_config_path}")
    checkpoint_volume.commit()

    # Get SFT config from finetuner
    sft_config_kwargs = finetuner.get_sft_config_kwargs(
        checkpoint_output_dir, eval_dataset is not None
    )

    # Print scheduler info if applicable
    if config.lr_scheduler_type == "reduce_lr_on_plateau":
        print(f"\nUsing reduce_lr_on_plateau scheduler:")
        print(f"  factor={config.reduce_lr_factor}, patience={config.reduce_lr_patience}")
        print(f"  threshold={config.reduce_lr_threshold}, min_lr={config.reduce_lr_min}")

    training_args = SFTConfig(**sft_config_kwargs)

    # Create trainer
    trainer = SFTTrainer(
        model=finetuner.model,
        processing_class=finetuner.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # Add early stopping callback if enabled
    if config.early_stopping and eval_dataset is not None:
        from transformers import EarlyStoppingCallback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
        trainer.add_callback(early_stopping_callback)
        print(f"\nEarly stopping enabled: patience={config.early_stopping_patience}, threshold={config.early_stopping_threshold}")

    # Add callback to save training_config.json to each checkpoint
    from transformers import TrainerCallback
    import shutil

    class SaveTrainingConfigCallback(TrainerCallback):
        """Saves training_config.json to each checkpoint directory."""

        def __init__(self, training_config_dict, checkpoint_volume):
            self.training_config_dict = training_config_dict
            self.checkpoint_volume = checkpoint_volume

        def on_save(self, args, state, control, **kwargs):
            # Find the checkpoint directory that was just saved
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint_dir.exists():
                config_path = checkpoint_dir / "training_config.json"
                with open(str(config_path), "w") as f:
                    json.dump(self.training_config_dict, f, indent=2)
                self.checkpoint_volume.commit()

    trainer.add_callback(SaveTrainingConfigCallback(training_config_dict, checkpoint_volume))

    # =========================================================================
    # 10. Train!
    # =========================================================================
    print("\n" + "="*60)
    if resume_checkpoint_path:
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
    elif load_existing_adapter:
        print(f"Continuing training from completed run: {load_existing_adapter}")
        print("  (Starting from step 0 with loaded adapter weights)")
    else:
        print("Starting fresh training...")
    print("="*60)

    trainer.train(resume_from_checkpoint=str(resume_checkpoint_path) if resume_checkpoint_path else None)

    # =========================================================================
    # 11. Save Model
    # =========================================================================
    print("\nSaving LoRA adapters...")

    # Create unique save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = config.model_name.split('/')[-1]

    run_name_parts = [
        f"{config.task_name}_finetune",
        model_short_name,
        timestamp,
        f"samples{config.num_samples if config.num_samples else 'all'}",
        f"epochs{config.num_train_epochs}",
        f"lr{config.learning_rate}",
        f"r{config.lora_r}",
    ]
    if config.wandb_run_name:
        run_name_parts.insert(2, config.wandb_run_name)

    final_model_path = CHECKPOINT_DIR / "_".join(run_name_parts)

    # Update timestamp in training_config_dict to reflect completion time
    training_config_dict["completion_timestamp"] = timestamp

    # Save using finetuner method (training_config_dict was created earlier)
    finetuner.save_model(final_model_path, training_config_dict, test_metadata)
    checkpoint_volume.commit()

    print("\n" + "="*60)
    print("Training completed!")
    print(f"LoRA adapters saved to: {final_model_path}")
    print(f"Size: ~50-200 MB (adapters only)")
    print(f"\nTo use for inference:")
    print(f"  1. Load base model: FastVisionModel.from_pretrained('{config.model_name}')")
    print(f"  2. Load adapters: FastVisionModel.load_adapter(model, '{final_model_path}')")
    print("="*60)

    if config.use_wandb:
        wandb.finish()

    return str(final_model_path)


@app.function(
    gpu="H100:2",
    timeout=3600,  # 1 hour
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def evaluate_adapter(
    adapter_path: str = None,
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    task_name: str = "segment_classification",
    num_samples: int = None,
    batch_size: int = 2,
    use_blank_images: bool = False,
    use_simple_prompt: bool = False,
    use_test_set_only: bool = False,
    test_indices_path: str = None,
    answer_only: bool = False,
    class_balance: bool = False,
):
    """
    Evaluate a fine-tuned LoRA adapter or base model on any configured task.

    Args:
        adapter_path: Path to the LoRA adapter folder (None = evaluate base model only)
        base_model: Base model name (used when adapter_path is None)
        task_name: Name of the task to evaluate on (e.g., "segment_classification", "split_action")
        num_samples: Number of samples to evaluate (None = all)
        batch_size: Batch size for inference
        use_blank_images: Use blank images for sanity check
        use_simple_prompt: Use simple prompt for sanity check
        use_test_set_only: If True, only evaluate on the held-out test set from training
        test_indices_path: Path to test_indices.json (None = auto-detect from adapter or use default)
        answer_only: If True, use answer-only prompt (no analysis instruction)
        class_balance: If True, balance evaluation samples across classes via oversampling
    """
    import torch
    import pandas as pd
    import numpy as np
    import json
    from datasets import load_dataset
    from PIL import Image
    from unsloth import FastVisionModel
    from peft import PeftModel

    # Load task configuration
    task = get_task(task_name)

    print("="*60)
    if adapter_path:
        print("Evaluating Fine-tuned LoRA Adapter")
    else:
        print("Evaluating Base Model (No Fine-tuning)")
    print(f"Task: {task.name} - {task.description}")
    if answer_only:
        print("Prompt mode: ANSWER ONLY (no analysis instruction)")
    print("="*60)

    # Set cache directories
    import os
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Determine which base model to use
    if adapter_path:
        # Load training config to get base model name
        full_adapter_path = CHECKPOINT_DIR / adapter_path
        config_path = full_adapter_path / "training_config.json"

        if config_path.exists():
            with open(str(config_path), "r") as f:
                training_config = json.load(f)
            base_model_name = training_config["model_name"]
            print(f"\nTraining config found:")
            print(f"  Base model: {base_model_name}")
            print(f"  Trained on: {training_config.get('num_samples', 'all')} samples")
            print(f"  Epochs: {training_config.get('num_train_epochs')}")
        else:
            print(f"\nWarning: No training_config.json found")
            base_model_name = "Qwen/Qwen3-VL-32B-Instruct"
    else:
        # Use provided base model
        base_model_name = base_model
        print(f"\nEvaluating base model: {base_model_name}")

    # Load base model
    print(f"\nLoading base model: {base_model_name}...")
    from transformers import AutoProcessor

    start_time = time.time()

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model_name,
        load_in_4bit=False,
        max_seq_length=2048,
    )

    # Also load the processor for proper image handling
    processor = AutoProcessor.from_pretrained(base_model_name)

    end_time = time.time()
    print(f"Time taken to load base model: {end_time - start_time} seconds")
    start_time = end_time
    

    # Load adapters if provided
    if adapter_path:
        print(f"Loading LoRA adapters from: {full_adapter_path}...")
        model = PeftModel.from_pretrained(model, str(full_adapter_path))

    FastVisionModel.for_inference(model)
    end_time = time.time()
    print(f"Time taken to load adapters: {end_time - start_time} seconds")
    # Load dataset using task configuration
    print(f"\nLoading dataset for task: {task.name}...")
    ds = task.load_dataset(cache_dir=str(DATASET_DIR))
    print(f"  Loaded {len(ds)} samples")

    # Apply task-specific filtering
    print(f"\nApplying task-specific filters...")
    ds = task.filter_dataset(ds)
    print(f"  After filtering: {len(ds)} samples")

    # If use_test_set_only, filter to only test set samples
    if use_test_set_only:
        # Determine where to look for test_indices.json
        if test_indices_path:
            # User explicitly provided a path
            test_indices_file = CHECKPOINT_DIR / test_indices_path
        elif adapter_path:
            # Look in the adapter folder
            test_indices_file = full_adapter_path / "test_indices.json"
        else:
            # Look in general CHECKPOINT_DIR
            test_indices_file = CHECKPOINT_DIR / "test_indices.json"

        if test_indices_file.exists():
            print(f"\nLoading test set indices from: {test_indices_file}")
            with open(str(test_indices_file), 'r') as f:
                test_metadata_loaded = json.load(f)

            test_identifiers = test_metadata_loaded['test_identifiers']
            print(f"  Test set has {len(test_identifiers)} samples")

            # Determine which fields to use for matching based on what's in test_identifiers
            # Use all fields except 'ground_truth' (which is the label, not an identifier)
            if test_identifiers:
                id_fields = [field for field in test_identifiers[0].keys() if field != 'ground_truth']
                print(f"  Using identifier fields: {id_fields}")
            else:
                raise ValueError("test_identifiers is empty")

            # Build a set of identifiers for fast lookup
            test_id_set = set()
            for item in test_identifiers:
                # Create a unique identifier from all available ID fields
                id_parts = tuple((field, item[field]) for field in id_fields if field in item)
                test_id_set.add(id_parts)

            # Filter dataset to only test samples
            def is_test_sample(sample):
                # Build identifier from sample using the same fields
                sample_id_parts = tuple((field, sample[field]) for field in id_fields if field in sample)
                return sample_id_parts in test_id_set

            ds = ds.filter(is_test_sample)
            print(f"  Filtered to {len(ds)} test samples")
        else:
            print(f"\nError: test_indices.json not found at {test_indices_file}")
            print("  Cannot proceed with --test-set-only without test indices.")
            print("  Either:")
            print("    1. Remove --test-set-only flag to evaluate on full dataset")
            print("    2. Provide --test-indices-path pointing to a valid test_indices.json file")
            raise FileNotFoundError(f"Test indices not found: {test_indices_file}")

    # Shuffle dataset to ensure balanced sampling (dataset may be ordered by label)
    ds = ds.shuffle(seed=42)

    # Apply class balancing via oversampling if enabled
    if class_balance:
        from collections import Counter
        import random
        random.seed(42)

        print("\nApplying class balancing via oversampling...")

        # Get class labels
        labels = [task.get_ground_truth(sample) for sample in ds]
        class_counts = Counter(labels)

        print(f"  Original class distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"    {label}: {count} samples")

        # Find majority class count
        max_class_count = max(class_counts.values())
        print(f"  Target count per class: {max_class_count} (majority class)")

        # Group indices by class
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Oversample minority classes
        oversampled_indices = []
        for label, indices in class_indices.items():
            current_count = len(indices)
            if current_count < max_class_count:
                num_to_add = max_class_count - current_count
                additional_indices = random.choices(indices, k=num_to_add)
                oversampled_indices.extend(indices + additional_indices)
            else:
                oversampled_indices.extend(indices)

        # Shuffle the oversampled indices
        random.shuffle(oversampled_indices)

        # Create new dataset with oversampled indices
        ds = ds.select(oversampled_indices)

        # Verify new distribution
        new_labels = [task.get_ground_truth(sample) for sample in ds]
        new_class_counts = Counter(new_labels)

        print(f"  Balanced class distribution:")
        for label, count in sorted(new_class_counts.items()):
            print(f"    {label}: {count} samples")
        print(f"  Total samples after balancing: {len(ds)}")

    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
        print(f"Evaluating on {num_samples} samples (shuffled)")
    else:
        print(f"Evaluating on all {len(ds)} samples")

    results = []

    # Process in batches for speed
    print(f"Processing with batch_size={batch_size}...")
    for batch_start in range(0, len(ds), batch_size):
        batch_end = min(batch_start + batch_size, len(ds))
        batch = ds.select(range(batch_start, batch_end))
        start_time = time.time()
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(ds) + batch_size - 1)//batch_size} (samples {batch_start + 1}-{batch_end})...")

        # Prepare batch data
        batch_texts = []
        batch_images = []
        batch_metadata = []

        for sample in batch:
            # Get ground truth and images using task methods
            ground_truth = task.get_ground_truth(sample)

            # Get images
            if use_blank_images:
                # Get number of images from task and create blank versions
                original_images = task.get_images(sample)
                images = [Image.new('RGB', (1024, 1024), color=(128, 128, 128)) for _ in original_images]
            else:
                images = task.get_images(sample)

            # Create prompt
            if use_simple_prompt:
                prompt = "What do you see in these images?"
            else:
                prompt = task.format_prompt(sample, answer_only=answer_only)

            # Prepare messages for this sample
            user_content = []
            for img in images:
                user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": prompt})

            messages = [{
                "role": "user",
                "content": user_content,
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)
            batch_images.extend(images)

            # Store metadata - include all sample fields for flexibility
            metadata = {
                'ground_truth': ground_truth,
            }
            # Add any ID fields that exist (for tracking)
            for id_field in ['proofread_root_id', 'current_root_id', 'species']:
                if id_field in sample:
                    metadata[id_field] = sample[id_field]

            batch_metadata.append(metadata)

        # Process batch through model
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        end_time = time.time()
        print(f"Time taken to generate answers: {end_time - start_time} seconds")

        # Process outputs
        for output_text, metadata in zip(output_texts, batch_metadata):
            # Extract answer from tags
            llm_answer = ""
            analysis = ""

            answer_start = output_text.find("<answer>")
            answer_end = output_text.find("</answer>")
            if answer_start != -1 and answer_end != -1:
                llm_answer = output_text[answer_start + len("<answer>"):answer_end].strip()
            print(f"LLM answer: {llm_answer}")

            analysis_start = output_text.find("<analysis>")
            analysis_end = output_text.find("</analysis>")
            if analysis_start != -1 and analysis_end != -1:
                analysis = output_text[analysis_start + len("<analysis>"):analysis_end].strip()
            elif answer_start != -1:
                analysis = output_text[:answer_start].strip()
            print(f"Analysis: {analysis}")

            # Map answer to description if task has CLASS_MAPPING
            predicted_description = None
            if hasattr(task, 'CLASS_MAPPING'):
                predicted_description = task.CLASS_MAPPING.get(llm_answer, None)
            else:
                # For tasks without mapping (e.g., yes/no), use answer directly
                predicted_description = llm_answer

            # Check if correct
            # For tasks with CLASS_MAPPING, compare descriptions; otherwise compare raw answers
            if hasattr(task, 'CLASS_MAPPING'):
                correct = (predicted_description == metadata['ground_truth']) if predicted_description and metadata['ground_truth'] else False
            else:
                # For split action (yes/no), convert to bool and compare
                if isinstance(metadata['ground_truth'], bool):
                    predicted_bool = (llm_answer.lower() == "yes")
                    correct = (predicted_bool == metadata['ground_truth'])
                else:
                    correct = (llm_answer == str(metadata['ground_truth']))

            results.append({
                **metadata,
                'llm_answer': llm_answer,
                'predicted_description': predicted_description,
                'correct': correct,
                'analysis': analysis,
                'full_response': output_text,
            })

            print(f"  Sample answer: {correct}")

    # Save results
    df = pd.DataFrame(results)

    if 'correct' in df.columns and df['correct'].notna().any():
        accuracy = df['correct'].mean()
        print(f"\n{'='*60}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Correct: {df['correct'].sum()}/{len(df)}")
        print(f"{'='*60}")

        print("\nPerformance by ground truth class:")
        for ground_truth in sorted(df['ground_truth'].unique()):
            subset = df[df['ground_truth'] == ground_truth]
            correct_count = subset['correct'].sum()
            total_count = len(subset)
            accuracy = correct_count / total_count if total_count > 0 else 0
            print(f"  {ground_truth}: {correct_count}/{total_count} correct ({accuracy:.1%})")

    # Save to file
    if adapter_path:
        adapter_name = adapter_path.replace("/", "_")
    else:
        adapter_name = base_model_name.replace("/", "_") + "_base"
    num_samples_str = f"{num_samples}samples" if num_samples else "all_samples"
    output_path = RESULTS_DIR / f"{adapter_name}_eval_{task_name}_{num_samples_str}.csv"
    df.to_csv(str(output_path), index=False)
    results_volume.commit()

    print(f"\nResults saved to: {output_path}")
    return df


@app.function(
    gpu="H100:2",
    timeout=3600 * 4,  # 4 hours for multiple checkpoints
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def evaluate_all_checkpoints(
    run_path: str,
    base_model: str = None,
    task_name: str = "merge_action",
    num_samples: int = 100,
    batch_size: int = 2,
    checkpoints: str = None,
    answer_only: bool = False,
    class_balance: bool = False,
    use_blank_images: bool = False,
    use_simple_prompt: bool = False,
):
    """
    Evaluate all checkpoints in a training run directory.

    Loads the base model once, then loops over all checkpoint-* directories
    and evaluates each one efficiently.

    Args:
        run_path: Path to training run directory (e.g., "Qwen3-VL-32B-Instruct_myrun")
                  Should contain checkpoint-* subdirectories
        base_model: Base model name (auto-detected from training_config.json if not provided)
        task_name: Task to evaluate on
        num_samples: Number of samples per checkpoint
        batch_size: Batch size for inference
        checkpoints: Filter checkpoints. Comma-separated steps (e.g., "100,200,300") or
                     range (e.g., "100-500"). If None, evaluates all checkpoints.
        answer_only: If True, use answer-only prompt (no analysis instruction)
        class_balance: If True, balance evaluation samples across classes via oversampling
        use_blank_images: If True, use blank gray images instead of real images (sanity check)
        use_simple_prompt: If True, use simple prompt "What do you see?" (sanity check)
    """
    import torch
    import pandas as pd
    import json
    import re
    import os
    import time
    from pathlib import Path
    from unsloth import FastVisionModel
    from peft import PeftModel
    from transformers import AutoProcessor

    print("=" * 60)
    print("Evaluating All Checkpoints")
    print("=" * 60)
    if answer_only:
        print("Prompt mode: ANSWER ONLY (no analysis instruction)")
    if use_blank_images:
        print("Image mode: BLANK IMAGES (sanity check)")
    if use_simple_prompt:
        print("Prompt mode: SIMPLE PROMPT (sanity check)")

    # Set cache directories
    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Load task configuration
    task = get_task(task_name)
    print(f"Task: {task.name} - {task.description}")

    # Find the run directory and checkpoints
    run_dir = CHECKPOINT_DIR / run_path
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Find all checkpoint directories
    checkpoint_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1])  # Sort by step number
    )

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint-* directories found in {run_dir}")

    print(f"Found {len(checkpoint_dirs)} checkpoints total")

    # Filter checkpoints if specified
    if checkpoints:
        if "-" in checkpoints and "," not in checkpoints:
            # Range format: "100-500"
            start, end = map(int, checkpoints.split("-"))
            checkpoint_dirs = [
                d for d in checkpoint_dirs
                if start <= int(d.name.split("-")[1]) <= end
            ]
        else:
            # Comma-separated: "100,200,300"
            selected_steps = set(int(s.strip()) for s in checkpoints.split(","))
            checkpoint_dirs = [
                d for d in checkpoint_dirs
                if int(d.name.split("-")[1]) in selected_steps
            ]
        print(f"Filtered to {len(checkpoint_dirs)} checkpoints: {checkpoints}")

    print(f"Evaluating checkpoints:")
    for cp in checkpoint_dirs:
        print(f"  - {cp.name}")

    # Determine base model from training config
    if base_model is None:
        config_path = run_dir / "training_config.json"
        if not config_path.exists():
            # Try first checkpoint
            config_path = checkpoint_dirs[0] / "training_config.json"

        if config_path.exists():
            with open(str(config_path), "r") as f:
                training_config = json.load(f)
            base_model = training_config["model_name"]
            print(f"\nAuto-detected base model: {base_model}")
        else:
            raise ValueError("Could not find training_config.json. Please specify --base-model")

    # Load base model ONCE
    print(f"\nLoading base model: {base_model}...")
    start_time = time.time()

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model,
        load_in_4bit=False,
        max_seq_length=2048,
        cache_dir=str(MODEL_DIR),
    )
    processor = AutoProcessor.from_pretrained(base_model, cache_dir=str(MODEL_DIR))

    print(f"Base model loaded in {time.time() - start_time:.1f}s")

    # Load dataset ONCE
    print(f"\nLoading dataset for task: {task.name}...")
    ds = task.load_dataset(cache_dir=str(DATASET_DIR))
    ds = task.filter_dataset(ds)

    # Shuffle dataset to ensure balanced sampling (dataset may be ordered by label)
    ds = ds.shuffle(seed=42)

    # Apply class balancing via oversampling if enabled
    if class_balance:
        from collections import Counter
        import random
        random.seed(42)

        print("\nApplying class balancing via oversampling...")

        # Get class labels
        labels = [task.get_ground_truth(sample) for sample in ds]
        class_counts = Counter(labels)

        print(f"  Original class distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"    {label}: {count} samples")

        # Find majority class count
        max_class_count = max(class_counts.values())
        print(f"  Target count per class: {max_class_count} (majority class)")

        # Group indices by class
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Oversample minority classes
        oversampled_indices = []
        for label, indices in class_indices.items():
            current_count = len(indices)
            if current_count < max_class_count:
                num_to_add = max_class_count - current_count
                additional_indices = random.choices(indices, k=num_to_add)
                oversampled_indices.extend(indices + additional_indices)
            else:
                oversampled_indices.extend(indices)

        # Shuffle the oversampled indices
        random.shuffle(oversampled_indices)

        # Create new dataset with oversampled indices
        ds = ds.select(oversampled_indices)

        # Verify new distribution
        new_labels = [task.get_ground_truth(sample) for sample in ds]
        new_class_counts = Counter(new_labels)

        print(f"  Balanced class distribution:")
        for label, count in sorted(new_class_counts.items()):
            print(f"    {label}: {count} samples")
        print(f"  Total samples after balancing: {len(ds)}")

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    print(f"Evaluating on {len(ds)} samples per checkpoint (shuffled)")

    # Prepare evaluation data once (images, prompts, ground truth)
    print("\nPreparing evaluation data...")
    eval_data = []
    from PIL import Image

    for sample in ds:
        ground_truth = task.get_ground_truth(sample)

        # Get images (use blank images if requested for sanity check)
        if use_blank_images:
            original_images = task.get_images(sample)
            images = [Image.new('RGB', (1024, 1024), color=(128, 128, 128)) for _ in original_images]
        else:
            images = task.get_images(sample)

        # Create prompt
        if use_simple_prompt:
            prompt = "What do you see in these images?"
        else:
            prompt = task.format_prompt(sample, answer_only=answer_only)

        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        eval_data.append({
            "text": text,
            "images": images,
            "ground_truth": ground_truth,
        })

    # Results summary
    all_results = []
    peft_model = None  # Will hold the PeftModel instance

    # Evaluate each checkpoint
    for cp_idx, checkpoint_dir in enumerate(checkpoint_dirs):
        checkpoint_name = checkpoint_dir.name
        step_num = int(checkpoint_name.split("-")[1])

        print(f"\n{'=' * 60}")
        print(f"Checkpoint {cp_idx + 1}/{len(checkpoint_dirs)}: {checkpoint_name} (step {step_num})")
        print("=" * 60)

        # Load adapter weights
        start_time = time.time()
        print(f"Loading adapter from {checkpoint_dir}...")

        if peft_model is None:
            # First checkpoint: create PeftModel
            peft_model = PeftModel.from_pretrained(model, str(checkpoint_dir))
            FastVisionModel.for_inference(peft_model)
        else:
            # Subsequent checkpoints: hotswap adapter weights in-place
            # This avoids issues with delete_adapter + load_adapter
            peft_model.load_adapter(str(checkpoint_dir), adapter_name="default", hotswap=True)

        print(f"Adapter loaded in {time.time() - start_time:.1f}s")

        # Run evaluation
        correct = 0
        total = 0
        checkpoint_results = []

        for batch_start in range(0, len(eval_data), batch_size):

            batch_end = min(batch_start + batch_size, len(eval_data))
            batch = eval_data[batch_start:batch_end]

            # Prepare batch inputs
            batch_texts = [item["text"] for item in batch]
            batch_images = []
            for item in batch:
                batch_images.extend(item["images"])

            inputs = processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = peft_model.generate(
                    **inputs, max_new_tokens=512, do_sample=False
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )

            # Check correctness
            for output_text, item in zip(output_texts, batch):
                ground_truth = item["ground_truth"]

                # Extract answer
                answer_match = re.search(r'<answer>\s*(yes|no|[a-g])\s*</answer>', output_text.lower())
                llm_answer = answer_match.group(1) if answer_match else ""

                # Extract analysis
                analysis = ""
                analysis_start = output_text.find("<analysis>")
                analysis_end = output_text.find("</analysis>")
                if analysis_start != -1 and analysis_end != -1:
                    analysis = output_text[analysis_start + len("<analysis>"):analysis_end].strip()

                # Print answer and analysis
                print(f"Output text: {output_text}")
                print(f"Ground truth: {ground_truth}")
                print(f"LLM answer: {llm_answer}")
                print(f"Analysis: {analysis}")

                # Check if correct
                if hasattr(task, 'CLASS_MAPPING'):
                    predicted = task.CLASS_MAPPING.get(llm_answer)
                    is_correct = predicted == ground_truth
                else:
                    if isinstance(ground_truth, bool):
                        is_correct = (llm_answer == "yes") == ground_truth
                    else:
                        is_correct = llm_answer == str(ground_truth)

                if is_correct:
                    correct += 1
                total += 1

                print(f"Correct: {is_correct}")

                checkpoint_results.append({
                    "checkpoint": checkpoint_name,
                    "step": step_num,
                    "ground_truth": ground_truth,
                    "llm_answer": llm_answer,
                    "correct": is_correct,
                    "analysis": analysis,
                })

        accuracy = correct / total if total > 0 else 0
        print(f"\nCheckpoint {checkpoint_name}: {correct}/{total} correct ({accuracy:.1%})")

        all_results.append({
            "checkpoint": checkpoint_name,
            "step": step_num,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY: All Checkpoints")
    print("=" * 60)

    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))

    best_idx = summary_df['accuracy'].idxmax()
    best = summary_df.loc[best_idx]
    print(f"\nBest checkpoint: {best['checkpoint']} with {best['accuracy']:.1%} accuracy")

    # Save results
    output_path = RESULTS_DIR / f"{run_path.replace('/', '_')}_all_checkpoints_eval_{task_name}.csv"
    summary_df.to_csv(str(output_path), index=False)
    results_volume.commit()

    print(f"\nResults saved to: {output_path}")
    return summary_df


@app.local_entrypoint()
def evaluate_checkpoints(
    run_path: str,
    base_model: str = None,
    task: str = "merge_action",
    num_samples: int = 100,
    batch_size: int = 2,
    checkpoints: str = None,
    answer_only: bool = False,
    class_balance: bool = False,
    use_blank_images: bool = False,
    use_simple_prompt: bool = False,
):
    """
    Evaluate all checkpoints in a training run.

    Usage:
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate_checkpoints \\
            --run-path "Qwen3-VL-32B-Instruct_myrun" \\
            --task merge_action \\
            --num-samples 100 \\
            --checkpoints "100,200,300"  # or "100-500" for range

        # With answer-only mode (no analysis instruction)
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate_checkpoints \\
            --run-path "Qwen3-VL-32B-Instruct_myrun" \\
            --answer-only

        # With class balancing
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate_checkpoints \\
            --run-path "Qwen3-VL-32B-Instruct_myrun" \\
            --class-balance

        # With blank images (sanity check)
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate_checkpoints \\
            --run-path "Qwen3-VL-32B-Instruct_myrun" \\
            --use-blank-images

        # With simple prompt (sanity check)
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate_checkpoints \\
            --run-path "Qwen3-VL-32B-Instruct_myrun" \\
            --use-simple-prompt
    """
    print(f"Evaluating all checkpoints in: {run_path}")
    print(f"Task: {task}")
    print(f"Samples per checkpoint: {num_samples}")
    if checkpoints:
        print(f"Filtering checkpoints: {checkpoints}")
    if answer_only:
        print("Prompt mode: ANSWER ONLY (no analysis instruction)")
    if class_balance:
        print("Class balancing: ENABLED")
    if use_blank_images:
        print("Image mode: BLANK IMAGES (sanity check)")
    if use_simple_prompt:
        print("Prompt mode: SIMPLE PROMPT (sanity check)")

    result_df = evaluate_all_checkpoints.remote(
        run_path=run_path,
        base_model=base_model,
        task_name=task,
        num_samples=num_samples,
        batch_size=batch_size,
        checkpoints=checkpoints,
        answer_only=answer_only,
        class_balance=class_balance,
        use_blank_images=use_blank_images,
        use_simple_prompt=use_simple_prompt,
    )

    print("\nFinal Results:")
    print(result_df.to_string(index=False))


@app.local_entrypoint()
def main(
    task: str = "segment_classification",  # Task to train on
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    num_samples: int = None,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    # LoRA component selection - which parts of the model get LoRA adapters
    finetune_vision: bool = True,     # Apply LoRA to vision encoder (ViT blocks)
    finetune_language: bool = True,   # Apply LoRA to language model layers
    finetune_attention: bool = True,  # Apply LoRA to attention layers
    finetune_mlp: bool = True,        # Apply LoRA to MLP/FFN layers
    finetune_merger: bool = True,     # Apply LoRA to vision-language merger (CRITICAL for gradient flow!)
    merger_full_finetune: bool = False,  # If True, fully train merger weights instead of LoRA
    gpu_count: int = 2,  # Number of GPUs to use (must match @app.function decorator)
    use_4bit: bool = False,
    use_unsloth: bool = True,  # Use Unsloth backend (default: standard transformers + PEFT)
    use_wandb: bool = False,
    run_name: str = None,
    augmented_dataset: str = None,  # Path to parquet file with teacher responses
    use_teacher: bool = False,  # Whether to use teacher model responses
    teacher_column: str = 'teacher_analysis',  # Column name (default works with merge script output)
    teacher_name: str = None,  # Optional: Teacher model name for W&B tracking
    teacher_model: str = None,  # Optional: Specific teacher model suffix to use (e.g., 'o4_mini')
    min_correct: int = 1,  # Minimum correct responses required per sample (filters out samples with fewer)
    train_split: float = 0.8,  # Fraction of data for training (default: 80%)
    val_split: float = 0.1,   # Fraction of data for validation (ignored if val_samples set)
    val_samples: int = None,  # Absolute number of validation samples (takes precedence over val_split)
    test_split: float = 0.0,  # Fraction of data for test (ignored if test_samples set)
    test_samples: int = None,  # Absolute number of test samples (takes precedence over test_split)
    eval_steps: int = 25,  # How often to evaluate (if eval_strategy="steps")
    eval_strategy: str = "steps",  # "steps", "epoch", or "no"
    early_stopping: bool = False,  # Enable early stopping based on validation loss
    early_stopping_patience: int = 3,  # Number of eval steps with no improvement before stopping
    early_stopping_threshold: float = 0.0,  # Minimum improvement to count as improvement
    lr_scheduler: str = "linear",  # LR scheduler: linear, cosine, reduce_lr_on_plateau, etc.
    reduce_lr_factor: float = 0.1,  # Factor to reduce LR by (for reduce_lr_on_plateau)
    reduce_lr_patience: int = 3,  # Eval steps before reducing LR (for reduce_lr_on_plateau)
    reduce_lr_threshold: float = 0.0001,  # Min improvement threshold (for reduce_lr_on_plateau)
    reduce_lr_min: float = 0.0,  # Minimum LR floor (for reduce_lr_on_plateau)
    resume_from: str = None,  # Path to checkpoint folder to resume training from
    wandb_resume_id: str = None,  # W&B run ID to resume (find in W&B URL or UI)
    class_balance: bool = False,  # Enable oversampling to balance classes
    positive_train_samples: int = None,  # Exact number of positive training samples (applied after split)
    positive_val_samples: int = None,  # Exact number of positive validation samples (applied after split)
    rationale_dropout: float = 0.0,  # Probability of dropping <analysis> section (0.0-1.0)
):
    """
    Local entry point to start fine-tuning.

    ⚠️ IMPORTANT: When changing --gpu-count, you MUST also update the decorator:
       @app.function(gpu="H100:N") to match, or you'll get an assertion error.

    Default: 2x H100 GPUs with distributed data parallel (DDP).
    Default effective batch size: 2 per GPU * 2 GPUs * 4 grad accum = 16.

    Usage:
        # List available tasks
        Available tasks: segment_classification, split_action, merge_action

        # Basic fine-tuning on segment classification (default)
        modal run scripts/model-post-training/modal_qwen_finetune.py --num-samples 100 --epochs 3

        # Fine-tune on split action task
        modal run scripts/model-post-training/modal_qwen_finetune.py \
            --task split_action \
            --num-samples 100 \
            --epochs 3

        # Fine-tune with more data and custom name
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --num-samples 1000 \\
            --epochs 5 \\
            --batch-size 4 \\
            --run-name "experiment1"

        # Full fine-tuning with W&B tracking
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --epochs 3 \\
            --use-wandb \\
            --run-name "full_training_v1"

        # Fine-tune with teacher model responses (legacy single-model format)
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --augmented-dataset "data/with_teacher_responses.parquet" \\
            --use-teacher \\
            --epochs 3 \\
            --run-name "teacher_distill_v1"

        # Fine-tune with multi-model teacher data (uses ALL correct responses by default)
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --augmented-dataset "data/multi_teacher_responses.parquet" \\
            --use-teacher \\
            --epochs 3

        # Fine-tune with specific teacher model from multi-model parquet
        # (use --teacher-model with the model suffix, e.g., 'o4_mini', 'claude_37_sonnet')
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --augmented-dataset "data/multi_teacher_responses.parquet" \\
            --use-teacher \\
            --teacher-model "o4_mini" \\
            --epochs 3

        # Fine-tune Qwen2-VL-7B instead
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --model "Qwen/Qwen2-VL-7B-Instruct" \\
            --num-samples 100 \\
            --epochs 3

        # Train with custom 70-15-15 split and evaluation
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --train-split 0.7 \\
            --val-split 0.15 \\
            --test-split 0.15 \\
            --eval-steps 25 \\
            --eval-strategy "steps" \\
            --epochs 5

        # Train without evaluation (use full dataset)
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --train-split 1.0 \\
            --val-split 0.0 \\
            --test-split 0.0 \\
            --epochs 3

        # Train with reduce LR on plateau scheduler
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --lr-scheduler "reduce_lr_on_plateau" \\
            --reduce-lr-factor 0.5 \\
            --reduce-lr-patience 3 \\
            --epochs 10

        # Use different number of GPUs (update decorator first: gpu=modal.gpu.H100(count=4))
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --gpu-count 4 \\
            --epochs 3

        # Control positive sample counts (maintains group-based splitting)
        modal run scripts/model-post-training/modal_qwen_finetune.py \\
            --positive-train-samples 500 \\
            --positive-val-samples 100 \\
            --epochs 3
    """
    # Validate task
    available_tasks = list_tasks()
    if task not in available_tasks:
        print(f"Error: Unknown task '{task}'")
        print(f"Available tasks: {', '.join(available_tasks)}")
        return

    print(f"Starting fine-tuning with {model}...")
    print(f"Task: {task}")
    if use_teacher and augmented_dataset:
        print(f"Mode: Teacher distillation")
        print(f"Teacher data: {augmented_dataset}")
        print(f"Teacher column: {teacher_column}")
        if teacher_model:
            print(f"Teacher model filter: {teacher_model} (single-model mode)")
        else:
            print(f"Teacher model filter: all available (multi-teacher mode)")
        if min_correct > 1:
            print(f"Min correct responses: {min_correct} (samples with fewer will be filtered)")
        if teacher_name:
            print(f"Teacher model (for tracking): {teacher_name}")
    else:
        print(f"Mode: Standard fine-tuning with ground truth labels")
    print(f"GPUs: {gpu_count}x H100")
    print(f"Samples: {num_samples if num_samples else 'all'}")
    print(f"Epochs: {epochs}")
    effective_batch = batch_size * gpu_count * gradient_accumulation
    print(f"Batch size: {batch_size} per GPU (effective: {effective_batch} with {gpu_count} GPUs)")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_r}")
    print(f"LoRA components: vision={finetune_vision}, language={finetune_language}, attention={finetune_attention}, mlp={finetune_mlp}")
    print(f"Merger training: {'full finetune' if merger_full_finetune else 'LoRA' if finetune_merger else 'frozen'}")
    print(f"4-bit quantization: {use_4bit}")
    print(f"Backend: {'Unsloth' if use_unsloth else 'transformers + PEFT'}")
    if val_samples is not None or val_split > 0 or test_samples is not None or test_split > 0:
        if val_samples is not None or test_samples is not None:
            val_str = f"{val_samples} samples" if val_samples is not None else f"{val_split*100:.0f}%"
            test_str = f"{test_samples} samples" if test_samples is not None else f"{test_split*100:.0f}%"
            print(f"Validation: {val_str}, Test: {test_str}")
        else:
            print(f"Data split: {train_split*100:.0f}% train / {val_split*100:.0f}% val / {test_split*100:.0f}% test")
        print(f"Eval strategy: {eval_strategy}")
        if eval_strategy == "steps":
            print(f"Eval frequency: every {eval_steps} steps")
        if early_stopping:
            print(f"Early stopping: patience={early_stopping_patience}, threshold={early_stopping_threshold}")
    else:
        print("No train/val/test split - using all data for training")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print(f"LR scheduler: {lr_scheduler}")
    if lr_scheduler == "reduce_lr_on_plateau":
        print(f"  factor={reduce_lr_factor}, patience={reduce_lr_patience}, threshold={reduce_lr_threshold}, min_lr={reduce_lr_min}")
    if class_balance:
        print(f"Class balancing: enabled (oversample to majority class)")
    if positive_train_samples is not None:
        print(f"Positive training samples: {positive_train_samples} (filtering after split)")
    if positive_val_samples is not None:
        print(f"Positive validation samples: {positive_val_samples} (filtering after split)")
    if rationale_dropout > 0:
        print(f"Rationale dropout: {rationale_dropout:.1%} ({'dynamic per-epoch' if use_unsloth else 'static per-sample'})")

    # Create config
    config = TrainingConfig(
        model_name=model,
        task_name=task,
        num_samples=num_samples,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        lora_r=lora_r,
        # LoRA component selection
        finetune_vision_layers=finetune_vision,
        finetune_language_layers=finetune_language,
        finetune_attention_modules=finetune_attention,
        finetune_mlp_modules=finetune_mlp,
        finetune_merger=finetune_merger,
        merger_modules_to_save=merger_full_finetune,
        gpu_count=gpu_count,
        load_in_4bit=use_4bit,
        use_unsloth=use_unsloth,
        use_wandb=use_wandb,
        wandb_run_name=run_name,
        train_split_ratio=train_split,
        val_split_ratio=val_split,
        val_samples=val_samples,
        test_split_ratio=test_split,
        test_samples=test_samples,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        lr_scheduler_type=lr_scheduler,
        reduce_lr_factor=reduce_lr_factor,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_threshold=reduce_lr_threshold,
        reduce_lr_min=reduce_lr_min,
        resume_from_checkpoint=resume_from,
        wandb_resume_id=wandb_resume_id,
        min_correct_responses=min_correct,
        class_balance=class_balance,
        positive_train_samples=positive_train_samples,
        positive_val_samples=positive_val_samples,
        rationale_dropout_prob=rationale_dropout,
    )

    # Run fine-tuning
    final_path = finetune_qwen.remote(
        config=config,
        augmented_dataset_path=augmented_dataset,
        use_teacher_responses=use_teacher,
        teacher_response_column=teacher_column,
        teacher_model_name=teacher_name,
        teacher_model=teacher_model,
    )

    print("\n" + "="*60)
    print("Fine-tuning job completed!")
    print(f"Model saved at: {final_path}")
    print("\nTo use the fine-tuned model, you can:")
    print("1. Use the merged model for inference (no LoRA adapters needed)")
    print("2. Use the checkpoint with LoRA adapters (smaller, requires loading adapters)")
    print("="*60)


@app.function(
    gpu="A10G",  # Small GPU - SigLIP-2 is only ~400M params
    timeout=3600*24,
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def linear_probe_visual_encoder(
    vision_model: str = "google/siglip2-so400m-patch16-512",
    task_name: str = "merge_action",
    num_samples: int = None,
    batch_size: int = 64,
    pooling: str = "mean",  # "mean", "max", "cls"
    classifier_type: str = "linear",  # "linear", "mlp"
    mlp_hidden_dim: int = 256,
    train_split: float = 0.8,
    seed: int = 42,
    use_cache: bool = True,  # Load/save features from cache
    class_balance: bool = False,  # Subsample majority class to balance
    save_probe_weights: bool = True,  # Save trained classifier weights
    load_probe_from: str = None,  # Load probe weights from previous run (path relative to RESULTS_DIR)
    dataset_path: str = None,  # Override dataset path (instead of using task's default dataset_source)
):
    """
    Frozen vision encoder linear probe using SigLIP-2.

    SigLIP-2 so400m has the SAME architecture as Qwen3-VL's vision encoder:
    - hidden_size: 1152
    - num_layers: 27
    - num_heads: 16
    - patch_size: 16

    This is ~400M params vs 32B - runs in minutes, not hours.

    Interpretation:
    - Val accuracy >> 50%: Vision encoder has signal → focus on reasoning head
    - Val accuracy ~ 50%: Vision encoder lacks signal → need to adapt vision tower

    Probe Weight Saving/Loading:
    - save_probe_weights: If True (default), saves trained classifier to probe_weights/
    - load_probe_from: Path to saved probe weights (e.g., "probe_weights/siglip2_merge_action_linear.npz")
                       When provided, loads weights and evaluates on current dataset without training.
                       Useful for comparing probe performance across different datasets.

    References:
    - Qwen3-VL tech report: https://arxiv.org/abs/2511.21631
    - SigLIP-2 blog: https://huggingface.co/blog/siglip2
    - Model config: https://huggingface.co/google/siglip2-so400m-patch16-512
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import os
    from PIL import Image
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    def compute_rate_metrics(y_true, y_pred):
        """Compute TPR, TNR, FPR, FNR from predictions."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall/Sensitivity)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        return {"tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr}

    print("=" * 60)
    print("Frozen Vision Encoder Linear Probe (SigLIP-2)")
    print("=" * 60)
    print(f"Vision model: {vision_model}")
    print(f"  (Same architecture as Qwen3-VL vision encoder)")
    print(f"Task: {task_name}")
    print(f"Pooling: {pooling}")
    print(f"Classifier: {classifier_type}")
    print(f"Class balance: {class_balance}")

    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Get task config (needed for dataset_source even when loading from cache)
    task = get_task(task_name)
    print(f"\nTask: {task.name} - {task.description}")

    # Check for cached features first (before loading model/dataset)
    cache_dir = RESULTS_DIR / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    model_slug = vision_model.replace("/", "_")
    # Include dataset info in filenames to avoid collisions across species/datasets
    if dataset_path:
        # Use the folder name from dataset_path as identifier
        dataset_slug = Path(dataset_path).name.replace("/", "_")
    else:
        # Use the task's default dataset_source
        dataset_slug = task.dataset_source.replace("/", "_")

    n_str = f"{num_samples}" if num_samples else "all"
    bal_str = "_balanced" if class_balance else ""
    cache_file = cache_dir / f"{model_slug}_{task_name}_{dataset_slug}_{n_str}_{pooling}_seed{seed}{bal_str}.npz"

    # Setup probe weights directory
    probe_weights_dir = RESULTS_DIR / "probe_weights"
    probe_weights_dir.mkdir(exist_ok=True)

    probe_weights_file = probe_weights_dir / f"{model_slug}_{task_name}_{dataset_slug}_{classifier_type}_{pooling}.npz"
    probe_mlp_file = probe_weights_dir / f"{model_slug}_{task_name}_{dataset_slug}_{classifier_type}_{pooling}.pt"

    X, y = None, None
    image_hashes, image_paths_json = None, None
    group_keys = None  # For group-based splitting to prevent data leakage
    class_counts = None
    if use_cache and cache_file.exists():
        print(f"\nLoading cached features from {cache_file}...")
        cached = np.load(str(cache_file), allow_pickle=True)
        X = cached["X"]
        y = cached["y"]
        # Load image hashes/paths if available (for leakage detection)
        if "image_hashes" in cached:
            image_hashes = cached["image_hashes"]
            image_paths_json = cached["image_paths"]
        # Load group keys if available (for group-based splitting)
        if "group_keys" in cached:
            group_keys = cached["group_keys"]
            print(f"  Loaded group keys for proper train/val splitting")
        class_counts = Counter(y.tolist())
        print(f"Loaded {X.shape[0]} samples, feature dim {X.shape[1]}")
        print(f"\nClass distribution (from cache):")
        print(f"  Class 0 (no): {class_counts[0]} ({class_counts[0]/len(y):.1%})")
        print(f"  Class 1 (yes): {class_counts[1]} ({class_counts[1]/len(y):.1%})")

    if X is None:
        # Need to load model and dataset to extract features
        # Load SigLIP-2 - same architecture as Qwen3-VL vision encoder
        print(f"\nLoading vision encoder: {vision_model}...")
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(
            vision_model,
            torch_dtype=torch.float16,
            cache_dir=str(MODEL_DIR),
        ).cuda()
        processor = AutoProcessor.from_pretrained(vision_model, cache_dir=str(MODEL_DIR))

        model.eval()
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Loaded: {num_params:.0f}M params (vs 32B for full Qwen3-VL)")

        # Load dataset
        print(f"\nLoading dataset...")
        ds = task.load_dataset(cache_dir=str(DATASET_DIR), dataset_path=dataset_path)
        ds = task.filter_dataset(ds)
        ds = ds.shuffle(seed=seed)

        # Class balance via subsampling majority class
        if class_balance:
            print(f"Balancing classes via subsampling...")
            labels_pre = [1 if task.get_ground_truth(sample) else 0 for sample in ds]
            class0_idx = [i for i, l in enumerate(labels_pre) if l == 0]
            class1_idx = [i for i, l in enumerate(labels_pre) if l == 1]
            min_count = min(len(class0_idx), len(class1_idx))
            # Subsample majority class to match minority
            balanced_idx = class0_idx[:min_count] + class1_idx[:min_count]
            # Shuffle the balanced indices
            np.random.seed(seed)
            np.random.shuffle(balanced_idx)
            ds = ds.select(balanced_idx)
            print(f"  Subsampled to {len(ds)} samples ({min_count} per class)")

        if num_samples is not None and num_samples < len(ds):
            ds = ds.select(range(num_samples))
        print(f"Using {len(ds)} samples")

        # Class distribution
        labels = [1 if task.get_ground_truth(sample) else 0 for sample in ds]
        class_counts = Counter(labels)
        print(f"\nClass distribution:")
        print(f"  Class 0 (no): {class_counts[0]} ({class_counts[0]/len(labels):.1%})")
        print(f"  Class 1 (yes): {class_counts[1]} ({class_counts[1]/len(labels):.1%})")

        # Extract features
        print(f"\nExtracting visual features...")
        all_features = []
        all_labels = []
        all_image_hashes = []  # Track image hashes for leakage detection
        all_image_paths = []   # Track image paths

        import hashlib

        for i in range(0, len(ds), batch_size):
            batch_end = min(i + batch_size, len(ds))
            print(f"  Batch {i//batch_size + 1}/{(len(ds) + batch_size - 1)//batch_size} "
                  f"(samples {i+1}-{batch_end})...")

            batch_images = []
            batch_labels = []
            batch_hashes = []
            batch_paths = []

            for j in range(i, batch_end):
                sample = ds[j]

                # Store image paths for debugging
                image_paths = sample.get('images', [])
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                batch_paths.append(tuple(image_paths))

                images = task.get_images(sample)
                label = 1 if task.get_ground_truth(sample) else 0

                # Concatenate multiple images into a grid (3 per row)
                if len(images) > 1:
                    images_per_row = 3

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
                        # Single row - horizontal concatenation (original behavior)
                        total_w = sum(img.size[0] for img in resized)
                        combined = Image.new('RGB', (total_w, max_h))
                        x_offset = 0
                        for img in resized:
                            combined.paste(img, (x_offset, 0))
                            x_offset += img.size[0]
                    else:
                        # Multiple rows - grid layout (e.g., 6 images -> 2x3 grid)
                        num_rows = (len(resized) + images_per_row - 1) // images_per_row

                        # Build rows
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

                    batch_images.append(combined)
                else:
                    batch_images.append(images[0])

                # Compute hash of the combined image for leakage detection
                img_bytes = batch_images[-1].tobytes()
                img_hash = hashlib.md5(img_bytes).hexdigest()
                batch_hashes.append(img_hash)

                batch_labels.append(label)

            all_image_hashes.extend(batch_hashes)
            all_image_paths.extend(batch_paths)

            # Process through vision encoder
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                hidden = outputs.last_hidden_state  # (batch, num_patches, 1152)

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
                    all_features.append(feat.cpu().float().numpy())

            all_labels.extend(batch_labels)
            torch.cuda.empty_cache()

        # Convert to numpy arrays
        X = np.stack(all_features)
        y = np.array(all_labels)
        image_hashes = np.array(all_image_hashes)
        # Convert paths to JSON strings for storage
        import json
        image_paths_json = np.array([json.dumps(p) for p in all_image_paths])

        # Extract group keys for proper train/val splitting (prevents data leakage)
        group_keys = []
        for sample in ds:
            key = task.get_split_group_key(sample)
            # Convert to JSON string for storage (handles None and tuples)
            group_keys.append(json.dumps(key) if key is not None else None)
        group_keys = np.array(group_keys, dtype=object)

        # Save to cache
        if use_cache:
            print(f"\nSaving features to cache: {cache_file}")
            np.savez(str(cache_file), X=X, y=y, image_hashes=image_hashes, image_paths=image_paths_json, group_keys=group_keys)
            results_volume.commit()

    print(f"\nFeature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # When loading pre-trained weights, use ALL data for evaluation (no split needed)
    # When training, split into train/val
    if load_probe_from:
        print(f"\nUsing all {len(X)} samples for evaluation (no train/val split)")
        X_eval = X
        y_eval = y
        # Set these to None to indicate we're in eval-only mode
        X_train, X_val, y_train, y_val = None, None, None, None
        train_idx, val_idx = None, None
    else:
        # Split into train/val - use group-based splitting if group keys available
        indices = np.arange(len(X))

        # Check if we have group keys for proper splitting (prevents data leakage)
        use_group_splitting = group_keys is not None and any(k is not None for k in group_keys)

        if use_group_splitting:
            print(f"\n(Using group-based splitting to prevent data leakage)")
            from collections import defaultdict
            import json as json_module

            # Check if task requires connected component merging
            if task.uses_connected_component_splitting():
                print(f"  (Using connected components to merge overlapping groups)")
                # Parse group keys back from JSON
                parsed_keys = []
                for k in group_keys:
                    if k is None:
                        parsed_keys.append(None)
                    else:
                        parsed_keys.append(tuple(json_module.loads(k)) if isinstance(json_module.loads(k), list) else json_module.loads(k))
                merged_group_ids = _merge_overlapping_groups(parsed_keys)
                sample_group_keys = merged_group_ids
            else:
                # Use group keys as-is (already JSON strings, hashable)
                sample_group_keys = list(group_keys)

            # Build mapping from group key to sample indices
            group_to_indices = defaultdict(list)
            for idx, key in enumerate(sample_group_keys):
                group_to_indices[key].append(idx)

            unique_groups = list(group_to_indices.keys())
            print(f"  Unique groups: {len(unique_groups)}")
            print(f"  Samples per group: min={min(len(v) for v in group_to_indices.values())}, "
                  f"max={max(len(v) for v in group_to_indices.values())}, "
                  f"avg={len(X)/len(unique_groups):.1f}")

            # Split groups (not samples)
            val_ratio = 1.0 - train_split
            train_groups, val_groups = train_test_split(
                unique_groups,
                test_size=val_ratio,
                random_state=seed,
                shuffle=True
            )

            # Convert groups back to sample indices
            train_set = set(train_groups)
            val_set = set(val_groups)

            train_idx = np.array([idx for idx, key in enumerate(sample_group_keys) if key in train_set])
            val_idx = np.array([idx for idx, key in enumerate(sample_group_keys) if key in val_set])

            print(f"  Groups: Train={len(train_groups)}, Val={len(val_groups)}")
        else:
            # Fallback to stratified sample-level splitting (no group keys available)
            if group_keys is None:
                print(f"\n(WARNING: No group keys available - using sample-level splitting)")
                print(f"  Re-run with --no-cache to generate group keys for proper splitting")
            train_idx, val_idx, y_train, y_val = train_test_split(
                indices, y,
                train_size=train_split,
                random_state=seed,
                stratify=y
            )

        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        X_eval, y_eval = None, None  # Not used in training mode

        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Val set: {len(X_val)} samples")

    # === DIAGNOSTIC CHECKS (only when training, not when loading weights) ===
    # Initialize CV results (will be populated if training, None if loading weights)
    cv_results = None

    if not load_probe_from:
        print(f"\n{'=' * 60}")
        print("DIAGNOSTIC: Checking for data leakage")
        print("=" * 60)

        # 0. Check for duplicate IMAGES between train and val (most important check!)
        if image_hashes is not None:
            print(f"\n0. IMAGE-LEVEL LEAKAGE CHECK (most important):")
            train_hashes = set(image_hashes[train_idx])
            val_hashes = set(image_hashes[val_idx])
            overlap_hashes = train_hashes & val_hashes
            print(f"   Unique images in train: {len(train_hashes)}")
            print(f"   Unique images in val: {len(val_hashes)}")
            print(f"   Images appearing in BOTH splits: {len(overlap_hashes)}")
            if len(overlap_hashes) > 0:
                print(f"   *** WARNING: {len(overlap_hashes)} DUPLICATE IMAGES BETWEEN TRAIN AND VAL! ***")
                # Show which samples have duplicates
                for h in list(overlap_hashes)[:3]:
                    train_samples = np.where(image_hashes[train_idx] == h)[0]
                    val_samples = np.where(image_hashes[val_idx] == h)[0]
                    print(f"      Hash {h[:8]}... in train idx {train_samples[:3]} and val idx {val_samples[:3]}")
            else:
                print(f"   OK: No duplicate images between train and val")

            # Also check for duplicate images WITHIN the dataset
            unique_hashes = len(set(image_hashes))
            print(f"\n   Total unique images in dataset: {unique_hashes}/{len(image_hashes)}")
            if unique_hashes < len(image_hashes):
                print(f"   *** WARNING: {len(image_hashes) - unique_hashes} duplicate images in dataset! ***")
        else:
            print(f"\n0. IMAGE-LEVEL CHECK: Skipped (no image hashes in cache, re-run with --no-cache)")

        # 0b. Check for GROUP-LEVEL leakage (same biological location in both splits)
        if group_keys is not None and use_group_splitting:
            print(f"\n0b. GROUP-LEVEL LEAKAGE CHECK:")
            train_groups_set = set(sample_group_keys[i] for i in train_idx)
            val_groups_set = set(sample_group_keys[i] for i in val_idx)
            overlap_groups = train_groups_set & val_groups_set
            print(f"   Unique groups in train: {len(train_groups_set)}")
            print(f"   Unique groups in val: {len(val_groups_set)}")
            print(f"   Groups appearing in BOTH splits: {len(overlap_groups)}")
            if len(overlap_groups) > 0:
                print(f"   *** WARNING: {len(overlap_groups)} GROUPS IN BOTH TRAIN AND VAL! ***")
                print(f"   This indicates data leakage at the biological location level.")
            else:
                print(f"   OK: No group overlap between train and val (proper splitting)")
        elif group_keys is None:
            print(f"\n0b. GROUP-LEVEL CHECK: Skipped (no group keys - re-run with --no-cache)")

        # 1. Feature statistics - verify features are diverse
        print(f"\n1. Feature statistics:")
        print(f"   X_train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
        print(f"   X_train min: {X_train.min():.4f}, max: {X_train.max():.4f}")
        print(f"   X_val mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")

        # 2. Check for duplicate features (exact matches between train and val)
        print(f"\n2. Checking for duplicate features between train/val...")
        n_duplicates = 0
        for i, val_feat in enumerate(X_val):
            # Check if this val feature exactly matches any train feature
            matches = np.all(np.isclose(X_train, val_feat, atol=1e-5), axis=1)
            if matches.any():
                n_duplicates += 1
                if n_duplicates <= 3:
                    print(f"   WARNING: Val sample {i} matches train sample {np.where(matches)[0][0]}")
        print(f"   Total duplicates: {n_duplicates}/{len(X_val)} ({n_duplicates/len(X_val):.1%})")

        # 3. Feature uniqueness - how many unique features?
        print(f"\n3. Feature uniqueness:")
        # Round to check for near-duplicates
        X_rounded = np.round(X, decimals=3)
        unique_features = len(np.unique(X_rounded, axis=0))
        print(f"   Unique features (rounded): {unique_features}/{len(X)} ({unique_features/len(X):.1%})")

        # 4. Per-class feature means - are classes trivially separable?
        print(f"\n4. Per-class feature analysis:")
        class0_mean = X[y == 0].mean(axis=0)
        class1_mean = X[y == 1].mean(axis=0)
        class_diff = np.linalg.norm(class1_mean - class0_mean)
        within_class0_std = X[y == 0].std()
        within_class1_std = X[y == 1].std()
        print(f"   Class 0 (no) feature mean norm: {np.linalg.norm(class0_mean):.2f}")
        print(f"   Class 1 (yes) feature mean norm: {np.linalg.norm(class1_mean):.2f}")
        print(f"   Distance between class means: {class_diff:.2f}")
        print(f"   Within-class std (class 0): {within_class0_std:.4f}")
        print(f"   Within-class std (class 1): {within_class1_std:.4f}")

        # 5. Sanity check: first few features of random samples
        print(f"\n5. Sample features (first 5 dims):")
        for i in [0, len(X)//2, len(X)-1]:
            print(f"   Sample {i} (y={y[i]}): {X[i][:5]}")

        # 6. LABEL PERMUTATION TEST - shuffle labels, accuracy should drop to chance
        print(f"\n6. Label permutation test (shuffled labels):")
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        y_shuffled = y.copy()
        np.random.seed(seed + 999)  # Different seed for shuffling
        np.random.shuffle(y_shuffled)
        clf_permuted = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
        # Use same train/val split indices but with shuffled labels
        y_train_shuffled = y_shuffled[train_idx]
        y_val_shuffled = y_shuffled[val_idx]
        clf_permuted.fit(X_train, y_train_shuffled)
        permuted_train_acc = accuracy_score(y_train_shuffled, clf_permuted.predict(X_train))
        permuted_val_acc = accuracy_score(y_val_shuffled, clf_permuted.predict(X_val))
        majority_baseline = max(class_counts[0], class_counts[1]) / sum(class_counts.values())
        print(f"   Shuffled labels train acc: {permuted_train_acc:.1%}")
        print(f"   Shuffled labels val acc: {permuted_val_acc:.1%}")
        print(f"   Majority baseline: {majority_baseline:.1%}")
        if permuted_val_acc > majority_baseline + 0.05:
            print(f"   *** WARNING: Permuted accuracy {permuted_val_acc:.1%} > baseline+5% - possible leakage! ***")
        else:
            print(f"   OK: Permuted accuracy at or below baseline as expected")

        # 7. K-FOLD CROSS-VALIDATION - more robust accuracy estimate
        print(f"\n7. 5-Fold Cross-Validation ({classifier_type}):")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        if classifier_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            # Use sklearn MLPClassifier for CV - same architecture as PyTorch MLP:
            # input -> hidden_dim -> hidden_dim//2 -> output
            # With L2 regularization (alpha) similar to weight_decay=0.01
            clf_cv = MLPClassifier(
                hidden_layer_sizes=(mlp_hidden_dim, mlp_hidden_dim // 2),
                activation='relu',
                solver='adam',
                alpha=0.01,  # L2 regularization, similar to weight_decay
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,  # patience, similar to PyTorch MLP
                random_state=seed,
            )
        else:
            clf_cv = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')

        # Manually iterate over folds to compute all metrics including TPR/TNR/FPR/FNR
        cv_acc, cv_precision, cv_recall, cv_f1 = [], [], [], []
        cv_tpr, cv_tnr, cv_fpr, cv_fnr = [], [], [], []

        from sklearn.metrics import precision_score, recall_score, f1_score
        from sklearn.base import clone

        for fold_idx, (train_idx_cv, val_idx_cv) in enumerate(cv.split(X, y)):
            X_train_cv, X_val_cv = X[train_idx_cv], X[val_idx_cv]
            y_train_cv, y_val_cv = y[train_idx_cv], y[val_idx_cv]

            clf_fold = clone(clf_cv)
            clf_fold.fit(X_train_cv, y_train_cv)
            y_pred_cv = clf_fold.predict(X_val_cv)

            cv_acc.append(accuracy_score(y_val_cv, y_pred_cv))
            cv_precision.append(precision_score(y_val_cv, y_pred_cv, zero_division=0))
            cv_recall.append(recall_score(y_val_cv, y_pred_cv, zero_division=0))
            cv_f1.append(f1_score(y_val_cv, y_pred_cv, zero_division=0))

            # Compute rate metrics
            rates = compute_rate_metrics(y_val_cv, y_pred_cv)
            cv_tpr.append(rates["tpr"])
            cv_tnr.append(rates["tnr"])
            cv_fpr.append(rates["fpr"])
            cv_fnr.append(rates["fnr"])

        cv_acc = np.array(cv_acc)
        cv_precision = np.array(cv_precision)
        cv_recall = np.array(cv_recall)
        cv_f1 = np.array(cv_f1)
        cv_tpr = np.array(cv_tpr)
        cv_tnr = np.array(cv_tnr)
        cv_fpr = np.array(cv_fpr)
        cv_fnr = np.array(cv_fnr)

        print(f"   Fold accuracies:  {[f'{s:.1%}' for s in cv_acc]}")
        print(f"   Fold precisions:  {[f'{s:.1%}' for s in cv_precision]}")
        print(f"   Fold recalls:     {[f'{s:.1%}' for s in cv_recall]}")
        print(f"   Fold F1 scores:   {[f'{s:.1%}' for s in cv_f1]}")
        print(f"   Fold TPR:         {[f'{s:.1%}' for s in cv_tpr]}")
        print(f"   Fold TNR:         {[f'{s:.1%}' for s in cv_tnr]}")
        print(f"   Fold FPR:         {[f'{s:.1%}' for s in cv_fpr]}")
        print(f"   Fold FNR:         {[f'{s:.1%}' for s in cv_fnr]}")
        print(f"\n   Mean CV accuracy:   {cv_acc.mean():.1%} (+/- {cv_acc.std()*2:.1%})")
        print(f"   Mean CV precision:  {cv_precision.mean():.1%} (+/- {cv_precision.std()*2:.1%})")
        print(f"   Mean CV recall:     {cv_recall.mean():.1%} (+/- {cv_recall.std()*2:.1%})")
        print(f"   Mean CV F1:         {cv_f1.mean():.1%} (+/- {cv_f1.std()*2:.1%})")
        print(f"   Mean CV TPR:        {cv_tpr.mean():.1%} (+/- {cv_tpr.std()*2:.1%})")
        print(f"   Mean CV TNR:        {cv_tnr.mean():.1%} (+/- {cv_tnr.std()*2:.1%})")
        print(f"   Mean CV FPR:        {cv_fpr.mean():.1%} (+/- {cv_fpr.std()*2:.1%})")
        print(f"   Mean CV FNR:        {cv_fnr.mean():.1%} (+/- {cv_fnr.std()*2:.1%})")
        if cv_acc.std() > 0.15:
            print(f"   *** WARNING: High variance across folds - results may be unstable ***")

        # Store CV results for saving
        cv_results = {
            "cv_accuracy_mean": float(cv_acc.mean()),
            "cv_accuracy_std": float(cv_acc.std()),
            "cv_accuracy_folds": [float(x) for x in cv_acc],
            "cv_precision_mean": float(cv_precision.mean()),
            "cv_precision_std": float(cv_precision.std()),
            "cv_precision_folds": [float(x) for x in cv_precision],
            "cv_recall_mean": float(cv_recall.mean()),
            "cv_recall_std": float(cv_recall.std()),
            "cv_recall_folds": [float(x) for x in cv_recall],
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
            "cv_f1_folds": [float(x) for x in cv_f1],
            "cv_tpr_mean": float(cv_tpr.mean()),
            "cv_tpr_std": float(cv_tpr.std()),
            "cv_tpr_folds": [float(x) for x in cv_tpr],
            "cv_tnr_mean": float(cv_tnr.mean()),
            "cv_tnr_std": float(cv_tnr.std()),
            "cv_tnr_folds": [float(x) for x in cv_tnr],
            "cv_fpr_mean": float(cv_fpr.mean()),
            "cv_fpr_std": float(cv_fpr.std()),
            "cv_fpr_folds": [float(x) for x in cv_fpr],
            "cv_fnr_mean": float(cv_fnr.mean()),
            "cv_fnr_std": float(cv_fnr.std()),
            "cv_fnr_folds": [float(x) for x in cv_fnr],
        }

        print(f"\n{'=' * 60}")

    # Define MLP class (needed for both training and loading)
    input_dim = X.shape[1]

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim=2):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    # Check if we should load existing probe weights
    clf = None
    mlp = None
    loaded_from_weights = False

    if load_probe_from:
        load_path = RESULTS_DIR / load_probe_from
        print(f"\nLoading probe weights from: {load_path}")

        if classifier_type == "linear":
            if not load_path.exists():
                raise FileNotFoundError(f"Probe weights not found: {load_path}")
            saved = np.load(str(load_path), allow_pickle=True)
            clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
            # Manually set the fitted attributes
            clf.classes_ = saved["classes"]
            clf.coef_ = saved["coef"]
            clf.intercept_ = saved["intercept"]
            clf.n_features_in_ = saved["n_features_in"]
            print(f"  Loaded LogisticRegression weights (features: {clf.n_features_in_})")
            loaded_from_weights = True

        elif classifier_type == "mlp":
            if not load_path.exists():
                raise FileNotFoundError(f"Probe weights not found: {load_path}")
            # Load metadata to get architecture
            meta_path = load_path.with_suffix('.json')
            if meta_path.exists():
                import json
                with open(str(meta_path), 'r') as f:
                    mlp_meta = json.load(f)
                saved_hidden_dim = mlp_meta.get('hidden_dim', mlp_hidden_dim)
                saved_input_dim = mlp_meta.get('input_dim', input_dim)
            else:
                saved_hidden_dim = mlp_hidden_dim
                saved_input_dim = input_dim

            mlp = MLP(saved_input_dim, saved_hidden_dim).cuda()
            mlp.load_state_dict(torch.load(str(load_path), weights_only=True))
            mlp.eval()
            print(f"  Loaded MLP weights (input: {saved_input_dim}, hidden: {saved_hidden_dim})")
            loaded_from_weights = True

    # Train classifier if not loaded from weights
    if not loaded_from_weights:
        print(f"\nTraining {classifier_type} classifier...")

        if classifier_type == "linear":
            # Use sklearn logistic regression
            clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
            clf.fit(X_train, y_train)

        elif classifier_type == "mlp":
            mlp = MLP(input_dim, mlp_hidden_dim).cuda()
            optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)

            # Learning rate scheduler - reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6
            )

            # Compute class weights for balanced training
            class_weights = torch.tensor([
                len(y_train) / (2 * (y_train == 0).sum()),
                len(y_train) / (2 * (y_train == 1).sum()),
            ], dtype=torch.float32).cuda()
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            # Convert to tensors
            X_train_t = torch.tensor(X_train, dtype=torch.float32).cuda()
            y_train_t = torch.tensor(y_train, dtype=torch.long).cuda()
            X_val_t = torch.tensor(X_val, dtype=torch.float32).cuda()

            # Training loop
            mlp.train()
            best_val_acc = 0
            best_epoch = 0
            best_train_acc = 0
            patience = 100  # Increased patience
            patience_counter = 0
            min_epochs = 100  # Don't early stop before this

            for epoch in range(1000):  # Increased max epochs
                optimizer.zero_grad()
                logits = mlp(X_train_t)
                loss = criterion(logits, y_train_t)
                loss.backward()
                optimizer.step()

                # Validate
                mlp.eval()
                with torch.no_grad():
                    val_logits = mlp(X_val_t)
                    val_preds_t = val_logits.argmax(dim=1)
                    val_acc = (val_preds_t.cpu().numpy() == y_val).mean()
                    train_logits = mlp(X_train_t)
                    train_preds_t = train_logits.argmax(dim=1)
                    train_acc = (train_preds_t.cpu().numpy() == y_train).mean()

                # Update LR scheduler
                scheduler.step(val_acc)

                is_new_best = val_acc > best_val_acc
                if is_new_best:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                    best_epoch = epoch
                    best_train_acc = train_acc
                else:
                    patience_counter += 1

                # Only early stop after min_epochs
                if epoch >= min_epochs and patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                    break

                if epoch % 50 == 0 or is_new_best:
                    current_lr = optimizer.param_groups[0]['lr']
                    marker = " *BEST*" if is_new_best else ""
                    print(f"  Epoch {epoch}: loss={loss.item():.4f}, train_acc={train_acc:.1%}, val_acc={val_acc:.1%}, lr={current_lr:.2e}{marker}")

                mlp.train()

            # Load best model
            mlp.load_state_dict(best_state)
            mlp.eval()
            print(f"  Loaded best model from epoch {best_epoch} (train_acc={best_train_acc:.1%}, val_acc={best_val_acc:.1%})")

        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        # Save probe weights if requested
        if save_probe_weights:
            print(f"\nSaving probe weights...")
            if classifier_type == "linear":
                np.savez(
                    str(probe_weights_file),
                    coef=clf.coef_,
                    intercept=clf.intercept_,
                    classes=clf.classes_,
                    n_features_in=clf.n_features_in_,
                    # Metadata for reference
                    vision_model=vision_model,
                    task_name=task_name,
                    pooling=pooling,
                    seed=seed,
                )
                print(f"  Saved to: {probe_weights_file}")
            elif classifier_type == "mlp":
                torch.save(mlp.state_dict(), str(probe_mlp_file))
                # Save metadata
                import json
                mlp_meta = {
                    'input_dim': input_dim,
                    'hidden_dim': mlp_hidden_dim,
                    'vision_model': vision_model,
                    'task_name': task_name,
                    'pooling': pooling,
                    'seed': seed,
                }
                with open(str(probe_mlp_file.with_suffix('.json')), 'w') as f:
                    json.dump(mlp_meta, f, indent=2)
                print(f"  Saved to: {probe_mlp_file}")
            results_volume.commit()

    # Get predictions - different paths for eval-only vs train mode
    if load_probe_from:
        # Eval-only mode: evaluate on ALL data
        if classifier_type == "linear":
            eval_preds = clf.predict(X_eval)
        elif classifier_type == "mlp":
            X_eval_t = torch.tensor(X_eval, dtype=torch.float32).cuda()
            with torch.no_grad():
                eval_preds = mlp(X_eval_t).argmax(dim=1).cpu().numpy()

        eval_acc = accuracy_score(y_eval, eval_preds)
        eval_rates = compute_rate_metrics(y_eval, eval_preds)

        print(f"\n{'=' * 60}")
        print("RESULTS: Pre-trained Probe Evaluation")
        print("=" * 60)
        print(f"\nEval Accuracy: {eval_acc:.1%} (on {len(X_eval)} samples)")
        print(f"Eval TPR: {eval_rates['tpr']:.1%}  TNR: {eval_rates['tnr']:.1%}")
        print(f"Eval FPR: {eval_rates['fpr']:.1%}  FNR: {eval_rates['fnr']:.1%}")

        # Detailed classification report
        target_names = ['no/bad (0)', 'yes/good (1)']
        print(classification_report(y_eval, eval_preds, target_names=target_names))

        # For results dict compatibility
        train_acc = None
        val_acc = eval_acc
        train_preds = None
        val_preds = eval_preds
        train_rates = None
        val_rates = eval_rates
    else:
        # Train mode: evaluate on train and val sets
        if classifier_type == "linear":
            train_preds = clf.predict(X_train)
            val_preds = clf.predict(X_val)
        elif classifier_type == "mlp":
            X_train_t = torch.tensor(X_train, dtype=torch.float32).cuda()
            X_val_t = torch.tensor(X_val, dtype=torch.float32).cuda()
            with torch.no_grad():
                train_preds = mlp(X_train_t).argmax(dim=1).cpu().numpy()
                val_preds = mlp(X_val_t).argmax(dim=1).cpu().numpy()

        # Compute metrics
        train_acc = accuracy_score(y_train, train_preds)
        val_acc = accuracy_score(y_val, val_preds)
        train_rates = compute_rate_metrics(y_train, train_preds)
        val_rates = compute_rate_metrics(y_val, val_preds)

        print(f"\n{'=' * 60}")
        print("RESULTS: Frozen-Encoder Linear Probe")
        print("=" * 60)
        print(f"\nTrain Accuracy: {train_acc:.1%}")
        print(f"Train TPR: {train_rates['tpr']:.1%}  TNR: {train_rates['tnr']:.1%}")
        print(f"Train FPR: {train_rates['fpr']:.1%}  FNR: {train_rates['fnr']:.1%}")
        print(f"\nVal Accuracy:   {val_acc:.1%}")
        print(f"Val TPR: {val_rates['tpr']:.1%}  TNR: {val_rates['tnr']:.1%}")
        print(f"Val FPR: {val_rates['fpr']:.1%}  FNR: {val_rates['fnr']:.1%}")

        # Detailed classification report on validation set
        target_names = ['no/bad (0)', 'yes/good (1)']
        print(classification_report(y_val, val_preds, target_names=target_names))

        # Interpretation (only when training, not when loading pre-trained weights)
        print(f"\n{'=' * 60}")
        print("INTERPRETATION")
        print("=" * 60)

        if val_acc > 0.70:
            print(f"✓ Val accuracy {val_acc:.1%} > 70%")
            print("  The visual encoder ALREADY contains enough signal!")
            print("  → Focus on improving reasoning/decision head (language-side LoRA)")
            print("  → Vision features are informative; the model just needs to learn to use them")
        elif val_acc > 0.55:
            print(f"~ Val accuracy {val_acc:.1%} is marginally above chance")
            print("  The visual encoder has SOME signal, but it's weak.")
            print("  → May need to adapt vision tower (finetune_vision_layers=True)")
            print("  → Or try larger probe (MLP) to see if signal is non-linear")
        else:
            print(f"✗ Val accuracy {val_acc:.1%} is near chance (~50%)")
            print("  The visual encoder does NOT contain sufficient signal.")
            print("  → Need to adapt the vision tower / cross-attention")
            print("  → Language-side LoRA alone is unlikely to help")
            print("  → Consider full vision encoder finetuning or different architecture")

    # Save results
    import json
    import pandas as pd
    from datetime import datetime

    num_total_samples = len(X)
    results = {
        "model": vision_model,
        "task": task_name,
        "num_samples": num_total_samples,
        "train_samples": len(X_train) if X_train is not None else None,
        "val_samples": len(X_val) if X_val is not None else None,
        "eval_samples": len(X_eval) if X_eval is not None else None,
        "feature_dim": X.shape[1],
        "pooling": pooling,
        "classifier_type": classifier_type,
        "train_accuracy": float(train_acc) if train_acc is not None else None,
        "val_accuracy": float(val_acc),
        "eval_accuracy": float(eval_acc) if load_probe_from else None,
        # Rate metrics for train set
        "train_tpr": float(train_rates["tpr"]) if train_rates is not None else None,
        "train_tnr": float(train_rates["tnr"]) if train_rates is not None else None,
        "train_fpr": float(train_rates["fpr"]) if train_rates is not None else None,
        "train_fnr": float(train_rates["fnr"]) if train_rates is not None else None,
        # Rate metrics for val set
        "val_tpr": float(val_rates["tpr"]) if val_rates is not None else None,
        "val_tnr": float(val_rates["tnr"]) if val_rates is not None else None,
        "val_fpr": float(val_rates["fpr"]) if val_rates is not None else None,
        "val_fnr": float(val_rates["fnr"]) if val_rates is not None else None,
        "class_distribution": dict(class_counts),
        "timestamp": datetime.now().isoformat(),
        "loaded_from_weights": loaded_from_weights,
        "weights_loaded_from": load_probe_from if loaded_from_weights else None,
        "dataset_path": dataset_path,
        "dataset_slug": dataset_slug,
        "cross_validation": cv_results,
    }

    # Add probe weights path to results
    if save_probe_weights and not loaded_from_weights:
        if classifier_type == "linear":
            results["probe_weights_path"] = str(probe_weights_file)
        elif classifier_type == "mlp":
            results["probe_weights_path"] = str(probe_mlp_file)

    output_name = f"linear_probe_{vision_model.replace('/', '_')}_{task_name}_{dataset_slug}_{num_total_samples}samples.json"
    output_path = RESULTS_DIR / output_name

    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)

    results_volume.commit()
    print(f"\nResults saved to: {output_path}")

    # Print probe weights info
    if loaded_from_weights:
        print(f"Probe weights loaded from: {load_probe_from}")
    elif save_probe_weights:
        weights_path = probe_weights_file if classifier_type == "linear" else probe_mlp_file
        print(f"Probe weights saved to: {weights_path}")

    return results


@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        str(MODEL_DIR): model_volume,
        str(DATASET_DIR): dataset_volume,
        str(RESULTS_DIR): results_volume,
    }
)
def evaluate_linear_probe_calibration(
    probe_weights_path: str,  # Path to probe weights (relative to RESULTS_DIR)
    vision_model: str = "google/siglip2-so400m-patch16-512",
    task_name: str = "merge_action",
    pooling: str = "mean",
    classifier_type: str = "linear",
    mlp_hidden_dim: int = 256,
    dataset_path: str = None,
    n_bins: int = 10,
    use_test_set: bool = False,
    use_cache: bool = True,
    cache_path: str = None,  # Override cache path (relative to RESULTS_DIR/feature_cache)
    seed: int = 42,
):
    """
    Evaluate calibration metrics (ECE, MCE, Brier score) for a trained linear probe.

    This function:
    1. Loads pre-trained linear probe weights
    2. Loads cached features (or extracts if not cached)
    3. Computes probabilities/confidences
    4. Computes calibration metrics

    Args:
        probe_weights_path: Path to saved probe weights (relative to RESULTS_DIR)
        vision_model: Vision encoder to use for feature extraction
        task_name: Task name (for loading dataset)
        pooling: Feature pooling method ("mean", "max", "cls")
        classifier_type: "linear" or "mlp"
        mlp_hidden_dim: Hidden dimension for MLP classifier
        dataset_path: Override dataset path (for cross-species evaluation)
        n_bins: Number of bins for ECE computation
        use_test_set: If True, derive test set from cache using same split as training (train_split=0.8)
        use_cache: If True, use cached features (default: True)
        cache_path: Specify exact cache file (relative to feature_cache dir). If None, auto-detect.
        seed: Random seed

    Returns:
        dict with confidences, predictions, labels, and calibration metrics
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import os
    from PIL import Image
    from collections import Counter

    def compute_calibration_metrics(confidences, predictions, labels, n_bins=10):
        """Compute ECE, MCE, and Brier score."""
        assert len(confidences) == len(predictions) == len(labels)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = predictions == labels

        ece = 0.0
        mce = 0.0
        bin_metrics = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                gap = abs(avg_confidence_in_bin - accuracy_in_bin)

                ece += gap * prop_in_bin
                mce = max(mce, gap)

                bin_metrics.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'count': int(in_bin.sum()),
                    'gap': float(gap)
                })
            else:
                bin_metrics.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'count': 0,
                    'gap': 0.0
                })

        # Brier score: mean squared error between confidence and correctness
        brier_score = ((confidences - accuracies) ** 2).mean()

        return {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier_score),
            'bin_metrics': bin_metrics,
            'n_bins': n_bins
        }

    print("=" * 60)
    print("Linear Probe Calibration Analysis")
    print("=" * 60)
    print(f"Probe weights: {probe_weights_path}")
    print(f"Vision model: {vision_model}")
    print(f"Task: {task_name}")

    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Get task config
    task = get_task(task_name)

    # Check for cached features first (same logic as linear_probe_visual_encoder)
    cache_dir = RESULTS_DIR / "feature_cache"
    cache_dir.mkdir(exist_ok=True)

    # Use provided cache path or construct automatically
    if cache_path:
        # User-provided cache path (relative to feature_cache dir)
        cache_file = cache_dir / cache_path
        print(f"\nUsing specified cache: {cache_path}")
    else:
        # Auto-construct cache filename
        model_slug = vision_model.replace("/", "_")

        # Include dataset info in cache filename to avoid collisions
        if dataset_path:
            dataset_slug = Path(dataset_path).name.replace("/", "_")
        else:
            dataset_slug = task.dataset_source.replace("/", "_")

        # Note: We don't include test_set in cache key because we filter after loading cache
        cache_file = cache_dir / f"{model_slug}_{task_name}_{dataset_slug}_all_{pooling}_seed{seed}.npz"

    X, y = None, None
    group_keys = None
    if use_cache and cache_file.exists():
        print(f"\n✓ Loading cached features from {cache_file.name}...")
        cached = np.load(str(cache_file), allow_pickle=True)
        X = cached["X"]
        y = cached["y"]
        # Load group keys if available (for proper train/test splitting)
        if "group_keys" in cached:
            group_keys = cached["group_keys"]
        print(f"  Loaded {X.shape[0]} samples, feature dim {X.shape[1]}")
    elif use_cache and cache_path:
        # Cache path was specified but doesn't exist
        print(f"\n⚠ Specified cache not found: {cache_file}")
        print(f"  Will extract features and save to this location")

    # If cache miss, extract features
    if X is None:
        print(f"\n⚠ Cache miss - extracting features (this will be slow)...")
        print(f"  Cache would be saved to: {cache_file.name}")

        # Load dataset
        print(f"\nLoading dataset...")
        ds = task.load_dataset(cache_dir=str(DATASET_DIR), dataset_path=dataset_path)

        # Handle test set if requested
        if use_test_set:
            # Load saved test indices
            test_indices_path = CHECKPOINT_DIR / "test_indices" / f"{task_name}_test_indices.parquet"
            if test_indices_path.exists():
                import pandas as pd
                test_df = pd.read_parquet(str(test_indices_path))
                test_indices = test_df['dataset_index'].tolist()

                # Map parquet indices to current dataset
                id_to_idx = {sample['id']: i for i, sample in enumerate(ds)}
                test_ids = test_df['id'].tolist()
                mapped_indices = [id_to_idx[id] for id in test_ids if id in id_to_idx]

                ds = ds.select(mapped_indices)
                print(f"  Loaded saved test set: {len(ds)} samples")
            else:
                print(f"  WARNING: Test indices not found, using full dataset")
                ds = task.filter_dataset(ds)
        else:
            ds = task.filter_dataset(ds)

        print(f"Using {len(ds)} samples")

        # Load vision encoder
        print(f"\nLoading vision encoder: {vision_model}...")
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(
            vision_model,
            torch_dtype=torch.float16,
            cache_dir=str(MODEL_DIR),
        ).cuda()
        processor = AutoProcessor.from_pretrained(vision_model, cache_dir=str(MODEL_DIR))
        model.eval()

        # Extract features and compute predictions with confidences
        print(f"\nExtracting features...")
        all_features = []
        all_labels = []
        batch_size = 64

        import hashlib

        for i in range(0, len(ds), batch_size):
            batch_end = min(i + batch_size, len(ds))
            if (i // batch_size) % 10 == 0:
                print(f"  Batch {i//batch_size + 1}/{(len(ds) + batch_size - 1)//batch_size}...")

            batch_images = []
            batch_labels = []

            for j in range(i, batch_end):
                sample = ds[j]
                images = task.get_images(sample)
                label = 1 if task.get_ground_truth(sample) else 0

                # Concatenate multiple images into a grid (same as linear probe)
                if len(images) > 1:
                    images_per_row = 3
                    max_h = max(img.size[1] for img in images)
                    resized = []
                    for img in images:
                        if img.size[1] != max_h:
                            ratio = max_h / img.size[1]
                            new_w = int(img.size[0] * ratio)
                            img = img.resize((new_w, max_h), Image.LANCZOS)
                        resized.append(img)

                    if len(resized) <= images_per_row:
                        total_w = sum(img.size[0] for img in resized)
                        combined = Image.new('RGB', (total_w, max_h))
                        x_offset = 0
                        for img in resized:
                            combined.paste(img, (x_offset, 0))
                            x_offset += img.size[0]
                    else:
                        num_rows = (len(resized) + images_per_row - 1) // images_per_row
                        rows = []
                        for row_idx in range(num_rows):
                            start_idx = row_idx * images_per_row
                            end_idx = min(start_idx + images_per_row, len(resized))
                            row_images = resized[start_idx:end_idx]

                            row_w = sum(img.size[0] for img in row_images)
                            row_img = Image.new('RGB', (row_w, max_h))
                            x_offset = 0
                            for img in row_images:
                                row_img.paste(img, (x_offset, 0))
                                x_offset += img.size[0]
                            rows.append(row_img)

                        max_row_w = max(r.size[0] for r in rows)
                        total_h = sum(r.size[1] for r in rows)
                        combined = Image.new('RGB', (max_row_w, total_h))
                        y_offset = 0
                        for row_img in rows:
                            combined.paste(row_img, (0, y_offset))
                            y_offset += row_img.size[1]

                    batch_images.append(combined)
                else:
                    batch_images.append(images[0])

                batch_labels.append(label)

            # Process through vision encoder
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.vision_model(**inputs)
                hidden = outputs.last_hidden_state  # (batch, num_patches, hidden_dim)

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
                    all_features.append(feat.cpu().float().numpy())

            all_labels.extend(batch_labels)
            torch.cuda.empty_cache()

        # Convert to numpy
        X = np.stack(all_features)
        y = np.array(all_labels)

        print(f"Extracted {len(X)} samples")

        # Save to cache
        if use_cache:
            print(f"\nSaving features to cache: {cache_file.name}")
            np.savez(str(cache_file), X=X, y=y)
            results_volume.commit()

    print(f"\nFeature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # If use_test_set=True, derive the test set using the same splitting logic as training
    if use_test_set:
        print(f"\nDeriving test set from cached features using seed={seed}...")
        from sklearn.model_selection import train_test_split

        # Use same train_split as linear probe training (default 0.8)
        train_split = 0.8

        # Check if we have group keys for proper splitting
        use_group_splitting = group_keys is not None and any(k is not None for k in group_keys)

        if use_group_splitting:
            print(f"  Using group-based splitting (prevents data leakage)")
            from collections import defaultdict
            import json as json_module

            # Check if task requires connected component merging
            if task.uses_connected_component_splitting():
                # Parse group keys back from JSON
                parsed_keys = []
                for k in group_keys:
                    if k is None:
                        parsed_keys.append(None)
                    else:
                        parsed_keys.append(tuple(json_module.loads(k)) if isinstance(json_module.loads(k), list) else json_module.loads(k))
                merged_group_ids = _merge_overlapping_groups(parsed_keys)
                sample_group_keys = merged_group_ids
            else:
                # Use group keys as-is
                sample_group_keys = list(group_keys)

            # Build mapping from group key to sample indices
            group_to_indices = defaultdict(list)
            for idx, key in enumerate(sample_group_keys):
                group_to_indices[key].append(idx)

            unique_groups = list(group_to_indices.keys())

            # Split groups (not samples) - same as training
            val_ratio = 1.0 - train_split
            train_groups, val_groups = train_test_split(
                unique_groups,
                test_size=val_ratio,
                random_state=seed,
                shuffle=True
            )

            # Convert val groups to sample indices (val = test set)
            val_set = set(val_groups)
            test_idx = np.array([idx for idx, key in enumerate(sample_group_keys) if key in val_set])
        else:
            # Fallback to stratified sample-level splitting
            if group_keys is None:
                print(f"  WARNING: No group keys in cache - using sample-level splitting")
                print(f"  For proper splitting, re-run linear probe training with latest code")

            indices = np.arange(len(X))
            _, test_idx = train_test_split(
                indices,
                test_size=1.0 - train_split,
                random_state=seed,
                stratify=y,
                shuffle=True
            )

        # Select test set
        n_total = len(X)
        X = X[test_idx]
        y = y[test_idx]
        print(f"  Test set: {len(X)} samples ({len(X)/n_total*100:.1f}% of total)")

    # Load probe weights
    print(f"\nLoading probe weights from: {probe_weights_path}")
    load_path = RESULTS_DIR / probe_weights_path

    if classifier_type == "linear":
        from sklearn.linear_model import LogisticRegression

        if not load_path.exists():
            raise FileNotFoundError(f"Probe weights not found: {load_path}")

        weights = np.load(str(load_path))
        input_dim = weights['n_features_in']

        # Create sklearn classifier and load weights
        clf = LogisticRegression()
        clf.coef_ = weights['coef']
        clf.intercept_ = weights['intercept']
        clf.classes_ = weights['classes']
        clf.n_features_in_ = int(weights['n_features_in'])
        print(f"  Loaded LogisticRegression (features: {clf.n_features_in_})")

    elif classifier_type == "mlp":
        # Define MLP class (same as in linear_probe_visual_encoder)
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, 2)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        # Load MLP metadata
        load_path_pt = load_path.with_suffix('.pt') if load_path.suffix != '.pt' else load_path
        meta_path = load_path_pt.with_suffix('.json')

        if meta_path.exists():
            import json
            with open(str(meta_path), 'r') as f:
                mlp_meta = json.load(f)
            input_dim = mlp_meta.get('input_dim')
            saved_hidden_dim = mlp_meta.get('hidden_dim', mlp_hidden_dim)
        else:
            raise FileNotFoundError(f"MLP metadata not found: {meta_path}")

        mlp = MLP(input_dim, saved_hidden_dim).cuda()
        mlp.load_state_dict(torch.load(str(load_path_pt), weights_only=True))
        mlp.eval()
        print(f"  Loaded MLP (input: {input_dim}, hidden: {saved_hidden_dim})")
    else:
        raise ValueError(f"Unknown classifier_type: {classifier_type}")

    # Get predictions and confidences
    print(f"\nComputing predictions with confidences...")
    if classifier_type == "linear":
        # Use predict_proba for calibrated probabilities
        probs = clf.predict_proba(X)  # shape: (n_samples, 2)
        predictions = probs.argmax(axis=1)
        # Confidence = probability of predicted class
        confidences = probs[np.arange(len(predictions)), predictions]

    elif classifier_type == "mlp":
        X_t = torch.tensor(X, dtype=torch.float32).cuda()
        with torch.no_grad():
            logits = mlp(X_t)  # shape: (n_samples, 2)
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
            predictions = logits.argmax(dim=1).cpu().numpy()
            # Confidence = probability of predicted class
            confidences = probs[torch.arange(len(predictions)), predictions].cpu().numpy()

    # Compute calibration metrics
    metrics = compute_calibration_metrics(confidences, predictions, y, n_bins=n_bins)
    accuracy = (predictions == y).mean()

    print(f"\nCalibration Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")

    # Return results
    return {
        'confidences': confidences.tolist(),
        'predictions': predictions.tolist(),
        'labels': y.tolist(),
        'metrics': metrics,
        'accuracy': float(accuracy),
        'probe_weights_path': probe_weights_path,
        'vision_model': vision_model,
        'task_name': task_name,
        'classifier_type': classifier_type,
        'pooling': pooling,
        'dataset_path': dataset_path,
        'use_test_set': use_test_set,
    }


@app.function(
    gpu="A100",  # Need more memory for Qwen3-VL with adapter
    timeout=3600*24,
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        CHECKPOINT_DIR: checkpoint_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def linear_probe_with_adapter(
    adapter_path: str,  # Path to LoRA adapter (relative to CHECKPOINT_DIR)
    task_name: str = "merge_error_identification",
    num_samples: int = None,
    batch_size: int = 8,  # Smaller batch for Qwen3-VL
    pooling: str = "mean",
    classifier_type: str = "linear",
    mlp_hidden_dim: int = 256,
    train_split: float = 0.8,
    seed: int = 42,
    class_balance: bool = False,
    compare_to_frozen: bool = True,  # Also extract frozen features for comparison
    dataset_path: str = None,
):
    """
    Linear probe using Qwen3-VL's vision encoder with a LoRA adapter applied.

    This function helps diagnose if fine-tuning the VLM degrades vision features.
    It extracts vision features from Qwen3-VL (with adapter) and trains a linear probe.

    Key difference from linear_probe_visual_encoder:
    - Uses Qwen3-VL's vision encoder (not SigLIP-2)
    - Can apply LoRA adapters to the vision encoder
    - Computes feature comparison with/without adapter

    Interpretation:
    - If adapted features work WORSE than frozen: adapter is degrading vision quality
    - If adapted features work BETTER: adapter is learning useful visual features
    - If no difference: adapter is not modifying vision encoder (check LoRA_B weights!)

    Usage:
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe_adapter \\
            --adapter-path "Qwen3-VL-32B-Instruct_merge_error_identification_32B_no_contamination/checkpoint-1900" \\
            --task merge_error_identification

    Args:
        adapter_path: Path to LoRA adapter directory (relative to CHECKPOINT_DIR)
        task_name: Task to evaluate on
        compare_to_frozen: If True, also extracts frozen features for comparison
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import os
    import json
    from PIL import Image
    from collections import Counter
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from safetensors import safe_open
    from datetime import datetime

    def compute_rate_metrics(y_true, y_pred):
        """Compute TPR, TNR, FPR, FNR from predictions."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return {"tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr}

    print("=" * 70)
    print("Vision Encoder Linear Probe with LoRA Adapter")
    print("=" * 70)
    print(f"Adapter path: {adapter_path}")
    print(f"Task: {task_name}")
    print(f"Pooling: {pooling}")
    print(f"Compare to frozen: {compare_to_frozen}")

    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Full adapter path
    full_adapter_path = CHECKPOINT_DIR / adapter_path

    # Load and analyze adapter configuration
    config_path = full_adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")

    with open(config_path, "r") as f:
        adapter_config = json.load(f)

    print(f"\n[1] Adapter Configuration")
    print("-" * 50)
    print(f"  Base model: {adapter_config.get('base_model_name_or_path', 'unknown')}")
    print(f"  LoRA r: {adapter_config.get('r', 'unknown')}")
    print(f"  LoRA alpha: {adapter_config.get('lora_alpha', 'unknown')}")

    # Analyze vision LoRA_B weights to check if they're zero
    adapter_weights_path = full_adapter_path / "adapter_model.safetensors"
    vision_lora_B_nonzero = False
    vision_lora_B_norm = 0.0

    if adapter_weights_path.exists():
        print(f"\n[2] Analyzing Adapter Weights")
        print("-" * 50)

        with safe_open(adapter_weights_path, framework="pt") as f:
            vision_layers = [n for n in f.keys() if "visual" in n.lower()]
            vision_lora_B = [n for n in vision_layers if "lora_B" in n]

            print(f"  Vision LoRA layers: {len(vision_layers)}")
            print(f"  Vision LoRA_B layers: {len(vision_lora_B)}")

            lora_B_norms = []
            for name in vision_lora_B:
                tensor = f.get_tensor(name)
                norm = np.linalg.norm(tensor.numpy())
                lora_B_norms.append(norm)

            if lora_B_norms:
                vision_lora_B_norm = np.mean(lora_B_norms)
                max_B_norm = max(lora_B_norms)
                vision_lora_B_nonzero = max_B_norm > 1e-6

                print(f"  Mean Vision LoRA_B norm: {vision_lora_B_norm:.6f}")
                print(f"  Max Vision LoRA_B norm: {max_B_norm:.6f}")

                if not vision_lora_B_nonzero:
                    print(f"\n  *** CRITICAL: Vision LoRA_B weights are ALL ZEROS! ***")
                    print(f"      The vision encoder is NOT being modified by the adapter.")
                    print(f"      Features will be identical to frozen encoder.")

    # Get task config
    task = get_task(task_name)
    print(f"\n[3] Loading Data")
    print("-" * 50)
    print(f"Task: {task.name} - {task.description}")

    # Load dataset
    ds = task.load_dataset(cache_dir=str(DATASET_DIR), dataset_path=dataset_path)
    ds = task.filter_dataset(ds)
    ds = ds.shuffle(seed=seed)

    # Class balance
    if class_balance:
        labels_pre = [1 if task.get_ground_truth(sample) else 0 for sample in ds]
        class0_idx = [i for i, l in enumerate(labels_pre) if l == 0]
        class1_idx = [i for i, l in enumerate(labels_pre) if l == 1]
        min_count = min(len(class0_idx), len(class1_idx))
        balanced_idx = class0_idx[:min_count] + class1_idx[:min_count]
        np.random.seed(seed)
        np.random.shuffle(balanced_idx)
        ds = ds.select(balanced_idx)
        print(f"  Balanced to {len(ds)} samples ({min_count} per class)")

    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    print(f"Using {len(ds)} samples")

    # Class distribution
    labels = [1 if task.get_ground_truth(sample) else 0 for sample in ds]
    class_counts = Counter(labels)
    print(f"Class distribution: 0={class_counts[0]}, 1={class_counts[1]}")

    # Load Qwen3-VL model with adapter
    print(f"\n[4] Loading Qwen3-VL with Adapter")
    print("-" * 50)

    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
    from peft import PeftModel

    base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2-VL-7B-Instruct')
    # Map unsloth models to HF models
    if 'unsloth' in base_model_name:
        base_model_name = base_model_name.replace('unsloth/', 'Qwen/')

    print(f"Loading base model: {base_model_name}")

    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=str(MODEL_DIR),
    )
    processor = Qwen2VLProcessor.from_pretrained(base_model_name, cache_dir=str(MODEL_DIR))

    # Load adapter
    print(f"Loading adapter from: {full_adapter_path}")
    model = PeftModel.from_pretrained(model, str(full_adapter_path))
    model.eval()

    print(f"Model loaded with adapter")

    # Extract features
    print(f"\n[5] Extracting Features")
    print("-" * 50)

    def extract_vision_features(model, images, pooling="mean"):
        """Extract vision encoder features from Qwen3-VL."""
        # Process images
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(model.device, dtype=torch.float16)

        with torch.no_grad():
            # Get vision encoder output
            vision_outputs = model.visual(pixel_values)
            # vision_outputs shape: (batch, num_patches, hidden_dim)

            if pooling == "mean":
                features = vision_outputs.mean(dim=1)
            elif pooling == "max":
                features = vision_outputs.max(dim=1).values
            elif pooling == "cls":
                features = vision_outputs[:, 0]
            else:
                features = vision_outputs.mean(dim=1)

        return features.cpu().float().numpy()

    all_features = []
    all_labels = []

    for i in range(0, len(ds), batch_size):
        batch_end = min(i + batch_size, len(ds))
        print(f"  Batch {i//batch_size + 1}/{(len(ds) + batch_size - 1)//batch_size}")

        batch_images = []
        batch_labels = []

        for j in range(i, batch_end):
            sample = ds[j]
            images = task.get_images(sample)
            label = 1 if task.get_ground_truth(sample) else 0

            # Concatenate multiple images horizontally
            if len(images) > 1:
                max_h = max(img.size[1] for img in images)
                resized = []
                for img in images:
                    if img.size[1] != max_h:
                        ratio = max_h / img.size[1]
                        new_w = int(img.size[0] * ratio)
                        img = img.resize((new_w, max_h), Image.LANCZOS)
                    resized.append(img)
                total_w = sum(img.size[0] for img in resized)
                combined = Image.new('RGB', (total_w, max_h))
                x_offset = 0
                for img in resized:
                    combined.paste(img, (x_offset, 0))
                    x_offset += img.size[0]
                batch_images.append(combined)
            else:
                batch_images.append(images[0])

            batch_labels.append(label)

        # Extract features
        features = extract_vision_features(model, batch_images, pooling)
        all_features.extend(features)
        all_labels.extend(batch_labels)

        torch.cuda.empty_cache()

    X = np.stack(all_features)
    y = np.array(all_labels)

    print(f"\nFeature shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Split data
    indices = np.arange(len(X))
    train_idx, val_idx, y_train, y_val = train_test_split(
        indices, y, train_size=train_split, random_state=seed, stratify=y
    )
    X_train = X[train_idx]
    X_val = X[val_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Train classifier
    print(f"\n[6] Training Classifier")
    print("-" * 50)

    if classifier_type == "linear":
        clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
        clf.fit(X_train, y_train)

        train_preds = clf.predict(X_train)
        val_preds = clf.predict(X_val)
    else:
        raise ValueError(f"Only linear classifier supported for now")

    # Compute metrics
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_rates = compute_rate_metrics(y_val, val_preds)

    print(f"\nTrain Accuracy: {train_acc:.1%}")
    print(f"Val Accuracy:   {val_acc:.1%}")
    print(f"Val TPR: {val_rates['tpr']:.1%}  TNR: {val_rates['tnr']:.1%}")

    # Cross-validation
    print(f"\n[7] Cross-Validation")
    print("-" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf_cv = LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced')
    cv_scores = cross_val_score(clf_cv, X, y, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

    # Save results
    results = {
        "task": task_name,
        "adapter_path": adapter_path,
        "adapter_config": {
            "r": adapter_config.get('r'),
            "alpha": adapter_config.get('lora_alpha'),
            "base_model": adapter_config.get('base_model_name_or_path'),
        },
        "vision_adapter_analysis": {
            "lora_B_nonzero": vision_lora_B_nonzero,
            "lora_B_mean_norm": float(vision_lora_B_norm),
            "vision_effectively_modified": vision_lora_B_nonzero,
        },
        "num_samples": len(X),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "feature_dim": X.shape[1],
        "pooling": pooling,
        "classifier_type": classifier_type,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "val_tpr": float(val_rates["tpr"]),
        "val_tnr": float(val_rates["tnr"]),
        "val_fpr": float(val_rates["fpr"]),
        "val_fnr": float(val_rates["fnr"]),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "cv_accuracy_folds": [float(s) for s in cv_scores],
        "class_distribution": dict(class_counts),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    adapter_name = adapter_path.replace("/", "_")
    output_name = f"linear_probe_with_adapter_{adapter_name}_{task_name}_{len(X)}samples.json"
    output_path = RESULTS_DIR / output_name

    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)

    results_volume.commit()
    print(f"\nResults saved to: {output_path}")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)

    if not vision_lora_B_nonzero:
        print(f"\n⚠ Vision LoRA_B weights are ZERO!")
        print(f"  The adapter did NOT modify the vision encoder.")
        print(f"  Features are identical to frozen Qwen3-VL vision encoder.")
        print(f"  → Check training code - vision gradients may not be flowing.")
        print(f"  → The poor VLM performance is NOT due to vision feature degradation.")
        print(f"  → Issue is likely in how language model interprets frozen vision features.")
    else:
        print(f"\nVision adapter is active (LoRA_B mean norm: {vision_lora_B_norm:.4f})")
        print(f"Val accuracy with adapter: {val_acc:.1%}")
        # Compare interpretation would go here if compare_to_frozen was implemented

    return results


@app.local_entrypoint()
def linear_probe_adapter(
    adapter_path: str,
    task: str = "merge_error_identification",
    num_samples: int = None,
    batch_size: int = 8,
    pooling: str = "mean",
    classifier: str = "linear",
    mlp_hidden: int = 256,
    train_split: float = 0.8,
    seed: int = 42,
    class_balance: bool = False,
    compare_frozen: bool = True,
    dataset_path: str = None,
):
    """
    Linear probe with LoRA adapter applied to Qwen3-VL vision encoder.

    This helps diagnose if fine-tuning degrades vision features.

    Usage:
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe_adapter \\
            --adapter-path "Qwen3-VL-32B-Instruct_merge_error_identification_32B_no_contamination/checkpoint-1900" \\
            --task merge_error_identification
    """
    print(f"Running vision encoder probe with adapter")
    print(f"Adapter: {adapter_path}")
    print(f"Task: {task}")

    results = linear_probe_with_adapter.remote(
        adapter_path=adapter_path,
        task_name=task,
        num_samples=num_samples,
        batch_size=batch_size,
        pooling=pooling,
        classifier_type=classifier,
        mlp_hidden_dim=mlp_hidden,
        train_split=train_split,
        seed=seed,
        class_balance=class_balance,
        compare_to_frozen=compare_frozen,
        dataset_path=dataset_path,
    )

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"Val Accuracy: {results['val_accuracy']:.1%}")
    print(f"CV Accuracy: {results['cv_accuracy_mean']:.1%} (+/- {results['cv_accuracy_std']*2:.1%})")

    if not results.get('vision_adapter_analysis', {}).get('vision_effectively_modified', True):
        print(f"\n⚠ CRITICAL: Vision encoder was NOT modified by adapter!")
        print(f"  All vision LoRA_B weights are zero.")

    return results


@app.local_entrypoint()
def linear_probe(
    model: str = "google/siglip2-so400m-patch16-512",
    task: str = "merge_action",
    num_samples: int = None,
    batch_size: int = 64,
    pooling: str = "mean",
    classifier: str = "linear",
    mlp_hidden: int = 256,
    train_split: float = 0.8,
    use_cache: bool = True,
    no_cache: bool = False,
    class_balance: bool = False,
    seed: int = 42,
    save_weights: bool = True,
    load_from: str = None,
    dataset_path: str = None,
):
    """
    Frozen vision encoder linear probe using SigLIP-2.

    SigLIP-2 so400m has the SAME architecture as Qwen3-VL's vision encoder
    (1152 hidden, 27 layers, 16 heads). This runs in minutes vs hours.

    Features are cached to disk (~14KB per sample) for fast re-runs.
    Probe weights can be saved and loaded to compare across datasets.

    Usage:
        # Basic probe (default: SigLIP-2 so400m, same as Qwen3-VL vision)
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --task merge_action

        # With sample limit for quick test
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --num-samples 500

        # Try MLP classifier (tests non-linear separability)
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --classifier mlp

        # Force re-extract features (ignore cache)
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --no-cache

        # Balance classes via subsampling majority
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --class-balance

        # Train on one dataset and save weights
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --task merge_action \\
            --save-weights

        # Load saved weights and evaluate on different dataset
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --task split_action \\
            --load-from "probe_weights/google_siglip2-so400m-patch16-512_merge_action_linear_mean.npz"

        # Compare probe trained on fly data against mouse data
        # First train on fly:
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --task merge_action_fly --save-weights
        # Then evaluate on mouse:
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --task merge_action_mouse \\
            --load-from "probe_weights/google_siglip2-so400m-patch16-512_merge_action_fly_linear_mean.npz"

        # Use custom dataset path (instead of default task dataset_source)
        modal run scripts/model-post-training/modal_qwen_finetune.py::linear_probe \\
            --task merge_action \\
            --dataset-path /path/to/custom/dataset
    """
    print(f"Running frozen vision encoder linear probe")
    print(f"Vision model: {model}")
    print(f"Task: {task}")
    print(f"Pooling: {pooling}")
    print(f"Classifier: {classifier}")
    print(f"Cache: {'disabled' if no_cache else 'enabled'}")
    print(f"Class balance: {class_balance}")
    if dataset_path:
        print(f"Dataset path: {dataset_path}")
    if load_from:
        print(f"Loading probe from: {load_from}")
    if save_weights:
        print(f"Save weights: enabled")

    results = linear_probe_visual_encoder.remote(
        vision_model=model,
        task_name=task,
        num_samples=num_samples,
        batch_size=batch_size,
        pooling=pooling,
        classifier_type=classifier,
        mlp_hidden_dim=mlp_hidden,
        train_split=train_split,
        use_cache=use_cache and not no_cache,
        class_balance=class_balance,
        seed=seed,
        save_probe_weights=save_weights,
        load_probe_from=load_from,
        dataset_path=dataset_path,
    )

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)

    # Show appropriate metrics based on mode
    if results.get('loaded_from_weights'):
        # Eval-only mode
        print(f"Eval Accuracy: {results['eval_accuracy']:.1%} (on {results['eval_samples']} samples)")
        print(f"\nProbe loaded from: {results.get('weights_loaded_from')}")
    else:
        # Train mode
        print(f"Train Accuracy: {results['train_accuracy']:.1%}")
        print(f"Val Accuracy:   {results['val_accuracy']:.1%}")
        if results.get('probe_weights_path'):
            print(f"\nProbe weights saved to: {results.get('probe_weights_path')}")
            print(f"  To load these weights on a different dataset:")
            print(f"    --load-from \"{results.get('probe_weights_path').split('/results/')[-1]}\"")

    return results


@app.local_entrypoint()
def analyze_linear_probe_calibration(
    probe_weights: str = None,
    task: str = "merge_action",
    vision_model: str = "google/siglip2-so400m-patch16-512",
    pooling: str = "mean",
    classifier: str = "linear",
    mlp_hidden: int = 256,
    output_dir: str = "linear_probe_calibration_results",
    dataset_path: str = None,
    use_test_set: bool = False,
    use_cache: bool = True,
    no_cache: bool = False,
    cache_path: str = None,
    n_bins: int = 10,
    seed: int = 42,
):
    """
    Analyze calibration (ECE, MCE, Brier score) for a trained linear probe.

    Features are cached to disk for fast re-runs. Use --no-cache to force re-extraction.

    Usage:
        # Analyze specific probe weights (uses cached features, full dataset)
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_linear_probe_calibration \\
            --probe-weights "probe_weights/google_siglip2-so400m-patch16-512_merge_action_fly-256nm_linear_mean.npz" \\
            --task merge_action

        # Analyze on test set only (derives test set from cache using same split as training)
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_linear_probe_calibration \\
            --probe-weights "..." \\
            --task merge_action \\
            --use-test-set

        # Analyze on different dataset (cross-species)
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_linear_probe_calibration \\
            --probe-weights "probe_weights/google_siglip2-so400m-patch16-512_merge_action_fly-256nm_linear_mean.npz" \\
            --task merge_action \\
            --dataset-path "merge-parquet-mouse"

        # Use specific cache file
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_linear_probe_calibration \\
            --probe-weights "..." \\
            --task merge_action \\
            --cache-path "google_siglip2-so400m-patch16-512_merge_action_fly-256nm_all_mean_seed42.npz"

        # Force re-extract features (ignore cache)
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_linear_probe_calibration \\
            --probe-weights "..." \\
            --task merge_action \\
            --no-cache
    """
    import json
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt

    def plot_reliability_diagram(metrics, title="", save_path=None):
        """Plot 3-panel reliability diagram."""
        bin_metrics = metrics['bin_metrics']

        # Extract data
        bin_centers = [(b['bin_lower'] + b['bin_upper']) / 2 for b in bin_metrics]
        accuracies = [b['accuracy'] for b in bin_metrics]
        confidences = [b['confidence'] for b in bin_metrics]
        counts = [b['count'] for b in bin_metrics]
        gaps = [b['gap'] for b in bin_metrics]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Panel 1: Reliability curve
        ax = axes[0]
        # Only plot bins with samples
        valid_mask = np.array(counts) > 0
        valid_centers = np.array(bin_centers)[valid_mask]
        valid_accuracies = np.array(accuracies)[valid_mask]
        valid_confidences = np.array(confidences)[valid_mask]
        valid_counts = np.array(counts)[valid_mask]

        if len(valid_centers) > 0:
            # Size points by number of samples
            sizes = 100 * (valid_counts / valid_counts.max())
            ax.scatter(valid_confidences, valid_accuracies, s=sizes, alpha=0.6, color='steelblue')
            ax.plot(valid_confidences, valid_accuracies, 'o-', alpha=0.5, color='steelblue')

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Reliability Curve\nECE={metrics["ece"]:.4f}', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Panel 2: Calibration gap
        ax = axes[1]
        colors = ['green' if g < 0 else 'red' for g in gaps]
        ax.bar(bin_centers, gaps, width=0.08, color=colors, alpha=0.6, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Confidence Bin', fontsize=12)
        ax.set_ylabel('Confidence - Accuracy', fontsize=12)
        ax.set_title(f'Calibration Gap\nMCE={metrics["mce"]:.4f}', fontsize=12)
        ax.grid(alpha=0.3, axis='y')
        ax.set_xlim(-0.05, 1.05)

        # Panel 3: Sample distribution
        ax = axes[2]
        ax.bar(bin_centers, counts, width=0.08, alpha=0.6, color='gray', edgecolor='black')
        ax.set_xlabel('Confidence Bin', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Sample Distribution', fontsize=12)
        ax.grid(alpha=0.3, axis='y')
        ax.set_xlim(-0.05, 1.05)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved reliability diagram to: {save_path}")

        return fig

    def plot_confidence_histogram(confidences, predictions, labels, title="", save_path=None):
        """Plot confidence histogram split by correct/incorrect."""
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        labels = np.array(labels)

        correct_mask = predictions == labels
        correct_confs = confidences[correct_mask]
        incorrect_confs = confidences[~correct_mask]

        fig, ax = plt.subplots(figsize=(10, 6))

        bins = np.linspace(0, 1, 21)
        ax.hist(correct_confs, bins=bins, alpha=0.6, color='green', label=f'Correct ({len(correct_confs)})', edgecolor='black')
        ax.hist(incorrect_confs, bins=bins, alpha=0.6, color='red', label=f'Incorrect ({len(incorrect_confs)})', edgecolor='black')

        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
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

    print("="*60)
    print("Linear Probe Calibration Analysis")
    print("="*60)
    print(f"Probe weights: {probe_weights}")
    print(f"Task: {task}")
    print(f"Vision model: {vision_model}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    if not probe_weights:
        raise ValueError("Must specify --probe-weights path")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run calibration evaluation on Modal
    print(f"\nRunning calibration evaluation...")
    if cache_path:
        print(f"Cache: using specified path: {cache_path}")
    else:
        print(f"Cache: {'disabled' if no_cache else 'auto-detect'}")

    result = evaluate_linear_probe_calibration.remote(
        probe_weights_path=probe_weights,
        vision_model=vision_model,
        task_name=task,
        pooling=pooling,
        classifier_type=classifier,
        mlp_hidden_dim=mlp_hidden,
        dataset_path=dataset_path,
        n_bins=n_bins,
        use_test_set=use_test_set,
        use_cache=use_cache and not no_cache,
        cache_path=cache_path,
        seed=seed,
    )

    # Extract results
    confidences = np.array(result['confidences'])
    predictions = np.array(result['predictions'])
    labels = np.array(result['labels'])
    metrics = result['metrics']
    accuracy = result['accuracy']

    print(f"\nCalibration Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")

    # Generate plots
    probe_name = Path(probe_weights).stem
    plot_title = f"Linear Probe: {probe_name}"
    if dataset_path:
        plot_title += f" on {dataset_path}"

    # Reliability diagram
    reliability_path = output_path / f"{probe_name}_reliability.png"
    plot_reliability_diagram(metrics, title=plot_title, save_path=reliability_path)

    # Confidence histogram
    histogram_path = output_path / f"{probe_name}_confidence_hist.png"
    plot_confidence_histogram(confidences, predictions, labels, title=plot_title, save_path=histogram_path)

    print(f"\n✓ Generated plots:")
    print(f"  - {reliability_path}")
    print(f"  - {histogram_path}")

    # Save summary JSON
    summary_path = output_path / f"{probe_name}_calibration_summary.json"
    summary = {
        'probe_weights_path': probe_weights,
        'vision_model': vision_model,
        'task_name': task,
        'dataset_path': dataset_path,
        'accuracy': accuracy,
        'ece': metrics['ece'],
        'mce': metrics['mce'],
        'brier_score': metrics['brier_score'],
        'n_samples': len(labels),
        'n_bins': n_bins,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary to: {summary_path}")
    print("="*60)


# ============================================================================
# Verbalized Confidence Calibration
# ============================================================================

@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={
        str(MODEL_DIR): model_volume,
        str(DATASET_DIR): dataset_volume,
        str(CHECKPOINT_DIR): checkpoint_volume,
        str(RESULTS_DIR): results_volume,
    }
)
def evaluate_verbalized_confidence_calibration(
    adapter_path: str,  # Path to LoRA adapter (relative to CHECKPOINT_DIR)
    base_model: str = "Qwen/Qwen3-VL-32B-Instruct",
    task_name: str = "merge_action",
    dataset_path: str = None,
    n_bins: int = 10,
    max_samples: int = None,
    use_test_set: bool = False,
    batch_size: int = 8,
):
    """
    Evaluate calibration metrics for verbalized confidence from a fine-tuned VLM.

    This function:
    1. Loads VLM model with LoRA adapter
    2. Runs inference with request_confidence=True
    3. Parses both answers and confidence scores from responses
    4. Computes calibration metrics (ECE, MCE, Brier)

    Args:
        adapter_path: Path to LoRA adapter (relative to CHECKPOINT_DIR)
        base_model: Base model name
        task_name: Task name (for loading dataset)
        dataset_path: Override dataset path (for cross-species evaluation)
        n_bins: Number of bins for ECE computation
        max_samples: Limit number of samples (for quick testing)
        use_test_set: If True, use saved test set
        batch_size: Batch size for inference

    Returns:
        dict with confidences, predictions, labels, and calibration metrics
    """
    import torch
    import numpy as np
    import os
    from PIL import Image
    from unsloth import FastVisionModel
    from peft import PeftModel

    def compute_calibration_metrics(confidences, predictions, labels, n_bins=10):
        """Compute ECE, MCE, and Brier score."""
        assert len(confidences) == len(predictions) == len(labels)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = predictions == labels

        ece = 0.0
        mce = 0.0
        bin_metrics = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                gap = abs(avg_confidence_in_bin - accuracy_in_bin)

                ece += gap * prop_in_bin
                mce = max(mce, gap)

                bin_metrics.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': float(accuracy_in_bin),
                    'confidence': float(avg_confidence_in_bin),
                    'count': int(in_bin.sum()),
                    'gap': float(gap)
                })
            else:
                bin_metrics.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'count': 0,
                    'gap': 0.0
                })

        # Brier score: mean squared error between confidence and correctness
        brier_score = ((confidences - accuracies) ** 2).mean()

        return {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier_score),
            'bin_metrics': bin_metrics,
            'n_bins': n_bins
        }

    print("=" * 60)
    print("Verbalized Confidence Calibration Analysis")
    print("=" * 60)
    print(f"Adapter path: {adapter_path}")
    print(f"Base model: {base_model}")
    print(f"Task: {task_name}")

    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    # Get task config
    task = get_task(task_name)

    # Load dataset
    print(f"\nLoading dataset...")
    ds = task.load_dataset(cache_dir=str(DATASET_DIR), dataset_path=dataset_path)

    # Handle test set if requested
    if use_test_set:
        test_indices_path = CHECKPOINT_DIR / "test_indices" / f"{task_name}_test_indices.parquet"
        if test_indices_path.exists():
            import pandas as pd
            test_df = pd.read_parquet(str(test_indices_path))
            test_indices = test_df['dataset_index'].tolist()

            # Map parquet indices to current dataset
            id_to_idx = {sample['id']: i for i, sample in enumerate(ds)}
            test_ids = test_df['id'].tolist()
            mapped_indices = [id_to_idx[id] for id in test_ids if id in id_to_idx]

            ds = ds.select(mapped_indices)
            print(f"  Loaded saved test set: {len(ds)} samples")
        else:
            print(f"  WARNING: Test indices not found, using filtered dataset")
            ds = task.filter_dataset(ds)
    else:
        ds = task.filter_dataset(ds)

    # Limit samples if requested
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        print(f"  Limited to {max_samples} samples")

    print(f"Using {len(ds)} samples")

    # Load model with adapter
    print(f"\nLoading model with adapter...")
    full_adapter_path = CHECKPOINT_DIR / adapter_path

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model,
        load_in_4bit=False,
        cache_dir=str(MODEL_DIR),
    )

    if full_adapter_path.exists():
        print(f"  Loading adapter from: {full_adapter_path}")
        model = PeftModel.from_pretrained(model, str(full_adapter_path))
    else:
        raise ValueError(f"Adapter not found: {full_adapter_path}")

    FastVisionModel.for_inference(model)
    model = model.to("cuda")

    # Load processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        base_model,
        cache_dir=str(MODEL_DIR),
    )

    # Run inference with confidence requests
    print(f"\nRunning inference with confidence requests...")
    all_confidences = []
    all_predictions = []
    all_labels = []
    n_missing_confidence = 0

    for i in range(0, len(ds), batch_size):
        batch_samples = ds[i:min(i + batch_size, len(ds))]
        batch_texts = []
        batch_images = []
        batch_ground_truths = []

        for sample in batch_samples:
            # Get images
            images = task.get_images(sample)

            # Get ground truth
            ground_truth = task.extract_ground_truth(sample)
            batch_ground_truths.append(ground_truth)

            # Format prompt WITH confidence request
            prompt = task.format_prompt(sample, answer_only=False, request_confidence=True)

            # Prepare messages for this sample
            user_content = []
            for img in images:
                user_content.append({"type": "image", "image": img})
            user_content.append({"type": "text", "text": prompt})

            messages = [{
                "role": "user",
                "content": user_content,
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_texts.append(text)
            batch_images.extend(images)

        # Process batch through model
        inputs = processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        # Process outputs
        for output_text, ground_truth in zip(output_texts, batch_ground_truths):
            # Extract answer
            llm_answer = ""
            answer_start = output_text.find("<answer>")
            answer_end = output_text.find("</answer>")
            if answer_start != -1 and answer_end != -1:
                llm_answer = output_text[answer_start + len("<answer>"):answer_end].strip()

            # Extract confidence using task config method
            confidence = task.extract_confidence(output_text)

            # Only include samples where we successfully extracted confidence
            if confidence is not None:
                all_confidences.append(confidence)
                all_predictions.append(llm_answer)
                all_labels.append(ground_truth)
            else:
                n_missing_confidence += 1

        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(ds))}/{len(ds)} samples...")

    print(f"\n✓ Inference complete")
    print(f"  Valid confidences: {len(all_confidences)}/{len(ds)} ({len(all_confidences)/len(ds):.1%})")
    print(f"  Missing confidences: {n_missing_confidence}")

    if len(all_confidences) == 0:
        raise ValueError("No valid confidence scores extracted! Check that the model is outputting <confidence>XX</confidence> tags.")

    # Convert to numpy arrays
    confidences = np.array(all_confidences)
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    # Compute accuracy
    correct = predictions == labels
    accuracy = correct.mean()

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Mean confidence: {confidences.mean():.3f}")
    print(f"  Std confidence: {confidences.std():.3f}")

    # Compute calibration metrics
    print(f"\nComputing calibration metrics...")
    metrics = compute_calibration_metrics(confidences, correct, correct, n_bins=n_bins)

    print(f"\nCalibration Metrics:")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")

    return {
        'confidences': confidences.tolist(),
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'accuracy': float(accuracy),
        'metrics': metrics,
        'n_samples': len(ds),
        'n_valid_confidences': len(all_confidences),
        'confidence_parse_rate': len(all_confidences) / len(ds),
    }


@app.local_entrypoint()
def analyze_verbalized_confidence_calibration(
    adapter_path: str = None,
    task: str = "merge_action",
    model: str = "Qwen/Qwen3-VL-32B-Instruct",
    output_dir: str = "verbalized_confidence_results",
    dataset_path: str = None,
    use_test_set: bool = False,
    n_bins: int = 10,
    max_samples: int = None,
    batch_size: int = 8,
):
    """
    Analyze verbalized confidence calibration for a fine-tuned VLM.

    Usage:
        # Analyze specific adapter
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_verbalized_confidence_calibration \\
            --adapter-path "merge_action_Qwen3-VL-32B-Instruct_merge_action_20250115_164415" \\
            --task merge_action

        # Quick test with limited samples
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_verbalized_confidence_calibration \\
            --adapter-path "..." \\
            --task merge_action \\
            --max-samples 100

        # Cross-species evaluation
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_verbalized_confidence_calibration \\
            --adapter-path "..." \\
            --task merge_action \\
            --dataset-path "merge-parquet-mouse"

        # Use test set for fair comparison with other methods
        modal run scripts/model-post-training/modal_qwen_finetune.py::analyze_verbalized_confidence_calibration \\
            --adapter-path "..." \\
            --task merge_action \\
            --use-test-set
    """
    import json
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt

    def plot_reliability_diagram(metrics, title="", save_path=None):
        """Plot 3-panel reliability diagram."""
        bin_metrics = metrics['bin_metrics']

        # Extract data
        bin_centers = [(b['bin_lower'] + b['bin_upper']) / 2 for b in bin_metrics]
        accuracies = [b['accuracy'] for b in bin_metrics]
        confidences = [b['confidence'] for b in bin_metrics]
        counts = [b['count'] for b in bin_metrics]
        gaps = [b['gap'] for b in bin_metrics]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Panel 1: Reliability curve
        ax = axes[0]
        # Only plot bins with samples
        valid_mask = np.array(counts) > 0
        valid_centers = np.array(bin_centers)[valid_mask]
        valid_accuracies = np.array(accuracies)[valid_mask]
        valid_confidences = np.array(confidences)[valid_mask]
        valid_counts = np.array(counts)[valid_mask]

        if len(valid_centers) > 0:
            # Size points by number of samples
            sizes = 100 * (valid_counts / valid_counts.max())
            ax.scatter(valid_confidences, valid_accuracies, s=sizes, alpha=0.6, color='steelblue')
            ax.plot(valid_confidences, valid_accuracies, 'o-', alpha=0.5, color='steelblue')

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
        ax.set_xlabel('Verbalized Confidence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Reliability Curve\nECE={metrics["ece"]:.4f}', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Panel 2: Calibration gap
        ax = axes[1]
        colors = ['green' if g < 0 else 'red' for g in gaps]
        ax.bar(bin_centers, gaps, width=0.08, color=colors, alpha=0.6, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Confidence Bin', fontsize=12)
        ax.set_ylabel('Confidence - Accuracy', fontsize=12)
        ax.set_title(f'Calibration Gap\nMCE={metrics["mce"]:.4f}', fontsize=12)
        ax.grid(alpha=0.3, axis='y')
        ax.set_xlim(-0.05, 1.05)

        # Panel 3: Sample distribution
        ax = axes[2]
        ax.bar(bin_centers, counts, width=0.08, alpha=0.6, color='gray', edgecolor='black')
        ax.set_xlabel('Confidence Bin', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Sample Distribution', fontsize=12)
        ax.grid(alpha=0.3, axis='y')
        ax.set_xlim(-0.05, 1.05)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved reliability diagram to: {save_path}")

        return fig

    def plot_confidence_histogram(confidences, predictions, labels, title="", save_path=None):
        """Plot confidence histogram split by correct/incorrect."""
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        labels = np.array(labels)

        correct_mask = predictions == labels
        correct_confs = confidences[correct_mask]
        incorrect_confs = confidences[~correct_mask]

        fig, ax = plt.subplots(figsize=(10, 6))

        bins = np.linspace(0, 1, 21)
        ax.hist(correct_confs, bins=bins, alpha=0.6, color='green', label=f'Correct ({len(correct_confs)})', edgecolor='black')
        ax.hist(incorrect_confs, bins=bins, alpha=0.6, color='red', label=f'Incorrect ({len(incorrect_confs)})', edgecolor='black')

        ax.set_xlabel('Verbalized Confidence', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        if title:
            ax.set_title(title, fontsize=13)
        else:
            accuracy = correct_mask.mean()
            ax.set_title(f'Verbalized Confidence Distribution (Accuracy: {accuracy:.2%})', fontsize=13)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved confidence histogram to: {save_path}")

        return fig

    print("="*60)
    print("Verbalized Confidence Calibration Analysis")
    print("="*60)
    print(f"Adapter: {adapter_path}")
    print(f"Task: {task}")
    print(f"Model: {model}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    if not adapter_path:
        raise ValueError("Must specify --adapter-path")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run calibration evaluation on Modal
    print(f"\nRunning calibration evaluation...")

    result = evaluate_verbalized_confidence_calibration.remote(
        adapter_path=adapter_path,
        base_model=model,
        task_name=task,
        dataset_path=dataset_path,
        n_bins=n_bins,
        max_samples=max_samples,
        use_test_set=use_test_set,
        batch_size=batch_size,
    )

    # Extract results
    confidences = np.array(result['confidences'])
    predictions = np.array(result['predictions'])
    labels = np.array(result['labels'])
    metrics = result['metrics']
    accuracy = result['accuracy']

    print(f"\nCalibration Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print(f"  MCE: {metrics['mce']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Confidence parse rate: {result['confidence_parse_rate']:.1%}")

    # Generate plots
    model_name = Path(adapter_path).name
    plot_title = f"Verbalized Confidence: {model_name}"
    if dataset_path:
        plot_title += f" on {dataset_path}"

    # Reliability diagram
    reliability_path = output_path / f"{model_name}_reliability.png"
    plot_reliability_diagram(metrics, title=plot_title, save_path=reliability_path)

    # Confidence histogram
    histogram_path = output_path / f"{model_name}_confidence_hist.png"
    plot_confidence_histogram(confidences, predictions, labels, title=plot_title, save_path=histogram_path)

    print(f"\n✓ Generated plots:")
    print(f"  - {reliability_path}")
    print(f"  - {histogram_path}")

    # Save summary JSON
    summary_path = output_path / f"{model_name}_calibration_summary.json"
    summary = {
        'adapter_path': adapter_path,
        'base_model': model,
        'task_name': task,
        'dataset_path': dataset_path,
        'accuracy': accuracy,
        'ece': metrics['ece'],
        'mce': metrics['mce'],
        'brier_score': metrics['brier_score'],
        'n_samples': result['n_samples'],
        'n_valid_confidences': result['n_valid_confidences'],
        'confidence_parse_rate': result['confidence_parse_rate'],
        'n_bins': n_bins,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Saved summary to: {summary_path}")
    print("="*60)


@app.function(
    volumes={DATASET_DIR: dataset_volume},
    timeout=600
)
def _write_file_to_volume(path: str, data: bytes):
    """Helper function to write data to volume."""
    with open(path, 'wb') as f:
        f.write(data)
    dataset_volume.commit()
    return path


@app.local_entrypoint()
def upload_teacher_data(
    local_path: str,
    remote_name: str = None,
):
    """
    Upload teacher response data to Modal volume.

    Usage:
        modal run scripts/model-post-training/modal_qwen_finetune.py::upload_teacher_data \\
            --local-path "data/claude_37_teacher_responses.parquet" \\
            --remote-name "claude_37_teacher_responses.parquet"
    """
    from pathlib import Path

    local_file = Path(local_path)
    if not local_file.exists():
        print(f"Error: File not found: {local_path}")
        return

    # Use filename if remote name not specified
    if remote_name is None:
        remote_name = local_file.name

    remote_path = DATASET_DIR / remote_name

    print(f"Uploading {local_path} to volume...")
    print(f"  Remote path: {remote_path}")
    print(f"  File size: {local_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Read local file
    with open(local_path, 'rb') as f:
        data = f.read()

    # Write to volume
    _write_file_to_volume.remote(str(remote_path), data)

    print(f"✓ Uploaded successfully!")
    print(f"\nUse in training with:")
    print(f"  --augmented-dataset \"{remote_path}\"")


@app.local_entrypoint()
def evaluate(
    adapter_path: str = None,
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct",
    task: str = "segment_classification",
    num_samples: int = None,
    batch_size: int = 2,
    blank_images: bool = False,
    simple_prompt: bool = False,
    test_set_only: bool = False,
    test_indices_path: str = None,
):
    """
    Evaluate a fine-tuned LoRA adapter or base model on any configured task.

    Usage:
        # List available tasks
        Available tasks: segment_classification, split_action, merge_action

        # Evaluate fine-tuned adapter on segment classification (default)
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate \\
            --adapter-path "Qwen3-VL-8B-Instruct_20251112_191538_samplesall_epochs1_lr0.0002_r16" \\
            --num-samples 100 \\
            --batch-size 4

        # Evaluate adapter on split action task
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate \\
            --adapter-path "Qwen3-VL-8B-Instruct_split_action_..." \\
            --task split_action \\
            --num-samples 100

        # Evaluate fine-tuned adapter only on held-out test set (auto-finds test_indices.json in adapter folder)
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate \\
            --adapter-path "Qwen3-VL-8B-Instruct_20251112_191538_samplesall_epochs1_lr0.0002_r16" \\
            --test-set-only

        # Evaluate with specific test indices file
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate \\
            --adapter-path "Qwen3-VL-8B-Instruct_20251112_191538_samplesall_epochs1_lr0.0002_r16" \\
            --test-set-only \\
            --test-indices-path "other_run_folder/test_indices.json"

        # Evaluate base model (no fine-tuning) for baseline
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate \\
            --num-samples 100

        # Evaluate different base model
        modal run scripts/model-post-training/modal_qwen_finetune.py::evaluate \\
            --base-model "Qwen/Qwen2-VL-7B-Instruct" \\
            --num-samples 100
    """
    # Validate task
    available_tasks = list_tasks()
    if task not in available_tasks:
        print(f"Error: Unknown task '{task}'")
        print(f"Available tasks: {', '.join(available_tasks)}")
        return

    if adapter_path:
        print(f"Evaluating adapter: {adapter_path}")
    else:
        print(f"Evaluating base model: {base_model}")
    print(f"Task: {task}")
    print(f"Samples: {num_samples if num_samples else 'all'}")
    print(f"Batch size: {batch_size}")
    if test_set_only:
        print(f"Mode: Test set only (held-out from training)")

    result_df = evaluate_adapter.remote(
        adapter_path=adapter_path,
        base_model=base_model,
        task_name=task,
        num_samples=num_samples,
        batch_size=batch_size,
        use_blank_images=blank_images,
        use_simple_prompt=simple_prompt,
        use_test_set_only=test_set_only,
        test_indices_path=test_indices_path,
    )

    print("\nCompleted!")
    if len(result_df) > 0:
        # Display relevant columns (not all tasks have all columns)
        display_cols = ['llm_answer', 'ground_truth', 'correct']
        optional_cols = ['species', 'current_root_id', 'predicted_description']
        for col in optional_cols:
            if col in result_df.columns:
                display_cols.insert(-1, col)  # Insert before 'correct'
        print(result_df[display_cols].head(10))


@app.function(
    gpu="H100:2",
    timeout=3600,
    volumes={
        MODEL_DIR: model_volume,
        CHECKPOINT_DIR: checkpoint_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def merge_lora_adapter(
    adapter_path: str,
    output_name: str = None,
):
    """
    Merge LoRA adapter weights into base model for vLLM compatibility.

    vLLM only supports LoRA on language backbone, not vision layers.
    This function merges the adapter into the base model so it can be
    served with vLLM as a regular model.
    """
    import torch
    import json
    import os
    from peft import PeftModel

    os.environ["HF_HOME"] = str(MODEL_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)

    full_adapter_path = CHECKPOINT_DIR / adapter_path

    # Load training config to get base model name
    config_path = full_adapter_path / "training_config.json"
    if config_path.exists():
        with open(str(config_path), "r") as f:
            training_config = json.load(f)
        base_model_name = training_config["model_name"]
        print(f"Base model: {base_model_name}")
    else:
        base_model_name = "Qwen/Qwen3-VL-32B-Instruct"
       

    print(f"\n{'='*60}")
    print("Merging LoRA Adapter into Base Model")
    print(f"{'='*60}")
    print(f"Adapter: {adapter_path}")
    print(f"Base model: {base_model_name}")

    # Load base model with transformers (not Unsloth) for merging
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"\nLoading base model...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=str(MODEL_DIR),
    )

    processor = AutoProcessor.from_pretrained(
        base_model_name,
        cache_dir=str(MODEL_DIR),
    )

    print(f"Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(full_adapter_path))

    print(f"Merging weights...")
    model = model.merge_and_unload()

    # Determine output path
    if output_name is None:
        output_name = f"{adapter_path}_merged"
    output_path = CHECKPOINT_DIR / output_name

    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    # Copy training config to merged model folder
    import shutil
    if config_path.exists():
        shutil.copy(str(config_path), str(output_path / "training_config.json"))
        # Update config to note this is merged
        merged_config_path = output_path / "training_config.json"
        with open(str(merged_config_path), "r") as f:
            merged_config = json.load(f)
        merged_config["merged_from_adapter"] = adapter_path
        merged_config["is_merged"] = True
        with open(str(merged_config_path), "w") as f:
            json.dump(merged_config, f, indent=2)

    checkpoint_volume.commit()

    print(f"\n{'='*60}")
    print("Merge completed!")
    print(f"Merged model saved to: {output_path}")
    print(f"\nTo use with vLLM evaluation:")
    print(f"  modal run scripts/analysis/evaluate_classification.py \\")
    print(f"      --base-model \"{output_path}\" \\")
    print(f"      --use-modal --task merge_action")
    print(f"{'='*60}")

    return str(output_path)


@app.local_entrypoint()
def merge(
    adapter_path: str,
    output_name: str = None,
):
    """
    Merge LoRA adapter into base model for vLLM compatibility.

    vLLM only supports dynamic LoRA on language layers, not vision layers.
    Use this to merge a fine-tuned adapter into the base model so it can
    be served with vLLM as a regular model.

    Usage:
        modal run scripts/model-post-training/modal_qwen_finetune.py::merge \\
            --adapter-path "merge_action_finetune_Qwen3-VL-32B-Instruct_..." \\
            --output-name "Qwen3-VL-32B-merge-action-merged"
    """
    print(f"Merging adapter: {adapter_path}")
    if output_name:
        print(f"Output name: {output_name}")

    merged_path = merge_lora_adapter.remote(
        adapter_path=adapter_path,
        output_name=output_name,
    )

    print(f"\nMerged model saved to: {merged_path}")


def _parse_sweep_config(config_path: str) -> tuple[dict, list[dict]]:
    """
    Parse a sweep config file (YAML or JSON) and return base config + list of run configs.

    Expected format:
    ```yaml
    base:  # Optional base config applied to all runs
      task: merge_action
      epochs: 3
      use_wandb: true
      wandb_project: my-sweep

    runs:  # List of run configs (override base)
      - run_name: lr_1e4
        learning_rate: 1e-4
      - run_name: lr_2e4
        learning_rate: 2e-4
      - run_name: lora_r32
        lora_r: 32
        learning_rate: 2e-4
    ```
    """
    import json
    from pathlib import Path

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    # Parse based on extension
    if config_file.suffix in ['.yaml', '.yml']:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML configs. Install with: pip install pyyaml")
        with open(config_file) as f:
            config = yaml.safe_load(f)
    elif config_file.suffix == '.json':
        with open(config_file) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_file.suffix}. Use .yaml, .yml, or .json")

    base_config = config.get('base', {})
    runs = config.get('runs', [])

    if not runs:
        raise ValueError("Sweep config must have at least one run in 'runs' list")

    return base_config, runs


def _build_training_config(base: dict, run_overrides: dict, run_index: int) -> TrainingConfig:
    """Build a TrainingConfig from base config + run-specific overrides."""
    # Merge base with overrides (overrides take precedence)
    merged = {**base, **run_overrides}

    # Map CLI-style param names to TrainingConfig field names
    param_mapping = {
        'model': 'model_name',
        'task': 'task_name',
        'epochs': 'num_train_epochs',
        'batch_size': 'per_device_train_batch_size',
        'gradient_accumulation': 'gradient_accumulation_steps',
        'use_4bit': 'load_in_4bit',
        'train_split': 'train_split_ratio',
        'val_split': 'val_split_ratio',
        'test_split': 'test_split_ratio',
        'resume_from': 'resume_from_checkpoint',
        'lr_scheduler': 'lr_scheduler_type',
    }

    # Convert CLI-style names to TrainingConfig field names
    config_kwargs = {}
    for key, value in merged.items():
        # Skip non-TrainingConfig fields (these are passed separately to finetune_qwen)
        if key in ['run_name', 'augmented_dataset', 'use_teacher', 'teacher_column',
                   'teacher_name', 'teacher_model']:
            continue
        # Map param name if needed
        field_name = param_mapping.get(key, key)
        config_kwargs[field_name] = value

    # Generate unique run name if not specified
    if 'wandb_run_name' not in config_kwargs:
        run_name = merged.get('run_name', f'run_{run_index}')
        config_kwargs['wandb_run_name'] = run_name

    return TrainingConfig(**config_kwargs)


@app.local_entrypoint()
def sweep(
    config_file: str,
    max_parallel: int = 4,
    dry_run: bool = False,
):
    """
    Run multiple training jobs in parallel from a YAML/JSON config file.

    The config file should define a 'base' config (optional) and a 'runs' list.
    Each run inherits from base and can override any parameter.

    Usage:
        # Create a sweep config file
        cat > sweep_config.yaml << 'EOF'
        base:
          task: merge_action
          model: Qwen/Qwen3-VL-8B-Instruct
          epochs: 3
          batch_size: 2
          gradient_accumulation: 4
          use_wandb: true
          wandb_project: merge-action-sweep
          val_samples: 50
          eval_steps: 25

        runs:
          - run_name: lr_1e4
            learning_rate: 1e-4
          - run_name: lr_2e4
            learning_rate: 2e-4
          - run_name: lr_3e4_lora32
            learning_rate: 3e-4
            lora_r: 32
        EOF

        # Run the sweep (max 4 parallel jobs)
        modal run scripts/model-post-training/modal_qwen_finetune.py::sweep \\
            --config-file sweep_config.yaml \\
            --max-parallel 4

        # Dry run to preview what would be launched
        modal run scripts/model-post-training/modal_qwen_finetune.py::sweep \\
            --config-file sweep_config.yaml \\
            --dry-run

    Config file format (YAML or JSON):
        base:                          # Optional - applied to all runs
          task: segment_classification
          model: Qwen/Qwen3-VL-8B-Instruct
          epochs: 3
          learning_rate: 2e-4
          use_wandb: true
          wandb_project: my-sweep      # All runs share this W&B project
          # ... any other TrainingConfig or CLI params

        runs:                          # Required - list of run configs
          - run_name: experiment_1     # Unique name (used in W&B and output path)
            learning_rate: 1e-4        # Override base params
          - run_name: experiment_2
            learning_rate: 2e-4
            lora_r: 32

    Supported parameters (same as main entrypoint):
        model, task, num_samples, epochs, batch_size, gradient_accumulation,
        learning_rate, lora_r, gpu_count, use_4bit, use_unsloth, use_wandb,
        wandb_project, run_name, augmented_dataset, use_teacher, teacher_column,
        teacher_name, train_split, val_split, val_samples, test_split,
        eval_steps, eval_strategy, early_stopping, early_stopping_patience,
        early_stopping_threshold
    """
    # Parse config file
    print(f"Loading sweep config from: {config_file}")
    base_config, runs = _parse_sweep_config(config_file)

    print(f"\n{'='*60}")
    print(f"Sweep Configuration")
    print(f"{'='*60}")
    print(f"Number of runs: {len(runs)}")
    print(f"Max parallel: {max_parallel}")

    if base_config:
        print(f"\nBase config:")
        for key, value in base_config.items():
            print(f"  {key}: {value}")

    print(f"\nRuns:")
    for i, run in enumerate(runs):
        run_name = run.get('run_name', f'run_{i}')
        overrides = {k: v for k, v in run.items() if k != 'run_name'}
        print(f"  [{i}] {run_name}: {overrides if overrides else '(base config only)'}")

    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - No jobs will be launched")
        print(f"{'='*60}")

        # Show what configs would be created
        print("\nGenerated TrainingConfigs:")
        for i, run in enumerate(runs):
            config = _build_training_config(base_config, run, i)
            print(f"\n  Run {i} ({run.get('run_name', f'run_{i}')}):")
            print(f"    model_name: {config.model_name}")
            print(f"    task_name: {config.task_name}")
            print(f"    learning_rate: {config.learning_rate}")
            print(f"    lora_r: {config.lora_r}")
            print(f"    num_train_epochs: {config.num_train_epochs}")
            print(f"    wandb_run_name: {config.wandb_run_name}")
        return

    # Build training configs and call args
    print(f"\n{'='*60}")
    print("Launching parallel training jobs...")
    print(f"{'='*60}")

    # Prepare arguments for starmap: list of (config, augmented_dataset_path, use_teacher_responses, teacher_response_column, teacher_model_name, teacher_model)
    call_args = []
    for i, run in enumerate(runs):
        merged = {**base_config, **run}
        config = _build_training_config(base_config, run, i)

        call_args.append((
            config,
            merged.get('augmented_dataset'),
            merged.get('use_teacher', False),
            merged.get('teacher_column', 'teacher_analysis'),
            merged.get('teacher_name'),
            merged.get('teacher_model'),
        ))

        print(f"  Queued: {config.wandb_run_name}")

    # Run all jobs in parallel using starmap
    # Note: max_containers is set on the function decorator, but we can limit
    # parallelism by batching if needed
    print(f"\nStarting {len(call_args)} training jobs (max {max_parallel} parallel)...")

    # Use map with concurrency limit
    results = []
    for result in finetune_qwen.starmap(call_args, order_outputs=False):
        results.append(result)
        print(f"  Completed: {result}")

    print(f"\n{'='*60}")
    print("Sweep Completed!")
    print(f"{'='*60}")
    print(f"\nResults:")
    for i, result in enumerate(results):
        print(f"  [{i}] {result}")

    return results


# =============================================================================
# Probe Threshold Sweep Analysis
# =============================================================================

@app.function(
    gpu="A10G",  # Light GPU for feature extraction if needed
    timeout=3600,
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def probe_threshold_sweep(
    probe_weights_path: str,  # Path relative to RESULTS_DIR (e.g., "probe_weights/google_siglip2...npz")
    task_name: str = "merge_action",
    dataset_path: str = None,
    num_thresholds: int = 101,  # Number of threshold points to evaluate
    use_val_split: bool = True,  # If True, use val split; if False, use all data
    seed: int = 42,  # Seed for train/val splitting
    feature_seed: int = None,  # Seed used when extracting features (for finding cache file). If None, uses `seed`.
):
    """
    Sweep decision threshold on a trained linear probe to analyze FNR/FPR trade-off.

    Loads saved probe weights and cached features, then evaluates at different
    decision thresholds to show how FNR changes as you shift the bias.

    For a logistic regression:
        logit = X @ coef.T + intercept
        prediction = 1 if logit > threshold else 0  (default threshold=0)

    By varying threshold, we trade off FNR vs FPR:
        - Lower threshold → more positive predictions → lower FNR, higher FPR
        - Higher threshold → fewer positive predictions → higher FNR, lower FPR

    Returns:
        Dict with threshold sweep results including ROC curve data.
    """
    import numpy as np
    import json
    from datetime import datetime
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    # Default feature_seed to seed if not specified
    if feature_seed is None:
        feature_seed = seed

    print("=" * 60)
    print("Probe Threshold Sweep Analysis")
    print("=" * 60)
    print(f"Feature seed: {feature_seed}")
    print(f"Split seed: {seed}")

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

    # Extract metadata from probe filename to find matching cache
    # Format: {model_slug}_{task_name}_{dataset_slug}_{classifier_type}_{pooling}.npz
    probe_name = probe_path.stem  # e.g., "google_siglip2-so400m-patch16-512_merge_action_linear_mean"
    parts = probe_name.rsplit("_", 2)  # Split from right to get classifier_type and pooling
    pooling = parts[-1]  # e.g., "mean"
    classifier_type = parts[-2]  # e.g., "linear"
    model_task_dataset = parts[0]  # e.g., "google_siglip2-so400m-patch16-512_merge_action"

    # Get task for dataset loading
    task = get_task(task_name)

    # Determine dataset slug
    if dataset_path:
        dataset_slug = Path(dataset_path).name.replace("/", "_")
    else:
        dataset_slug = task.dataset_source.replace("/", "_")

    # Look for cached features
    cache_dir = RESULTS_DIR / "feature_cache"
    # Try to find a matching cache file with the specified feature_seed
    import glob
    cache_pattern = str(cache_dir / f"*_{task_name}_{dataset_slug}_*_{pooling}_seed{feature_seed}*.npz")
    cache_files = glob.glob(cache_pattern)

    if not cache_files:
        # Try without seed suffix (older cache format)
        cache_pattern_no_seed = str(cache_dir / f"*_{task_name}_{dataset_slug}_*_{pooling}*.npz")
        cache_files = glob.glob(cache_pattern_no_seed)
        if cache_files:
            print(f"\n  Warning: No cache with seed{feature_seed} found, using available cache files")
        else:
            raise FileNotFoundError(
                f"No cached features found matching pattern: {cache_pattern}\n"
                f"Run linear_probe first to extract and cache features."
            )

    # Use the most recent cache file (or first match)
    cache_file = sorted(cache_files)[-1]
    print(f"\nLoading cached features from: {cache_file}")

    cached = np.load(cache_file, allow_pickle=True)
    X = cached["X"]
    y = cached["y"]
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")

    # Load group keys if available for proper splitting
    group_keys = None
    if "group_keys" in cached:
        group_keys = cached["group_keys"]

    class_counts = Counter(y.tolist())
    print(f"\nClass distribution:")
    print(f"  Class 0 (no): {class_counts[0]} ({class_counts[0]/len(y):.1%})")
    print(f"  Class 1 (yes): {class_counts[1]} ({class_counts[1]/len(y):.1%})")

    # Split data if requested
    if use_val_split:
        use_group_split = False
        if group_keys is not None:
            # Convert to list if numpy array
            if hasattr(group_keys, 'tolist'):
                group_keys = group_keys.tolist()

            # Filter out None values and check if task uses meaningful group keys
            non_none_keys = [k for k in group_keys if k is not None]
            if len(non_none_keys) > 0:
                unique_groups = list(set(non_none_keys))
                # Need at least 5 groups to do a meaningful 80/20 split
                if len(unique_groups) >= 5:
                    use_group_split = True
                    print(f"\n  Task uses group-based splitting: {len(unique_groups)} unique groups")
                else:
                    print(f"\n  Only {len(unique_groups)} unique groups - falling back to random split")
            else:
                print(f"\n  Task does not define group keys - using random split")

        if use_group_split:
            # Group-based split (only on samples with non-None keys)
            train_groups, val_groups = train_test_split(
                unique_groups, test_size=0.2, random_state=seed
            )
            val_set = set(val_groups)
            val_mask = np.array([k in val_set for k in group_keys])
            X_eval = X[val_mask]
            y_eval = y[val_mask]
            print(f"Using val split: {len(X_eval)} samples (group-based, {len(val_groups)} groups)")
        else:
            # Random split
            _, X_eval, _, y_eval = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
            print(f"\nUsing val split: {len(X_eval)} samples (random)")
    else:
        X_eval = X
        y_eval = y
        print(f"\nUsing all data: {len(X_eval)} samples")

    # Compute logits
    logits = X_eval @ coef.T + intercept
    logits = logits.flatten()  # Shape: (n_samples,)

    # Compute probabilities (for ROC/PR curves)
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid

    # Sweep thresholds
    # Use a range that covers the actual logit distribution
    logit_min, logit_max = logits.min(), logits.max()
    logit_range = logit_max - logit_min
    # Extend range slightly beyond data
    threshold_min = logit_min - 0.1 * logit_range
    threshold_max = logit_max + 0.1 * logit_range
    thresholds = np.linspace(threshold_min, threshold_max, num_thresholds)

    print(f"\nSweeping {num_thresholds} thresholds from {threshold_min:.3f} to {threshold_max:.3f}")
    print(f"  Logit range in data: [{logit_min:.3f}, {logit_max:.3f}]")
    print(f"  Default threshold (0): {'within range' if logit_min < 0 < logit_max else 'outside range!'}")

    # Compute metrics at each threshold
    sweep_results = []
    for thresh in thresholds:
        preds = (logits > thresh).astype(int)

        # Confusion matrix elements
        tp = np.sum((preds == 1) & (y_eval == 1))
        tn = np.sum((preds == 0) & (y_eval == 0))
        fp = np.sum((preds == 1) & (y_eval == 0))
        fn = np.sum((preds == 0) & (y_eval == 1))

        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Precision and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        accuracy = (tp + tn) / len(y_eval)

        sweep_results.append({
            "threshold": float(thresh),
            "accuracy": float(accuracy),
            "tpr": float(tpr),
            "tnr": float(tnr),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        })

    # Find key operating points
    # 1. Default threshold (0)
    default_idx = np.argmin(np.abs(thresholds))
    default_result = sweep_results[default_idx]

    # 2. Best accuracy
    best_acc_idx = np.argmax([r["accuracy"] for r in sweep_results])
    best_acc_result = sweep_results[best_acc_idx]

    # 3. Best F1
    best_f1_idx = np.argmax([r["f1"] for r in sweep_results])
    best_f1_result = sweep_results[best_f1_idx]

    # 4. FNR targets (find thresholds that achieve specific FNR values)
    fnr_targets = [0.01, 0.05, 0.10, 0.20]
    fnr_target_results = {}
    for target_fnr in fnr_targets:
        # Find threshold closest to target FNR
        fnr_diffs = [abs(r["fnr"] - target_fnr) for r in sweep_results]
        closest_idx = np.argmin(fnr_diffs)
        if fnr_diffs[closest_idx] < 0.05:  # Only if within 5% of target
            fnr_target_results[f"fnr_{int(target_fnr*100):02d}pct"] = sweep_results[closest_idx]

    # Compute sklearn ROC curve for comparison/AUC
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_eval, probs)
    roc_auc = auc(fpr_roc, tpr_roc)

    # Compute PR curve
    precision_pr, recall_pr, thresholds_pr = precision_recall_curve(y_eval, probs)
    pr_auc = auc(recall_pr, precision_pr)

    print(f"\n{'='*60}")
    print("RESULTS: Threshold Sweep Analysis")
    print("=" * 60)

    print(f"\nDefault threshold (0):")
    print(f"  Accuracy: {default_result['accuracy']:.1%}")
    print(f"  TPR: {default_result['tpr']:.1%}  TNR: {default_result['tnr']:.1%}")
    print(f"  FPR: {default_result['fpr']:.1%}  FNR: {default_result['fnr']:.1%}")
    print(f"  Precision: {default_result['precision']:.1%}  F1: {default_result['f1']:.3f}")

    print(f"\nBest accuracy (threshold={best_acc_result['threshold']:.3f}):")
    print(f"  Accuracy: {best_acc_result['accuracy']:.1%}")
    print(f"  TPR: {best_acc_result['tpr']:.1%}  TNR: {best_acc_result['tnr']:.1%}")
    print(f"  FPR: {best_acc_result['fpr']:.1%}  FNR: {best_acc_result['fnr']:.1%}")

    print(f"\nBest F1 (threshold={best_f1_result['threshold']:.3f}):")
    print(f"  Accuracy: {best_f1_result['accuracy']:.1%}")
    print(f"  F1: {best_f1_result['f1']:.3f}")
    print(f"  TPR: {best_f1_result['tpr']:.1%}  FNR: {best_f1_result['fnr']:.1%}")

    print(f"\nFNR target operating points:")
    for name, result in fnr_target_results.items():
        print(f"  {name} (threshold={result['threshold']:.3f}):")
        print(f"    FNR: {result['fnr']:.1%}  FPR: {result['fpr']:.1%}  Accuracy: {result['accuracy']:.1%}")

    print(f"\nAUC Scores:")
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"  PR AUC:  {pr_auc:.3f}")

    # Prepare output
    results = {
        "probe_weights_path": probe_weights_path,
        "task_name": task_name,
        "dataset_path": dataset_path,
        "num_samples": len(X_eval),
        "use_val_split": use_val_split,
        "seed": seed,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "default_threshold": default_result,
        "best_accuracy": best_acc_result,
        "best_f1": best_f1_result,
        "fnr_targets": fnr_target_results,
        "sweep_results": sweep_results,
        # Include ROC curve data for plotting
        "roc_curve": {
            "fpr": fpr_roc.tolist(),
            "tpr": tpr_roc.tolist(),
        },
        "pr_curve": {
            "precision": precision_pr.tolist(),
            "recall": recall_pr.tolist(),
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    output_name = f"probe_threshold_sweep_{task_name}_{len(X_eval)}samples.json"
    output_path = RESULTS_DIR / output_name

    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2)

    results_volume.commit()
    print(f"\nResults saved to: {output_path}")

    return results


@app.function(
    gpu="A10G",  # Small GPU - just running sklearn
    timeout=3600*3,  # Longer timeout for multiple repeats
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def linear_probe_data_sweep(
    cache_file_name: str,  # Name of cached features file in RESULTS_DIR/feature_cache/
    task_name: str = "merge_action",
    test_size: int = 1024,  # Fixed test set size (deprecated - use test_fraction instead)
    train_sizes: list = None,  # List of training sizes to sweep (default: [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    n_repeats: int = 10,  # Number of random subsets to sample at each training size (for error bars)
    seed: int = 42,
    classifier_type: str = "linear",  # "linear" or "mlp"
    mlp_hidden_dim: int = 256,
    train_fraction: float = 0.8,  # Fraction of data to use for training pool (default: 0.8 to match ResNet)
    test_fraction: float = 0.1,  # Fraction of data to use for test (default: 0.1)
    # Note: val_fraction = 1 - train_fraction - test_fraction (currently unused, added to train pool)
):
    """
    Sweep over different training data sizes using cached features.

    Loads pre-extracted features from cache, creates an 80/10/10 train/val/test split
    (matching ResNet training splits), then trains linear probes on varying amounts
    of training data.

    For each training size, samples n_repeats different random subsets
    to compute mean and standard deviation (error bars).

    The final training size automatically uses all training data (80% of total),
    ensuring the curve connects with ResNet full-data points in plots.

    Args:
        cache_file_name: Name of the .npz file in RESULTS_DIR/feature_cache/
                        e.g., "google_siglip2-so400m-patch16-512_merge_action_all_mean_seed84_balanced.npz"
        task_name: Task name (for metadata)
        test_size: (Deprecated) Fixed test set size - use test_fraction instead
        train_sizes: List of training set sizes to try (default: powers of 2 from 16 to 8192)
        n_repeats: Number of random subsets to sample per training size (for computing error bars)
        seed: Random seed for train/test split (each repeat uses seed+repeat_idx)
        classifier_type: "linear" for logistic regression, "mlp" for 2-layer MLP
        mlp_hidden_dim: Hidden dimension for MLP classifier
        train_fraction: Fraction of data for training pool (default: 0.8)
        test_fraction: Fraction of data for test set (default: 0.1)

    Returns:
        Dictionary with results for each training size (including mean/std across repeats)

    Usage:
        modal run scripts/model-post-training/modal_qwen_finetune.py::data_sweep \\
            --cache-file-name "google_siglip2-so400m-patch16-512_merge_action_all_mean_seed84_balanced.npz" \\
            --task merge_action \\
            --n-repeats 10
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import json
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from datetime import datetime
    import torch.nn.functional as F

    def compute_rate_metrics(y_true, y_pred):
        """Compute TPR, TNR, FPR, FNR from predictions."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return {"tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr}

    def compute_calibration_metrics(confidences, predictions, labels, n_bins=10):
        """Compute ECE, MCE, and Brier score."""
        assert len(confidences) == len(predictions) == len(labels)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = predictions == labels

        ece = 0.0
        mce = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                gap = abs(avg_confidence_in_bin - accuracy_in_bin)

                ece += gap * prop_in_bin
                mce = max(mce, gap)

        # Brier score: mean squared error between confidence and correctness
        brier_score = ((confidences - accuracies) ** 2).mean()

        return {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier_score)
        }

    # MLP classifier class
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x):
            return self.net(x)

    # Default training sizes if not specified
    if train_sizes is None:
        train_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print("=" * 60)
    print("Linear Probe Data Sweep")
    print("=" * 60)
    print(f"Cache file: {cache_file_name}")
    print(f"Task: {task_name}")
    print(f"Split: {train_fraction:.0%} train / {1-train_fraction-test_fraction:.0%} val / {test_fraction:.0%} test")
    print(f"Training sizes: {train_sizes}")
    print(f"Repeats per size: {n_repeats}")
    print(f"Classifier: {classifier_type}")
    print(f"Seed: {seed}")

    # Load cached features
    cache_dir = RESULTS_DIR / "feature_cache"
    cache_path = cache_dir / cache_file_name

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    print(f"\nLoading cached features from {cache_path}...")
    cached = np.load(str(cache_path), allow_pickle=True)
    X = cached["X"]
    y = cached["y"]

    # Load group keys if available (for proper splitting)
    group_keys = None
    if "group_keys" in cached:
        group_keys = cached["group_keys"]
        print(f"  Loaded group keys for proper splitting")

    print(f"Loaded {X.shape[0]} samples, feature dim {X.shape[1]}")

    # Class distribution
    class_counts = Counter(y.tolist())
    print(f"\nClass distribution:")
    print(f"  Class 0 (no): {class_counts[0]} ({class_counts[0]/len(y):.1%})")
    print(f"  Class 1 (yes): {class_counts[1]} ({class_counts[1]/len(y):.1%})")

    # Get task config for group-based splitting
    task = get_task(task_name)

    # Split into train pool and test set using fractions (matching ResNet 80/10/10 split)
    # Note: val set (10%) is currently added to train pool since linear probes don't need validation
    indices = np.arange(len(X))

    # Calculate actual test size from fraction
    actual_test_size = max(1, int(len(X) * test_fraction))

    # Check if we need group-based splitting
    use_group_splitting = group_keys is not None and any(k is not None for k in group_keys)

    if use_group_splitting:
        print(f"\n(Using group-based splitting to prevent data leakage)")
        from collections import defaultdict
        import json as json_module

        # Check if task requires connected component merging
        if task.uses_connected_component_splitting():
            print(f"  (Using connected components to merge overlapping groups)")
            # Parse group keys back from JSON
            parsed_keys = []
            for k in group_keys:
                if k is None:
                    parsed_keys.append(None)
                else:
                    parsed_keys.append(tuple(json_module.loads(k)) if isinstance(json_module.loads(k), list) else json_module.loads(k))
            merged_group_ids = _merge_overlapping_groups(parsed_keys)
            sample_group_keys = merged_group_ids
        else:
            # Use group keys as-is
            sample_group_keys = list(group_keys)

        # Build mapping from group key to sample indices
        group_to_indices = defaultdict(list)
        for idx, key in enumerate(sample_group_keys):
            group_to_indices[key].append(idx)

        unique_groups = list(group_to_indices.keys())
        print(f"  Unique groups: {len(unique_groups)}")

        # Split groups into train+val pool and test using fraction
        avg_samples_per_group = len(X) / len(unique_groups)
        test_group_count = max(1, int(actual_test_size / avg_samples_per_group))

        trainval_groups, test_groups = train_test_split(
            unique_groups,
            test_size=test_group_count,
            random_state=seed,
            shuffle=True
        )

        # Convert groups back to sample indices
        trainval_set = set(trainval_groups)
        test_set = set(test_groups)

        trainval_idx = np.array([idx for idx, key in enumerate(sample_group_keys) if key in trainval_set])
        test_idx = np.array([idx for idx, key in enumerate(sample_group_keys) if key in test_set])

        print(f"  Groups: TrainVal={len(trainval_groups)}, Test={len(test_groups)}")
    else:
        # Stratified sample-level splitting using fraction
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=y,
            shuffle=True
        )

    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    print(f"\nSplit:")
    print(f"  Train+Val pool: {len(X_trainval)} samples")
    print(f"  Test (held out): {len(X_test)} samples")

    # Test set class distribution
    test_class_counts = Counter(y_test.tolist())
    print(f"\nTest set class distribution:")
    print(f"  Class 0: {test_class_counts[0]} ({test_class_counts[0]/len(y_test):.1%})")
    print(f"  Class 1: {test_class_counts[1]} ({test_class_counts[1]/len(y_test):.1%})")

    # Add full training data as final point if not already included
    # Use train_fraction of total data (e.g., 80%) to match ResNet training
    # This ensures the linear probe curve connects with ResNet full-data points in plots
    full_train_size = int(len(X) * train_fraction)

    # Make sure we don't exceed the trainval pool
    full_train_size = min(full_train_size, len(X_trainval))

    if train_sizes[-1] < full_train_size:
        train_sizes = train_sizes + [full_train_size]
        print(f"\nAdding full training data as final point: {full_train_size} samples ({train_fraction:.0%} of total)")
        print(f"Updated training sizes: {train_sizes}")

    # Sweep over training sizes
    results = []
    all_repeat_results = []  # Store all individual repeat results

    for n_train in train_sizes:
        if n_train > len(X_trainval):
            print(f"\nSkipping n_train={n_train} (exceeds train+val pool size {len(X_trainval)})")
            continue

        print(f"\n{'='*60}")
        print(f"Training with {n_train} samples ({n_repeats} repeats)")
        print(f"{'='*60}")

        # Store results for all repeats at this training size
        repeat_accuracies = []
        repeat_tprs = []
        repeat_tnrs = []
        repeat_fprs = []
        repeat_fnrs = []
        repeat_eces = []
        repeat_mces = []
        repeat_briers = []

        for repeat_idx in range(n_repeats):
            # Use different seed for each repeat
            repeat_seed = seed + repeat_idx

            # Sample training data (stratified)
            train_idx_subset, _ = train_test_split(
                np.arange(len(X_trainval)),
                train_size=n_train,
                random_state=repeat_seed,
                stratify=y_trainval,
                shuffle=True
            )

            X_train = X_trainval[train_idx_subset]
            y_train = y_trainval[train_idx_subset]

            # Train set class distribution (only print for first repeat)
            train_class_counts = Counter(y_train.tolist())
            if repeat_idx == 0:
                print(f"Train class distribution: 0={train_class_counts[0]}, 1={train_class_counts[1]}")

            # Train classifier
            if classifier_type == "linear":
                clf = LogisticRegression(max_iter=1000, random_state=repeat_seed, class_weight='balanced')
                clf.fit(X_train, y_train)

                # Predictions and probabilities
                train_preds = clf.predict(X_train)
                test_preds = clf.predict(X_test)
                test_probs = clf.predict_proba(X_test)  # Shape: (n_samples, 2)
                test_confidences = test_probs[np.arange(len(test_preds)), test_preds]

            elif classifier_type == "mlp":
                mlp = MLP(X.shape[1], mlp_hidden_dim).cuda()
                optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)

                # Compute class weights
                class_weights = torch.tensor([
                    len(y_train) / (2 * (y_train == 0).sum()),
                    len(y_train) / (2 * (y_train == 1).sum()),
                ], dtype=torch.float32).cuda()
                criterion = nn.CrossEntropyLoss(weight=class_weights)

                # Convert to tensors
                X_train_t = torch.tensor(X_train, dtype=torch.float32).cuda()
                y_train_t = torch.tensor(y_train, dtype=torch.long).cuda()
                X_test_t = torch.tensor(X_test, dtype=torch.float32).cuda()

                # Training loop (silent for repeats)
                mlp.train()
                best_test_acc = 0
                patience_counter = 0
                patience = 50

                for epoch in range(500):
                    optimizer.zero_grad()
                    logits = mlp(X_train_t)
                    loss = criterion(logits, y_train_t)
                    loss.backward()
                    optimizer.step()

                    # Validate on test set
                    if epoch % 10 == 0:
                        mlp.eval()
                        with torch.no_grad():
                            test_logits = mlp(X_test_t)
                            test_preds_t = test_logits.argmax(dim=1)
                            test_acc = (test_preds_t.cpu().numpy() == y_test).mean()

                            train_logits = mlp(X_train_t)
                            train_preds_t = train_logits.argmax(dim=1)
                            train_acc = (train_preds_t.cpu().numpy() == y_train).mean()

                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                            patience_counter = 0
                            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            break

                        mlp.train()

                # Load best model
                mlp.load_state_dict(best_state)
                mlp.eval()

                # Final predictions and probabilities
                with torch.no_grad():
                    train_logits = mlp(X_train_t)
                    train_preds = train_logits.argmax(dim=1).cpu().numpy()

                    test_logits = mlp(X_test_t)
                    test_probs = F.softmax(test_logits, dim=1).cpu().numpy()  # Shape: (n_samples, 2)
                    test_preds = test_logits.argmax(dim=1).cpu().numpy()
                    test_confidences = test_probs[np.arange(len(test_preds)), test_preds]

            else:
                raise ValueError(f"Unknown classifier_type: {classifier_type}")

            # Compute metrics
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)
            train_rates = compute_rate_metrics(y_train, train_preds)
            test_rates = compute_rate_metrics(y_test, test_preds)

            # Compute calibration metrics
            test_calibration = compute_calibration_metrics(test_confidences, test_preds, y_test, n_bins=10)

            # Store repeat results
            repeat_accuracies.append(test_acc)
            repeat_tprs.append(test_rates["tpr"])
            repeat_tnrs.append(test_rates["tnr"])
            repeat_fprs.append(test_rates["fpr"])
            repeat_fnrs.append(test_rates["fnr"])
            repeat_eces.append(test_calibration["ece"])
            repeat_mces.append(test_calibration["mce"])
            repeat_briers.append(test_calibration["brier_score"])

            # Store individual repeat result
            all_repeat_results.append({
                "n_train": n_train,
                "repeat_idx": repeat_idx,
                "repeat_seed": repeat_seed,
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
                "train_class_0": int(train_class_counts[0]),
                "train_class_1": int(train_class_counts[1]),
                "test_tpr": float(test_rates["tpr"]),
                "test_tnr": float(test_rates["tnr"]),
                "test_fpr": float(test_rates["fpr"]),
                "test_fnr": float(test_rates["fnr"]),
                "test_ece": float(test_calibration["ece"]),
                "test_mce": float(test_calibration["mce"]),
                "test_brier_score": float(test_calibration["brier_score"]),
            })

        # Compute mean and std across repeats
        mean_acc = np.mean(repeat_accuracies)
        std_acc = np.std(repeat_accuracies)
        mean_tpr = np.mean(repeat_tprs)
        std_tpr = np.std(repeat_tprs)
        mean_tnr = np.mean(repeat_tnrs)
        std_tnr = np.std(repeat_tnrs)
        mean_fpr = np.mean(repeat_fprs)
        std_fpr = np.std(repeat_fprs)
        mean_fnr = np.mean(repeat_fnrs)
        std_fnr = np.std(repeat_fnrs)
        mean_ece = np.mean(repeat_eces)
        std_ece = np.std(repeat_eces)
        mean_mce = np.mean(repeat_mces)
        std_mce = np.std(repeat_mces)
        mean_brier = np.mean(repeat_briers)
        std_brier = np.std(repeat_briers)

        print(f"\nResults (mean ± std over {n_repeats} repeats):")
        print(f"  Test Accuracy: {mean_acc:.1%} ± {std_acc:.1%}")
        print(f"  Test TPR: {mean_tpr:.1%} ± {std_tpr:.1%}  TNR: {mean_tnr:.1%} ± {std_tnr:.1%}")
        print(f"  Test FPR: {mean_fpr:.1%} ± {std_fpr:.1%}  FNR: {mean_fnr:.1%} ± {std_fnr:.1%}")
        print(f"  Test ECE: {mean_ece:.4f} ± {std_ece:.4f}  MCE: {mean_mce:.4f} ± {std_mce:.4f}")
        print(f"  Test Brier: {mean_brier:.4f} ± {std_brier:.4f}")

        # Store aggregated results
        results.append({
            "n_train": n_train,
            "n_repeats": n_repeats,
            "test_accuracy_mean": float(mean_acc),
            "test_accuracy_std": float(std_acc),
            "test_tpr_mean": float(mean_tpr),
            "test_tpr_std": float(std_tpr),
            "test_tnr_mean": float(mean_tnr),
            "test_tnr_std": float(std_tnr),
            "test_fpr_mean": float(mean_fpr),
            "test_fpr_std": float(std_fpr),
            "test_fnr_mean": float(mean_fnr),
            "test_fnr_std": float(std_fnr),
            "test_ece_mean": float(mean_ece),
            "test_ece_std": float(std_ece),
            "test_mce_mean": float(mean_mce),
            "test_mce_std": float(std_mce),
            "test_brier_mean": float(mean_brier),
            "test_brier_std": float(std_brier),
            "train_class_0": int(train_class_counts[0]),
            "train_class_1": int(train_class_counts[1]),
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Data Sweep Results (mean ± std)")
    print(f"{'='*80}")
    print(f"\n{'n_train':<10} {'Test Acc':<20} {'Test ECE':<20} {'Test Brier':<20}")
    print("-" * 80)
    for r in results:
        print(f"{r['n_train']:<10} "
              f"{r['test_accuracy_mean']:.1%} ± {r['test_accuracy_std']:.1%}    "
              f"{r['test_ece_mean']:.4f} ± {r['test_ece_std']:.4f}    "
              f"{r['test_brier_mean']:.4f} ± {r['test_brier_std']:.4f}")

    # Save results
    output_data = {
        "cache_file": cache_file_name,
        "task": task_name,
        "test_size": test_size,  # Deprecated - use test_fraction
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "val_fraction": 1 - train_fraction - test_fraction,
        "train_sizes": train_sizes,
        "n_repeats": n_repeats,
        "seed": seed,
        "classifier_type": classifier_type,
        "total_samples": len(X),
        "feature_dim": X.shape[1],
        "trainval_samples": len(X_trainval),
        "test_samples": len(X_test),
        "test_class_distribution": {
            "class_0": int(test_class_counts[0]),
            "class_1": int(test_class_counts[1]),
        },
        "aggregated_results": results,  # Mean and std across repeats
        "all_repeat_results": all_repeat_results,  # Individual repeat results
        "timestamp": datetime.now().isoformat(),
    }

    output_name = f"data_sweep_{task_name}_{classifier_type}_test{test_size}_repeats{n_repeats}_seed{seed}.json"
    output_path = RESULTS_DIR / output_name

    with open(str(output_path), 'w') as f:
        json.dump(output_data, f, indent=2)

    results_volume.commit()
    print(f"\nResults saved to: {output_path}")

    return output_data


@app.local_entrypoint()
def data_sweep(
    cache_file_name: str,
    task: str = "merge_action",
    test_size: int = 1024,  # Deprecated - kept for backwards compatibility
    train_sizes: str = None,  # Comma-separated list, e.g. "16,32,64,128,256,512,1024,2048"
    n_repeats: int = 10,  # Number of random subsets per training size
    seed: int = 42,
    classifier_type: str = "linear",
    mlp_hidden_dim: int = 256,
    train_fraction: float = 0.8,  # Fraction for training pool (matches ResNet)
    test_fraction: float = 0.1,  # Fraction for test set
):
    """
    Local entrypoint for data sweep with multiple repeats for error bars.

    Uses an 80/10/10 train/val/test split by default (matching ResNet),
    and automatically adds a final point using all training data.

    Usage:
        # Basic usage with default 80/10/10 split
        modal run scripts/model-post-training/modal_qwen_finetune.py::data_sweep \\
            --cache-file-name "google_siglip2-so400m-patch16-512_merge_action_all_mean_seed84_balanced.npz" \\
            --task merge_action \\
            --n-repeats 10

        # With custom training sizes
        modal run scripts/model-post-training/modal_qwen_finetune.py::data_sweep \\
            --cache-file-name "google_siglip2-so400m-patch16-512_merge_action_all_mean_seed84_balanced.npz" \\
            --train-sizes "16,32,64,128,256,512,1024,2048" \\
            --n-repeats 20

        # With MLP classifier
        modal run scripts/model-post-training/modal_qwen_finetune.py::data_sweep \\
            --cache-file-name "google_siglip2-so400m-patch16-512_merge_action_all_mean_seed84_balanced.npz" \\
            --classifier-type mlp \\
            --n-repeats 10
    """
    # Parse train_sizes if provided
    train_sizes_list = None
    if train_sizes:
        train_sizes_list = [int(x.strip()) for x in train_sizes.split(",")]

    print(f"Running data sweep")
    print(f"Cache file: {cache_file_name}")
    print(f"Task: {task}")
    print(f"Split: {train_fraction:.0%} train / {1-train_fraction-test_fraction:.0%} val / {test_fraction:.0%} test")
    print(f"Repeats per size: {n_repeats}")
    if train_sizes_list:
        print(f"Training sizes: {train_sizes_list}")

    results = linear_probe_data_sweep.remote(
        cache_file_name=cache_file_name,
        task_name=task,
        test_size=test_size,  # Kept for backwards compatibility but overridden by fractions
        train_sizes=train_sizes_list,
        n_repeats=n_repeats,
        seed=seed,
        classifier_type=classifier_type,
        mlp_hidden_dim=mlp_hidden_dim,
        train_fraction=train_fraction,
        test_fraction=test_fraction,
    )

    print("\n" + "="*60)
    print("Data Sweep Completed")
    print("="*60)

    return results


@app.local_entrypoint()
def threshold_sweep(
    probe_path: str,  # Path to probe weights relative to RESULTS_DIR
    task: str = "merge_action",
    dataset_path: str = None,
    num_thresholds: int = 101,
    all_data: bool = False,  # If True, use all data instead of val split
    seed: int = 42,  # Seed for train/val splitting
    feature_seed: int = None,  # Seed used when extracting features (for cache file lookup). Defaults to seed.
):
    """
    Sweep decision threshold on a trained probe to analyze FNR/FPR trade-off.

    Usage:
        # Basic sweep with default val split
        modal run scripts/model-post-training/modal_qwen_finetune.py::threshold_sweep \\
            --probe-path "probe_weights/google_siglip2-so400m-patch16-512_merge_action_linear_mean.npz" \\
            --task merge_action

        # Use all data (no split)
        modal run scripts/model-post-training/modal_qwen_finetune.py::threshold_sweep \\
            --probe-path "probe_weights/..." \\
            --all-data

        # Specify feature seed (if different from split seed)
        modal run scripts/model-post-training/modal_qwen_finetune.py::threshold_sweep \\
            --probe-path "probe_weights/..." \\
            --seed 86 --feature-seed 84

        # More threshold points for smoother curves
        modal run scripts/model-post-training/modal_qwen_finetune.py::threshold_sweep \\
            --probe-path "probe_weights/..." \\
            --num-thresholds 201
    """
    print(f"Running probe threshold sweep analysis")
    print(f"Probe: {probe_path}")
    print(f"Task: {task}")
    if feature_seed is not None:
        print(f"Feature seed: {feature_seed}")

    results = probe_threshold_sweep.remote(
        probe_weights_path=probe_path,
        task_name=task,
        dataset_path=dataset_path,
        num_thresholds=num_thresholds,
        use_val_split=not all_data,
        seed=seed,
        feature_seed=feature_seed,
    )

    print(f"\n{'='*60}")
    print("Summary")
    print("='*60")

    print(f"\nROC AUC: {results['roc_auc']:.3f}")
    print(f"PR AUC:  {results['pr_auc']:.3f}")

    print(f"\nKey operating points:")
    print(f"  Default (threshold=0): FNR={results['default_threshold']['fnr']:.1%}, FPR={results['default_threshold']['fpr']:.1%}")
    print(f"  Best accuracy: FNR={results['best_accuracy']['fnr']:.1%}, FPR={results['best_accuracy']['fpr']:.1%}")
    print(f"  Best F1: FNR={results['best_f1']['fnr']:.1%}, FPR={results['best_f1']['fpr']:.1%}")

    if results.get('fnr_targets'):
        print(f"\nFNR targets:")
        for name, r in results['fnr_targets'].items():
            print(f"  {name}: threshold={r['threshold']:.3f}, FPR={r['fpr']:.1%}")

    return results
