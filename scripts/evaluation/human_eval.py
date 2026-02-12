#!/usr/bin/env python3
"""
Human Evaluation Pipeline for Connectome Proofreading Tasks.

This GUI allows human evaluators to:
1. Select a task from the data inventory
2. View the same images shown to the finetuned model
3. Answer the same questions
4. Optionally evaluate model reasoning faithfulness
5. Save results with user IDs and timestamps

Usage:
    python scripts/analysis/human_eval.py

    # Specify a user ID
    python scripts/analysis/human_eval.py --user-id "evaluator1"

    # Run faithfulness evaluation mode
    python scripts/analysis/human_eval.py --mode faithfulness
"""

import json
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageDraw
from dataclasses import dataclass, field, asdict
import pandas as pd

try:
    import gradio as gr
except ImportError:
    print("Error: Gradio is not installed.")
    print("Please install it with: pip install gradio")
    print("Or: pixi add --pypi gradio")
    exit(1)

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.task_configs import (
    SplitActionTask,
    MergeActionTask,
    MergeErrorIdentificationTask,
    EndpointErrorIdentificationTask,
    EndpointErrorIdentificationWithEMTask,
)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "training_data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation_results" / "human_eval"
DEFAULT_INVENTORY_PATH = PROJECT_ROOT / "evaluation_results" / "final_data" / "data_inventory.json"


# Task configuration mapping
# Note: split-correction-proposal is excluded as it requires coordinate prediction
# which is not practical for human evaluation without specialized tools
TASK_CONFIGS = {
    "split-error-identification": EndpointErrorIdentificationTask,  # endpoints-parquet
    "endpoint_error_identification_with_em": EndpointErrorIdentificationWithEMTask,  # endpoints-with-em-parquet (6 images: 3 mesh + 3 EM)
    "split-error-correction": MergeActionTask,  # merge-parquet
    "merge-error-identification": MergeErrorIdentificationTask,  # merge-error-identification-parquet
    "split-correction-evaluation": SplitActionTask,  # splits-parquet
    # "split-correction-proposal": excluded - requires coordinate prediction
}

# Human-friendly task names
TASK_DISPLAY_NAMES = {
    "split-error-identification": "Split Error Identification (Endpoint Analysis)",
    "endpoint_error_identification_with_em": "Endpoint Error Identification with EM Context",
    "split-error-correction": "Split Error Correction (Merge Decision)",
    "merge-error-identification": "Merge Error Identification",
    "split-correction-evaluation": "Split Correction Evaluation",
}

# Task answer types
TASK_ANSWER_TYPES = {
    "split-error-identification": "yes_no",  # yes = split error, no = natural terminus
    "endpoint_error_identification_with_em": "yes_no",  # yes = split error, no = natural terminus
    "split-error-correction": "yes_no",  # yes = good merge, no = bad merge
    "merge-error-identification": "yes_no",  # yes = merge error, no = valid
    "split-correction-evaluation": "yes_no",  # yes = good split, no = bad split
}


@dataclass
class EvaluationResponse:
    """A single human evaluation response."""
    sample_idx: int
    parquet_idx: int
    user_id: str
    task_name: str
    human_answer: Any
    ground_truth: Any
    correct: bool
    response_time_seconds: float
    timestamp: str
    image_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For faithfulness evaluation
    model_response: Optional[str] = None
    faithfulness_rating: Optional[int] = None  # 1-5 scale
    faithfulness_notes: Optional[str] = None


@dataclass
class EvaluationSession:
    """Tracks a human evaluation session."""
    user_id: str
    task_name: str
    session_id: str
    start_time: str
    responses: List[EvaluationResponse] = field(default_factory=list)
    num_samples: int = 0
    num_correct: int = 0

    def add_response(self, response: EvaluationResponse):
        self.responses.append(response)
        self.num_samples += 1
        if response.correct:
            self.num_correct += 1

    @property
    def accuracy(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.num_correct / self.num_samples

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "task_name": self.task_name,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
            "responses": [asdict(r) for r in self.responses],
        }


class HumanEvaluator:
    """Human evaluation interface for connectome proofreading tasks."""

    def __init__(
        self,
        data_dir: Path,
        results_dir: Path,
        inventory_path: Path,
        default_user_id: str = "anonymous",
        mode: str = "task",  # "task" or "faithfulness"
        num_samples: int = 32,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.inventory_path = Path(inventory_path)
        self.default_user_id = default_user_id
        self.mode = mode
        self.num_samples = num_samples
        self.seed = seed

        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load data inventory
        self.inventory = self._load_inventory()

        # Current state
        self.current_task: Optional[str] = None
        self.current_task_config = None
        self.current_dataset = None
        self.current_samples: List[Dict] = []
        self.current_sample_idx: int = 0
        self.sample_start_time: Optional[datetime] = None

        # Session tracking
        self.session: Optional[EvaluationSession] = None

        # Model results for faithfulness evaluation
        self.model_results: Optional[Dict] = None

    def _load_inventory(self) -> Dict:
        """Load the data inventory."""
        if not self.inventory_path.exists():
            print(f"Warning: Inventory not found at {self.inventory_path}")
            return {"mouse": {}}

        with open(self.inventory_path) as f:
            return json.load(f)

    def get_available_tasks(self) -> List[str]:
        """Get list of tasks that have data available."""
        available = []
        for task_key in TASK_CONFIGS.keys():
            if task_key in self.inventory.get("mouse", {}):
                available.append(task_key)
        return available

    def load_task(self, task_name: str, user_id: str) -> str:
        """Load a task and prepare samples for evaluation."""
        if task_name not in TASK_CONFIGS:
            return f"Unknown task: {task_name}"

        self.current_task = task_name

        # Instantiate task config
        task_class = TASK_CONFIGS[task_name]
        self.current_task_config = task_class()

        # Load dataset
        try:
            dataset = self.current_task_config.load_dataset(
                cache_dir=str(self.data_dir),
            )
            # Apply task-specific filtering
            dataset = self.current_task_config.filter_dataset(dataset)
        except FileNotFoundError as e:
            return f"Dataset not found: {e}"
        except Exception as e:
            return f"Error loading dataset: {e}"

        self.current_dataset = dataset

        # Sample indices for evaluation
        random.seed(self.seed)
        total_samples = len(dataset)
        sample_count = min(self.num_samples, total_samples)
        sample_indices = random.sample(range(total_samples), sample_count)

        # Prepare samples
        self.current_samples = []
        for idx in sample_indices:
            sample = dataset[idx]
            self.current_samples.append({
                "dataset_idx": idx,
                "sample": sample,
            })

        self.current_sample_idx = 0
        self.sample_start_time = datetime.now()

        # Create new session
        session_id = hashlib.md5(
            f"{user_id}_{task_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        self.session = EvaluationSession(
            user_id=user_id,
            task_name=task_name,
            session_id=session_id,
            start_time=datetime.now().isoformat(),
        )

        # Load model results if in faithfulness mode
        if self.mode == "faithfulness":
            self._load_model_results(task_name)

        return f"Loaded {sample_count} samples for {TASK_DISPLAY_NAMES.get(task_name, task_name)}"

    def _load_model_results(self, task_name: str):
        """Load model results for faithfulness evaluation."""
        # Look for finetuned model results in inventory
        task_data = self.inventory.get("mouse", {}).get(task_name, {})
        finetuned_path = task_data.get("Finetuned Qwen3-VL-32B-Instruct")

        if finetuned_path and Path(PROJECT_ROOT / finetuned_path).exists():
            with open(PROJECT_ROOT / finetuned_path) as f:
                self.model_results = json.load(f)
            print(f"Loaded model results from {finetuned_path}")
        else:
            self.model_results = None
            print("No model results available for faithfulness evaluation")

    def get_current_sample(self) -> Tuple[List[Image.Image], str, str, Dict]:
        """Get the current sample's images, prompt, and metadata."""
        if not self.current_samples or self.current_sample_idx >= len(self.current_samples):
            return [], "No more samples", "", {}

        sample_data = self.current_samples[self.current_sample_idx]
        sample = sample_data["sample"]

        # Add base path for image loading
        sample_with_path = dict(sample)
        sample_with_path["_base_path"] = self.data_dir / self.current_task_config.dataset_source

        # Load images
        try:
            images = self.current_task_config.get_images(sample_with_path)
        except Exception as e:
            print(f"Error loading images: {e}")
            images = [self._create_placeholder_image(f"Error loading images:\n{e}")]

        # Get prompt
        prompt = self.current_task_config.format_prompt(sample, answer_only=True)

        # Get model response if available (for faithfulness mode)
        model_response = ""
        if self.mode == "faithfulness" and self.model_results:
            predictions = self.model_results.get("predictions", [])
            # Try to find matching prediction
            parquet_idx = sample.get("_original_parquet_idx", sample_data["dataset_idx"])
            for pred in predictions:
                if pred.get("sample_idx") == parquet_idx:
                    model_response = pred.get("response", "")
                    break

        # Build info dict
        info = {
            "sample_num": self.current_sample_idx + 1,
            "total_samples": len(self.current_samples),
            "dataset_idx": sample_data["dataset_idx"],
            "parquet_idx": sample.get("_original_parquet_idx", sample_data["dataset_idx"]),
        }

        # Start timer
        self.sample_start_time = datetime.now()

        return images, prompt, model_response, info

    def submit_answer(
        self,
        answer: str,
        faithfulness_rating: Optional[int] = None,
        faithfulness_notes: str = "",
    ) -> Tuple[bool, str, str]:
        """Submit an answer for the current sample.

        Returns: (correct, feedback_message, next_action)
        """
        if not self.current_samples or self.current_sample_idx >= len(self.current_samples):
            return False, "No active sample", "done"

        # Calculate response time
        response_time = 0.0
        if self.sample_start_time:
            response_time = (datetime.now() - self.sample_start_time).total_seconds()

        sample_data = self.current_samples[self.current_sample_idx]
        sample = sample_data["sample"]

        # Parse and evaluate answer
        ground_truth = self.current_task_config.get_ground_truth(sample)
        human_answer, correct = self._evaluate_answer(answer, ground_truth)

        # Get image paths
        try:
            sample_with_path = dict(sample)
            sample_with_path["_base_path"] = self.data_dir / self.current_task_config.dataset_source
            image_paths = self.current_task_config.get_image_paths(sample_with_path)
        except Exception:
            image_paths = []

        # Create response record
        response = EvaluationResponse(
            sample_idx=self.current_sample_idx,
            parquet_idx=sample.get("_original_parquet_idx", sample_data["dataset_idx"]),
            user_id=self.session.user_id,
            task_name=self.current_task,
            human_answer=human_answer,
            ground_truth=ground_truth,
            correct=correct,
            response_time_seconds=response_time,
            timestamp=datetime.now().isoformat(),
            image_paths=image_paths,
            metadata=dict(sample.get("metadata", {})) if isinstance(sample.get("metadata"), dict) else {},
            faithfulness_rating=faithfulness_rating,
            faithfulness_notes=faithfulness_notes if faithfulness_notes else None,
        )

        # Add model response if in faithfulness mode
        if self.mode == "faithfulness" and self.model_results:
            predictions = self.model_results.get("predictions", [])
            parquet_idx = sample.get("_original_parquet_idx", sample_data["dataset_idx"])
            for pred in predictions:
                if pred.get("sample_idx") == parquet_idx:
                    response.model_response = pred.get("response", "")
                    break

        self.session.add_response(response)

        # Format feedback
        if correct:
            feedback = f"Correct! Ground truth was: {self._format_ground_truth(ground_truth)}"
        else:
            feedback = f"Incorrect. Ground truth was: {self._format_ground_truth(ground_truth)}, you answered: {human_answer}"

        # Move to next sample
        self.current_sample_idx += 1

        if self.current_sample_idx >= len(self.current_samples):
            # Session complete - save results
            self._save_session()
            return correct, feedback, "done"

        return correct, feedback, "next"

    def _evaluate_answer(self, answer: str, ground_truth: Any) -> Tuple[Any, bool]:
        """Evaluate the human's answer against ground truth."""
        answer_type = TASK_ANSWER_TYPES.get(self.current_task, "yes_no")

        if answer_type == "yes_no":
            # Normalize answer
            answer_lower = answer.lower().strip()
            if answer_lower in ["yes", "y", "true", "1"]:
                human_answer = True
            elif answer_lower in ["no", "n", "false", "0"]:
                human_answer = False
            else:
                human_answer = None

            correct = human_answer == ground_truth
            return human_answer, correct

        elif answer_type == "multiple_choice":
            human_answer = answer.lower().strip()
            correct = human_answer == str(ground_truth).lower()
            return human_answer, correct

        return answer, False

    def _format_ground_truth(self, ground_truth: Any) -> str:
        """Format ground truth for display."""
        if isinstance(ground_truth, bool):
            return "yes" if ground_truth else "no"
        return str(ground_truth)

    def _save_session(self):
        """Save the evaluation session to disk."""
        if not self.session:
            return

        filename = f"human_eval_{self.session.task_name}_{self.session.user_id}_{self.session.session_id}.json"
        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.session.to_dict(), f, indent=2, default=str)

        print(f"Saved session to {filepath}")

    def get_session_stats(self, show_accuracy: bool = False) -> str:
        """Get current session statistics.

        Args:
            show_accuracy: If True, show accuracy (only at end of session)
        """
        if not self.session:
            return "No active session"

        stats = f"""**Session Info**
- User: {self.session.user_id}
- Task: {TASK_DISPLAY_NAMES.get(self.session.task_name, self.session.task_name)}
- Completed: {self.session.num_samples}
"""
        if show_accuracy:
            stats += f"""- Correct: {self.session.num_correct}
- Accuracy: {self.session.accuracy:.1%}
"""
        return stats

    def skip_sample(self) -> str:
        """Skip the current sample without answering."""
        self.current_sample_idx += 1
        if self.current_sample_idx >= len(self.current_samples):
            self._save_session()
            return "done"
        return "next"

    def _create_placeholder_image(self, text: str) -> Image.Image:
        """Create a placeholder image with text."""
        img = Image.new('RGB', (512, 384), color=(40, 40, 40))
        draw = ImageDraw.Draw(img)
        lines = text.split('\n')
        y = 150
        for line in lines:
            draw.text((30, y), line, fill=(200, 200, 200))
            y += 25
        return img


def create_interface(
    data_dir: Path,
    results_dir: Path,
    inventory_path: Path,
    default_user_id: str,
    mode: str,
    num_samples: int,
) -> gr.Blocks:
    """Create the Gradio interface."""

    evaluator = HumanEvaluator(
        data_dir=data_dir,
        results_dir=results_dir,
        inventory_path=inventory_path,
        default_user_id=default_user_id,
        mode=mode,
        num_samples=num_samples,
    )

    with gr.Blocks(
        title="Human Evaluation - Connectome Proofreading",
    ) as app:

        # Header
        gr.Markdown("""
# Human Evaluation Pipeline
### Connectome Proofreading Tasks

Evaluate your ability on the same tasks used to benchmark AI models.
        """)

        # State variables
        current_images = gr.State([])

        with gr.Row():
            # Left panel - Setup and Stats
            with gr.Column(scale=1):
                gr.Markdown("### Setup")

                user_id_input = gr.Textbox(
                    label="Your User ID",
                    value=default_user_id,
                    placeholder="Enter your name or ID",
                )

                available_tasks = evaluator.get_available_tasks()
                task_dropdown = gr.Dropdown(
                    choices=available_tasks,
                    label="Select Task",
                    interactive=True,
                    info="Choose a proofreading task to evaluate",
                )

                start_btn = gr.Button("Start Evaluation", variant="primary", size="lg")

                status_display = gr.Markdown("Select a task and click Start to begin.")

                gr.Markdown("---")
                gr.Markdown("### Progress")
                stats_display = gr.Markdown("No active session")

                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Progress",
                    interactive=False,
                )

            # Right panel - Evaluation interface
            with gr.Column(scale=3):
                # Images display
                gr.Markdown("### Sample Images")
                gr.Markdown("**Mesh Views**")
                with gr.Row():
                    img1 = gr.Image(label="View 1 (Front)", type="pil", height=300)
                    img2 = gr.Image(label="View 2 (Side)", type="pil", height=300)
                    img3 = gr.Image(label="View 3 (Top)", type="pil", height=300)
                gr.Markdown("**EM Views** (only shown for EM-enabled tasks)")
                with gr.Row():
                    img4 = gr.Image(label="EM Front", type="pil", height=300)
                    img5 = gr.Image(label="EM Side", type="pil", height=300)
                    img6 = gr.Image(label="EM Top", type="pil", height=300)

                # Question/Prompt
                gr.Markdown("### Question")
                prompt_display = gr.Textbox(
                    label="Task Prompt",
                    lines=6,
                    interactive=False,
                    elem_classes=["prompt-box"],
                )

                # Model response (for faithfulness mode)
                with gr.Accordion("Model Response (Faithfulness Mode)", open=False, visible=(mode == "faithfulness")) as model_response_accordion:
                    model_response_display = gr.Textbox(
                        label="Model's Response",
                        lines=8,
                        interactive=False,
                    )
                    faithfulness_rating = gr.Radio(
                        choices=[("1 - Unfaithful", 1), ("2", 2), ("3 - Neutral", 3), ("4", 4), ("5 - Faithful", 5)],
                        label="Faithfulness Rating",
                        info="How well does the model's reasoning reflect what's actually in the images?",
                    )
                    faithfulness_notes = gr.Textbox(
                        label="Notes (optional)",
                        placeholder="Any observations about the model's reasoning...",
                        lines=2,
                    )

                # Answer input
                gr.Markdown("### Your Answer")
                with gr.Row():
                    answer_input = gr.Radio(
                        choices=[("Yes", "yes"), ("No", "no")],
                        label="Select your answer",
                        interactive=True,
                    )

                with gr.Row():
                    submit_btn = gr.Button("Submit Answer", variant="primary", size="lg")
                    skip_btn = gr.Button("Skip", variant="secondary")

                # Feedback
                feedback_display = gr.Markdown("")

                # Sample info
                sample_info = gr.Markdown("")

        # Helper to pad images list to exactly 6
        def pad_images(images):
            """Ensure we have exactly 6 images (or None placeholders)."""
            result = list(images) if images else []
            while len(result) < 6:
                result.append(None)
            return result[:6]

        # Event handlers
        def on_start(task_name, user_id):
            """Start a new evaluation session."""
            print(f"[DEBUG] on_start called with task_name={task_name!r}, user_id={user_id!r}")

            try:
                if not task_name:
                    print("[DEBUG] No task_name provided")
                    return (
                        "Please select a task first.",
                        "No active session",
                        0,
                        None, None, None, None, None, None,
                        "",
                        "",
                        "",
                        [],
                    )

                status = evaluator.load_task(task_name, user_id)
                print(f"[DEBUG] load_task returned: {status!r}")

                if "Error" in status or "not found" in status.lower():
                    return (
                        status,
                        "No active session",
                        0,
                        None, None, None, None, None, None,
                        "",
                        "",
                        "",
                        [],
                    )

                # Get first sample
                images, prompt, model_response, info = evaluator.get_current_sample()
                print(f"[DEBUG] get_current_sample returned {len(images)} images, info={info}")

                images = pad_images(images)
                print(f"[DEBUG] After padding: {[type(img).__name__ if img else None for img in images]}")

                progress = (info["sample_num"] / info["total_samples"]) * 100 if info.get("total_samples") else 0
                sample_info_text = f"**Sample {info.get('sample_num', 1)} of {info.get('total_samples', '?')}**"

                print(f"[DEBUG] Returning images to Gradio: {[img.size if img else None for img in images]}")
                return (
                    status,
                    evaluator.get_session_stats(show_accuracy=False),
                    progress,
                    images[0], images[1], images[2], images[3], images[4], images[5],
                    prompt,
                    model_response,
                    sample_info_text,
                    images,
                )

            except Exception as e:
                import traceback
                print(f"[DEBUG] EXCEPTION in on_start: {e}")
                traceback.print_exc()
                return (
                    f"Error: {e}",
                    "Error occurred",
                    0,
                    None, None, None, None, None, None,
                    str(e),
                    "",
                    "",
                    [],
                )

        def on_submit(answer, faith_rating, faith_notes, images_state):
            """Submit the current answer."""
            if not answer:
                images = pad_images(images_state)
                return (
                    "Please select an answer.",
                    evaluator.get_session_stats(show_accuracy=False),
                    gr.Slider(),
                    images[0], images[1], images[2], images[3], images[4], images[5],
                    gr.Textbox(),
                    gr.Textbox(),
                    gr.Markdown(),
                    images,
                    gr.Radio(),  # Reset answer selection
                )

            correct, feedback, next_action = evaluator.submit_answer(
                answer,
                faithfulness_rating=faith_rating,
                faithfulness_notes=faith_notes,
            )

            # Style feedback (don't reveal correctness during eval)
            feedback_styled = f"Answer recorded: **{answer}**"

            if next_action == "done":
                # Session complete - NOW show accuracy
                final_stats = evaluator.get_session_stats(show_accuracy=True)
                return (
                    f"**Session Complete!** Results have been saved.\n\n{final_stats}",
                    final_stats,
                    100,
                    None, None, None, None, None, None,
                    "Session complete. Thank you for your evaluation!",
                    "",
                    "**Evaluation Complete**",
                    [],
                    None,  # Reset answer selection
                )

            # Get next sample
            images, prompt, model_response, info = evaluator.get_current_sample()
            images = pad_images(images)

            progress = (info["sample_num"] / info["total_samples"]) * 100 if info.get("total_samples") else 0
            sample_info_text = f"**Sample {info.get('sample_num', 1)} of {info.get('total_samples', '?')}**"

            return (
                feedback_styled,
                evaluator.get_session_stats(show_accuracy=False),
                progress,
                images[0], images[1], images[2], images[3], images[4], images[5],
                prompt,
                model_response,
                sample_info_text,
                images,
                None,  # Reset answer selection
            )

        def on_skip(images_state):
            """Skip current sample."""
            next_action = evaluator.skip_sample()

            if next_action == "done":
                final_stats = evaluator.get_session_stats(show_accuracy=True)
                return (
                    f"Session Complete! Results have been saved.\n\n{final_stats}",
                    final_stats,
                    100,
                    None, None, None, None, None, None,
                    "Session complete.",
                    "",
                    "**Evaluation Complete**",
                    [],
                )

            # Get next sample
            images, prompt, model_response, info = evaluator.get_current_sample()
            images = pad_images(images)

            progress = (info["sample_num"] / info["total_samples"]) * 100 if info.get("total_samples") else 0
            sample_info_text = f"**Sample {info.get('sample_num', 1)} of {info.get('total_samples', '?')}**"

            return (
                "Sample skipped.",
                evaluator.get_session_stats(show_accuracy=False),
                progress,
                images[0], images[1], images[2], images[3], images[4], images[5],
                prompt,
                model_response,
                sample_info_text,
                images,
            )

        # Wire up events
        start_btn.click(
            on_start,
            inputs=[task_dropdown, user_id_input],
            outputs=[
                status_display,
                stats_display,
                progress_bar,
                img1, img2, img3, img4, img5, img6,
                prompt_display,
                model_response_display,
                sample_info,
                current_images,
            ],
        )

        submit_btn.click(
            on_submit,
            inputs=[answer_input, faithfulness_rating, faithfulness_notes, current_images],
            outputs=[
                feedback_display,
                stats_display,
                progress_bar,
                img1, img2, img3, img4, img5, img6,
                prompt_display,
                model_response_display,
                sample_info,
                current_images,
                answer_input,
            ],
        )

        skip_btn.click(
            on_skip,
            inputs=[current_images],
            outputs=[
                feedback_display,
                stats_display,
                progress_bar,
                img1, img2, img3, img4, img5, img6,
                prompt_display,
                model_response_display,
                sample_info,
                current_images,
            ],
        )

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Human Evaluation Pipeline for Connectome Proofreading Tasks"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing training data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory to save evaluation results (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help=f"Path to data inventory JSON (default: {DEFAULT_INVENTORY_PATH})",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="anonymous",
        help="Default user ID for the evaluator",
    )
    parser.add_argument(
        "--mode",
        choices=["task", "faithfulness"],
        default="task",
        help="Evaluation mode: 'task' for standard evaluation, 'faithfulness' to also rate model reasoning",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples per evaluation session (default: 50)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the server on (default: 7861)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link",
    )

    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Mode: {args.mode}")
    print(f"Samples per session: {args.num_samples}")

    app = create_interface(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        inventory_path=args.inventory,
        default_user_id=args.user_id,
        mode=args.mode,
        num_samples=args.num_samples,
    )

    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
