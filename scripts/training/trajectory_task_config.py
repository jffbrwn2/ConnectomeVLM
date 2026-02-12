"""
Trajectory task configuration for SFT training on human gameplay data.

This module formats human trajectories from the connectome proofreading game
as single-turn examples (observation -> action) for supervised fine-tuning.
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
import re

# Add src/ directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import TaskConfig (works both locally via environment.task_configs and in Modal via /root/task_configs.py)
try:
    from task_configs import TaskConfig
except ImportError:
    from environment.task_configs import TaskConfig


# Action type mapping from environment
ACTION_TYPES = {
    0: "move",
    1: "load_neuron",
    2: "unload_neuron",
    3: "propose_merge",
    4: "split",
    5: "goto_node",
    6: "zoom",
}

MOVE_DIRECTIONS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
    4: "forward",
    5: "backward",
}


def format_action_as_response(action: Dict[str, Any]) -> str:
    """
    Format a game action as an XML-tagged response.

    Args:
        action: Action dict from trajectory step

    Returns:
        Formatted response string with <analysis> and <answer> tags
    """
    if action is None:
        return None

    action_type = action.get("action_type", 0)
    action_name = ACTION_TYPES.get(action_type, "unknown")

    # Build analysis based on action type
    if action_name == "move":
        direction = MOVE_DIRECTIONS.get(action.get("move_direction", 0), "unknown")
        analysis = f"I will move {direction} to explore more of the neuron and find potential errors."
        answer_json = {"action": "move", "direction": direction}

    elif action_name == "zoom":
        zoom_extent = action.get("zoom_extent_nm", 10000)
        if zoom_extent > 10000:
            analysis = f"I will zoom out to {zoom_extent}nm to get a broader view of the structure."
        else:
            analysis = f"I will zoom in to {zoom_extent}nm to examine details more closely."
        answer_json = {"action": "zoom", "extent_nm": zoom_extent}

    elif action_name == "goto_node":
        node_idx = action.get("graph_node_index", 0)
        analysis = f"I will navigate to node {node_idx} to examine that region of the neuron."
        answer_json = {"action": "goto_node", "node_index": node_idx}

    elif action_name == "load_neuron":
        root_id = action.get("load_root_id", 0)
        analysis = f"I will load adjacent neuron {root_id} to examine the connection."
        answer_json = {"action": "load_neuron", "root_id": root_id}

    elif action_name == "unload_neuron":
        root_id = action.get("unload_root_id", 0)
        analysis = f"I will unload neuron {root_id} to focus on the primary structure."
        answer_json = {"action": "unload_neuron", "root_id": root_id}

    elif action_name == "propose_merge":
        root_id = action.get("merge_root_id", 0)
        analysis = f"I have identified a split error. I will propose merging segment {root_id} with the active neuron."
        answer_json = {"action": "propose_merge", "root_id": root_id}

    elif action_name == "split":
        sources = action.get("split_source_coords", [])
        sinks = action.get("split_sink_coords", [])
        analysis = f"I have identified a merge error. I will perform a split operation to separate the incorrectly merged segments."
        answer_json = {"action": "split", "source_coords": sources, "sink_coords": sinks}

    else:
        analysis = f"Taking action: {action_name}"
        answer_json = action

    response = f"""<analysis>
{analysis}
</analysis>

<answer>{json.dumps(answer_json)}</answer>"""

    return response


def format_observation_as_prompt(step_info: Dict[str, Any], step_idx: int) -> str:
    """
    Format step information as an observation prompt.

    Args:
        step_info: Step data from trajectory (includes info, position, etc.)
        step_idx: Step number in episode

    Returns:
        Formatted prompt string
    """
    position = step_info.get("position", [0, 0, 0])
    if isinstance(position, list):
        pos_str = f"({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f}) nm"
    else:
        pos_str = str(position)

    active_neuron = step_info.get("active_neuron", "unknown")
    loaded_neurons = step_info.get("loaded_neurons", [])

    # Get available nodes from info
    info = step_info.get("info", {})
    available_nodes = info.get("available_nodes", {}).get("nodes", [])

    # Get error info from render metadata
    render_metadata = info.get("render_metadata", {})
    errors = render_metadata.get("errors", [])

    # Build node description
    node_descriptions = []
    for node in available_nodes[:5]:  # Limit to first 5 nodes
        node_id = node.get("node_id", "unknown")
        distance = node.get("distance_nm", 0)
        coord = node.get("coord_nm", [0, 0, 0])
        node_descriptions.append(f"  - Node {node_id}: {distance:.0f}nm away at ({coord[0]:.0f}, {coord[1]:.0f}, {coord[2]:.0f})")

    # Build error description (for context, agent shouldn't know exact locations)
    error_summary = ""
    if errors:
        merge_errors = sum(1 for e in errors if e.get("kind") == "merge_error")
        split_errors = sum(1 for e in errors if e.get("kind") == "split_error")
        if merge_errors > 0 or split_errors > 0:
            error_summary = f"\n\nThe neuron is known to have segmentation errors that need to be identified and fixed."

    prompt = f"""You are a connectome proofreading agent. Your task is to navigate the 3D neuronal structure and identify/fix segmentation errors.

**Current State (Step {step_idx}):**
- Position: {pos_str}
- Active Neuron: {active_neuron}
- Loaded Neurons: {loaded_neurons}

**Nearby Graph Nodes:**
{chr(10).join(node_descriptions) if node_descriptions else "  (none visible)"}
{error_summary}

**Available Actions:**
- move: Move the camera in a direction (up, down, left, right, forward, backward)
- zoom: Change the zoom level (extent_nm)
- goto_node: Jump to a specific graph node by index
- load_neuron: Load an adjacent neuron segment
- unload_neuron: Unload a loaded neuron segment
- propose_merge: Propose merging an adjacent segment (fixes split errors)
- split: Perform a split operation at specific coordinates (fixes merge errors)

Analyze the current state and decide on the best action to take.

Surround your reasoning with <analysis> and </analysis> tags.
Surround your action (as JSON) with <answer> and </answer> tags."""

    return prompt


class TrajectoryTask(TaskConfig):
    """Task config for human gameplay trajectories."""

    def __init__(self, trajectory_dir: str = None):
        """
        Args:
            trajectory_dir: Path to directory containing trajectory JSON files
        """
        super().__init__(
            name="trajectory_sft",
            description="SFT on human gameplay trajectories for connectome proofreading",
            dataset_source="trajectories",  # Volume path
            dataset_config=None
        )
        self.trajectory_dir = trajectory_dir

    def load_dataset(self, cache_dir: str):
        """Load trajectory files and flatten into step-level samples."""
        from datasets import Dataset

        # Determine trajectory directory
        if self.trajectory_dir:
            traj_dir = Path(self.trajectory_dir)
        else:
            traj_dir = Path(cache_dir) / "solved-problems"

        if not traj_dir.exists():
            raise FileNotFoundError(
                f"Trajectory directory not found at {traj_dir}. "
                f"Upload it first using:\n"
                f"  modal run scripts/model-post-training/modal_trajectory_sft.py::upload_trajectories \\\n"
                f"    --local-path training_data/solved-problems"
            )

        # Load all trajectory files
        samples = []
        for traj_file in sorted(traj_dir.glob("*.json")):
            try:
                with open(traj_file) as f:
                    trajectory = json.load(f)

                steps = trajectory.get("steps", [])
                episode_metadata = {
                    "episode": trajectory.get("episode"),
                    "neuron_id": trajectory.get("neuron_id"),
                    "species": trajectory.get("species"),
                    "session_name": trajectory.get("session_name"),
                    "total_steps": trajectory.get("total_steps"),
                    "total_reward": trajectory.get("total_reward"),
                }

                # Convert each step (except initial state) to a sample
                for i, step in enumerate(steps):
                    action = step.get("action")
                    if action is None:
                        # Skip initial state (no action taken yet)
                        continue

                    sample = {
                        "step_idx": step.get("step", i),
                        "action": action,
                        "position": step.get("position"),
                        "active_neuron": step.get("active_neuron"),
                        "loaded_neurons": step.get("loaded_neurons"),
                        "adjacent_segments": step.get("adjacent_segments"),
                        "reward": step.get("reward"),
                        "info": step.get("info", {}),
                        "images": step.get("images"),  # Pre-rendered images (base64)
                        "trajectory_file": str(traj_file.name),
                        **episode_metadata,
                    }
                    samples.append(sample)

            except Exception as e:
                print(f"Warning: Failed to load {traj_file}: {e}")
                continue

        if not samples:
            raise ValueError(f"No valid trajectory samples found in {traj_dir}")

        print(f"Loaded {len(samples)} trajectory steps from {traj_dir}")
        return Dataset.from_list(samples)

    def filter_dataset(self, dataset):
        """Filter out steps with invalid actions."""
        return dataset.filter(lambda x: x['action'] is not None)

    def get_images(self, sample: Dict) -> List:
        """
        Get images for a sample.

        If images are pre-rendered (base64 encoded in the 'images' field),
        decode and return them as PIL Images.

        Returns empty list if no images are available.
        """
        images_data = sample.get('images')
        if not images_data:
            return []

        import base64
        from io import BytesIO
        from PIL import Image

        images = []
        # Load in consistent order: front, side, top
        for view in ['front', 'side', 'top']:
            if view in images_data:
                try:
                    img_bytes = base64.b64decode(images_data[view])
                    img = Image.open(BytesIO(img_bytes))
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to decode {view} image: {e}")

        return images

    def get_ground_truth(self, sample: Dict) -> Dict:
        """Return the ground truth action."""
        return sample['action']

    def format_prompt(self, sample: Dict) -> str:
        """Format the observation as a prompt."""
        return format_observation_as_prompt(sample, sample.get('step_idx', 0))

    def format_response(self, sample: Dict, use_teacher_response: bool = False,
                       teacher_response_column: str = None) -> str:
        """Format the action as a response."""
        return format_action_as_response(sample['action'])

    def format_sample_for_training(self, sample: Dict, use_teacher_response: bool = False,
                                   teacher_response_column: str = None) -> Dict:
        """Format a trajectory step for SFT training."""
        prompt_text = self.format_prompt(sample)
        response_text = self.format_response(sample)

        if response_text is None:
            return None

        # Get images if available (pre-rendered as base64)
        images = self.get_images(sample)

        # Build user message content
        user_content = []

        # Add images first (if available)
        for img in images:
            user_content.append({"type": "image", "image": img})

        # Add text prompt
        user_content.append({"type": "text", "text": prompt_text})

        messages = [{
            "role": "user",
            "content": user_content
        }, {
            "role": "assistant",
            "content": response_text
        }]

        return {
            "messages": messages,
            "images": images,  # Include for reference
            "ground_truth": sample['action'],
        }

    def create_reward_function(self) -> Callable:
        """Create reward function for trajectory actions."""

        def reward_fn(completions, ground_truth=None, **kwargs):
            if ground_truth is None:
                return [0.0] * len(completions)

            rewards = []
            for completion, gt_action in zip(completions, ground_truth):
                # Extract answer from completion
                if isinstance(completion, list):
                    completion_text = " ".join(
                        str(msg.get("content", msg)) if isinstance(msg, dict) else str(msg)
                        for msg in completion
                    )
                else:
                    completion_text = str(completion)

                # Extract JSON from <answer> tags
                answer_match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', completion_text, re.DOTALL)

                reward = 0.0
                if answer_match:
                    try:
                        predicted_action = json.loads(answer_match.group(1))

                        # Compare action types
                        gt_type = gt_action.get("action_type", -1)
                        pred_type_name = predicted_action.get("action", "")

                        # Map action name to type
                        reverse_action_types = {v: k for k, v in ACTION_TYPES.items()}
                        pred_type = reverse_action_types.get(pred_type_name, -2)

                        if gt_type == pred_type:
                            reward = 0.5  # Correct action type

                            # Bonus for correct parameters
                            if pred_type == 0:  # move
                                gt_dir = gt_action.get("move_direction", -1)
                                pred_dir_name = predicted_action.get("direction", "")
                                reverse_dirs = {v: k for k, v in MOVE_DIRECTIONS.items()}
                                if reverse_dirs.get(pred_dir_name, -2) == gt_dir:
                                    reward = 1.0
                            elif pred_type == 6:  # zoom
                                if abs(predicted_action.get("extent_nm", 0) - gt_action.get("zoom_extent_nm", 0)) < 1000:
                                    reward = 1.0
                            else:
                                reward = 1.0  # Other actions: type match is sufficient

                    except json.JSONDecodeError:
                        pass

                # Bonus for including analysis
                if '<analysis>' in completion_text.lower() and '</analysis>' in completion_text.lower():
                    reward += 0.1

                rewards.append(min(1.0, reward))

            return rewards

        return reward_fn
