"""
Problem specification for ConnectomeEnv initialization.

This module provides a dataclass that mirrors all ConnectomeEnv initialization
parameters, ensuring consistent environment configuration across all runners
and training scripts.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Sequence, Any, Dict
from pathlib import Path
import json

from environment.connectome_env import ConnectomeEnv


@dataclass
class ProblemSpec:
    """
    Specification for a ConnectomeEnv problem instance.

    All parameters mirror ConnectomeEnv.__init__() to ensure consistent
    initialization. Required parameters must be provided; optional parameters
    use ConnectomeEnv defaults.
    """
    # Required parameters
    neuron_id: int

    # Optional parameters with ConnectomeEnv defaults
    species: str = "fly"
    dataset: str = "public"
    timestamp: Optional[int] = None
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    spawn_seed: Optional[int] = None
    window_size_nm: int = 512
    max_steps: int = 100
    render_size: Tuple[int, int] = (512, 512)
    views: List[str] = field(default_factory=lambda: ['front', 'side', 'top'])
    render_mode: Optional[str] = None
    distance_reward_threshold_nm: float = 10_000.0
    merge_reward_threshold: float = 0.5
    merge_reward_radius_nm: float = 5_000.0
    correct_action_reward_threshold_nm: float = 5_000.0
    save_images: bool = False
    save_logs: bool = True
    session_name: Optional[str] = None
    log_level: str = "INFO"
    observation_mode: str = "mesh_only"
    view_extent_nm: float = 16_384.0
    em_window_size_nm: Optional[int] = 512
    show_projection_legend: bool = True
    mesh_crop_enabled: bool = True
    split_window_size_nm: int = 7_500
    split_point_tolerance_nm: int = 300
    split_reward_threshold: float = 0.5
    max_split_points: int = 6
    node_count_limit: int = 20
    allowed_corrections: str = "both"  # "both", "merge_only", or "split_only"

    def __post_init__(self):
        # check that one of spawn_seed or initial_position (x, y, z) is provided
        initial_position = (self.x, self.y, self.z) if self.x is not None and self.y is not None and self.z is not None else None
        assert (self.spawn_seed is not None) ^ (initial_position is not None), "Exactly one of spawn_seed or initial_position (x, y, z) must be provided"
        self.initial_position = initial_position

    def to_env(self) -> ConnectomeEnv:
        """
        Convert ProblemSpec to ConnectomeEnv instance.
        """
        return ConnectomeEnv(**self.to_env_kwargs())

    def to_env_kwargs(self) -> Dict[str, Any]:
        """
        Convert ProblemSpec to kwargs dict for ConnectomeEnv initialization.

        Returns:
            Dict of all parameters suitable for **kwargs unpacking
        """
        return asdict(self)

    def to_dict_minimal(self) -> Dict[str, Any]:
        """
        Convert to dict with only non-default values for cleaner JSON export.

        Returns:
            Dict containing only parameters that differ from defaults
        """
        default_spec = ProblemSpec(neuron_id=0, spawn_seed=0)  # Create with defaults (spawn_seed required for validation)
        current_dict = asdict(self)
        default_dict = asdict(default_spec)

        # Include only values that differ from defaults or are required
        minimal = {}
        for key, value in current_dict.items():
            # Always include neuron_id (required)
            if key == 'neuron_id':
                minimal[key] = value
            # Include if different from default
            elif value != default_dict.get(key):
                minimal[key] = value

        return minimal

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProblemSpec":
        """
        Create ProblemSpec from a dictionary.

        Args:
            data: Dictionary with problem parameters

        Returns:
            ProblemSpec instance
        """
        # Filter to only include fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_path: Path) -> List["ProblemSpec"]:
        """
        Load a list of ProblemSpecs from a JSON file.

        Args:
            json_path: Path to JSON file containing list of problem dicts

        Returns:
            List of ProblemSpec instances
        """
        with open(json_path) as f:
            problems_data = json.load(f)

        if not isinstance(problems_data, list):
            raise ValueError("JSON must contain a list of problem specifications")

        return [cls.from_dict(p) for p in problems_data]

    def to_json(self, json_path: Path) -> None:
        """
        Save this ProblemSpec to a JSON file.

        Args:
            json_path: Path to output JSON file
        """
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def save_list(problems: List["ProblemSpec"], json_path: Path, minimal: bool = False) -> None:
        """
        Save a list of ProblemSpecs to a JSON file.

        Args:
            problems: List of ProblemSpec instances
            json_path: Path to output JSON file
            minimal: If True, save only non-default values for cleaner output
        """
        with open(json_path, 'w') as f:
            if minimal:
                json.dump([p.to_dict_minimal() for p in problems], f, indent=2)
            else:
                json.dump([asdict(p) for p in problems], f, indent=2)
