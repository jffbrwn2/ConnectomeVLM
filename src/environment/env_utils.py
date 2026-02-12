import logging,os, json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np


class ActionError(Exception):
    """Raised when the agent supplies an invalid action."""

def setup_logging(env: "ConnectomeEnv", log_level: str) -> logging.Logger:
    logger = logging.getLogger(f"ConnectomeEnv_{env.neuron_id}_{env.session_name}")
    lvl = getattr(logging, log_level.upper())
    logger.setLevel(lvl)
    logger.propagate = False

    # nuke old handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console)

    # file handler
    if env.save_images and env.session_dir:
        log_path = os.path.join(env.session_dir, "environment.log")
        f = logging.FileHandler(log_path)
        f.setLevel(lvl)
        f.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(funcName)s:%(lineno)d - %(message)s'
        ))
        logger.addHandler(f)

    logger.log(lvl, f"logging level set to {log_level.upper()}")
    return logger

def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np

@dataclass
class EnvStepLog:
    # core info
    session_name: str
    episode: int
    step: int
    active_neuron: int
    loaded_neurons: List[int]
    position: List[float]

    render_metadata: Dict[str, Any]

    # action related
    action: Dict[str, Any]
    action_context: Optional[Dict[str, Any]]
    
    # environment connectome state (optional, expensive to compute and can be derived)
    nodes_metadata: Optional[Dict[str, Any]]
    adjacent_segments: Optional[List[int]]
    errors_metadata: Optional[Dict[str, Any]]      
    
    # termination/reward
    terminated: bool
    truncated: bool
    max_steps: int
    reward: Optional[float] # can be derived
    distance_to_error: Optional[float] # can be derived
    
    reasoning: Optional[str] = None, # added in 1.1

    SCHEMA_VERSION: int = 1.1

    def __post_init__(self):
        if self.SCHEMA_VERSION not in [1, 1.1]:
            raise ValueError(f"Invalid schema version: {self.SCHEMA_VERSION}")

    @classmethod
    def from_env(
        cls,
        env: "ConnectomeEnv",
        observation: Dict,
        action_context: Dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        action: Optional[Dict[str, Any]] = None,
    ) -> "EnvStepLog":
        pos = observation["position"]
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()

        return cls(
            session_name=env.session_name,
            episode=env.episode_count,
            step=env.current_step,
            active_neuron=int(observation["active_neuron"]),
            loaded_neurons=[int(n) for n in observation["loaded_neurons"]],
            position=pos,
            render_metadata=getattr(env, "_last_render_metadata", {}),
            nodes_metadata=getattr(env, "last_available_nodes", {}),
            adjacent_segments=[int(n) for n in observation["adjacent_segments"]],
            errors_metadata=getattr(env, "_last_available_errors", {}),
            terminated=bool(terminated),
            truncated=bool(truncated),
            max_steps=env.max_steps,
            reward=float(reward),
            distance_to_error=(
                float(env._last_distance_to_error)
                if getattr(env, "_last_distance_to_error", None) is not None
                else None
            ),
            action=action,
            action_context=action_context,

            reasoning=getattr(env, "_step_reasoning", None),
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return _convert_to_json_serializable(asdict(self))

@dataclass
class EpisodeLog:
    session_name: str
    episode: int
    steps: List[EnvStepLog]
    error: Optional[str] = None
    
    def to_json_dict(self) -> Dict[str, Any]:
        self.steps = [step.to_json_dict() for step in self.steps]
        return _convert_to_json_serializable(asdict(self))

def log_step(
    env : "ConnectomeEnv",
    observation: Dict,
    action_context: Dict,
    reward: float,
    terminated: bool,
    truncated: bool,
    action: Optional[Dict[str, Any]] = None
):
    """Log a single step to memory and optionally to disk."""
    entry = EnvStepLog.from_env(env, observation, action_context, reward, terminated, truncated, action)
    step_data = entry.to_json_dict()

    # Clear reasoning after logging
    env._step_reasoning = None
    
    env.current_episode_steps.append(step_data)

    # Log key information
    if action is not None:
        action_type = action.get('action_type', 0)
        if action_type == 0:
            move_target = action.get('move_target')
            if move_target:
                env.logger.info(f"Step {env.current_step}: Move to {move_target}, Reward: {reward:.3f}")
            else:
                env.logger.info(f"Step {env.current_step}: Move (no target), Reward: {reward:.3f}")
        elif action_type == 1:
            merge_id = action.get('merge_root_id', 0)
            env.logger.info(f"Step {env.current_step}: Merge with {merge_id}, Reward: {reward:.3f}")
        elif action_type == 4:
            load_id = action.get('load_root_id', 0)
            env.logger.info(f"Step {env.current_step}: Load neuron {load_id}, Reward: {reward:.3f}")
        elif action_type == 5:
            unload_id = action.get('unload_root_id', 0)
            env.logger.info(f"Step {env.current_step}: Unload neuron {unload_id}, Reward: {reward:.3f}")

    if terminated or truncated:
        total_reward = sum(s['reward'] for s in env.current_episode_steps)
        env.logger.info(f"Episode {env.episode_count} ended: Total reward: {total_reward:.3f}, Steps: {env.current_step}")

def save_episode(env : "ConnectomeEnv"):
    """Save episode data to disk."""
    if not env.save_logs or not env.session_dir:
        return

    episode_data = {
        'episode': env.episode_count,
        'session_name': env.session_name,
        'session_timestamp': env.session_timestamp,
        'neuron_id': env.neuron_id,
        'species': env.species,
        'dataset': env.dataset,
        'total_steps': len(env.current_episode_steps),
        'total_reward': sum(s['reward'] for s in env.current_episode_steps),
        'steps': env.current_episode_steps,
        'config': {
            'em_window_size_nm': env.em_window_size_nm,
            'max_steps': env.max_steps,
            'reward_config' : asdict(env.reward_config),
            'distance_reward_threshold_nm': env.distance_reward_threshold_nm,
            'merge_reward_threshold': env.merge_reward_threshold,
            'merge_reward_radius_nm': env.merge_reward_radius_nm,
        }
    }
    episode_data=_convert_to_json_serializable(episode_data)

    env.episode_history.append(episode_data)

    # Save individual episode file in episodes subdirectory
    episodes_dir = os.path.join(env.session_dir, "episodes")
    Path(episodes_dir).mkdir(exist_ok=True)
    episode_file = os.path.join(episodes_dir, f"episode_{env.episode_count:03d}.json")
    with open(episode_file, 'w') as f:
        json.dump(episode_data, f, indent=2)

    env.logger.debug(f"Saved episode {env.episode_count} data to {episode_file}")


def coerce_coord_sequence(env: "ConnectomeEnv", value: Any, field_name: str) -> List[List[float]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            if value.size == 0:
                return []
            if value.size != 3:
                raise ActionError(f"{field_name} array must be of shape (N, 3).")
            value = [value]
        elif value.ndim == 2:
            if value.shape[1] != 3:
                raise ActionError(f"{field_name} array must be of shape (N, 3).")
            value = [row for row in value]
        else:
            raise ActionError(f"{field_name} array must be of shape (N, 3).")

    coords: List[List[float]] = []
    for idx, item in enumerate(value):
        try:
            arr = np.asarray(item, dtype=float)
        except Exception as exc:
            raise ActionError(f"{field_name}[{idx}] could not be converted to 3-vector") from exc
        if arr.shape != (3,):
            raise ActionError(f"{field_name}[{idx}] must have length 3.")
        coords.append([float(v) for v in arr])

    if len(coords) > env.max_split_points:
        raise ActionError(
            f"{field_name} may contain at most {env.max_split_points} points."
        )
    return coords

def coerce_scalar_int(value: Any, field_name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ActionError(f"{field_name} array must be scalar.")
        value = value.item()
    if isinstance(value, (list, tuple, dict)):
        raise ActionError(f"{field_name} must be scalar.")
    try:
        return int(value)
    except Exception as exc:
        raise ActionError(f"{field_name} must be convertible to int") from exc
