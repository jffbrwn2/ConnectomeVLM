import enum
from dataclasses import dataclass
import shutil
from typing import List, Dict, Any
from glob import glob
import re
import json
import pandas as pd
import os
import numpy as np
from pathlib import Path
import time
class QuestionType(enum.Enum):
    SPLIT_VERIFICATION = 'split_verification'
    MERGE_VERIFICATION = 'merge_verification'
    MERGE_ACTION_MULTIPLE_CHOICE = 'merge_action_multiple_choice'  # Select correct merge partner from candidates
    ENDPOINT_ERROR_IDENTIFICATION = 'endpoint_error_identification'  # Identify if endpoint has split error
    ENDPOINT_LOCALIZATION = 'endpoint_localization'  # Predict x,y,z coordinates of error location
    SPLIT_PROPOSAL = 'split_proposal'
    SEGMENT_IDENTITY = 'segment_identity'  # Determine if two images show the same segment

class AnswerSpace(enum.Enum):
    YES_OR_NO = 'yes_or_no'
    MULTIPLE_CHOICE = 'multiple_choice'  # a, b, c, d, or 'none'
    COORDINATES = 'coordinates'  # [x, y, z] coordinate values in nm
    SPLIT_POINTS = 'split_points'  # tuple of two lists of (x, y, z) coordinates (sources, sinks)

@dataclass
class DatasetQuestion:
    question_type: QuestionType
    answer_space: AnswerSpace
    answer: Any
    images: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        self.validate()

    def validate(self):
        """
        Validate that self.answer matches the expected type for the answer_space.
        Raises:
            TypeError: if the answer type does not match the expected answer_space type.
        """
        if self.metadata is None or self.metadata == "":
            self.metadata = {}

        if self.answer_space == AnswerSpace.YES_OR_NO:
            if isinstance(self.answer, np.bool_):
                self.answer = bool(self.answer)
            if not isinstance(self.answer, bool):
                raise TypeError(f"For answer_space YES_OR_NO, answer must be bool, got {type(self.answer).__name__}")
        elif self.answer_space == AnswerSpace.MULTIPLE_CHOICE:
            valid_choices = ['a', 'b', 'c', 'd', 'none']
            if isinstance(self.answer, np.str_):
                self.answer = str(self.answer)
            if not isinstance(self.answer, str) or self.answer.lower() not in valid_choices:
                raise TypeError(f"For answer_space MULTIPLE_CHOICE, answer must be one of {valid_choices}, got {self.answer}")
        elif self.answer_space == AnswerSpace.COORDINATES:
            # Validate answer is [x, y, z] coordinates
            # Convert numpy arrays or other array-like types (from parquet loading) to list
            if hasattr(self.answer, 'tolist'):
                self.answer = self.answer.tolist()
            if not isinstance(self.answer, (list, tuple)):
                raise TypeError(f"For answer_space COORDINATES, answer must be list/tuple, got {type(self.answer).__name__}")
            if len(self.answer) != 3:
                raise TypeError(f"For answer_space COORDINATES, answer must have 3 elements [x,y,z], got {len(self.answer)}")
            if not all(isinstance(v, (int, float, np.integer, np.floating)) for v in self.answer):
                raise TypeError(f"For answer_space COORDINATES, all elements must be numeric, got {[type(v).__name__ for v in self.answer]}")
        elif self.answer_space == AnswerSpace.SPLIT_POINTS:
            if not isinstance(self.answer, (tuple, list)) or len(self.answer) != 2:
                raise TypeError(f"For answer_space SPLIT_POINTS, answer must be tuple/list of (sources, sinks), got {type(self.answer).__name__}")
            sources, sinks = self.answer
            if not isinstance(sources, list) or not isinstance(sinks, list):
                raise TypeError(f"For answer_space SPLIT_POINTS, sources and sinks must be lists")
        else:
            raise NotImplementedError(f"Answer space {self.answer_space} not implemented")


    
    @staticmethod
    def _convert_to_json_serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, int):
            return str(obj) if abs(obj) > 2**53 - 1 else obj
        elif isinstance(obj, dict):
            return {k: DatasetQuestion._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DatasetQuestion._convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def to_dict(self, drop_keys: List[str] = None) -> Dict[str, Any]:
        if drop_keys is None:
            drop_keys = []

        metadata = {k: v for k, v in self.metadata.items() if k not in drop_keys}

        # parquet/arrow cannot handle a struct column with *zero* fields.
        # if metadata is empty, inject a dummy key.
        metadata = self.metadata or {"__dummy__": None}

        return {
            "question_type": self.question_type.value,
            "answer_space": self.answer_space.value,
            "answer": DatasetQuestion._convert_to_json_serializable(self.answer),
            "images": self.images,
            "metadata": DatasetQuestion._convert_to_json_serializable(metadata),
        }

class QuestionDataset:
    def __init__(self, questions: List[DatasetQuestion] = None):
        self.questions : List[DatasetQuestion] = questions or []
        self._has_normalized_paths : bool = False
        self._normalized_path_dir : str = None
    
    def to_pandas(self, drop_metadata_keys: List[str] = None) -> pd.DataFrame:
        return pd.DataFrame([question.to_dict(drop_keys=drop_metadata_keys) for question in self.questions])  


    def __add__(self, other: 'QuestionDataset') -> 'QuestionDataset':
        if self._has_normalized_paths or other._has_normalized_paths:
            raise NotImplementedError("Adding datasets with normalized paths is not supported, please use merge_parquets on their parquets instead.")
        return QuestionDataset(questions=self.questions + other.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def to_parquet(self, path: str, move_images: bool = False, drop_metadata_keys: List[str] = None):
        """
        Convert the question dataset to a parquet file. 
        Creates a directory called 'images' in the same directory as the parquet file, and moves/copies the images to this directory.
        Args:
            path: Path to the parquet file.
            move_images: If True, move the images to the 'images' directory. If False, copy the images to the 'images' directory.
            drop_metadata_keys: List of keys to drop from the metadata.
        """
        # image dir is a directory "images" in the same directory as the parquet file
        images_dir = os.path.join(os.path.dirname(path), "images")
        os.makedirs(images_dir, exist_ok=True)
        assert os.path.dirname(path) == os.path.dirname(images_dir), "parquet_path and images_dir must be in the same directory"
        assert os.path.basename(images_dir) == "images", "images_dir must be named 'images'"

        # move/copy the images to the images directory, convert image paths to relative paths
        if not self._has_normalized_paths:
            print(f"Moving/copying images to {images_dir}")
            all_question_image_paths = []
            for i, question in enumerate(self.questions):
                question_image_paths = []
                for j, image_path in enumerate(question.images):
                    if move_images:
                        image_path = shutil.move(image_path, os.path.join(images_dir, f"{i}_{j}_{os.path.basename(image_path)}"))
                    else:
                        image_path = shutil.copy(image_path, os.path.join(images_dir, f"{i}_{j}_{os.path.basename(image_path)}"))
                    
                    image_path = os.path.relpath(image_path, os.path.dirname(path))
                    question_image_paths.append(image_path)
                all_question_image_paths.append(question_image_paths)
            
            # check if all images were copied/moved for all questions
            assert len(all_question_image_paths) == len(self.questions), "number of image paths must match number of questions"
            for i, question_image_paths in enumerate(all_question_image_paths):
                assert len(question_image_paths) == len(self.questions[i].images), "number of image paths must match number of images in question"
                # update question image paths to relative paths
                self.questions[i].images = question_image_paths

            self._has_normalized_paths = True
            self._normalized_path_dir = Path(images_dir).absolute().as_posix()
        else:
            print("Images already normalized, skipping moving/copying images")

        # convert questions with updated image paths to df and to parquet, then return           
        self.to_pandas(drop_metadata_keys=drop_metadata_keys).to_parquet(path=path)
        print(f"Saved parquet file to {path}")
        return path
    
    @staticmethod
    def from_parquet(path: str) -> 'QuestionDataset':
        if not str(path).endswith(".parquet"):
            path = os.path.join(path, "questions.parquet")
        
        assert os.path.exists(path), f"parquet file {path} does not exist"

        print(f"Loading parquet file from {path}")
        df = pd.read_parquet(path)
        
        # check if images are present next to the parquet file
        images_dir = os.path.join(os.path.dirname(path), "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"images directory {images_dir} not found")
         
        if len(df) > 0:
            def has_images_cell(cell):
                return any(isinstance(p, str) and p.startswith("images/") for p in cell)
            images_present = any(has_images_cell(cell) for cell in df["images"])
            assert (
                not images_present
                or os.path.exists(os.path.join(os.path.dirname(path), "images"))
            ), "images directory must exist if images are present in parquet file"
            
        questions = []
        for row in df.to_dict(orient='records'):
            raw_meta = row.get("metadata", {}) or {}
            # strip dummy-only metadata back to {}
            if (
                isinstance(raw_meta, dict)
                and "__dummy__" in raw_meta
                and len(raw_meta) == 1
            ):
                metadata = {}
            else:
                metadata = raw_meta

            # coerce answer to proper type based on answer_space
            answer = row["answer"]
            answer_space = AnswerSpace(row["answer_space"])
            if answer_space == AnswerSpace.SPLIT_POINTS:
                # parquet converts tuples to ndarrays, coerce back to tuple of lists
                if isinstance(answer, np.ndarray):
                    sources, sinks = answer[0], answer[1]
                    if isinstance(sources, np.ndarray):
                        sources = sources.tolist()
                    if isinstance(sinks, np.ndarray):
                        sinks = sinks.tolist()
                    answer = (list(sources), list(sinks))
                elif isinstance(answer, (list, tuple)):
                    sources, sinks = answer[0], answer[1]
                    answer = (list(sources), list(sinks))

            q = DatasetQuestion(
                question_type=QuestionType(row["question_type"]),
                answer_space=answer_space,
                answer=answer,
                images=row["images"],
                metadata=metadata,
            )
            questions.append(q)
        
        ds = QuestionDataset(questions=questions)
        ds._has_normalized_paths = True
        ds._normalized_path_dir = Path(images_dir).absolute().as_posix()
        return ds

    @staticmethod
    def _move_images(source_path: str, target_dir: str, move_images: bool = False) -> Path:
        # move/copy all images in source_path to target_path
        num_current_images_in_target_dir = len(list(glob(os.path.join(target_dir, "*.png"))))
        image_paths = list(glob(os.path.join(source_path, "*.png")))
        length = len(image_paths)
        for image_path in image_paths:
            if move_images:
                shutil.move(image_path, os.path.join(target_dir, os.path.basename(image_path)))
            else:
                shutil.copy(image_path, os.path.join(target_dir, os.path.basename(image_path)))
        assert num_current_images_in_target_dir + length == len(list(glob(os.path.join(target_dir, "*.png")))), f"Expected {num_current_images_in_target_dir + length} images, found {len(list(glob(os.path.join(target_dir, '*.png'))))} in {target_dir}"
        return Path(target_dir)

    @staticmethod
    def merge_parquets(source_directories: List[str], target_directory: str, move_images: bool = False) -> Path:
        """
        Merge multiple parquet folders into a single merged parquet folder (contains images dir and parquet file).
        Args:
            paths: List of paths to the parquet files.
        Returns:
            QuestionDataset: A question dataset containing the merged parquet files.
        """
        target_dir = target_directory
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)

        assert len(glob(os.path.join(target_directory, "images", "*.png"))) == 0, f"Target directory {target_directory} is not empty, aborting."

        questions = []
        for directory in source_directories:
            assert os.path.isdir(directory), f"path {directory} is not a directory"
            image_dir = os.path.join(directory, "images")
            assert os.path.isdir(image_dir), f"images directory {image_dir} not found in {directory}"
            
            # glob match for parquet files
            parquet_paths = glob(os.path.join(directory, "*.parquet"))
            assert len(parquet_paths) == 1, f"Expected 1 parquet file, found {len(parquet_paths)} in {directory}"
            parquet_path = parquet_paths[0]

            # load parquet file
            ds = QuestionDataset.from_parquet(parquet_path)

            # move/copy images to target_dir and update question image paths to relative paths
            QuestionDataset._move_images(image_dir, os.path.join(target_dir, "images"), move_images)            
            for question in ds.questions:
                question.images = [os.path.join("images", os.path.basename(image_path)) for image_path in question.images]

            questions.extend(ds.questions)

        
        new_ds = QuestionDataset(questions=questions)
        new_ds.to_pandas().to_parquet(os.path.join(target_dir, "questions.parquet"))
        return Path(target_dir)

    @staticmethod
    def from_binary_splits_folder(path: str) -> 'QuestionDataset':
        assert os.path.isdir(path), f"path {path} is not a directory"

        good_dir = os.path.join(path, "good")
        bad_dir = os.path.join(path, "bad")

        assert os.path.isdir(good_dir), f"missing good/ directory at {good_dir}"
        assert os.path.isdir(bad_dir), f"missing bad/ directory at {bad_dir}"

        def load_split_dirs(root):
            # only subdirs, skip files
            dirs = [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]

            out = {}
            for split_dir in dirs:
                imgs = [
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir)
                    if f.endswith(".png")
                ]
                out[split_dir] = {
                    "images": imgs,
                    "metadata": {"split_dir": split_dir},
                }
            return out

        good_splits = load_split_dirs(good_dir)
        bad_splits = load_split_dirs(bad_dir)

        questions = []

        for split_dir, info in good_splits.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.SPLIT_VERIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=True,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        for split_dir, info in bad_splits.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.SPLIT_VERIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=False,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        return QuestionDataset(questions=questions)

    @staticmethod
    def from_endpoint_error_identification_folder(path: str) -> 'QuestionDataset':
        """
        Load a split error identification dataset from good/bad folder structure.

        This loads data generated by split_data_generator.py which identifies
        whether skeleton endpoints are split errors (need merge correction) or
        natural termini (don't need correction).

        Folder structure:
            path/
            ├── good/                    # Split errors (is_split_error=True)
            │   └── {root_id}_endpoint_{idx}/
            │       ├── *.png images
            │       └── metadata.json
            └── bad/                     # Natural termini (is_split_error=False)
                └── {root_id}_endpoint_{idx}/
                    ├── *.png images
                    └── metadata.json

        Args:
            path: Path to the dataset root directory containing good/ and bad/ subdirs

        Returns:
            QuestionDataset with ENDPOINT_ERROR_IDENTIFICATION questions
        """
        assert os.path.isdir(path), f"path {path} is not a directory"

        good_dir = os.path.join(path, "good")
        bad_dir = os.path.join(path, "bad")

        assert os.path.isdir(good_dir), f"missing good/ directory at {good_dir}"
        assert os.path.isdir(bad_dir), f"missing bad/ directory at {bad_dir}"

        def load_endpoint_dirs(root, is_split_error: bool):
            """Load endpoint directories with their metadata."""
            dirs = [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]

            out = {}
            for endpoint_dir in dirs:
                all_imgs = [
                    os.path.join(endpoint_dir, f)
                    for f in os.listdir(endpoint_dir)
                    if f.endswith(".png")
                ]
                
                # Sort images to ensure consistent ordering:
                # Mesh views first (front, side, top), then EM views (em_front, em_side, em_top)
                # This is important for the model to receive images in expected order
                def image_sort_key(path):
                    fname = os.path.basename(path).lower()
                    # Check EM patterns FIRST (more specific) to avoid 'front' matching 'em_front'
                    # Then check mesh patterns
                    # Order: mesh views (0,1,2), then EM views (3,4,5)
                    if 'em_front' in fname:
                        return 3
                    elif 'em_side' in fname:
                        return 4
                    elif 'em_top' in fname:
                        return 5
                    elif 'front' in fname:
                        return 0
                    elif 'side' in fname:
                        return 1
                    elif 'top' in fname:
                        return 2
                    return float('inf')  # Unknown files sort last                
                imgs = sorted(all_imgs, key=image_sort_key)

                # Load metadata.json if it exists
                metadata_path = os.path.join(endpoint_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {}

                # Add endpoint_dir to metadata for reference
                metadata["endpoint_dir"] = endpoint_dir
                metadata["is_split_error"] = is_split_error
                metadata["has_em_images"] = any("em_" in os.path.basename(p).lower() for p in imgs)

                out[endpoint_dir] = {
                    "images": imgs,
                    "metadata": metadata,
                }
            return out

        good_endpoints = load_endpoint_dirs(good_dir, is_split_error=True)
        bad_endpoints = load_endpoint_dirs(bad_dir, is_split_error=False)

        questions = []

        # Good = split errors that need correction (answer=True means "yes, this is a split error")
        for endpoint_dir, info in good_endpoints.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.ENDPOINT_ERROR_IDENTIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=True,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        # Bad = natural termini that don't need correction (answer=False means "no, this is not a split error")
        for endpoint_dir, info in bad_endpoints.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.ENDPOINT_ERROR_IDENTIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=False,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        return QuestionDataset(questions=questions)

    @staticmethod
    def from_rerendered_splits(
        images_dir: Path,
        metadata_dir: Path,
        drop_cut_edges: bool = True,
        exclude_small_splits: bool = True,
        min_candidate_size: int = 10,
    ) -> "QuestionDataset":
        assert images_dir.is_dir(), f"images_dir {images_dir} is not a directory"
        assert metadata_dir.is_dir(), f"metadata_dir {metadata_dir} is not a directory"

        print(f"Loading rerendered splits from images dir {images_dir} and metadata dir {metadata_dir}")

        views = ("front", "side", "top")

        def _candidate_size(meta: dict) -> int | None:
            try:
                return int(meta.get("evaluation_stats", {}).get("candidate_size"))
            except Exception:
                return None

        questions: list[DatasetQuestion] = []

        root_dirs = sorted(metadata_dir.iterdir(), key=lambda p: p.name)
        excluded_small_splits = 0
        excluded_missing_views = 0
        print(f"Found {len(root_dirs)} root directories")
        # iterate metadata first (require metadata to exist)
        for root_dir in root_dirs:
            if not root_dir.is_dir():
                continue

            root_id = root_dir.name
            img_root_dir = images_dir / root_id
            if not img_root_dir.is_dir():
                continue

            meta_paths = sorted(root_dir.glob("*.json"), key=lambda p: p.name)
            for meta_path in meta_paths:
                split_hash = meta_path.stem  # <split_hash>.json

                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Skipping {meta_path} (failed to load json): {e}")
                    continue

                if exclude_small_splits:
                    cs = _candidate_size(meta)
                    if cs is not None and cs < min_candidate_size:
                        excluded_small_splits += 1
                        continue

                # require all three views to exist:
                # images_dir/<root_id>/<split_hash>_<extra>_<view>.png
                view_paths: dict[str, Path] = {}
                missing = False
                for v in views:
                    matches = sorted(img_root_dir.glob(f"{split_hash}_*_{v}.png"))
                    if len(matches) == 0:
                        missing = True
                        excluded_missing_views += 1
                        break
                    if len(matches) > 1:
                        # deterministic, but noisy
                        print(
                            f"Warning: multiple '{v}' images for root {root_id}, split {split_hash}; using {matches[0].name}"
                        )
                    view_paths[v] = matches[0]

                if missing:
                    continue

                imgs_sorted = [view_paths["front"], view_paths["side"], view_paths["top"]]

                if "is_good" not in meta:
                    continue

                answer = bool(meta["is_good"])


                meta_out = dict(meta)
                meta_out["root_id"] = root_id
                meta_out["split_hash"] = split_hash
                meta_out["rerender_images_dir"] = str(img_root_dir)
                meta_out["rerender_metadata_path"] = str(meta_path)

                if drop_cut_edges:
                    meta_out.pop("cut_edges", None)

                questions.append(
                    DatasetQuestion(
                        question_type=QuestionType.SPLIT_VERIFICATION,
                        answer_space=AnswerSpace.YES_OR_NO,
                        answer=answer,
                        images=[str(p) for p in imgs_sorted],
                        metadata=meta_out,
                    )
                )
        print(f"Loaded {len(questions)} questions from rerendered splits (excluded {excluded_small_splits} small splits and {excluded_missing_views} missing views)")
        return QuestionDataset(questions=questions)

    @staticmethod
    def from_good_splits_to_split_proposal(
        images_dir: Path,
        metadata_dir: Path,
        drop_cut_edges: bool = True,
        exclude_small_splits: bool = True,
        min_candidate_size: int = 10,
    ) -> "QuestionDataset":
        """
        Create SPLIT_PROPOSAL questions from good splits only.
        Uses pre_split images and extracts sources/sinks as the answer.
        """
        assert images_dir.is_dir(), f"images_dir {images_dir} is not a directory"
        assert metadata_dir.is_dir(), f"metadata_dir {metadata_dir} is not a directory"

        print(f"Loading good splits for split proposals from images dir {images_dir} and metadata dir {metadata_dir}")

        views = ("front", "side", "top")

        def _candidate_size(meta: dict) -> int | None:
            try:
                return int(meta.get("evaluation_stats", {}).get("candidate_size"))
            except Exception:
                return None

        questions: list[DatasetQuestion] = []

        root_dirs = sorted(metadata_dir.iterdir(), key=lambda p: p.name)
        excluded_small_splits = 0
        excluded_missing_views = 0
        excluded_bad_splits = 0
        excluded_missing_split_points = 0
        excluded_missing_nodes = 0

        print(f"Found {len(root_dirs)} root directories")
        # iterate metadata first (require metadata to exist)
        for root_dir in root_dirs:
            if not root_dir.is_dir():
                continue

            root_id = root_dir.name
            img_root_dir = images_dir / root_id
            if not img_root_dir.is_dir():
                continue

            meta_paths = sorted(root_dir.glob("*.json"), key=lambda p: p.name)
            for meta_path in meta_paths:
                split_hash = meta_path.stem  # <split_hash>.json

                try:
                    with meta_path.open("r") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"Skipping {meta_path} (failed to load json): {e}")
                    continue

                # only include good splits
                if "is_good" not in meta or not bool(meta["is_good"]):
                    excluded_bad_splits += 1
                    continue

                if exclude_small_splits:
                    cs = _candidate_size(meta)
                    if cs is not None and cs < min_candidate_size:
                        excluded_small_splits += 1
                        continue

                # extract sources and sinks for answer
                if "sources" not in meta or "sinks" not in meta:
                    excluded_missing_split_points += 1
                    continue

                sources = meta["sources"]
                sinks = meta["sinks"]
                answer = (sources, sinks)

                # only match pre_split_ images
                # require all three views to exist:
                # images_dir/<root_id>/<split_hash>_*_pre_split_*_<view>.png
                view_paths: dict[str, Path] = {}
                missing = False
                for v in views:
                    matches = sorted(img_root_dir.glob(f"{split_hash}_*_pre_split_*_{v}.png"))
                    if len(matches) == 0:
                        missing = True
                        excluded_missing_views += 1
                        break
                    if len(matches) > 1:
                        # deterministic, but noisy
                        print(
                            f"Warning: multiple '{v}' pre_split images for root {root_id}, split {split_hash}; using {matches[0].name}"
                        )
                    view_paths[v] = matches[0]

                if missing:
                    continue

                # try to load neighbor metadata (optional but nice to have)
                neighbor_metadata = None
                try:
                    neighbors_matches = list(img_root_dir.glob(f"{split_hash}_*_nodes.json"))
                    if len(neighbors_matches) > 0:
                        with neighbors_matches[0].open("r") as f:
                            neighbor_metadata = json.load(f)
                except Exception as e:
                    excluded_missing_nodes += 1
                    # not critical, just skip

                imgs_sorted = [view_paths["front"], view_paths["side"], view_paths["top"]]

                meta_out = dict(meta)
                meta_out["root_id"] = root_id
                meta_out["split_hash"] = split_hash
                meta_out["rerender_images_dir"] = str(img_root_dir)
                meta_out["rerender_metadata_path"] = str(meta_path)
                if neighbor_metadata is not None:
                    meta_out["neighbors"] = neighbor_metadata

                if drop_cut_edges:
                    meta_out.pop("cut_edges", None)

                questions.append(
                    DatasetQuestion(
                        question_type=QuestionType.SPLIT_PROPOSAL,
                        answer_space=AnswerSpace.SPLIT_POINTS,
                        answer=answer,
                        images=[str(p) for p in imgs_sorted],
                        metadata=meta_out,
                    )
                )

        print(f"Loaded {len(questions)} split proposal questions from good splits")
        print(f"  Excluded: {excluded_bad_splits} bad splits, {excluded_small_splits} small splits, "
              f"{excluded_missing_views} missing views, {excluded_missing_split_points} missing split points, "
              f"{excluded_missing_nodes} missing nodes (non-critical)")
        return QuestionDataset(questions=questions)


    @staticmethod
    def from_binary_merge_identification_folder(path: str) -> 'QuestionDataset':
        """
        Generates a question dataset from a folder of binary merge identification directories as generated by split_merge_resolution.py, with the task setting --task merge-identification.
        Requires the updated generation script that sets first_one_correct to True.

        Args:
            path: Path to the folder of merge identification directories.
        Returns:
            QuestionDataset: A question dataset containing the merge identification directories.
        """
        assert os.path.isdir(path), f"path {path} is not a directory"

        merge_directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

        import json

        questions = []

        for merge_directory in merge_directories:
            try:
                with open(os.path.join(merge_directory, "generation_metadata.json"), "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata from {merge_directory}: {e}, skipping")
                continue
            
            if not metadata["first_one_correct"]:
                raise NotImplementedError("Only first one correct merge identification is supported, please update your merge generation script")
            
            if not len(metadata["incorrect_ids"]) == 1:
                print(f"Only one incorrect id is supported, found {len(metadata['incorrect_ids'])} ({merge_directory}), skipping")
                continue

            correct_id = metadata["correct_id"]
            incorrect_id = metadata["incorrect_ids"][0]

            correct_views_paths = []
            incorrect_views_paths = []

            correct_images_dict = metadata["image_paths"]["options"][correct_id]["zoomed"]
            incorrect_images_dict = metadata["image_paths"]["options"][incorrect_id]["zoomed"]

            for view, image_path in correct_images_dict.items():
                assert os.path.exists(image_path), f"image path {image_path} does not exist"
                correct_views_paths.append(image_path)

            for view, image_path in incorrect_images_dict.items():
                assert os.path.exists(image_path), f"image path {image_path} does not exist"
                incorrect_views_paths.append(image_path)

            # remove image_paths from metadata for cleanliness
            metadata["image_paths"] = None

            correct_question = DatasetQuestion(
                question_type=QuestionType.MERGE_VERIFICATION,
                answer_space=AnswerSpace.YES_OR_NO,
                answer=True,
                images=correct_views_paths,
                metadata=metadata,
            )

            incorrect_question = DatasetQuestion(
                question_type=QuestionType.MERGE_VERIFICATION,
                answer_space=AnswerSpace.YES_OR_NO,
                answer=False,
                images=incorrect_views_paths,
                metadata=metadata,
            )

            questions.append(correct_question)
            questions.append(incorrect_question)

        return QuestionDataset(questions=questions)

    @staticmethod
    def from_binary_merge_corrections_folder(path: str) -> 'QuestionDataset':
        """
        Generates a question dataset from a merge-corrections folder with good/bad subdirectories.

        This is for merge correction data generated by merge_sampler.py, which has the same
        structure as splits (good/bad folders) but represents whether segments should be merged.

        Args:
            path: Path to the merge-corrections folder containing good/ and bad/ subdirectories.
        Returns:
            QuestionDataset: A question dataset with MERGE_VERIFICATION questions.
        """
        assert os.path.isdir(path), f"path {path} is not a directory"

        good_dir = os.path.join(path, "good")
        bad_dir = os.path.join(path, "bad")

        assert os.path.isdir(good_dir), f"missing good/ directory at {good_dir}"
        assert os.path.isdir(bad_dir), f"missing bad/ directory at {bad_dir}"

        def load_sample_dirs(root):
            dirs = [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]

            out = {}
            for sample_dir in dirs:
                imgs = [
                    os.path.join(sample_dir, f)
                    for f in os.listdir(sample_dir)
                    if f.endswith(".png")
                ]

                # Read metadata.json if it exists, keeping only essential fields
                metadata = {"sample_dir": sample_dir}
                metadata_path = os.path.join(sample_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        full_meta = json.load(f)
                    # Only keep fields needed for split grouping and identification
                    for key in ("segment1_id", "interface_point"):
                        if key in full_meta:
                            metadata[key] = full_meta[key]

                out[sample_dir] = {
                    "images": imgs,
                    "metadata": metadata,
                }
            return out

        good_samples = load_sample_dirs(good_dir)
        bad_samples = load_sample_dirs(bad_dir)

        questions = []

        for sample_dir, info in good_samples.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.MERGE_VERIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=True,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        for sample_dir, info in bad_samples.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.MERGE_VERIFICATION,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=False,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        return QuestionDataset(questions=questions)

    @staticmethod
    def from_endpoint_localization_folder(path: str) -> 'QuestionDataset':
        """
        Load endpoint localization dataset from folder structure.

        This loads data generated by endpoint_localization_sampler.py which creates
        samples where the model must predict x,y,z coordinates of error locations
        from neuron images with graph overlays.

        Folder structure:
            path/
            └── samples/
                └── {sample_id}/
                    ├── front.png, side.png, top.png (or other image names)
                    └── metadata.json  # Contains interface_point_nm, neighbor_meta, etc.

        Args:
            path: Path to the dataset root directory containing samples/ subdir

        Returns:
            QuestionDataset with ENDPOINT_LOCALIZATION questions
        """
        assert os.path.isdir(path), f"path {path} is not a directory"

        samples_dir = os.path.join(path, "samples")
        if not os.path.isdir(samples_dir):
            # Fallback: samples might be directly in path
            samples_dir = path

        sample_dirs = [
            os.path.join(samples_dir, d)
            for d in os.listdir(samples_dir)
            if os.path.isdir(os.path.join(samples_dir, d))
        ]

        questions = []
        for sample_dir in sample_dirs:
            # Load images
            imgs = sorted([
                os.path.join(sample_dir, f)
                for f in os.listdir(sample_dir)
                if f.endswith(".png")
            ])

            if len(imgs) != 3:
                print(f"Skipping {sample_dir}: expected 3 images, got {len(imgs)}")
                continue

            # Load metadata
            metadata_path = os.path.join(sample_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                print(f"Skipping {sample_dir}: no metadata.json found")
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Extract ground truth coordinates
            interface_point = metadata.get("interface_point_nm")
            if interface_point is None:
                print(f"Skipping {sample_dir}: no interface_point_nm in metadata")
                continue

            # Ensure it's a list of 3 floats
            if isinstance(interface_point, (list, tuple)) and len(interface_point) == 3:
                answer = [float(v) for v in interface_point]
            else:
                print(f"Skipping {sample_dir}: invalid interface_point_nm format")
                continue

            # Add sample_dir to metadata for reference
            metadata["sample_dir"] = sample_dir

            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.ENDPOINT_LOCALIZATION,
                    answer_space=AnswerSpace.COORDINATES,
                    answer=answer,
                    images=imgs,
                    metadata=metadata,
                )
            )

        return QuestionDataset(questions=questions)

    @staticmethod
    def from_segment_identity_folder(path: str) -> 'QuestionDataset':
        """
        Load segment identity dataset from good/bad folder structure.

        This loads data generated by segment_identity_sampler.py which creates
        samples where the model must determine if two images show the same segment
        (different views/zoom) or different segments.

        Folder structure:
            path/
            ├── good/                    # Same segment pairs (answer=True)
            │   └── {sample_id}/
            │       ├── image1.png
            │       ├── image2.png
            │       └── metadata.json
            └── bad/                     # Different segment pairs (answer=False)
                └── {sample_id}/
                    ├── image1.png
                    ├── image2.png
                    └── metadata.json

        Args:
            path: Path to the dataset root directory containing good/ and bad/ subdirs

        Returns:
            QuestionDataset with SEGMENT_IDENTITY questions
        """
        assert os.path.isdir(path), f"path {path} is not a directory"

        good_dir = os.path.join(path, "good")
        bad_dir = os.path.join(path, "bad")

        assert os.path.isdir(good_dir), f"missing good/ directory at {good_dir}"
        assert os.path.isdir(bad_dir), f"missing bad/ directory at {bad_dir}"

        def load_sample_dirs(root, is_same_segment: bool):
            """Load sample directories with their metadata."""
            dirs = [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]

            out = {}
            for sample_dir in dirs:
                # Get image paths (expect image1.png and image2.png)
                imgs = sorted([
                    os.path.join(sample_dir, f)
                    for f in os.listdir(sample_dir)
                    if f.endswith(".png")
                ])

                if len(imgs) != 2:
                    print(f"Skipping {sample_dir}: expected 2 images, got {len(imgs)}")
                    continue

                # Load metadata.json if it exists
                metadata_path = os.path.join(sample_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                else:
                    metadata = {}

                # Add sample_dir to metadata for reference
                metadata["sample_dir"] = sample_dir
                metadata["is_same_segment"] = is_same_segment

                out[sample_dir] = {
                    "images": imgs,
                    "metadata": metadata,
                }
            return out

        good_samples = load_sample_dirs(good_dir, is_same_segment=True)
        bad_samples = load_sample_dirs(bad_dir, is_same_segment=False)

        questions = []

        # Good = same segment pairs (answer=True means "yes, these are the same segment")
        for sample_dir, info in good_samples.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.SEGMENT_IDENTITY,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=True,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        # Bad = different segment pairs (answer=False means "no, these are different segments")
        for sample_dir, info in bad_samples.items():
            questions.append(
                DatasetQuestion(
                    question_type=QuestionType.SEGMENT_IDENTITY,
                    answer_space=AnswerSpace.YES_OR_NO,
                    answer=False,
                    images=info["images"],
                    metadata=info["metadata"],
                )
            )

        return QuestionDataset(questions=questions)


import argparse

if __name__ == "__main__":
    """
    Example: python -c "from src.training.question_dataset import QuestionDataset; QuestionDataset.from_binary_merge_corrections_folder('training_data/merge-corrections').to_parquet('training_data/merge-parquet/questions.parquet')"               
    """
    parser = argparse.ArgumentParser(
        description="Convert binary splits to parquet format."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        nargs='+',
        help="Input directory or directories containing binary splits (and metadata)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for the generated parquet file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        help="Mode to run the script in.",
        choices=["split", "merge", "merge_corrections", "split_rerendered", "endpoint_error_identification", "endpoint_localization", "split_proposal", "segment_identity"],
    )
    parser.add_argument(
        "--move-images",
        action="store_true",
        help="Move images to the output directory instead of copying them."
    )
    args = parser.parse_args()

    start_time = time.time()
    input_paths = [Path(p) for p in args.input_path]
    output_path = Path(args.output_path)
    print(f"Input paths: {input_paths}")
    print(f"Output path: {output_path}")
    print(f"Mode: {args.mode}")
    print(f"Move images: {args.move_images}")

    for p in input_paths:
        if not os.path.isdir(p):
            parser.error(f"Input path {p} is not a directory")
    # Create output parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)


    if not os.path.exists(output_path):
        if args.mode == "split":
            assert len(input_paths) == 1, f"Expected 1 input path for split mode, got {len(input_paths)}"
            ds = QuestionDataset.from_binary_splits_folder(input_paths[0])
        elif args.mode == "merge":
            assert len(input_paths) == 1, f"Expected 1 input path for merge mode, got {len(input_paths)}"
            ds = QuestionDataset.from_binary_merge_identification_folder(input_paths[0])
        elif args.mode == "merge_corrections":
            assert len(input_paths) == 1, f"Expected 1 input path for merge_corrections mode, got {len(input_paths)}"
            ds = QuestionDataset.from_binary_merge_corrections_folder(str(input_paths[0]))
        elif args.mode == "split_rerendered":
            assert len(input_paths) == 2, f"Expected 2 input paths for split_rerendered mode, got {len(input_paths)}"
            images_path = input_paths[0]
            metadata_path = input_paths[1]
            assert os.path.isdir(images_path), f"images path {images_path} is not a directory"
            assert os.path.isdir(metadata_path), f"metadata path {metadata_path} is not a directory"
            ds = QuestionDataset.from_rerendered_splits(Path(images_path), Path(metadata_path), drop_cut_edges=True)
        elif args.mode == "endpoint_error_identification":
            assert len(input_paths) == 1, f"Expected 1 input path for endpoint errro mode, got {len(input_paths)}"
            ds = QuestionDataset.from_endpoint_error_identification_folder(str(input_paths[0]))
        elif args.mode == "endpoint_localization":
            assert len(input_paths) == 1, f"Expected 1 input path for endpoint localization mode, got {len(input_paths)}"
            ds = QuestionDataset.from_endpoint_localization_folder(str(input_paths[0]))
        elif args.mode == "split_proposal":
            assert len(input_paths) == 2, f"Expected 2 input paths for split_proposal mode (images_dir, metadata_dir), got {len(input_paths)}"
            images_path = Path(input_paths[0])
            metadata_path = Path(input_paths[1])
            assert os.path.isdir(images_path), f"images path {images_path} is not a directory"
            assert os.path.isdir(metadata_path), f"metadata path {metadata_path} is not a directory"
            ds = QuestionDataset.from_good_splits_to_split_proposal(
                Path(images_path),
                Path(metadata_path),
                drop_cut_edges=True,
                exclude_small_splits=True,
                min_candidate_size=10
            )
        elif args.mode == "segment_identity":
            assert len(input_paths) == 1, f"Expected 1 input path for segment_identity mode, got {len(input_paths)}"
            ds = QuestionDataset.from_segment_identity_folder(str(input_paths[0]))
        else:
            parser.error(f"Invalid mode provided: {args.mode}.")
        
        ds.to_parquet(output_path, move_images=args.move_images, drop_metadata_keys=["cut_edges"])
    else:
        print(f"Splits parquet file already exists at {output_path}, skipping conversion")


    ds = QuestionDataset.from_parquet(output_path)
    print(f"Loaded {len(ds.questions)} questions from parquet file at {output_path}.")
    elapsed = int(time.time() - start_time)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Done in {hours}:{minutes:02d}:{seconds:02d}.")
