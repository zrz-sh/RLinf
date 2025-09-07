"""
Simplified PyTorch dataset implementation that follows OpenPI's exact logic.

This implementation matches the original OpenPI data loading pipeline while being 
compatible with TRL and Transformers trainers.
"""

from typing import Dict, Any, Optional, List, Union
import torch
from torch.utils.data import Dataset
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np

from .config import DataConfigFactory
from .transforms import DataTransformFn, Normalize, PromptFromLeRobotTask, compose, load_task_descriptions


class TransformedDataset(Dataset):
    """Simple transformed dataset wrapper matching OpenPI's pattern."""
    
    def __init__(self, dataset: Dataset, transforms: List[DataTransformFn]):
        self._dataset = dataset
        self._transform = compose(transforms)
    
    def __getitem__(self, index):
        # Get sample from underlying dataset
        sample = self._dataset[index]
        
        # Apply transforms directly to tensor data
        # All transforms are tensor-aware and preserve device placement
        transformed_sample = self._transform(sample)
        
        return transformed_sample
    
    def __len__(self):
        return len(self._dataset)
        
class LeRobotPyTorchDataset(Dataset):
    """Simple PyTorch dataset that follows OpenPI's exact logic."""
    
    def __init__(
        self,
        repo_id: str,
        action_horizon: int = 10,
        split: str = "train",
        data_config_factory: Optional[DataConfigFactory] = None,
        action_dim: Optional[int] = None,
        max_token_len: Optional[int] = 256
    ):
        """
        Initialize dataset following OpenPI's create_torch_dataset + transform_dataset pattern.
        
        Args:
            repo_id: LeRobot dataset repository ID
            action_horizon: Number of future actions to predict
            split: Dataset split ("train", "test", etc.)
            data_config_factory: Factory for dataset configuration (optional)
            action_dim: Action dimensionality (used if config_factory provided)
            max_token_len: Maximum token length (used if config_factory provided)
        """
        self.repo_id = repo_id
        self.action_horizon = action_horizon
        self.split = split
        
        # Check if this is a local path or remote repo ID
        self.is_local = self._is_local_path(repo_id)
            
        # Load dataset metadata and create base dataset
        if self.is_local:
            # For local datasets, use just the folder name as repo_id and set root to the full path
            local_path = Path(repo_id).resolve()
            folder_name = local_path.name
            self.dataset_meta = LeRobotDatasetMetadata(folder_name, root=local_path)
        else:
            self.dataset_meta = LeRobotDatasetMetadata(repo_id)
        
        # Create data config if factory provided
        if data_config_factory is not None:
            self.data_config = data_config_factory.create(
                action_dim=action_dim,
                action_horizon=action_horizon,
                max_token_len=max_token_len
            )
        else:
            self.data_config = None
        
        # For delta_timestamps, we need to use the RAW dataset keys (before repack transforms)
        # The most common key in LeRobot datasets is "action" (singular)
        raw_action_keys = []
        if "action" in self.dataset_meta.features:
            raw_action_keys = ["action"]
        elif "actions" in self.dataset_meta.features:
            raw_action_keys = ["actions"]
        else:
            raise ValueError(f"No action key found in dataset metadata: {self.dataset_meta.features}")
        
        # Calculate delta_timestamps using FPS from metadata (OpenPI pattern)
        delta_timestamps = {
            key: [t / self.dataset_meta.fps for t in range(action_horizon)] 
            for key in raw_action_keys
        }
        
        # Create base LeRobot dataset
        if self.is_local:
            # For local datasets, use just the folder name as repo_id and set root to the full path
            local_path = Path(repo_id).resolve()
            folder_name = local_path.name
            self.base_dataset = LeRobotDataset(
                folder_name,
                root=local_path,
                delta_timestamps=delta_timestamps,
                download_videos=False  # Skip videos for faster loading
            )
        else:
            self.base_dataset = LeRobotDataset(
                repo_id,
                delta_timestamps=delta_timestamps,
                download_videos=False  # Skip videos for faster loading
            )
        
        # Step 2: Add prompt from task if needed (following OpenPI pattern)
        # This must happen BEFORE the main transform pipeline (like in OpenPI data_loader.py)
        if self.data_config and getattr(self.data_config, "prompt_from_task", True):
            # Determine which tasks to use
            tasks_to_use = None
            
            if self.is_local:
                # For local datasets, try loading from our universal task loader
                tasks_to_use = load_task_descriptions(Path(repo_id).resolve())
                
            # If no local tasks found, try using metadata tasks (for remote datasets or fallback)
            if not tasks_to_use and hasattr(self.dataset_meta, 'tasks') and self.dataset_meta.tasks:
                tasks_to_use = self.dataset_meta.tasks
            
            # Apply prompt transform if we have tasks
            if tasks_to_use:
                print(f"Adding prompt transform with {len(tasks_to_use)} tasks")
                self.base_dataset = TransformedDataset(
                    self.base_dataset,
                    [PromptFromLeRobotTask(tasks_to_use)]
                )
            
        # Step 3: Apply transform pipeline (following OpenPI's transform_dataset pattern)
        # This should wrap the dataset with TransformedDataset, not apply transforms in __getitem__
        transforms = self._create_transform_list()
        if transforms:
            self.base_dataset = TransformedDataset(self.base_dataset, transforms)
        
        print(f"Loaded dataset: {repo_id}")
        print(f"  Type: {'Local' if self.is_local else 'Remote'}")
        print(f"  Episodes: {self.dataset_meta.total_episodes}")
        print(f"  Frames: {self.dataset_meta.total_frames}")
        print(f"  FPS: {self.dataset_meta.fps}")
        print(f"  Available keys: {list(self.dataset_meta.features.keys())}")
        print(f"  Split '{split}': {len(self.base_dataset)} samples")
    
    def _is_local_path(self, path_or_id: str) -> bool:
        """Check if the input is a local path or a remote repo ID."""
        # Check multiple indicators of a local path
        path = Path(path_or_id)
        return (
            path.exists() or  # Path exists
            path.is_absolute() or  # Absolute path
            path_or_id.startswith("./") or  # Relative path with ./
            path_or_id.startswith("../") or  # Relative path with ../
            (path_or_id.startswith("data/") and not "/" in path_or_id[5:]) or  # data/ prefix without additional slashes
            ("/" not in path_or_id and not path_or_id.startswith("lerobot/"))  # No slashes and not a huggingface repo
        )
    
    def _create_transform_list(self) -> List[DataTransformFn]:
        """Create transform list following OpenPI's transform_dataset logic."""
        transforms = []
        
        if self.data_config is not None:
            # Add repack transforms (following OpenPI pattern)
            transforms.extend(self.data_config.repack_transforms.inputs)
            
            # Add data transforms  
            transforms.extend(self.data_config.data_transforms.inputs)
            
            # Add normalization (following OpenPI pattern) 
            if self.data_config.norm_stats is not None:
                transforms.append(
                    Normalize(self.data_config.norm_stats, self.data_config.use_quantile_norm)
                )
            
            # Add model transforms
            transforms.extend(self.data_config.model_transforms.inputs)
        
        return transforms
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        # All transforms are now handled by TransformedDataset wrappers
        # Just delegate to the base dataset (which may be wrapped with transforms)
        return self.base_dataset[idx]


def create_lerobot_dataset(
    repo_id: str,
    action_horizon: int = 10,
    split: str = "train", 
    data_config_factory: Optional[DataConfigFactory] = None,
    action_dim: Optional[int] = None,
    max_token_len: Optional[int] = 256
) -> LeRobotPyTorchDataset:
    """
    Create a LeRobot dataset following OpenPI's pattern.
    
    This function replicates OpenPI's create_torch_dataset + transform_dataset logic
    in a single, simplified interface that's compatible with TRL/Transformers.
    """
    return LeRobotPyTorchDataset(
        repo_id=repo_id,
        action_horizon=action_horizon,
        split=split,
        data_config_factory=data_config_factory,
        action_dim=action_dim,
        max_token_len=max_token_len
    )


def vla_data_collator(features):
    """
    Custom data collator for VLA training that handles multi-modal inputs.
    
    This collator handles:
    - Images (multiple cameras)
    - State vectors 
    - Actions
    - Text prompts
    """
    batch = {}
    
    # Handle different types of inputs
    for key in features[0].keys():
        values = [f[key] for f in features]
        
        if key in ["image", "wrist_image"]:
            # Stack images
            batch[key] = torch.stack(values)
        elif key in ["state", "actions"]:
            # Stack numerical arrays
            batch[key] = torch.stack(values)
        elif key == "prompt":
            # Keep prompts as list for tokenization
            batch[key] = values
        else:
            # Default behavior
            batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    
    return batch
