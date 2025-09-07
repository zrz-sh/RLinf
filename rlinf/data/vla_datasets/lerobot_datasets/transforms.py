"""
PyTorch implementation of OpenPI transforms for dataset preprocessing.

This module provides PyTorch-based transforms that match the functionality
of the original JAX-based OpenPI transforms, designed to work with 
HuggingFace datasets and the broader PyTorch ecosystem.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Sequence
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import copy
import re
import json


class DataTransformFn:
    """Base class for data transforms."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation to the data.
        
        Args:
            data: Dictionary containing the data to transform
            
        Returns:
            Transformed data dictionary
        """
        raise NotImplementedError


@dataclass
class Group:
    """A group of transforms matching OpenPI's structure."""
    
    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()
    
    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()
    
    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.
        
        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.
            
        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


class CompositeTransform(DataTransformFn):
    """Applies a sequence of transforms in order."""
    
    def __init__(self, transforms: List[DataTransformFn]):
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(list(transforms))


class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary structure.
    
    This matches the OpenPI RepackTransform functionality, allowing us to
    remap keys from dataset-specific formats to a common format.
    """
    
    def __init__(self, structure: Dict[str, str]):
        """
        Args:
            structure: Mapping from new keys to old keys (flattened paths)
        """
        self.structure = structure
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        
        # Simple flat key mapping for LeRobot datasets
        for new_key, old_key in self.structure.items():
            if old_key in data:
                result[new_key] = data[old_key]
            else:
                print(f"Warning: Key '{old_key}' not found in data")
        
        return result
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class InjectDefaultPrompt(DataTransformFn):
    """Injects a default prompt if none is present."""
    
    def __init__(self, prompt: Optional[str]):
        self.prompt = prompt
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = self.prompt
        return data


class ResizeImages(DataTransformFn):
    """Resizes images to the specified dimensions."""
    
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in data:
            if isinstance(data["image"], dict):
                # Multiple images case
                for key, img in data["image"].items():
                    data["image"][key] = self._resize_image(img)
            else:
                # Single image case
                data["image"] = self._resize_image(data["image"])
        return data
    
    def _resize_image(self, img: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        """Resize image maintaining PyTorch CHW format."""
        if isinstance(img, Image.Image):
            img = torch.from_numpy(np.array(img))
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        
        # Ensure CHW format for torchvision
        if len(img.shape) == 3:
            if img.shape[-1] == 3:  # HWC -> CHW
                img = img.permute(2, 0, 1)
            # If already CHW (shape[0] == 3), keep as is
        
        # Resize using torchvision (expects CHW)
        import torchvision.transforms.functional as TF
        img = TF.resize(img, (self.height, self.width))
        
        # Return in CHW format (PyTorch standard)
        return img


class DeltaActions(DataTransformFn):
    """Converts absolute actions to delta actions relative to current state."""
    
    def __init__(self, mask: Optional[List[bool]]):
        self.mask = mask
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "actions" not in data or self.mask is None:
            return data
        
        if "state" not in data:
            print("Warning: DeltaActions requires 'state' but it's not present")
            return data
        
        state = data["state"]
        actions = data["actions"]
        
        # Convert to tensors if needed, preserving device of actions tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        device = actions.device
        dtype = actions.dtype
        
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=dtype, device=device)
        else:
            state = state.to(device=device, dtype=dtype)
        
        mask = torch.tensor(self.mask, dtype=torch.bool, device=device)
        dims = len(mask)
        
        # Apply delta conversion only to masked dimensions
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add time dimension
        
        # Subtract current state from actions for masked dimensions
        state_expanded = state[:dims].unsqueeze(0).expand(actions.shape[0], -1)
        mask_expanded = mask.unsqueeze(0).expand(actions.shape[0], -1)
        
        actions_copy = actions.clone()
        actions_copy[:, :dims] = torch.where(
            mask_expanded, 
            actions[:, :dims] - state_expanded, 
            actions[:, :dims]
        )
        
        data["actions"] = actions_copy
        data["state"] = state  # Preserve the converted tensor state
        return data


class AbsoluteActions(DataTransformFn):
    """Converts delta actions to absolute actions by adding current state."""
    
    def __init__(self, mask: Optional[List[bool]]):
        self.mask = mask
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "actions" not in data or self.mask is None:
            return data
        
        if "state" not in data:
            print("Warning: AbsoluteActions requires 'state' but it's not present")
            return data
        
        state = data["state"]
        actions = data["actions"]
        
        # Convert to tensors if needed, preserving device of actions tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        device = actions.device
        dtype = actions.dtype
        
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=dtype, device=device)
        else:
            state = state.to(device=device, dtype=dtype)
        
        mask = torch.tensor(self.mask, dtype=torch.bool, device=device)
        dims = len(mask)
        
        # Apply absolute conversion only to masked dimensions
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add time dimension
        
        # Add current state to actions for masked dimensions
        state_expanded = state[:dims].unsqueeze(0).expand(actions.shape[0], -1)
        mask_expanded = mask.unsqueeze(0).expand(actions.shape[0], -1)
        
        actions_copy = actions.clone()
        actions_copy[:, :dims] = torch.where(
            mask_expanded, 
            actions[:, :dims] + state_expanded, 
            actions[:, :dims]
        )
        
        data["actions"] = actions_copy
        data["state"] = state  # Preserve the converted tensor state
        return data


class PromptFromLeRobotTask(DataTransformFn):
    """Extracts prompt from LeRobot dataset task following OpenPI implementation."""
    
    def __init__(self, tasks: Dict[int, str]):
        self.tasks = tasks
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')
        
        task_index = data["task_index"]
        if isinstance(task_index, torch.Tensor):
            task_index = task_index.item()
        elif isinstance(task_index, (list, tuple)):
            task_index = task_index[0]
        
        task_index = int(task_index)
        
        # Following OpenPI pattern: check if task exists, with fallback for -1
        if task_index in self.tasks:
            prompt = self.tasks[task_index]
        elif task_index == -1:
            # Handle special case of -1 (unknown task) with default prompt
            prompt = "Complete the task"
        else:
            raise ValueError(f"task_index={task_index} not found in task mapping: {self.tasks}")
        
        # Return new dict with prompt added (following OpenPI pattern)
        return {**data, "prompt": prompt}


class Normalize(DataTransformFn):
    """Normalizes data using precomputed statistics."""
    
    def __init__(self, norm_stats: Optional[Dict[str, Dict[str, float]]], use_quantiles: bool = False, strict: bool = False):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.norm_stats is None:
            return data
        
        # Keys that should NOT be normalized (categorical variables)
        SKIP_NORMALIZATION = {'task_index', 'episode_index', 'frame_index', 'index'}
        
        def _normalize_value(value, stats):
            # Preserve device and dtype of input tensor
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            
            device = value.device
            dtype = value.dtype
            
            if self.use_quantiles:
                q01, q99 = stats.get('q01', 0), stats.get('q99', 1)
                # Convert to tensors with same device/dtype if they are lists/arrays
                if not isinstance(q01, torch.Tensor):
                    q01 = torch.tensor(q01, dtype=dtype, device=device)
                else:
                    q01 = q01.to(device=device, dtype=dtype)
                if not isinstance(q99, torch.Tensor):
                    q99 = torch.tensor(q99, dtype=dtype, device=device)
                else:
                    q99 = q99.to(device=device, dtype=dtype)
                return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                mean, std = stats.get('mean', 0), stats.get('std', 1)
                # Convert to tensors with same device/dtype if they are lists/arrays
                if not isinstance(mean, torch.Tensor):
                    mean = torch.tensor(mean, dtype=dtype, device=device)
                else:
                    mean = mean.to(device=device, dtype=dtype)
                if not isinstance(std, torch.Tensor):
                    std = torch.tensor(std, dtype=dtype, device=device)
                else:
                    std = std.to(device=device, dtype=dtype)
                return (value - mean) / (std + 1e-6)
        
        # Apply normalization to each key that has stats, except categorical variables
        result = copy.deepcopy(data)
        flat_data = self._flatten_dict(result)
        
        for key, stats in self.norm_stats.items():
            if key in flat_data and key not in SKIP_NORMALIZATION:
                flat_data[key] = _normalize_value(flat_data[key], stats)
        
        # Reconstruct nested structure
        result = self._unflatten_dict(flat_data)
        return result
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _unflatten_dict(self, d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
        """Unflatten a dictionary."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result


def pad_to_dim(x: torch.Tensor, target_dim: int, axis: int = -1) -> torch.Tensor:
    """Pad a tensor to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_size = target_dim - current_dim
        # Create padding list for torch.nn.functional.pad (works from last dim to first)
        pad_list = [0, 0] * len(x.shape)
        # Convert negative axis to positive
        if axis < 0:
            axis = len(x.shape) + axis
        # Set padding for the target axis (counting from the end)
        pad_index = (len(x.shape) - 1 - axis) * 2 + 1
        pad_list[pad_index] = pad_size
        
        # Create zeros with same device and dtype as input tensor
        return torch.nn.functional.pad(x, pad_list, value=0.0)
    return x


def make_bool_mask(*dims: int) -> List[bool]:
    """Make a boolean mask for the given dimensions.
    
    Example:
        make_bool_mask(2, -2, 2) == [True, True, False, False, True, True]
        make_bool_mask(2, 0, 2) == [True, True, True, True]
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * dim)
        else:
            result.extend([False] * (-dim))
    return result


def load_task_descriptions(dataset_path: Union[str, Path]) -> Dict[int, str]:
    """Load task descriptions from dataset, handling multiple file formats.
    
    Supports:
    - tasks.jsonl: JSON Lines format with {"task_index": int, "task": str}
    - tasks.parquet: Parquet format with task descriptions as index and task_index as column
    """
    dataset_path = Path(dataset_path)
    meta_path = dataset_path / "meta"
    
    # Try different task file formats
    tasks_jsonl = meta_path / "tasks.jsonl"
    tasks_parquet = meta_path / "tasks.parquet"
    
    if tasks_jsonl.exists():
        return _load_tasks_jsonl(tasks_jsonl)
    elif tasks_parquet.exists():
        return _load_tasks_parquet(tasks_parquet)
    else:
        print(f"Warning: No task files found in {meta_path}")
        return {}

def _load_tasks_jsonl(tasks_file: Path) -> Dict[int, str]:
    """Load tasks from JSON Lines format."""
    tasks = {}
    with open(tasks_file, 'r') as f:
        for line in f:
            if line.strip():
                task_data = json.loads(line.strip())
                tasks[task_data['task_index']] = task_data['task']
    
    print(f"Loaded {len(tasks)} task descriptions from {tasks_file}")
    return tasks

def _load_tasks_parquet(tasks_file: Path) -> Dict[int, str]:
    """Load tasks from Parquet format."""
    try:
        import pyarrow.parquet as pq
        
        # Read parquet file
        table = pq.read_table(tasks_file)
        df = table.to_pandas()
        
        # In DROID format, task descriptions are the index and task_index is the column
        tasks = {}
        for task_description, row in df.iterrows():
            task_index = int(row['task_index'])  # Convert numpy int64 to Python int
            tasks[task_index] = task_description
        
        print(f"Loaded {len(tasks)} task descriptions from {tasks_file}")
        return tasks
        
    except ImportError:
        print("Warning: pyarrow not available, cannot load parquet task files")
        return {}
    except Exception as e:
        print(f"Warning: Failed to load parquet task file {tasks_file}: {e}")
        return {}