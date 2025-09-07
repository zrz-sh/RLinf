"""
PyTorch implementation of OpenPI dataset configurations.

This module provides dataset configuration classes that match the original
OpenPI configuration system, designed to work with HuggingFace datasets
and PyTorch data loaders.
"""

from typing import Dict, Any, Optional, List, Union, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import json
import abc

from .transforms import (
    DataTransformFn, 
    Group,
    RepackTransform, 
    InjectDefaultPrompt, 
    ResizeImages,
    DeltaActions,
    AbsoluteActions,
    PromptFromLeRobotTask,
    Normalize,
    compose,
    make_bool_mask,
    load_task_descriptions
)
from .io_processing.libero import LiberoInputs, LiberoOutputs
from .io_processing.droid import DroidInputs, DroidOutputs
from .io_processing.aloha import AlohaInputs, AlohaOutputs


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset preprocessing and transforms, matching OpenPI structure."""
    
    # LeRobot repo id. If None, fake data will be created.
    repo_id: Optional[str] = None
    # Directory within the assets directory containing the data assets.
    asset_id: Optional[str] = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: Optional[Dict[str, Any]] = None
    
    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: Group = field(default_factory=Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized.
    data_transforms: Group = field(default_factory=Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: Group = field(default_factory=Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False
    
    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    
    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = True
    
    def create_input_transform(self) -> DataTransformFn:
        """Create the complete input transform pipeline."""
        transforms = []
        transforms.extend(self.repack_transforms.inputs)
        transforms.extend(self.data_transforms.inputs)
        
        # Add normalization if stats are available
        if self.norm_stats is not None:
            transforms.append(Normalize(self.norm_stats, self.use_quantile_norm))
        
        transforms.extend(self.model_transforms.inputs)
        
        return compose(transforms)
    
    def create_output_transform(self) -> DataTransformFn:
        """Create the output transform pipeline (for inference)."""
        # Output transforms are applied in reverse order
        output_transforms = []
        output_transforms.extend(self.model_transforms.outputs)
        output_transforms.extend(self.data_transforms.outputs)
        output_transforms.extend(self.repack_transforms.outputs)
        
        return compose(output_transforms)


@dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """Base class for dataset configuration factories, matching OpenPI structure."""
    
    # The LeRobot repo id.
    repo_id: str
    # Asset configuration  
    asset_id: Optional[str] = None
    
    @abc.abstractmethod
    def create(self, action_dim: int, action_horizon: int, max_token_len: int = 256) -> DataConfig:
        """Create a data configuration."""
        
    def _load_norm_stats(self, dataset_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load normalization statistics if available."""
        stats_file = Path(dataset_path) / "meta" / "norm_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                print(f"Loaded normalization stats from {stats_file}")
                return stats['norm_stats']
            except Exception as e:
                print(f"Failed to load stats from {stats_file}: {e}")
        return None


@dataclass(frozen=True)
class LiberoDataConfig(DataConfigFactory):
    """Configuration factory for Libero datasets."""
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: Optional[str] = None
    
    def create(self, action_dim: int, action_horizon: int, max_token_len: int = 256) -> DataConfig:
        """Create Libero dataset configuration."""
        
        # Load task descriptions if using prompt from task
        tasks = {}
        if Path(self.repo_id).exists():
            tasks = load_task_descriptions(self.repo_id)
        
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment.
        repack_transforms = Group(inputs=[
            RepackTransform({
                "observation/image": "image",
                "observation/wrist_image": "wrist_image", 
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            })
        ])
        
        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        data_transforms = Group(
            inputs=[LiberoInputs(action_dim=action_dim, mask_padding=True)],
            outputs=[LiberoOutputs()],
        )
        
        # Apply delta actions transform to match OpenPI training
        # Models are trained on delta actions (relative to the first state in each action chunk).
        # The first 6 dimensions (joints) are converted to delta actions, while the last dimension
        # (gripper) remains absolute, matching OpenPI's configuration.
        delta_action_mask = make_bool_mask(6, -1)  # First 6 dims delta, last 1 absolute
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask)],
            outputs=[AbsoluteActions(delta_action_mask)],
        )
        
        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = Group(inputs=[
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
        ])
        
        # Load normalization stats
        norm_stats = self._load_norm_stats(self.repo_id) if Path(self.repo_id).exists() else None
        
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            action_sequence_keys=["actions"],
            prompt_from_task=True
        )


@dataclass(frozen=True)
class DroidDataConfig(DataConfigFactory):
    """Configuration factory for DROID datasets."""
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: Optional[str] = None
    
    def create(self, action_dim: int, action_horizon: int, max_token_len: int = 256) -> DataConfig:
        """Create DROID dataset configuration."""
        
        # Load task descriptions if using prompt from task
        tasks = {}
        if Path(self.repo_id).exists():
            tasks = load_task_descriptions(self.repo_id)
        
        # Repack transforms - map DROID dataset keys to standard format
        repack_transforms = Group(inputs=[
            RepackTransform({
                "observation/image": "observation/exterior_image_1_left",
                "observation/wrist_image": "observation/wrist_image_left",
                "observation/joint_position": "observation/joint_position", 
                "observation/gripper_position": "observation/gripper_position",
                "actions": "action",  # Map singular 'action' to plural 'actions'
                "prompt": "prompt"
            })
        ])
        
        # Data transforms - DROID-specific processing
        data_transforms = Group(
            inputs=[DroidInputs(action_dim=action_dim)],
            outputs=[DroidOutputs()],
        )
        
        # DROID typically uses delta actions for all dimensions
        delta_action_mask = make_bool_mask(7, -1)  # First 7 dims delta, last 1 absolute
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask)],
            outputs=[AbsoluteActions(delta_action_mask)],
        )
        
        # Model transforms - standard preprocessing
        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
        ]
        
        # Add prompt from task if tasks are available
        if tasks:
            model_transforms_list.insert(0, PromptFromLeRobotTask(tasks))
            
        model_transforms = Group(inputs=model_transforms_list)
        
        # Load normalization stats
        norm_stats = self._load_norm_stats(self.repo_id) if Path(self.repo_id).exists() else None
        
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            action_sequence_keys=["actions"],
            prompt_from_task=bool(tasks)
        )


@dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    """Simple configuration factory for basic datasets."""
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: Optional[str] = None
    # Robot type for basic categorization
    robot_type: str = "generic"
    
    def create(self, action_dim: int, action_horizon: int, max_token_len: int = 256) -> DataConfig:
        """Create simple dataset configuration."""
        
        # Basic repack transforms - assumes standard LeRobot format (PushT style)
        repack_transforms = Group(inputs=[
            RepackTransform({
                "image": "observation.image",
                "state": "observation.state", 
                "actions": "action",  # Map singular 'action' to plural 'actions'
            })
        ])
        
        # Basic data transforms - minimal processing
        data_transforms = Group()
        
        # Model transforms
        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
        ]
        
        # Load normalization stats and tasks
        norm_stats = None
        tasks = {}
        if Path(self.repo_id).exists():
            norm_stats = self._load_norm_stats(self.repo_id)
            tasks = load_task_descriptions(self.repo_id)
        
        # Add prompt from task if available
        if tasks:
            model_transforms_list.insert(0, PromptFromLeRobotTask(tasks))
        
        model_transforms = Group(inputs=model_transforms_list)
        
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            action_sequence_keys=("actions",),
            prompt_from_task=bool(tasks)
        )


@dataclass(frozen=True)
class AlohaDataConfig(DataConfigFactory):
    """Configuration factory for ALOHA datasets."""
    
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: Optional[str] = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True
    
    def create(self, action_dim: int, action_horizon: int, max_token_len: int = 256) -> DataConfig:
        """Create ALOHA dataset configuration."""
        
        # Load task descriptions if using prompt from task
        tasks = {}
        if Path(self.repo_id).exists():
            tasks = load_task_descriptions(self.repo_id)
        
        # Repack transforms - map ALOHA dataset keys to standard format
        repack_transforms = Group(inputs=[
            RepackTransform({
                "state": "state",  # ALOHA state is already in standard format
                "images": "images",  # ALOHA images dict is already in standard format
                "actions": "action",  # Map singular 'action' to plural 'actions'
                "prompt": "prompt"
            })
        ])
        
        # Data transforms - ALOHA-specific processing
        data_transforms = Group(
            inputs=[AlohaInputs(action_dim=action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        
        # ALOHA typically uses delta actions for joint angles but absolute for grippers
        # First 6 dims (left arm joints) + next 6 dims (right arm joints) = 12 delta
        # Last 2 dims (grippers) = absolute
        delta_action_mask = make_bool_mask(12, -2)  # First 12 dims delta, last 2 absolute
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask)],
            outputs=[AbsoluteActions(delta_action_mask)],
        )
        
        # Model transforms - standard preprocessing
        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
        ]
        
        # Add prompt from task if tasks are available
        if tasks:
            model_transforms_list.insert(0, PromptFromLeRobotTask(tasks))
            
        model_transforms = Group(inputs=model_transforms_list)
        
        # Load normalization stats
        norm_stats = self._load_norm_stats(self.repo_id) if Path(self.repo_id).exists() else None
        
        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            action_sequence_keys=["actions"],
            prompt_from_task=bool(tasks)
        )


# Predefined configurations for common datasets
DATASET_CONFIGS = {
    "libero": LiberoDataConfig,
    "droid": DroidDataConfig,
    "aloha": AlohaDataConfig,
}


def get_dataset_config(dataset_type: str, repo_id: str, **kwargs) -> DataConfigFactory:
    """Get a dataset configuration factory by type."""
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    
    return DATASET_CONFIGS[dataset_type](repo_id, **kwargs)