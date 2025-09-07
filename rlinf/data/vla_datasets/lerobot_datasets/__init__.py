"""
Simplified VLA Datasets with OpenPI-style preprocessing and TRL compatibility.

This package provides a clean, simplified dataset implementation for Vision-Language-Action (VLA)
models that exactly replicates the original OpenPI data loading logic while being fully compatible
with TRL and Transformers trainers.

Key Features:
- Exact replication of OpenPI's data loading pipeline (create_torch_dataset + transform_dataset)
- Full compatibility with TRL's SFTTrainer and Transformers' Trainer
- Support for multimodal data (images, text, proprioceptive state, actions)
- Simplified architecture with fewer classes and cleaner interfaces
- Correct FPS handling from dataset metadata
- Robot-specific preprocessing (Libero, DROID, etc.)

Main Functions:
- create_lerobot_dataset: Create datasets following OpenPI's exact pattern
- vla_data_collator: Handle multi-modal data batching
- LeRobotPyTorchDataset: Core dataset class matching OpenPI logic
"""

# Simplified dataset following OpenPI's exact pattern
from .lerobot_dataset import (
    LeRobotPyTorchDataset,
    create_lerobot_dataset,
    vla_data_collator
)

# Configuration and transforms
from .config import (
    DataConfig,
    DataConfigFactory,
    LiberoDataConfig,
    DroidDataConfig,
    SimpleDataConfig,
    get_dataset_config,
    DATASET_CONFIGS,
)

from .transforms import (
    DataTransformFn,
    Group,
    RepackTransform,
    InjectDefaultPrompt,
    ResizeImages,
    DeltaActions,
    AbsoluteActions,
    Normalize,
    PromptFromLeRobotTask,
    compose,
    make_bool_mask,
    load_task_descriptions,
)

# Policy-specific transforms
from .io_processing.libero import LiberoInputs, LiberoOutputs
from .io_processing.droid import DroidInputs, DroidOutputs

# Version info
__version__ = "0.1.0"

# All exports
__all__ = [
    # Main dataset functions
    "create_lerobot_dataset",
    "vla_data_collator",
    
    # Dataset classes
    "LeRobotPyTorchDataset",
    
    # Configuration
    "DataConfig",
    "DataConfigFactory", 
    "LiberoDataConfig",
    "DroidDataConfig",
    "SimpleDataConfig",
    "get_dataset_config",
    "DATASET_CONFIGS",
    
    # Transforms
    "DataTransformFn",
    "Group",
    "RepackTransform",
    "InjectDefaultPrompt", 
    "ResizeImages",
    "DeltaActions",
    "AbsoluteActions",
    "Normalize",
    "PromptFromLeRobotTask",
    "compose",
    "make_bool_mask",
    "load_task_descriptions",
    
    # Policy transforms
    "LiberoInputs",
    "LiberoOutputs",
    "DroidInputs", 
    "DroidOutputs",
]