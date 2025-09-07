"""
DROID-specific data transforms that match the original OpenPI implementation.

This module provides transforms specifically designed for DROID datasets,
converting them to the standard format expected by PI0 models.
"""

from typing import Dict, Any, Optional
import torch
import numpy as np
# from ..transforms import DataTransformFn
from .. import transforms as _transforms


class DroidInputs(_transforms.DataTransformFn):
    """DROID input transforms that match OpenPI's droid_policy.DroidInputs."""
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform DROID data format to PyTorch tensor format.
        
        Expected input format (after repack):
        - observation/image: main exterior camera image (mapped from observation/exterior_image_1_left)
        - observation/wrist_image: wrist camera image (mapped from observation/wrist_image_left)
        - observation/joint_position: joint positions
        - observation/gripper_position: gripper position
        - actions: action sequence (mapped from action)
        - prompt: task description
        """
        # Process state - concatenate joint and gripper positions like OpenPI
        state = self._process_state(data)
        
        # Process images - handle both tensor and numpy inputs
        base_image = self._process_image(data["observation/image"])
        wrist_image = self._process_image(data["observation/wrist_image"])
        
        # Create right wrist placeholder (same device as base image)
        right_wrist_image = torch.zeros_like(base_image)
        
        # Create inputs dict in format expected by PI0 data collator
        # Use OpenPI's camera naming convention with dictionaries
        inputs = {
            "state": state,
            # OpenPI expects 'images' as a dict with camera keys
            "images": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_masks": {
                "base_0_rgb": torch.tensor(True, dtype=torch.bool),
                "left_wrist_0_rgb": torch.tensor(True, dtype=torch.bool),
                "right_wrist_0_rgb": torch.tensor(False, dtype=torch.bool),
            },
        }
        
        # Handle actions - pad to model action dimension if needed
        if "actions" in data:
            actions = self._process_actions(data["actions"])
            inputs["actions"] = actions
        
        # PI0 collator expects 'task' key, not 'prompt'
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs
    
    def _process_state(self, data: Dict[str, Any]) -> torch.Tensor:
        """Process state by concatenating joint and gripper positions, matching OpenPI logic."""
        # Try to get joint and gripper positions
        joint_pos = data["observation/joint_position"]
        gripper_pos = data["observation/gripper_position"]
        
        # Convert to tensors if needed
        if joint_pos is not None and not isinstance(joint_pos, torch.Tensor):
            joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
        if gripper_pos is not None and not isinstance(gripper_pos, torch.Tensor):
            gripper_pos = torch.tensor(gripper_pos, dtype=torch.float32)
        
        # Concatenate joint and gripper positions like OpenPI does
        state = torch.cat([joint_pos, gripper_pos], dim=-1)
        
        # Pad to action dimension using codebase utility
        state = _transforms.pad_to_dim(state, self.action_dim, axis=-1)
        
        return state
    
    def _process_actions(self, actions) -> torch.Tensor:
        """Process actions tensor with device-aware padding."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        
        # Ensure proper shape
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add time dimension
        
        actions = _transforms.pad_to_dim(actions, self.action_dim, axis=-1)
        
        return actions
    
    def _process_image(self, img) -> torch.Tensor:
        """
        Process image following OpenPI's _parse_image logic but using PyTorch.
        
        Keeps images as uint8[0,255] and ensures CHW format, matching OpenPI's behavior.
        The conversion to float32[-1,1] will happen later in the image processor.
        """
        # Convert to tensor if needed
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        elif not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        
        # Follow OpenPI's logic: if floating point, scale to [0,255] and convert to uint8
        if img.dtype.is_floating_point:
            img = (255 * img).to(torch.uint8)
        
        # Ensure CHW format (standard PyTorch convention) but keep as uint8
        if len(img.shape) == 3:
            if img.shape[-1] == 3:  # HWC -> CHW
                img = img.permute(2, 0, 1)
            # If already CHW (shape[0] == 3), keep as is
        
        # Ensure uint8 dtype (matching OpenPI's approach)
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        
        return img


class DroidOutputs(_transforms.DataTransformFn):
    """DROID output transforms that match OpenPI's droid_policy.DroidOutputs."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform model outputs back to DROID format.
        
        Extract the first 8 actions (matching OpenPI behavior) and ensure proper tensor format.
        """
        actions = data["actions"]
        if isinstance(actions, torch.Tensor):
            # Keep as tensor but extract first 8 dimensions (matching OpenPI)
            actions = actions[..., :8]
        else:
            # Convert to tensor if needed
            actions = torch.tensor(actions, dtype=torch.float32)[..., :8]
        
        return {"actions": actions}