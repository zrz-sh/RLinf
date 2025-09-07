"""
Libero-specific data transforms that match the original OpenPI implementation.

This module provides transforms specifically designed for Libero datasets,
converting them to the standard format expected by PI0 models.
"""

from typing import Dict, Any, Optional
import torch
import numpy as np
# from ..transforms import DataTransformFn
from .. import transforms as _transforms


class LiberoInputs(_transforms.DataTransformFn):
    """Libero input transforms for preprocessing Libero dataset format."""
    
    def __init__(self, action_dim: int, mask_padding: bool = True):
        self.action_dim = action_dim
        self.mask_padding = mask_padding
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Libero data format to PyTorch tensor format.
        
        Expected input format (after repack):
        - observation/image: main camera image (tensor)
        - observation/wrist_image: wrist camera image (tensor)
        - observation/state: robot proprioceptive state (tensor)
        - actions: action sequence (tensor)
        - prompt: task description (string)
        """
        # Process state - pad to action dimension if needed (OpenPI logic)
        state = self._process_state(data["observation/state"])

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
                "right_wrist_0_rgb": torch.tensor(False, dtype=torch.bool) if self.mask_padding else torch.tensor(True, dtype=torch.bool),
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
    
    def _process_state(self, state) -> torch.Tensor:
        """Process state tensor with device-aware padding."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        state = _transforms.pad_to_dim(state, self.action_dim, axis=-1)
        
        return state
    
    def _process_actions(self, actions) -> torch.Tensor:
        """Process actions tensor with device-aware padding."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        
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


class LiberoOutputs(_transforms.DataTransformFn):
    """Libero output transforms for PyTorch tensor outputs."""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform model outputs back to Libero format.
        
        Extract the first 7 actions (remove any padding) and ensure proper tensor format.
        """
        if "actions" in data:
            actions = data["actions"]
            if isinstance(actions, torch.Tensor):
                # Keep as tensor but extract first 7 dimensions
                actions = actions[..., :7]
            else:
                # Convert to tensor if needed
                actions = torch.tensor(actions, dtype=torch.float32)[..., :7]
            
            return {"actions": actions}
        
        return data