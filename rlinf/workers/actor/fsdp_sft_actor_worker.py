# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
from typing import Dict, Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlinf.data.vla_datasets.lerobot_datasets.lerobot_dataset import create_lerobot_dataset, vla_data_collator
from rlinf.data.vla_datasets.lerobot_datasets.config import DataConfigFactory
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.models.embodiment.openvla_action_model import PrismaticProcessor
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict


class EmbodiedFSDPSFTActor(FSDPModelManager, Worker):
    """FSDP Actor Worker for Supervised Fine-Tuning of OpenVLA models."""
    
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)
        
        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
        
        # SFT-specific setup
        self.current_epoch = 0
        self.global_step = 0
        self.dataset = None
        self.dataloader = None
        self.processor = None
        
        # Channel setup (simplified for SFT)
        self.channel = self.connect_channel(cfg.actor.channel.name)
        self.channel.create_queue(
            cfg.actor.channel.queue_name, maxsize=cfg.actor.channel.queue_size
        )

    def init_worker(self):
        """Initialize model, optimizer, and dataset."""
        self.setup_model_and_optimizer()
        self._setup_dataset()
        self._setup_processor()
        
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def model_provider_func(self):
        """Provide the model for FSDP."""
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def _setup_dataset(self):
        """Setup LeRobot dataset for SFT training."""
        # Create data config factory
        data_config_factory = DataConfigFactory()
        
        # Get dataset configuration from config
        dataset_cfg = self.cfg.actor.get("dataset", {})
        repo_id = dataset_cfg.get("repo_id", "lerobot/aloha_sim_insertion_human_v0")
        action_horizon = dataset_cfg.get("action_horizon", 10)
        action_dim = dataset_cfg.get("action_dim", 7)
        max_token_len = dataset_cfg.get("max_token_len", 256)
        batch_size = dataset_cfg.get("batch_size", 4)
        
        # Create dataset
        self.dataset = create_lerobot_dataset(
            repo_id=repo_id,
            action_horizon=action_horizon,
            split="train",
            data_config_factory=data_config_factory,
            action_dim=action_dim,
            max_token_len=max_token_len
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=vla_data_collator,
            num_workers=dataset_cfg.get("num_workers", 4),
            pin_memory=True
        )
        
        print(f"SFT Dataset initialized: {len(self.dataset)} samples, {len(self.dataloader)} batches per epoch")

    def _setup_processor(self):
        """Setup the processor for tokenizing and processing inputs."""
        from transformers import AutoTokenizer
        from rlinf.models.embodiment.openvla_action_model import PrismaticImageProcessor
        
        # Load tokenizer and image processor
        model_path = self.cfg.actor.checkpoint_load_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        image_processor = PrismaticImageProcessor.from_pretrained(model_path)
        
        # Create processor
        self.processor = PrismaticProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer
        )

    def compute_sft_loss(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Compute supervised fine-tuning loss."""
        # Extract data from batch
        images = batch.get("image", batch.get("wrist_image"))  # Try both keys
        prompts = batch["prompt"]
        actions = batch["actions"]
        
        # Move to device
        images = images.to(self.device)
        actions = actions.to(self.device)
        
        # Process inputs with the processor
        processed = self.processor(
            text=prompts,
            images=images,
            proprio_states=torch.zeros(images.shape[0], 1).to(self.device),  # Placeholder
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Move processed data to device
        input_ids = processed["input_ids"].to(self.device)
        attention_mask = processed["attention_mask"].to(self.device) 
        pixel_values = processed["pixel_values"].to(self.device)
        
        # Discretize actions to tokens (following OpenVLA convention)
        # Actions are normalized to [-1, 1] and then discretized to tokens
        action_tokens = self._discretize_actions(actions)
        
        # Create labels for causal language modeling
        # Labels are input_ids shifted by one position, with action tokens appended
        labels = torch.cat([input_ids[:, 1:], action_tokens], dim=-1)
        full_input_ids = torch.cat([input_ids, action_tokens], dim=-1)
        full_attention_mask = torch.cat([
            attention_mask, 
            torch.ones_like(action_tokens).to(attention_mask.dtype)
        ], dim=-1)
        
        # Forward pass
        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            use_cache=False
        )
        
        loss = outputs.loss
        
        # Compute metrics
        metrics = {
            "sft/loss": loss.detach().item(),
            "sft/perplexity": torch.exp(loss).detach().item(),
            "sft/batch_size": images.shape[0]
        }
        
        return {"loss": loss, "metrics": metrics}

    def _discretize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert continuous actions to discrete tokens following OpenVLA convention."""
        # Actions should be normalized to [-1, 1] 
        # OpenVLA uses the last n_action_bins tokens of vocabulary for actions
        
        # Get model config
        n_action_bins = getattr(self.model.config, 'n_action_bins', 256)
        vocab_size = self.model.config.text_config.vocab_size
        
        # Normalize actions to [-1, 1] if not already
        actions_normalized = torch.clamp(actions, -1, 1)
        
        # Map to discrete bins [0, n_action_bins-1]
        action_bins = ((actions_normalized + 1) / 2 * (n_action_bins - 1)).round().long()
        
        # Convert to vocabulary tokens (use the last n_action_bins tokens)
        action_tokens = vocab_size - n_action_bins + action_bins
        
        return action_tokens

    def run_training_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single training step."""
        # Load model parameters if offloaded
        if next(self.model.parameters()).is_cpu:
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        loss_dict = self.compute_sft_loss(batch)
        loss = loss_dict["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.cfg.actor, 'grad_clip_norm') and self.cfg.actor.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.actor.grad_clip_norm)
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update global step
        self.global_step += 1
        
        # Add step info to metrics
        loss_dict["metrics"]["sft/global_step"] = self.global_step
        loss_dict["metrics"]["sft/epoch"] = self.current_epoch
        if self.scheduler is not None:
            loss_dict["metrics"]["sft/lr"] = self.scheduler.get_last_lr()[0]
        
        return loss_dict["metrics"]

    def run_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        epoch_metrics = {}
        step_metrics_list = []
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Run training step
            step_metrics = self.run_training_step(batch)
            step_metrics_list.append(step_metrics)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{step_metrics['sft/loss']:.4f}",
                "ppl": f"{step_metrics['sft/perplexity']:.2f}",
                "step": step_metrics['sft/global_step']
            })
        
        # Aggregate epoch metrics
        if step_metrics_list:
            for key in step_metrics_list[0].keys():
                if key.startswith("sft/") and key not in ["sft/global_step", "sft/epoch"]:
                    epoch_metrics[key] = np.mean([m[key] for m in step_metrics_list])
            
            # Add epoch info
            epoch_metrics["sft/epoch"] = self.current_epoch
            epoch_metrics["sft/global_step"] = self.global_step
        
        # All-reduce metrics across processes
        epoch_metrics = all_reduce_dict(epoch_metrics)
        
        self.current_epoch += 1
        return epoch_metrics

    def save_checkpoint(self, output_dir: str, global_step: int):
        """Save model checkpoint."""
        if next(self.model.parameters()).is_cpu:
            self.load_fsdp_param_and_grad(self.device)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FSDP model state dict
        state_dict = self.get_model_state_dict()
        
        if self._rank == 0:
            checkpoint = {
                "model_state_dict": state_dict,
                "global_step": global_step,
                "epoch": self.current_epoch,
            }
            
            # Save optimizer state if needed
            if hasattr(self, 'optimizer'):
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
            
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            
            torch.save(checkpoint, os.path.join(output_dir, "checkpoint.pt"))
            print(f"Checkpoint saved at step {global_step} to {output_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Load model state
            self.load_model_state_dict(checkpoint["model_state_dict"])
            
            # Load optimizer state
            if hasattr(self, 'optimizer') and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state  
            if hasattr(self, 'scheduler') and self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Load step info
            self.global_step = checkpoint.get("global_step", 0)
            self.current_epoch = checkpoint.get("epoch", 0)
            
            print(f"Checkpoint loaded from {checkpoint_path} (step {self.global_step}, epoch {self.current_epoch})")
        else:
            print(f"No checkpoint found at {checkpoint_path}")