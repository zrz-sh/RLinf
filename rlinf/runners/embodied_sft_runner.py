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

import os
from typing import Optional

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress
from rlinf.workers.actor.fsdp_sft_actor_worker import EmbodiedFSDPSFTActor


class EmbodiedSFTRunner:
    """Runner for Supervised Fine-Tuning of OpenVLA models using LeRobot datasets."""
    
    def __init__(
        self,
        cfg: DictConfig,
        actor: EmbodiedFSDPSFTActor,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.run_timer = run_timer
        
        self.current_epoch = 0
        self.global_step = 0
        
        # Compute max steps and epochs
        self.set_max_steps()
        
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self):
        """Initialize the actor worker."""
        self.actor.init_worker().wait()

    def run_training_epoch(self):
        """Run one training epoch."""
        with self.timer("epoch"):
            # Run epoch on actor
            epoch_future = self.actor.run_epoch()
            epoch_metrics = epoch_future.wait()
        
        return epoch_metrics[0] if isinstance(epoch_metrics, list) else epoch_metrics

    def run(self):
        """Run the complete SFT training."""
        print(f"Starting SFT training for {self.max_epochs} epochs...")
        
        # Load checkpoint if specified
        if hasattr(self.cfg.actor, 'resume_from_checkpoint') and self.cfg.actor.resume_from_checkpoint:
            checkpoint_path = self.cfg.actor.resume_from_checkpoint
            load_future = self.actor.load_checkpoint(checkpoint_path)
            load_future.wait()
        
        for epoch in tqdm(range(self.max_epochs), desc="Training Epochs", ncols=120):
            self.current_epoch = epoch
            
            # Run training epoch
            epoch_metrics = self.run_training_epoch()
            
            # Update global step from actor
            if "sft/global_step" in epoch_metrics:
                self.global_step = epoch_metrics["sft/global_step"]
            
            # Log metrics
            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
            
            # Add epoch info to metrics
            epoch_metrics["sft/epoch"] = epoch
            
            self.metric_logger.log(epoch_metrics, step=epoch)
            self.metric_logger.log(time_metrics, step=epoch)
            
            # Check if we should save checkpoint
            _, save_model, is_train_end = check_progress(
                epoch + 1,
                self.max_epochs,
                val_check_interval=-1,  # No validation in SFT
                save_interval=self.cfg.runner.save_interval,
                progress_ratio=1.0,
                run_time_exceeded=False,
            )
            
            if save_model:
                self._save_checkpoint(epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch}: Loss={epoch_metrics.get('sft/loss', 0.0):.4f}, "
                  f"Perplexity={epoch_metrics.get('sft/perplexity', 0.0):.2f}, "
                  f"Step={self.global_step}")
        
        # Save final checkpoint
        self._save_checkpoint(self.max_epochs - 1, final=True)
        
        self.metric_logger.finish()
        print("SFT training completed!")

    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save training checkpoint."""
        if final:
            checkpoint_dir = os.path.join(
                self.cfg.runner.logger.log_path,
                "checkpoints/final"
            )
        else:
            checkpoint_dir = os.path.join(
                self.cfg.runner.logger.log_path,
                f"checkpoints/epoch_{epoch}"
            )
        
        save_future = self.actor.save_checkpoint(checkpoint_dir, self.global_step)
        save_future.wait()
        
        print(f"{'Final checkpoint' if final else 'Checkpoint'} saved to {checkpoint_dir}")

    def set_max_steps(self):
        """Set maximum training steps and epochs."""
        # For SFT, we typically train for a fixed number of epochs
        self.max_epochs = self.cfg.runner.get("max_epochs", 10)
        
        # Calculate max steps if needed (for compatibility with existing infrastructure)
        # This is an approximation since we don't know exact dataset size at this point
        estimated_steps_per_epoch = self.cfg.runner.get("estimated_steps_per_epoch", 1000)
        self.max_steps = self.max_epochs * estimated_steps_per_epoch
        
        # Override with explicit max_steps if provided
        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = max_steps
            self.max_epochs = max(1, max_steps // estimated_steps_per_epoch)
        
        print(f"Training configuration: {self.max_epochs} epochs, ~{self.max_steps} steps")

    @property
    def epoch(self):
        """Current epoch number."""
        return self.current_epoch