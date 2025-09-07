# OpenVLA Supervised Fine-Tuning (SFT) with LeRobot Datasets

This implementation provides a complete pipeline for supervised fine-tuning of OpenVLA models using LeRobot-format datasets. The SFT approach trains the model to predict actions directly from demonstrations without reinforcement learning.

## Overview

The SFT pipeline includes:
- **Data Loading**: Seamless integration with LeRobot datasets
- **Model Training**: Standard supervised learning with cross-entropy loss
- **Distributed Training**: FSDP support for multi-GPU training
- **Checkpointing**: Save/resume functionality
- **Logging**: Metrics tracking and visualization

## Quick Start

### 1. Install Dependencies

```bash
# Install RLinf with embodiment dependencies
pip install -e .
pip install -r requirements/openvla.txt

# Install LeRobot for dataset loading
pip install lerobot
```

### 2. Run SFT Training

```bash
cd examples/embodiment

# Basic training with default config
./run_sft.sh

# Use custom dataset
./run_sft.sh openvla_sft dataset.repo_id=lerobot/your_dataset

# Multi-GPU training
./run_sft.sh openvla_sft cluster.num_gpus_per_node=4

# Resume from checkpoint
./run_sft.sh openvla_sft actor.resume_from_checkpoint=./logs/sft_training/checkpoints/epoch_5/checkpoint.pt
```

## Configuration

### Dataset Configuration

The SFT pipeline supports any LeRobot dataset. Configure your dataset in the YAML file:

```yaml
actor:
  dataset:
    repo_id: lerobot/aloha_sim_insertion_human_v0  # LeRobot dataset ID
    action_horizon: 10  # Number of future actions to predict
    action_dim: 7       # Robot action dimensionality
    batch_size: 4       # Training batch size
    max_token_len: 256  # Maximum sequence length
```

### Model Configuration

Configure the OpenVLA model parameters:

```yaml
actor:
  model:
    action_dim: 7              # Must match dataset
    num_action_chunks: 10      # Should match action_horizon
    hidden_size: 4096          # Model hidden size
  
  checkpoint_load_path: openvla/openvla-7b  # HuggingFace model path
```

### Training Configuration

Control the training process:

```yaml
runner:
  max_epochs: 10           # Number of training epochs
  save_interval: 2         # Save checkpoint every N epochs

actor:
  optimizer:
    lr: 1e-5               # Learning rate
    weight_decay: 0.01     # Weight decay
  
  grad_clip_norm: 1.0      # Gradient clipping
```

## Architecture

### Key Components

1. **EmbodiedFSDPSFTActor** (`rlinf/workers/actor/fsdp_sft_actor_worker.py`)
   - Handles model training with FSDP
   - Implements supervised loss computation
   - Manages dataset loading and processing

2. **EmbodiedSFTRunner** (`rlinf/runners/embodied_sft_runner.py`)
   - Orchestrates the training loop
   - Handles checkpointing and logging
   - Manages training epochs

3. **LeRobot Dataset Integration** (`rlinf/data/vla_datasets/lerobot_datasets/`)
   - Seamless loading of LeRobot datasets
   - Handles action tokenization
   - Supports various robot platforms (ALOHA, DROID, etc.)

### Data Flow

```
LeRobot Dataset → DataLoader → Processor → OpenVLA Model → Loss → Optimizer
                                    ↓
                              Action Tokens ← Discretization ← Continuous Actions
```

## Supported Datasets

The implementation supports all LeRobot datasets:

- **ALOHA**: `lerobot/aloha_sim_*` or `lerobot/aloha_real_*`
- **DROID**: `lerobot/droid_*`
- **BridgeData**: `lerobot/bridge_*`
- **Custom datasets**: Any dataset following LeRobot format

### Local Datasets

You can also use local LeRobot datasets:

```yaml
actor:
  dataset:
    repo_id: /path/to/local/dataset  # Local path
```

## Advanced Usage

### Multi-GPU Training

For distributed training across multiple GPUs:

```yaml
cluster:
  num_nodes: 1
  num_gpus_per_node: 4

actor:
  fsdp_config:
    sharding_strategy: FULL_SHARD
    mixed_precision:
      param_dtype: torch.bfloat16
```

### Memory Optimization

For limited GPU memory:

```yaml
actor:
  enable_offload: true      # Offload parameters to CPU
  dataset:
    batch_size: 1           # Reduce batch size
  
  fsdp_config:
    cpu_offload: true
```

### Custom Action Spaces

For robots with different action spaces:

```yaml
actor:
  dataset:
    action_dim: 14          # Your robot's action dimension
  model:
    action_dim: 14          # Must match dataset
    num_action_chunks: 20   # Adjust based on your needs
```

## Monitoring and Logging

### Metrics

The SFT pipeline tracks:
- `sft/loss`: Cross-entropy loss
- `sft/perplexity`: Model perplexity 
- `sft/epoch`: Current epoch
- `sft/global_step`: Global training step
- `sft/lr`: Learning rate
- `time/epoch`: Time per epoch

### Checkpointing

Checkpoints are saved to `{log_path}/checkpoints/`:
- `epoch_N/`: Regular epoch checkpoints
- `final/`: Final model checkpoint

### Resume Training

Resume from any checkpoint:

```bash
./run_sft.sh openvla_sft actor.resume_from_checkpoint=./logs/sft_training/checkpoints/epoch_5/checkpoint.pt
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `batch_size`
   - Enable `enable_offload: true`
   - Use mixed precision training

2. **Dataset Loading Issues**
   - Verify LeRobot dataset exists
   - Check internet connection for remote datasets
   - Ensure local dataset paths are correct

3. **Action Dimension Mismatch**
   - Verify `action_dim` matches your robot
   - Check dataset action space in LeRobot

### Performance Tips

1. **Faster Training**
   - Increase `batch_size` if memory allows
   - Use multiple `num_workers` for data loading
   - Enable mixed precision training

2. **Better Convergence**
   - Adjust learning rate based on dataset size
   - Use learning rate scheduling
   - Monitor training metrics

## Integration with RL Pipeline

After SFT training, you can use the fine-tuned model for RL training:

1. Save the SFT checkpoint
2. Use it as initialization for RL training:

```yaml
actor:
  checkpoint_load_path: ./logs/sft_training/checkpoints/final/
```

## Citation

If you use this SFT implementation, please cite:

```bibtex
@misc{rlinf_sft,
  title={OpenVLA Supervised Fine-Tuning with LeRobot Datasets},
  author={RLinf Team},
  year={2025},
  url={https://github.com/your-repo/rlinf}
}
```