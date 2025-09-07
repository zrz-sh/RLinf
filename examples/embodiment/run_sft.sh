#!/bin/bash

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

# OpenVLA Supervised Fine-Tuning Training Script
# Example usage:
#   ./run_sft.sh                                    # Use default config
#   ./run_sft.sh openvla_sft_custom                # Use custom config
#   ./run_sft.sh openvla_sft dataset.repo_id=my_dataset  # Override config values

set -e

# Default configuration
CONFIG_NAME=${1:-openvla_sft}
shift || true  # Remove the first argument, keep the rest for hydra overrides

echo "Starting OpenVLA Supervised Fine-Tuning..."
echo "Configuration: $CONFIG_NAME"
echo "Additional arguments: $@"
echo "==========================================="

# Check if we're in the correct directory
if [ ! -f "train_sft_agent.py" ]; then
    echo "Error: train_sft_agent.py not found. Make sure you're in the examples/embodiment directory."
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# Run the training
python train_sft_agent.py \
    --config-name=$CONFIG_NAME \
    "$@"

echo "==========================================="
echo "SFT Training completed!"