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

import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.runners.embodied_sft_runner import EmbodiedSFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_sft_actor_worker import EmbodiedFSDPSFTActor

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="openvla_sft"
)
def main(cfg) -> None:
    # cfg = validate_cfg(cfg)

    cluster = Cluster(
        num_nodes=cfg.cluster.num_nodes, num_gpus_per_node=cfg.cluster.num_gpus_per_node
    )
    component_placement = HybridComponentPlacement(cfg)

    # Create SFT actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = EmbodiedFSDPSFTActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Create SFT runner (no rollout or env workers needed)
    runner = EmbodiedSFTRunner(
        cfg=cfg,
        actor=actor_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()