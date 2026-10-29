# scalable-signal-free-marl

Code for the paper *Scalable padding-free networked multi-agent reinforcement learning for signal-free traffic: one policy from a single junction to a 25-junction network* (H. Adel, M. I. Awad, D. Emad, O. M. Shehata).

## About this repository

This repository is a code-only snapshot extracted from our private development repository, at the version used to produce the paper's results. The development history and unrelated work in progress are not included, which is why the history starts with a single commit. Trained checkpoints and evaluation records are not included; they are available from the corresponding author on reasonable request.

## Contents

- `dmpo_autonomous_driving/`: training (`dmpo_full_training.py`), the SUMO environment, observations, rewards and the learning-baseline modules
- `scripts/`: evaluation engines and scenario generators
- `configs/consistency_block.sh`, `configs/training_block.sh`: settings that must match between training and evaluation
- `configs/runs/`: the exact settings and command line of each trained learner
- `configs/sumo_configs/`: the SUMO scenarios
- `third_party/`: our changes to the Networked-MB-MARL framework, as a patch

## Setup

1. Python 3.10, then `pip install -r requirements.txt` (installs SUMO 1.27.1).
2. The underlying framework (DMPO; Ma et al., *Nature Machine Intelligence*, 2024) is not redistributed here. Download it at the version our changes are based on, and apply the patch:

   ```
   git clone https://github.com/CDM1619/Networked-MB-MARL third_party/Networked-MB-MARL
   cd third_party/Networked-MB-MARL
   git checkout 25d2e37a14901ab06b21e94307d82772d62183b0
   git apply ../networked_mb_marl.patch
   ```

   The code looks for the framework in `third_party/Networked-MB-MARL`; set `DMPO_FRAMEWORK_PATH` to use another location.
3. Run from the repository root with `PYTHONPATH=.`. Each file in `configs/runs/` gives the command line in its header and the environment variables to export.

## Licence

Our code is released under the MIT licence (see `LICENSE`). The licence covers our changes in `third_party/networked_mb_marl.patch`, not the Networked-MB-MARL framework itself; the framework, including the unchanged lines the patch shows as context, remains under its authors' terms.
