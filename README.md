# ARC Humanoid Robot Training using Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository contains a project developed for training _**humanoid robots**_ from the _[KIST Advanced Robot Control Lab.](https://sites.google.com/view/kist-arc/)_ using Isaac Lab. It follows the structure of the Isaac Lab extension template.

**Humanoid Robots:**

- **New KIST Humanoid Robot**: Coming soon!
- **MAHRU**: MAHRU is a wheel-legged humanoid robot developed by KIST. This robot combines wheels and legs for enhanced mobility!
- **G1**: 29-DoF G1 robot from Unitree Robotics.
> Also contains the 'Anymal-D' robot as an example!

**Keywords:** humanoid, reinforcement learning, isaaclab, sim2sim, sim2real
**Maintainer, Author:** [Sol Choi](https://github.com/S-CHOI-S)

</br>

## Installation

**Step 1.** Clone this git repository
```
git clone https://github.com/S-CHOI-S/Humarconoid.git
```

**Step 2.** (Optional) Rename all occurrences of humarconoid (in files/directories) to your_fancy_extension_name
```
python scripts/rename_template.py your_fancy_extension_name
```

**Step 3.** Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html)

**Step 4.** Using a python interpreter that has Isaac Lab installed, install the library
- `humarcscripts` library
    ```
    python -m pip install -e .
    ```
- `humarconoid` library
    ```
    cd source/humarconoid
    python -m pip install -e .
    ```


</br>

## Usage
### Train
**Step 1.** Check out the environments available in the Humarconoid project.
```
python scripts/list_envs.py
```
**Step 2.** Start training with humarconoid `TASK`!
```
python scripts/rsl_rl/train.py --task TASK --headless
```
>[!Tip]
> Humarconoid is compatible with `rsl_rl`, `Stable Baselines3`, as well as custom reinforcement learning algorithms.

### Play
**Step 1.** Choose your `TASK` and `LOGDIR` from your log.
```
python scripts/rsl_rl/play.py --task TASK --log_dir LOGDIR --num_envs NUM_ENVS
```

### Monitor
You can monitor real-time training logs via TensorBoard!
```
tensorboard --logdir logs/rsl_rl/TASK/
```


</br>

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
