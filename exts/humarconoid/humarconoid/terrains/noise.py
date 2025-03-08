"""*******************************************************************************
* HumARConoid
*
* Advanced Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
*******************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen

from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

NOISE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(2.0, 2.0),
    border_width=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.45, grid_height_range=(0.00, 0.05), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.00, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)
"""Noise terrains configuration."""

