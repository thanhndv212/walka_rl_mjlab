"""Walka-local additions on top of mjlab's stock velocity task mdp.

Re-exports everything from mjlab.tasks.velocity.mdp (the stock template
walka_rl_mjlab is built on) and adds a small set of terms ported from
unitree_rl_mjlab's own local velocity-task fork that stock mjlab doesn't
have: a gait-clock phase observation and its matching clock-based gait
reward, plus a stand-still pose penalty. See observations.py/rewards.py
docstrings for why only these three were ported rather than the whole
fork (stock has since gained features unitree's fork predates, e.g.
TerrainHeightSensorCfg-based foot height, foot_swing_height, world/forward
-frame velocity commands, out_of_terrain_bounds termination).
"""

from mjlab.tasks.velocity.mdp import *

from .observations import *
from .rewards import *
