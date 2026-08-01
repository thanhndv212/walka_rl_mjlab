"""Walka-local MDP terms for the get-up task.

Re-exports stock mjlab envs.mdp and tasks.velocity.mdp terms, plus custom
get-up-specific reward/observation/termination/metric functions. The
get-up task needs height/contact observations and height/upright/stand-on-feet
rewards that stock mjlab's velocity template doesn't provide.
"""

from mjlab.envs.mdp import *
from mjlab.tasks.velocity.mdp import *

from .events import *
from .metrics import *
from .observations import *
from .rewards import *
from .terminations import *
