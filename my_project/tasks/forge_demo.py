"""A minimal project task: Isaac Lab's Forge peg insert under a project name.

This is the smallest possible "add a task": subclass the env and its task cfg, change
nothing, register under a new id (see ``__init__.py``). A real project replaces the
``pass`` bodies with its own observations, rewards and resets -- the subclass boundary is
where your task differs from stock Forge.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.direct.forge.forge_env import ForgeEnv
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeTaskPegInsertCfg


@configclass
class ForgeDemoTaskCfg(ForgeTaskPegInsertCfg):
    """The task cfg: override defaults here (episode length, randomization, ctrl...)."""

    pass


class ForgeDemoEnv(ForgeEnv):
    """The env: override _get_rewards / _get_observations / _reset_idx here."""

    pass
