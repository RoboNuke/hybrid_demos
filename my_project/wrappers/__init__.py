"""The project's env wrappers.

* :mod:`~my_project.wrappers.env_wrappers` -- the policy interface (world-pose actions), the success reset, the
  observation history, the measured wrench, the Forge hybrid controller, and :func:`project_env_wrappers`, which
  applies every enabled wrapper to every env the package builds;
* :mod:`~my_project.wrappers.residual_policy` -- residual RL on a frozen diffusion policy: the base policy (innermost)
  and the two residuals, additive and multiplicative (outermost of the residual wrappers);
* :mod:`~my_project.wrappers.action_spaces` -- dropped action dims (and observation columns), tracked targets;
* :mod:`~my_project.wrappers.observation_spaces` -- the observation relative to the EE.

The residual wrappers stack in one fixed order, innermost first: base policy, drop columns, target tracking, EE-relative
observation, then the additive or the multiplicative residual; each is on or off by its own config section
(:mod:`my_project.cfg`). :func:`~my_project.wrappers.register.register_project` registers every section and builder an
entry script needs.
"""

from my_project.wrappers.env_wrappers import (  # noqa: F401  (the names the scripts and modules import from here)
    ForceObservation,
    HistoryBuffer,
    HybridController,
    ObservationHistory,
    ResetOnSuccess,
    WorldPoseAction,
    apply_policy_wrappers,
    history_index,
    hybrid_demo_block,
    obs_term_slices,
    project_env_wrappers,
)
