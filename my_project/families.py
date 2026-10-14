"""The two env families this project runs on, and the few things that differ between them.

* **forge** -- Isaac Lab's Direct-env Forge tasks (peg insert, nut thread): the observation
  is ``task.cfg.obs_order`` of Factory's terms, a Direct env's hooks (``_get_observations``,
  integer ``cfg.observation_space``), Forge's reward recomputable offline.
* **furniture** -- ``Isaac-Furniture-*-FB-v0`` from Robonuke_Furniture_Bench, manager-based:
  the observation is FurnitureBench's ``DEFAULT_STATE_OBS`` (``FB_OBS_LAYOUT`` there), the
  observation manager computes it, the reward is the demo's recorded sparse success. The
  ``-FBAbs-v0`` task is the same family with the absolute rot6d interface (``FBABS_*`` there:
  observation 61, action 10); :func:`is_abs` tells them apart where the layout matters.

Everything that must know the family asks here (:func:`family_of`, :func:`obs_layout`), so
the wrappers, the residual demo conversion and the config checks keep one code path each
(``my_project/envhooks.py`` is the live-env side of the same split).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = ["FORGE", "FURNITURE", "family_of", "is_abs", "obs_layout", "obs_term_slices_of", "action_layout", "action_metric_groups"]

FORGE = "forge"
FURNITURE = "furniture"


def family_of(task_name: str) -> str:
    """``Isaac-Furniture-*`` is the furniture family; anything else is the Forge family the
    project was built on (its Direct-env tasks, and the fakes the tests register)."""
    return FURNITURE if "Furniture" in str(task_name) else FORGE


def is_abs(task_name: str) -> bool:
    """The furniture family's absolute-target, rot6d interface (``Isaac-Furniture-*-FBAbs-v0``)."""
    return family_of(task_name) == FURNITURE and "FBAbs" in str(task_name)


def obs_layout(task_name: str, task_cfg: Any) -> List[Tuple[str, int]]:
    """``[(term, width), ...]`` of the policy observation's leading terms, in order."""
    if family_of(task_name) == FURNITURE:
        from furniture_lab.tasks.assembly.assembly_env_cfg import FB_OBS_LAYOUT, FBABS_OBS_LAYOUT

        return list(FBABS_OBS_LAYOUT if is_abs(task_name) else FB_OBS_LAYOUT)
    from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG

    return [(name, int(OBS_DIM_CFG[name])) for name in task_cfg.obs_order]


def obs_term_slices_of(task_name: str, task_cfg: Any) -> Dict[str, slice]:
    slices, start = {}, 0
    for name, width in obs_layout(task_name, task_cfg):
        slices[name] = slice(start, start + width)
        start += width
    return slices


def action_layout(task_name: str) -> Dict[str, Any]:
    """What the policy's action dims mean, for metrics and config checks."""
    if is_abs(task_name):
        # the absolute target: [goal_pos 3, goal_rot6d 6, gripper 1]
        return {"dim": 10, "pos": slice(0, 3), "rot6d": slice(3, 9), "gripper": 9}
    if family_of(task_name) == FURNITURE:
        # FurnitureBench: [dpos 3, dquat xyzw 4, gripper 1]
        return {"dim": 8, "pos": slice(0, 3), "quat_xyzw": slice(3, 7), "gripper": 7}
    # the project's world pose [x, y, z, yaw]
    return {"dim": 4, "pos": slice(0, 3), "yaw": 3}


def action_metric_groups(task_name: str) -> Dict[str, List[int]]:
    """The BC trainer's per-group action error breakdown (``bc.action_metric_groups``)."""
    if is_abs(task_name):
        return {"pos": [0, 1, 2], "rot": [3, 4, 5, 6, 7, 8], "gripper": [9]}
    if family_of(task_name) == FURNITURE:
        return {"pos": [0, 1, 2], "rot": [3, 4, 5, 6], "gripper": [7]}
    return {"pos": [0, 1, 2], "yaw": [3]}
