"""The two env families this project runs on, and the few things that differ between them.

* **forge** -- Isaac Lab's Direct-env Forge tasks (peg insert, nut thread): the observation
  is ``task.cfg.obs_order`` of Factory's terms, a Direct env's hooks (``_get_observations``,
  integer ``cfg.observation_space``), Forge's reward recomputable offline.
* **furniture** -- ``Isaac-Furniture-*-FB-v0`` from Robonuke_Furniture_Bench, manager-based:
  the observation is FurnitureBench's ``DEFAULT_STATE_OBS`` (``FB_OBS_LAYOUT`` there), the
  observation manager computes it, the reward is the demo's recorded sparse success. The
  ``-FBAbs-v0`` task is the same family with the absolute rot6d interface (``FBABS_*`` there:
  observation 61, action 10); :func:`is_abs` tells them apart where the layout matters. The
  ``-FBRel-ResiP-v0`` task is the ResiP FBAbs task with the observation and the action relative to
  the end-effector (``FBREL_*`` there: observation 55, action 10); :func:`is_rel`. The
  ``-FBDelta-ResiP-v0`` task is the ResiP FBAbs task (its observation, 63) with only the action
  relative to the end-effector, its rotation an axis-angle (``FBDELTA_ACTION_DIM`` there: 7);
  :func:`is_delta`.

Everything that must know the family asks here (:func:`family_of`, :func:`obs_layout`), so
the wrappers, the residual demo conversion and the config checks keep one code path each
(``my_project/envhooks.py`` is the live-env side of the same split).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = [
    "FORGE", "FURNITURE", "ROBOSUITE", "family_of", "is_abs", "is_rel", "is_delta", "obs_layout", "obs_term_slices_of", "action_layout",
    "action_metric_groups", "obs_units", "action_units",
]

FORGE = "forge"
FURNITURE = "furniture"
#: robosuite tasks behind a Direct-env surface (``my_project/tasks/robosuite_can.py``): ``Robosuite-*``
ROBOSUITE = "robosuite"


def family_of(task_name: str) -> str:
    """``Isaac-Furniture-*`` is the furniture family, ``Robosuite-*`` the robosuite one; anything else is the Forge
    family the project was built on (its Direct-env tasks, and the fakes the tests register)."""
    if str(task_name).startswith("Robosuite-"):
        return ROBOSUITE
    return FURNITURE if "Furniture" in str(task_name) else FORGE


def is_abs(task_name: str) -> bool:
    """The furniture family's absolute-target, rot6d interface (``Isaac-Furniture-*-FBAbs-v0``)."""
    return family_of(task_name) == FURNITURE and "FBAbs" in str(task_name)


def is_rel(task_name: str) -> bool:
    """The furniture family's interface relative to the end-effector (``Isaac-Furniture-*-FBRel-*``):
    the action ``[dpos 3, drot6d 6, gripper]`` in the EE's frame, every observed pose in it."""
    return family_of(task_name) == FURNITURE and "FBRel" in str(task_name)


def is_delta(task_name: str) -> bool:
    """The furniture family's ResiP FBAbs observation with the action relative to the end-effector
    (``Isaac-Furniture-*-FBDelta-*``): ``[dpos 3, drot 3 (axis-angle), gripper]`` in the EE's frame."""
    return family_of(task_name) == FURNITURE and "FBDelta" in str(task_name)


def observes_obstacle(task_name: str) -> bool:
    """The ResiP FBAbs task (``Isaac-Furniture-*-FBAbs-ResiP-*``) also observes the front
    obstacle's x, y, every pose in the robot-base frame (``FBABS_RESIP_OBS_LAYOUT``, 63): the rule the sim-free demo importers use;
    with an env cfg in hand, ``obs_layout`` reads the cfg itself."""
    return is_abs(task_name) and "ResiP" in str(task_name)


def obs_layout(task_name: str, task_cfg: Any) -> List[Tuple[str, int]]:
    """``[(term, width), ...]`` of the policy observation's leading terms, in order."""
    if family_of(task_name) == FURNITURE:
        from furniture_lab.tasks.assembly.assembly_env_cfg import FB_OBS_LAYOUT, FBREL_RESIP_OBS_LAYOUT, fbabs_obs_layout

        if is_rel(task_name):
            return list(FBREL_RESIP_OBS_LAYOUT)
        return list(fbabs_obs_layout(task_cfg) if is_abs(task_name) or is_delta(task_name) else FB_OBS_LAYOUT)
    if family_of(task_name) == ROBOSUITE:
        # one term: the env checks obs_keys' widths against observation_space itself
        return [("observation", int(task_cfg.observation_space))]
    from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG

    return [(name, int(OBS_DIM_CFG[name])) for name in task_cfg.obs_order]


#: the physical unit of every dim of an FBAbs observation term (``bc.normalization: per_unit``)
_POSE9 = ["pos"] * 3 + ["rot6d"] * 6
TERM_UNITS: Dict[str, List[str]] = {
    "ee_pos": ["pos"] * 3,
    "ee_rot6d": ["rot6d"] * 6,
    "ee_pos_vel": ["lin_vel"] * 3,
    "ee_ori_vel": ["ang_vel"] * 3,
    "gripper_width": ["finger_width"],
    "parts_poses": _POSE9 * 5,
    "parts_poses_b": _POSE9 * 5,
    "obstacle_xy": ["pos"] * 2,
}


def obs_units(task_name: str, task_cfg: Any) -> List[str]:
    """Every observation dim's physical unit, in order, from the task's layout (FBAbs only)."""
    if not is_abs(task_name):
        raise ValueError(f"per-unit scaling knows the furniture FBAbs layouts only, got task {task_name!r}")
    units: List[str] = []
    for name, width in obs_layout(task_name, task_cfg):
        if name not in TERM_UNITS or len(TERM_UNITS[name]) != width:
            raise KeyError(f"observation term {name!r} ({width} wide) has no unit map in families.TERM_UNITS")
        units += TERM_UNITS[name]
    return units


#: the ``-FBAbs-ResiP-`` observation's terms (63), the same as ``assembly_env_cfg.FBABS_RESIP_OBS_LAYOUT``, for code
#: that runs without the sim (the per-phase demo builder); the env wrapper checks its width against the live env's
FBABS_RESIP_TERMS = (("ee_pos", 3), ("ee_rot6d", 6), ("ee_pos_vel", 3), ("ee_ori_vel", 3), ("gripper_width", 1),
                     ("parts_poses_b", 45), ("obstacle_xy", 2))


def resip_obs_units(task_name: str) -> List[str]:
    """Every dim's physical unit of the ``-FBAbs-ResiP-`` observation (:data:`FBABS_RESIP_TERMS`), sim-free."""
    if not observes_obstacle(task_name):
        raise ValueError(f"the ResiP FBAbs observation's units are known for Isaac-Furniture-*-FBAbs-ResiP-* only, got {task_name!r}")
    units: List[str] = []
    for name, width in FBABS_RESIP_TERMS:
        if len(TERM_UNITS[name]) != width:
            raise KeyError(f"observation term {name!r}: {width} wide, its unit map {len(TERM_UNITS[name])}")
        units += TERM_UNITS[name]
    return units


def action_units(task_name: str) -> List[str]:
    """Every action dim's unit: the FBAbs target ``[pos 3, rot6d 6, gripper]``; its position and
    rot6d share the observation's ranges."""
    if not is_abs(task_name):
        raise ValueError(f"per-unit scaling knows the furniture FBAbs action only, got task {task_name!r}")
    return ["pos"] * 3 + ["rot6d"] * 6 + ["gripper_cmd"]


def obs_term_slices_of(task_name: str, task_cfg: Any) -> Dict[str, slice]:
    slices, start = {}, 0
    for name, width in obs_layout(task_name, task_cfg):
        slices[name] = slice(start, start + width)
        start += width
    return slices


def action_layout(task_name: str) -> Dict[str, Any]:
    """What the policy's action dims mean, for metrics and config checks."""
    if is_abs(task_name) or is_rel(task_name):
        # the absolute target [goal_pos 3, goal_rot6d 6, gripper 1], or the relative one in the EE's
        # frame [dpos 3 (units of the term's pos_limit), drot 6 (rot6d minus the identity's), gripper 1]
        return {"dim": 10, "pos": slice(0, 3), "rot6d": slice(3, 9), "gripper": 9}
    if is_delta(task_name):
        # the relative target in the EE's frame [dpos 3 (units of the term's pos_limit), drot 3
        # (axis-angle, rad), gripper 1]
        return {"dim": 7, "pos": slice(0, 3), "axis_angle": slice(3, 6), "gripper": 6}
    if family_of(task_name) == FURNITURE:
        # FurnitureBench: [dpos 3, dquat xyzw 4, gripper 1]
        return {"dim": 8, "pos": slice(0, 3), "quat_xyzw": slice(3, 7), "gripper": 7}
    if family_of(task_name) == ROBOSUITE:
        if "-BoxCleanup-" in str(task_name):
            # two Panda arms with dexterous hands, each [OSC_POSE dpos 3, drot 3 (0.05 m / 0.5 rad per unit), hand 6]:
            # the metrics read the first arm's pose
            return {"dim": 24, "pos": slice(0, 3), "axis_angle": slice(3, 6), "pos_m_per_unit": 0.05, "rad_per_unit": 0.5}
        if "-CanSort-" in str(task_name) or "-Coffee-" in str(task_name):
            # GR1's whole-body IK, absolute world targets: [right hand pos 3 (m), rot 3 (axis-angle, rad), left hand pos 3,
            # rot 3, right fingers 6, left fingers 6 (joint targets, rad)]: the metrics read the right hand's pose
            return {"dim": 24, "pos": slice(0, 3), "axis_angle": slice(3, 6), "pos_m_per_unit": 1.0, "rad_per_unit": 1.0}
        # robosuite's OSC_POSE delta inputs [dpos 3, drot axis-angle 3, gripper 1], each in [-1, 1]; the controller
        # moves 0.05 m and 0.5 rad per unit (robosuite 1.5.1 default_panda.json output_max)
        return {"dim": 7, "pos": slice(0, 3), "axis_angle": slice(3, 6), "gripper": 6, "pos_m_per_unit": 0.05,
                "rad_per_unit": 0.5}
    # the project's world pose [x, y, z, yaw]
    return {"dim": 4, "pos": slice(0, 3), "yaw": 3}


def action_metric_groups(task_name: str) -> Dict[str, List[int]]:
    """The BC trainer's per-group action error breakdown (``bc.action_metric_groups``)."""
    if is_abs(task_name) or is_rel(task_name):
        return {"pos": [0, 1, 2], "rot": [3, 4, 5, 6, 7, 8], "gripper": [9]}
    if is_delta(task_name):
        return {"pos": [0, 1, 2], "rot": [3, 4, 5], "gripper": [6]}
    if family_of(task_name) == FURNITURE:
        return {"pos": [0, 1, 2], "rot": [3, 4, 5, 6], "gripper": [7]}
    if family_of(task_name) == ROBOSUITE:
        if "-CanSort-" in str(task_name) or "-Coffee-" in str(task_name):  # [right pose 6, left pose 6, fingers 12]
            return {"pos": [0, 1, 2, 6, 7, 8], "rot": [3, 4, 5, 9, 10, 11], "hand": list(range(12, 24))}
        if action_layout(task_name)["dim"] == 24:  # two arms of 12: pose 6, hand 6
            return {"pos": [0, 1, 2, 12, 13, 14], "rot": [3, 4, 5, 15, 16, 17], "hand": list(range(6, 12)) + list(range(18, 24))}
        return {"pos": [0, 1, 2], "rot": [3, 4, 5], "gripper": [6]}
    return {"pos": [0, 1, 2], "yaw": [3]}
