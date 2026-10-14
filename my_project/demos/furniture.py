"""FurnitureBench-format demos (``scripts/collect_scripted.py`` in Robonuke_Furniture_Bench)
-> the recorder's LeRobot-style dataset, for the BC trainer and the residual pipeline.

A demo pkl is one episode in FurnitureBench's own layout::

    observations[t]  {robot_state: {ee_pos 3, ee_quat 4 (xyzw), ee_pos_vel 3, ee_ori_vel 3,
                                    gripper_width 1, joint_positions 7, ...},
                      parts_poses 35 (5 parts x [pos 3, quat xyzw 4], AprilTag frame)}
    actions[t]       8 = [dpos 3, dquat 4 (xyzw, EE frame), gripper]   (T of them)
    rewards, skills  (T)        success, furniture

Written columns, per frame ``t`` (one row per action; the final observation, which has no
action, is dropped, as the recorder drops it):

* ``observation.state`` (49): FurnitureBench's ``DEFAULT_STATE_OBS`` -- ``ee_pos, ee_quat,
  ee_pos_vel, ee_ori_vel, gripper_width, parts_poses`` -- exactly their low-dim policy input;
* ``action`` (8): the action as recorded;
* the absolute rot6d interface (``Isaac-Furniture-OneLeg-FBAbs-v0``, ResiP's ``control_mode:
  pos``), derived exactly from the same frames by ``furniture_lab.demos``:
  ``observation.state_abs`` (61: every orientation as rot6d) and ``action.abs`` (10: the
  target pose the delta commanded, ``[ee_pos + dpos, rot6d(R_ee R_dq), gripper]``);
* raw material, never trained on by default: ``observation.joint_positions`` (7),
  ``observation.next_state`` (49) / ``observation.next_state_abs`` (61) (the frame after --
  so transitions need no neighbour lookup), ``reward`` (1), ``skill`` (1).

A dataset written before the ``*_abs`` columns existed has different features: import into
a fresh directory.

No Isaac Sim is needed: this is a file conversion.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .writer import DemoWriter

__all__ = ["STATE_KEYS", "STATE_DIM", "ACTION_DIM", "ABS_STATE_DIM", "ABS_ACTION_DIM", "find_demos", "load_pkl", "state_vector", "import_demos"]

#: FurnitureBench's DEFAULT_STATE_OBS, in order (furniture_bench/envs/observation.py)
STATE_KEYS = ("ee_pos", "ee_quat", "ee_pos_vel", "ee_ori_vel", "gripper_width", "parts_poses")
STATE_DIM = 3 + 4 + 3 + 3 + 1 + 35
ACTION_DIM = 8
ABS_STATE_DIM, ABS_ACTION_DIM = 61, 10
CONTROL_HZ = 20.0  # FurnitureSim's env rate, the collector's too (physics 1/120 x 6)


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def state_vector(obs: Dict[str, Any]) -> np.ndarray:
    rs = obs["robot_state"]
    parts = [_np(rs[k]) for k in STATE_KEYS[:-1]] + [_np(obs["parts_poses"])]
    v = np.concatenate(parts)
    if v.shape[0] != STATE_DIM:
        raise ValueError(f"state is {v.shape[0]} wide, expected {STATE_DIM}: {[p.shape[0] for p in parts]}")
    return v


def load_pkl(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_demos(src: Path, include_failures: bool = False) -> List[Path]:
    """``<src>/<task>/<stamp>/<stamp>.pkl`` (successes) and, if asked, ``<src>/<task>/fails/*.pkl``."""
    src = Path(src).expanduser()
    files = sorted(p for p in src.glob("*/*/*.pkl") if p.parent.name != "fails")
    if include_failures:
        files += sorted(src.glob("*/fails/*.pkl"))
    return files


def import_demos(
    src: Path,
    out: Path,
    include_failures: bool = False,
    task_name: str = "Isaac-Furniture-OneLeg-v0",
    max_demos: Optional[int] = None,
    files: Optional[Iterable[Path]] = None,
    control_hz: float = CONTROL_HZ,
) -> Dict[str, Any]:
    """Convert every demo under ``src`` into the dataset at ``out`` (created or appended)."""
    files = list(files) if files is not None else find_demos(src, include_failures)
    if max_demos is not None:
        files = files[:max_demos]
    if not files:
        raise FileNotFoundError(f"no demo pkls under {src} (looked for <task>/<stamp>/<stamp>.pkl)")
    writer = DemoWriter(
        out,
        fps=control_hz,
        info={
            "task": task_name,
            "control_dt": 1.0 / control_hz,
            "physics_dt": 1.0 / 120.0,
            "decimation": round(120.0 / control_hz),
            "source": "Robonuke_Furniture_Bench scripts/collect_scripted.py (FurnitureBench pkl)",
            "observation.state": list(STATE_KEYS),
            "action": "[dpos 3 (m, robot frame), dquat 4 (xyzw, EE frame), gripper (<0 open)]",
            "observation.state_abs": "[ee_pos 3, ee_rot6d 6, ee_pos_vel 3, ee_ori_vel 3, gripper_width 1, parts 5 x [pos 3, rot6d 6]]",
            "action.abs": "[goal_pos 3 (m, robot frame), goal_rot6d 6, gripper (<0 open)]",
        },
    )
    already = {e.get("source") for e in writer.episodes}
    saved, skipped = 0, 0
    for path in files:
        source = str(path)
        if source in already:
            skipped += 1
            continue
        d = load_pkl(path)
        obs, acts = d["observations"], d["actions"]
        T = len(acts)
        if len(obs) not in (T, T + 1):
            raise ValueError(f"{path}: {len(obs)} observations for {T} actions")
        states = np.stack([state_vector(o) for o in obs])
        if len(obs) == T:  # no final observation recorded: repeat the last one for next_state
            states = np.concatenate([states, states[-1:]])
        from furniture_lab.demos import fb_abs_frames  # the port's own conversion (NumPy, no sim)

        states_abs, acts_abs = fb_abs_frames(d)
        if states_abs.shape != (T + 1, ABS_STATE_DIM) or acts_abs.shape != (T, ABS_ACTION_DIM):
            raise ValueError(f"{path}: abs frames {states_abs.shape} / {acts_abs.shape}, expected {(T + 1, ABS_STATE_DIM)} / {(T, ABS_ACTION_DIM)}")
        rewards = np.asarray([float(_np(r)[0]) for r in d.get("rewards", [0.0] * T)], dtype=np.float32)
        skills = np.asarray([float(_np(s)[0]) for s in d.get("skills", [0.0] * T)], dtype=np.float32)
        writer.start()
        for t in range(T):
            writer.add(
                {
                    "observation.state": states[t],
                    "action": _np(acts[t]),
                    "observation.joint_positions": _np(obs[t]["robot_state"]["joint_positions"]),
                    "observation.next_state": states[t + 1],
                    "observation.state_abs": states_abs[t],
                    "action.abs": acts_abs[t],
                    "observation.next_state_abs": states_abs[t + 1],
                    "reward": rewards[t],
                    "skill": skills[t],
                }
            )
        success = bool(d.get("success", False))
        meta = {
            "success": success,
            "ended_by": "success" if success else ("error" if d.get("error") else "timeout"),
            "error_description": str(d.get("error_description", "")),
            "furniture": str(d.get("furniture", "")),
            "source": source,
            "seed": None,
            "reward_sum": float(rewards.sum()),
        }
        start_state = {"observation": obs[0], "note": "FurnitureBench observation of the first frame (no env snapshot)"}
        writer.save(meta, start_state)
        saved += 1
    return {"saved": saved, "skipped": skipped, "episodes": writer.num_episodes, "frames": writer.num_frames, "root": str(writer.root)}
