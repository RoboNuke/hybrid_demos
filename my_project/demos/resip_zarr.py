"""ResiP's processed demos (a ``.zarr`` from robust-rearrangement's ``process_pickles.py``)
-> the recorder's LeRobot-style dataset, in the absolute rot6d interface (``*-FBAbs-*``).

ResiP (Ankile et al., https://github.com/ankile/robust-rearrangement) publishes its one_leg
teleop demos only processed: one flat zarr per dataset, every episode concatenated, with
``episode_ends`` marking the boundaries. Per frame ``t`` (one row per action; the observation
after the last action is already dropped):

* ``robot_state`` (16): ``ee_pos 3, ee_rot6d 6, ee_pos_vel 3, ee_ori_vel 3, gripper_width 1``
  -- the same terms, order and rot6d convention (first two **rows** of R) as ours;
* ``parts_poses`` (35): 5 parts x ``[pos 3, quat 4 (xyzw)]``, FurnitureBench's AprilTag frame;
* ``action/pos`` (10): ``[ee_pos + dpos, rot6d(R_ee R_dq), sign(gripper)]`` -- the absolute
  target, built exactly as ``furniture_lab.demos.fb_abs_action`` builds ours (theirs clips the
  delta first: 2.5 cm, 0.35 rad);
* ``reward``, ``skill`` (per frame), ``success`` (per episode); images, ignored here.

Written columns: ``observation.state_abs`` (61: ``robot_state`` + each part as
``[pos, rot6d]``), ``action.abs`` (10: ``action/pos``), ``observation.next_state_abs`` (the
next frame's; the last frame repeats itself, since the zarr holds no observation after the
last action), ``reward``, ``skill``. No ``observation.state`` / ``action`` (the 8-d delta
interface) and no joint positions: the zarr has neither, so these demos cannot be replayed
from their start state (``replay_abs.py`` needs the pkl).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .writer import DemoWriter

__all__ = ["zarr_states_abs", "import_resip_zarr"]

ROBOT_STATE_DIM, PARTS_DIM, ABS_STATE_DIM, ABS_ACTION_DIM = 16, 35, 61, 10


def zarr_states_abs(robot_state: np.ndarray, parts_poses: np.ndarray) -> np.ndarray:
    """``(T, 16)`` robot state and ``(T, 35)`` part poses -> ``(T, 61)`` FBAbs observations."""
    from furniture_lab.demos import mat_to_rot6d, quat_to_mat, quat_xyzw_to_wxyz  # NumPy, no sim

    T = robot_state.shape[0]
    if robot_state.shape != (T, ROBOT_STATE_DIM) or parts_poses.shape != (T, PARTS_DIM):
        raise ValueError(f"robot_state {robot_state.shape} / parts_poses {parts_poses.shape}, expected (T, 16) / (T, 35)")
    parts = parts_poses.reshape(T, 5, 7)
    rot = np.stack(
        [np.stack([mat_to_rot6d(quat_to_mat(quat_xyzw_to_wxyz(parts[t, i, 3:7]))) for i in range(5)]) for t in range(T)]
    )
    parts_abs = np.concatenate([parts[:, :, :3], rot], axis=-1).reshape(T, 45)
    return np.concatenate([robot_state, parts_abs], axis=-1).astype(np.float32)


def import_resip_zarr(
    zarr_path: Path,
    out: Path,
    task_name: str = "Isaac-Furniture-OneLeg-FBAbs-ResiP-v0",
    control_hz: float = 10.0,
    max_demos: Optional[int] = None,
    successes_only: bool = True,
) -> Dict[str, Any]:
    """Convert every episode of ResiP's zarr at ``zarr_path`` into the dataset at ``out``."""
    import zarr

    z = zarr.open(str(Path(zarr_path).expanduser()), mode="r")
    for key in ("robot_state", "parts_poses", "action/pos", "episode_ends"):
        if key not in z:
            raise KeyError(f"{zarr_path} has no {key!r}: not a robust-rearrangement processed zarr?")
    ends = np.asarray(z["episode_ends"][:], dtype=np.int64)
    robot_state = np.asarray(z["robot_state"][:], dtype=np.float32)
    parts_poses = np.asarray(z["parts_poses"][:], dtype=np.float32)
    actions = np.asarray(z["action/pos"][:], dtype=np.float32)
    n_frames = int(ends[-1])
    for name, arr in (("robot_state", robot_state), ("parts_poses", parts_poses), ("action/pos", actions)):
        if arr.shape[0] != n_frames:
            raise ValueError(f"{zarr_path}: {name} has {arr.shape[0]} frames, episode_ends says {n_frames}")
    if actions.shape[1] != ABS_ACTION_DIM:
        raise ValueError(f"{zarr_path}: action/pos is {actions.shape[1]} wide, expected {ABS_ACTION_DIM}")
    reward = np.asarray(z["reward"][:], dtype=np.float32) if "reward" in z else np.zeros(n_frames, np.float32)
    skill = np.asarray(z["skill"][:], dtype=np.float32) if "skill" in z else np.zeros(n_frames, np.float32)
    success = np.asarray(z["success"][:], dtype=np.int64) if "success" in z else np.ones(len(ends), np.int64)
    try:  # an object array (JSON codec): optional, only used to name each episode's source
        sources = [str(s) for s in z["pickle_file"][:]]
    except Exception:  # noqa: BLE001 -- absent, or a codec this zarr version cannot read
        sources = [None] * len(ends)

    writer = DemoWriter(
        out,
        fps=control_hz,
        info={
            "task": task_name,
            "control_dt": 1.0 / control_hz,
            "physics_dt": 1.0 / 120.0,
            "decimation": round(120.0 / control_hz),
            "source": f"ResiP processed zarr ({zarr_path})",
            "observation.state_abs": "[ee_pos 3, ee_rot6d 6, ee_pos_vel 3, ee_ori_vel 3, gripper_width 1, parts 5 x [pos 3, rot6d 6]]",
            "action.abs": "[goal_pos 3 (m, robot frame), goal_rot6d 6, gripper (sign; > 0 closes)]",
        },
    )
    already = {e.get("source") for e in writer.episodes}
    saved, skipped, failures = 0, 0, 0
    start = 0
    for e, end in enumerate(ends):
        lo, hi = start, int(end)
        start = hi
        if max_demos is not None and saved >= max_demos:
            break
        if successes_only and not success[e]:
            failures += 1
            continue
        source = f"{zarr_path}#{sources[e] or e}"
        if source in already:
            skipped += 1
            continue
        states = zarr_states_abs(robot_state[lo:hi], parts_poses[lo:hi])
        nxt = np.concatenate([states[1:], states[-1:]])
        writer.start()
        for t in range(hi - lo):
            writer.add(
                {
                    "observation.state_abs": states[t],
                    "action.abs": actions[lo + t],
                    "observation.next_state_abs": nxt[t],
                    "reward": reward[lo + t],
                    "skill": skill[lo + t],
                }
            )
        meta = {
            "success": bool(success[e]),
            "ended_by": "success" if success[e] else "timeout",
            "furniture": "one_leg",
            "source": source,
            "seed": None,
            "reward_sum": float(reward[lo:hi].sum()),
        }
        writer.save(meta, {"note": "ResiP processed zarr: no start-state snapshot or joint positions"})
        saved += 1
    return {"saved": saved, "skipped": skipped, "failures_dropped": failures, "episodes": writer.num_episodes,
            "frames": writer.num_frames, "root": str(writer.root)}
