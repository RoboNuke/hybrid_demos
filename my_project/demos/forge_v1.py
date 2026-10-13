"""Read the forge-v1 NutThread demos (https://huggingface.co/datasets/angledusgar/forge-v1) and
recover, per episode, where the hand was relative to the bolt -- the dataset records neither the
bolt's pose nor the nut's place in the gripper.

What an episode holds (LeRobot v2.1, 15 Hz, stock Forge with its randomized dynamics):
``state`` = true fingertip position (env frame) + its orientation as an axis-angle + the two
finger joints; ``actions`` = the expert's raw Forge action (before Forge's smoothing); ``ft`` =
the force on the hand in world axes (+ a torque we do not use). Episodes 1980-2967 are nut
threading; 2091-2201 are byte-identical copies of 1980-2090.

The reconstruction (pure numpy):

* **contact**: the first step whose force exceeds ``contact_force``;
* **the bolt's axis and the nut's offset**: while the hand turns, the fingertip circles the bolt
  axis at the nut's offset in the gripper, ``p_xy = c + Rz(heading) o``; a least-squares fit
  from contact to the last turning step gives the axis ``c`` and the offset ``o``;
* **the nut-centred path**: ``q = p - Rz(heading) o``, where the fingertip would be with the nut
  centred in the gripper (how the replay env holds it);
* **the bolt tip's height**: only the nut's height in the gripper plus the bolt's height is
  observable, so the tip is put ``contact_height`` below the fingertip's height at contact,
  one constant for the dataset (see :func:`contact_height`).

All positions returned are relative to the recovered bolt tip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

__all__ = ["NUT_EPISODES", "Demo", "episode_path", "unique_nut_episodes", "load_raw", "reconstruct", "contact_height"]

#: nut-threading episode indices in forge-v1 (inclusive)
NUT_EPISODES = (1980, 2967)
#: stock NutThread hand start: 15 mm above the bolt tip, +-10 mm (factory_tasks_cfg.py NutThread)
HAND_INIT_HEIGHT = 0.015


def episode_path(root: str | Path, index: int) -> Path:
    return Path(root).expanduser() / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"


def load_raw(root: str | Path, index: int) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(episode_path(root, index)).to_pydict()
    return {key: np.asarray(table[key], dtype=np.float64) for key in ("state", "actions", "ft")}


def unique_nut_episodes(root: str | Path) -> List[int]:
    """The nut-threading episodes present under ``root``, dropping exact duplicates (same bytes)."""
    import hashlib

    seen, keep = set(), []
    for index in range(NUT_EPISODES[0], NUT_EPISODES[1] + 1):
        path = episode_path(root, index)
        if not path.exists():
            raise FileNotFoundError(f"{path}: forge-v1 episode {index} is missing")
        raw = load_raw(root, index)
        digest = hashlib.sha1(b"".join(raw[k].tobytes() for k in ("state", "actions", "ft"))).hexdigest()
        if digest not in seen:
            seen.add(digest)
            keep.append(index)
    return keep


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.concatenate((xyzw[:, 3:4], xyzw[:, 0:3]), axis=1)


def _heading(rotvec: np.ndarray) -> np.ndarray:
    """The hand x axis' heading in the world xy plane, unwrapped (rad)."""
    from scipy.spatial.transform import Rotation

    x = Rotation.from_rotvec(rotvec).apply([1.0, 0.0, 0.0])
    return np.unwrap(np.arctan2(x[:, 1], x[:, 0]))


def _fingertip_yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    """``my_project.demos.teleop.fingertip_yaw`` in numpy: undo the 180 deg roll, take yaw."""
    w, x, y, z = quat_wxyz.T
    # (0, -1, 0, 0) * q
    w2, x2, y2, z2 = x, -w, z, -y
    return np.arctan2(2.0 * (w2 * z2 + x2 * y2), 1.0 - 2.0 * (y2 * y2 + z2 * z2))


@dataclass
class Demo:
    index: int
    #: (T, 3) nut-centred fingertip path, relative to the bolt tip, m
    path: np.ndarray
    #: (T,) fingertip yaw in the world-pose action's convention, rad
    yaw: np.ndarray
    #: (T, 4) fingertip orientation, (w, x, y, z)
    quat: np.ndarray
    #: (T, 3) force on the hand, world axes, N
    force: np.ndarray
    contact_step: int
    #: nut offset in the gripper (hand frame, xy), m; median circle-fit residual, m
    nut_offset: np.ndarray
    fit_residual: float


def _contact_step(force: np.ndarray, contact_force: float) -> int:
    above = np.linalg.norm(force[:, 0:3], axis=1) > contact_force
    if not above.any():
        raise ValueError(f"the force never exceeds {contact_force} N")
    return int(np.argmax(above))


def contact_height(root: str | Path, episodes: Sequence[int], contact_force: float) -> float:
    """The fingertip's height above the bolt tip at first contact, for the whole dataset.

    Stock starts the fingertip ``HAND_INIT_HEIGHT`` above the tip (+-10 mm, uniform), so on
    average ``start - contact = HAND_INIT_HEIGHT - contact_height``: the median over episodes.
    """
    drops = []
    for index in episodes:
        raw = load_raw(root, index)
        step = _contact_step(raw["ft"], contact_force)
        drops.append(raw["state"][0, 2] - raw["state"][step, 2])
    return float(HAND_INIT_HEIGHT - np.median(drops))


def reconstruct(raw: dict, index: int, contact_force: float, height_at_contact: float) -> Demo:
    state, force = raw["state"], raw["ft"][:, 0:3]
    position, rotvec = state[:, 0:3], state[:, 3:6]
    contact = _contact_step(force, contact_force)
    heading = _heading(rotvec)
    turning = np.nonzero(np.abs(np.diff(heading)) > np.deg2rad(0.2))[0]
    stop = int(turning[-1] + 1) if len(turning) else len(heading) - 1
    if stop - contact < 10:
        raise ValueError(f"episode {index}: the hand barely turns after contact (steps {contact}-{stop})")
    rows = np.arange(contact, stop + 1)
    c, s = np.cos(heading[rows]), np.sin(heading[rows])
    system = np.zeros((2 * len(rows), 4))
    target = np.zeros(2 * len(rows))
    system[0::2, 0], system[0::2, 2], system[0::2, 3], target[0::2] = 1.0, c, -s, position[rows, 0]
    system[1::2, 1], system[1::2, 2], system[1::2, 3], target[1::2] = 1.0, s, c, position[rows, 1]
    solution, *_ = np.linalg.lstsq(system, target, rcond=None)
    residual = np.linalg.norm((system @ solution - target).reshape(-1, 2), axis=1)
    axis, offset = solution[0:2], solution[2:4]

    c, s = np.cos(heading), np.sin(heading)
    offset_world = np.stack((c * offset[0] - s * offset[1], s * offset[0] + c * offset[1]), axis=1)
    centred = position.copy()
    centred[:, 0:2] -= offset_world
    tip = np.array([axis[0], axis[1], position[contact, 2] - height_at_contact])
    quat = _rotvec_to_quat_wxyz(rotvec)
    return Demo(
        index=index,
        path=centred - tip,
        yaw=_fingertip_yaw(quat),
        quat=quat,
        force=force,
        contact_step=contact,
        nut_offset=offset,
        fit_residual=float(np.median(residual)),
    )


def read_info(root: str | Path) -> dict:
    return json.loads((Path(root).expanduser() / "meta" / "info.json").read_text())
