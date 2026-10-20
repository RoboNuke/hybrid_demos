"""Check that ResiP's released BC checkpoint is loaded correctly, offline: no sim.

    python scripts/check_resip_checkpoint.py --checkpoint ~/resip_ckpt/bc_one_leg_low.pt \\
        --zarr ~/resip_data/one_leg_teleop_low/success.zarr

The checkpoint (``resip_checkpoint.load_resip_policy``: their U-Net weights, their normalizer
stats, their horizons) is run on **their own** data, in **their** layout and frames -- the zarr's
``robot_state`` (16) and ``parts_poses`` (42) concatenated, exactly what their ``bc.py`` trained
on; no adapter, no z shift, no rotation rescale -- and its predicted chunks are compared with the
zarr's recorded ``action/pos`` (the absolute targets it was trained to predict, rotations as
stored). The checkpoint was trained on these demos, so a correctly loaded one reproduces them
closely; a loading error (wrong weights, normalizer, horizons, obs order) shows up as errors
near or above the "hold still" baseline.

Reported over ``--windows`` random (episode, frame) windows, the chunk sampled from pure noise
(``--samples`` per window), median over windows:

* position error (mm) at chunk steps 0, ``action_horizon - 1`` and the last;
* rotation error (deg) at step 0;
* gripper sign agreement over the executed part;
* the "hold" baseline: predicting the current EE position for the whole chunk, for scale.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np
import torch

__all__ = ["windows_from_zarr", "check", "main"]


def windows_from_zarr(robot_state: np.ndarray, parts: np.ndarray, actions: np.ndarray, ends: np.ndarray,
                      horizon: int, n: int, rng: np.random.Generator):
    """``n`` random windows: obs ``(n, 58)`` at frame t and the recorded actions ``(n, horizon, 10)``
    from t on, the episode's last action repeated past its end (their ``pad_after``)."""
    starts = np.concatenate([[0], ends[:-1]])
    ep = rng.integers(0, len(ends), size=n)
    t = np.array([rng.integers(starts[e], ends[e]) for e in ep])
    obs = np.concatenate([robot_state[t], parts[t]], axis=1)
    idx = np.minimum(t[:, None] + np.arange(horizon)[None], ends[ep][:, None] - 1)
    return obs.astype(np.float32), actions[idx].astype(np.float32), t


def _rot_deg(a6: torch.Tensor, b6: torch.Tensor) -> torch.Tensor:
    from my_project.bc.resip_checkpoint import rot6d_to_mat

    R = rot6d_to_mat(a6).transpose(-1, -2) @ rot6d_to_mat(b6)
    c = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(c))


@torch.no_grad()
def check(checkpoint: str, zarr_path: str, windows: int = 512, samples: int = 2, seed: int = 0,
          device: str = "cpu", inference_steps: int = 4) -> Dict[str, Any]:
    import zarr

    from my_project.bc.resip_checkpoint import load_resip_policy

    policy, _, info = load_resip_policy(checkpoint, device, inference_steps)
    z = zarr.open(os.path.expanduser(zarr_path), mode="r")
    robot_state = np.asarray(z["robot_state"][:], dtype=np.float32)
    parts = np.asarray(z["parts_poses"][:], dtype=np.float32)
    actions = np.asarray(z["action/pos"][:], dtype=np.float32)
    ends = np.asarray(z["episode_ends"][:], dtype=np.int64)
    if robot_state.shape[1] != 16 or parts.shape[1] != 42 or actions.shape[1] != 10:
        raise ValueError(
            f"{zarr_path}: robot_state {robot_state.shape[1]} / parts_poses {parts.shape[1]} / action/pos "
            f"{actions.shape[1]}; the checkpoint needs 16 / 42 (5 parts + the obstacle) / 10"
        )
    H, A = policy.horizon, policy.action_horizon
    obs, target, _ = windows_from_zarr(robot_state, parts, actions, ends, H, windows, np.random.default_rng(seed))
    obs_t = torch.tensor(obs, device=device)
    target_t = torch.tensor(target, device=device)
    nobs = policy.normalizer.normalize(obs_t[:, None, :], "obs").clamp(-policy.cfg.obs_clip, policy.cfg.obs_clip)
    torch.manual_seed(seed)
    pred = torch.stack([policy.predict_chunk(nobs) for _ in range(samples)])  # (S, N, H, 10)

    pos_err = torch.linalg.norm(pred[..., :3] - target_t[None, ..., :3], dim=-1).mean(0) * 1000  # (N, H) mm
    hold = torch.linalg.norm(obs_t[:, None, :3] - target_t[..., :3], dim=-1) * 1000  # the current EE, held
    rot0 = _rot_deg(pred[:, :, 0, 3:9], target_t[None, :, 0, 3:9]).mean(0)
    grip = (torch.sign(pred[..., :A, 9]) == torch.sign(target_t[None, :, :A, 9])).float().mean()
    step = torch.linalg.norm(target_t[:, 1:, :3] - target_t[:, :-1, :3], dim=-1).median() * 1000
    med = lambda x: float(x.median())  # noqa: E731
    return {
        "info": info,
        "windows": windows,
        "pos_mm": {"step0": med(pos_err[:, 0]), f"step{A - 1}": med(pos_err[:, A - 1]), f"step{H - 1}": med(pos_err[:, H - 1])},
        "hold_mm": {"step0": med(hold[:, 0]), f"step{A - 1}": med(hold[:, A - 1]), f"step{H - 1}": med(hold[:, H - 1])},
        "rot_step0_deg": med(rot0),
        "gripper_agreement": float(grip),
        "demo_step_mm": float(step),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="ResiP's actor_chkpt.pt")
    parser.add_argument("--zarr", required=True, help="their processed one_leg .zarr (the demos it was trained on)")
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    r = check(args.checkpoint, args.zarr, args.windows, args.samples, args.seed, args.device, args.inference_steps)
    print(f"[check] {args.checkpoint}: {r['info']}")
    print(f"[check] {r['windows']} windows of their own demos (their layout and frames, no adapter); "
          f"the demos move a median {r['demo_step_mm']:.1f} mm per step")
    print("[check] position error, median (mm):  " + "   ".join(f"{k} {v:6.1f}" for k, v in r["pos_mm"].items()))
    print("[check] 'hold still' baseline (mm):    " + "   ".join(f"{k} {v:6.1f}" for k, v in r["hold_mm"].items()))
    print(f"[check] rotation error at step 0: {r['rot_step0_deg']:.2f} deg; gripper sign agreement {100 * r['gripper_agreement']:.1f}%")
    return 0
