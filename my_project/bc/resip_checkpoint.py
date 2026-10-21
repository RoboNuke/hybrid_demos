"""Roll out ResiP's own released BC checkpoint in this port's ResiP task.

    python scripts/eval_resip_checkpoint.py --config configs/experiments/pivot_demos.yaml \\
        --checkpoint ~/resip_ckpt/bc/one_leg/low/actor_chkpt.pt

The question it answers: does the policy that scores ~54% in ResiP's sim (Ankile et al., the
one_leg low state BC: robust-rearrangement ``src/train/bc.py``, U-Net) keep that score in this
port? If it does, the port is faithful and our gap is our data or training; if it collapses,
the port is what caps BC.

The checkpoint (``model_state_dict``: the actor, normalizer included; ``config``: its run's
resolved config) is loaded into this project's :class:`DiffusionPolicy` -- the U-Net is the
same class -- and acted through :class:`ChunkedBase` as ResiP's ``evaluate_model.py`` acts it:
4 DDIM steps, warm start from timestep 50 (their actor always warm-starts), observations
clamped to [-3, 3] after normalizing.

Per step, the port's observation (63: the ResiP task, every pose in the robot-base frame) is
rebuilt in ResiP's (58):

* ``robot_state`` 16 -- same terms, order and rot6d convention; the EE z moves into ResiP's
  base frame (their base is 15 mm lower: ``frames.RESIP_BASE_Z_TO_PORT``);
* ``parts_poses`` 42 -- 5 parts then the front obstacle, each ``[pos 3, quat xyzw 4]`` in the
  AprilTag frame: our base-frame ``[pos 3, rot6d 6]`` parts moved into it, and the obstacle's
  x, y with its fixed z; each quaternion with the sign their sim gives it
  (:func:`mat_to_quat_xyzw`: w >= 0). The obstacle's rotation is THEIR body's: their sim
  creates the front obstacle rotated 90 deg about z (``furniture_rl_sim_env.py``
  ``obstacle_front_pose.r``), ours is an unrotated box of the same shape, so its observed
  rotation is ``Rz(90)``, not ours. Its quaternion sits at w = 0, where the sign of x is
  float noise in their sim; when the checkpoint's normalizer shows the 4 dims constant
  (padded range 2), the adapter feeds exactly that training value.

The predicted action (10: target pos, rot6d, gripper) is sent as their sim received it, its z
moved back into the port's base frame. Their rotation targets are the episode-wide shrunk ones
(see ``my_project/demos/resip_zarr.py``): that is what the policy learned and what their own
evaluation executed, so they are not rescaled here.
"""

from __future__ import annotations

import argparse
import sys
import os
import traceback
from typing import Any, Dict, Tuple

import torch

__all__ = ["RESIP_OBS_DIM", "PORT_OBS_DIM", "rot6d_to_mat", "rot6d_to_quat_xyzw", "mat_to_quat_xyzw", "ResipObsAdapter", "load_resip_policy", "main"]

ROBOT, N_PARTS, N_POSES = 16, 5, 6  # robot_state; 5 parts (+ the front obstacle in theirs)
PORT_OBS_DIM = ROBOT + N_PARTS * 9 + 2  # 63
RESIP_OBS_DIM = ROBOT + N_POSES * 7  # 58
ACTION_DIM = 10


def rot6d_to_mat(d6: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` rot6d (the matrix's first two rows) -> ``(..., 3, 3)`` (Gram-Schmidt)."""
    b1 = torch.nn.functional.normalize(d6[..., 0:3], dim=-1)
    a2 = d6[..., 3:6]
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)  # rows


def rot6d_to_quat_xyzw(d6: torch.Tensor) -> torch.Tensor:
    """``(..., 6)`` rot6d -> ``(..., 4)`` quaternion xyzw in their sim's convention."""
    return mat_to_quat_xyzw(rot6d_to_mat(d6))


def mat_to_quat_xyzw(m: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(..., 4)`` quaternion xyzw, exactly as ResiP's sim writes
    ``parts_poses``: its fork of FurnitureBench converts every pose matrix with
    ``control_utils.matrix_to_quaternion_xyzw`` (w = sqrt(1 + trace) / 2 >= 0; x, y, z take their
    magnitude from the diagonal and their sign from the off-diagonal differences)."""

    def sqrt_pos(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.clamp(min=0.0))

    def copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # theirs: no care for -0
        return torch.where((a < 0) != (b < 0), -a, a)

    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    w = 0.5 * sqrt_pos(1 + m00 + m11 + m22)
    x = copysign(0.5 * sqrt_pos(1 + m00 - m11 - m22), m[..., 2, 1] - m[..., 1, 2])
    y = copysign(0.5 * sqrt_pos(1 - m00 + m11 - m22), m[..., 0, 2] - m[..., 2, 0])
    z = copysign(0.5 * sqrt_pos(1 - m00 - m11 + m22), m[..., 1, 0] - m[..., 0, 1])
    return torch.stack((x, y, z, w), dim=-1)


class ResipObsAdapter:
    """The port's 63-wide observation -> ResiP's 58-wide one; their action -> the port's.

    ``tag_rot`` / ``tag_pos``: the AprilTag frame in the robot-base frame (``frames.TAG_*_W``);
    ``obstacle_z``: the front obstacle's fixed base-frame z (``frames.OBSTACLE_FRONT_POS``)."""

    #: their front obstacle's rotation in the base frame: 90 deg about z (ours is unrotated)
    OBSTACLE_R_B = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    def __init__(self, base_z_to_port: float, tag_rot=None, tag_pos=None, obstacle_z: float | None = None,
                 obstacle_quat: torch.Tensor | None = None) -> None:
        from furniture_lab.tasks.furniture import frames

        self.base_z_to_port = float(base_z_to_port)
        self.R_tw = torch.as_tensor(frames.TAG_ROT_W if tag_rot is None else tag_rot, dtype=torch.float32)
        self.p_tw = torch.as_tensor(frames.TAG_POS_W if tag_pos is None else tag_pos, dtype=torch.float32)
        self.obstacle_z = float(frames.OBSTACLE_FRONT_POS[2] if obstacle_z is None else obstacle_z)
        R_ob = self.R_tw.T @ torch.tensor(self.OBSTACLE_R_B, dtype=torch.float32)
        #: the obstacle's tag-frame quaternion (xyzw) fed to their policy: a constant
        self.obstacle_quat = (mat_to_quat_xyzw(R_ob) if obstacle_quat is None else torch.as_tensor(obstacle_quat)).float()

    def _to_tag(self, pos_b: torch.Tensor, R_b: torch.Tensor):
        R, p = self.R_tw.to(pos_b.device), self.p_tw.to(pos_b.device)
        return (pos_b - p) @ R, R.T @ R_b  # R^T (p_b - p_tw), R^T R_b

    def obs(self, port: torch.Tensor) -> torch.Tensor:
        if port.shape[-1] != PORT_OBS_DIM:
            raise ValueError(f"expected the ResiP task's {PORT_OBS_DIM}-wide observation, got {port.shape[-1]}")
        lead = port.shape[:-1]
        robot = port[..., :ROBOT].clone()
        robot[..., 2] -= self.base_z_to_port  # port base frame -> ResiP's (15 mm lower base)
        poses = port[..., ROBOT : ROBOT + 9 * N_PARTS].reshape(*lead, N_PARTS, 9)
        pos_t, R_t = self._to_tag(poses[..., :3], rot6d_to_mat(poses[..., 3:9]))
        parts = torch.cat((pos_t, mat_to_quat_xyzw(R_t)), dim=-1)
        # the front obstacle: its observed x, y, its fixed z; their body's fixed rotation
        xy = port[..., ROBOT + 9 * N_PARTS :]
        ob_b = torch.cat((xy, torch.full((*lead, 1), self.obstacle_z, device=port.device, dtype=port.dtype)), dim=-1)
        ob_t, _ = self._to_tag(ob_b[..., None, :], torch.eye(3, device=port.device).expand(*lead, 1, 3, 3))
        ob = torch.cat((ob_t, self.obstacle_quat.to(port.device, port.dtype).expand(*lead, 1, 4)), dim=-1)
        parts = torch.cat((parts, ob), dim=-2)
        return torch.cat((robot, parts.reshape(*lead, N_POSES * 7)), dim=-1)

    def action(self, resip: torch.Tensor) -> torch.Tensor:
        out = resip.clone()
        out[..., 2] += self.base_z_to_port  # ResiP base frame -> the port's
        return out


def load_resip_policy(path: str, device: Any, inference_steps: int = 4) -> Tuple[Any, ResipObsAdapter, Dict[str, Any]]:
    """ResiP's ``actor_chkpt.pt`` -> (this project's DiffusionPolicy, the obs adapter, info)."""
    from furniture_lab.tasks.furniture.frames import RESIP_BASE_Z_TO_PORT

    from my_project.models.cfg import DiffusionUnetActorCfg
    from my_project.models.diffusion_policy import DiffusionPolicy

    ckpt = torch.load(os.path.expanduser(path), map_location=device, weights_only=False)
    for key in ("model_state_dict", "config"):
        if key not in ckpt:
            raise KeyError(f"{path} has no {key!r} (keys: {sorted(ckpt)}): not a robust-rearrangement bc.py checkpoint?")
    sd, cfg = ckpt["model_state_dict"], ckpt["config"]
    actor = cfg["actor"]
    if actor.get("name") != "diffusion" or actor["diffusion_model"]["name"] != "unet":
        raise ValueError(f"{path}: expected a diffusion U-Net actor, got {actor.get('name')} / {actor['diffusion_model'].get('name')}")
    if cfg.get("observation_type") != "state":
        raise ValueError(f"{path}: observation_type {cfg.get('observation_type')!r}, expected 'state'")
    if cfg["control"]["control_mode"] != "pos" or cfg["control"]["act_rot_repr"] != "rot_6d":
        raise ValueError(f"{path}: control {cfg['control']}, expected pos / rot_6d")

    stats = {k[len("normalizer.stats."):]: v for k, v in sd.items() if k.startswith("normalizer.stats.")}
    model = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    other = [k for k in sd if not (k.startswith("model.") or k.startswith("normalizer."))]
    if other:
        raise KeyError(f"{path}: state-dict entries this loader does not map: {other}")
    for key in ("robot_state.min", "robot_state.max", "parts_poses.min", "parts_poses.max", "action.min", "action.max"):
        if key not in stats:
            raise KeyError(f"{path}: no normalizer stat {key!r} (has {sorted(stats)})")
    widths = (stats["robot_state.min"].numel(), stats["parts_poses.min"].numel(), stats["action.min"].numel())
    if widths != (ROBOT, N_POSES * 7, ACTION_DIM):
        raise ValueError(f"{path}: robot_state / parts_poses / action widths {widths}, expected {(ROBOT, N_POSES * 7, ACTION_DIM)}")

    actor_cfg = DiffusionUnetActorCfg(
        horizon=int(actor["pred_horizon"]),
        n_obs_steps=int(actor["obs_horizon"]),
        action_horizon=int(actor["action_horizon"]),
        num_diffusion_iters=int(actor["num_diffusion_iters"]),
        beta_schedule=str(actor["beta_schedule"]),
        prediction_type=str(actor["prediction_type"]),
        clip_sample=bool(actor["clip_sample"]),
        inference_steps=int(inference_steps),  # evaluate_model.py sets 4 for every diffusion actor
        warm_start=True,  # their actor always warm-starts ...
        warm_start_timestep=50,  # ... from this timestep
        obs_clip=3.0,
        down_dims=[int(d) for d in actor["diffusion_model"]["down_dims"]],
        # diffusion_step_embed_dim / kernel_size / n_groups: their get_diffusion_backbone passes
        # none of them, so ConditionalUnet1D's defaults (256 / 5 / 8) -- ours are the same
    )
    if actor_cfg.n_obs_steps != 1:
        raise ValueError(f"{path}: obs_horizon {actor_cfg.n_obs_steps}; this loader handles 1")
    if actor.get("predict_past_actions", False):
        raise ValueError(f"{path}: predict_past_actions is on; this loader handles future-only chunks")
    policy = DiffusionPolicy(actor_cfg, RESIP_OBS_DIM, ACTION_DIM).to(device)
    policy.model.load_state_dict(model, strict=True)
    with torch.no_grad():
        policy.normalizer.obs_min.copy_(torch.cat((stats["robot_state.min"], stats["parts_poses.min"])).to(device))
        policy.normalizer.obs_max.copy_(torch.cat((stats["robot_state.max"], stats["parts_poses.max"])).to(device))
        policy.normalizer.action_min.copy_(stats["action.min"].to(device))
        policy.normalizer.action_max.copy_(stats["action.max"].to(device))
    policy.eval()

    # the obstacle quaternion: the training value when the checkpoint saw it constant
    # (LinearNormalizer pads a constant column to [v - 1, v + 1]), else Rz(90) in the tag frame
    lo_q, hi_q = stats["parts_poses.min"][38:42].float().cpu(), stats["parts_poses.max"][38:42].float().cpu()
    constant = bool(torch.allclose(hi_q - lo_q, torch.full((4,), 2.0), atol=1e-4))
    adapter = ResipObsAdapter(RESIP_BASE_Z_TO_PORT, obstacle_quat=(lo_q + hi_q) / 2 if constant else None)
    info = {
        "epoch": ckpt.get("epoch"),
        "best_success_rate": ckpt.get("best_success_rate"),
        "task": cfg.get("task"),
        "randomness": cfg.get("randomness"),
        "horizon": actor_cfg.horizon,
        "action_horizon": actor_cfg.action_horizon,
        "inference_steps": actor_cfg.inference_steps,
        "obstacle_quat_stats": {"min": [round(float(v), 4) for v in lo_q], "max": [round(float(v), 4) for v in hi_q]},
        "obstacle_quat_fed": [round(float(v), 4) for v in adapter.obstacle_quat],
        "obstacle_quat_source": "the checkpoint's constant training value" if constant else "Rz(90) in the tag frame",
    }
    return policy, adapter, info


def main(argv=None, setup=None) -> int:
    from robonuke_rl_core.config import add_config_args

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True, help="ResiP's actor_chkpt.pt (robust-rearrangement bc.py)")
    parser.add_argument("--inference_steps", type=int, default=4, help="DDIM steps (their evaluate_model.py: 4)")
    parser.add_argument(
        "--save_rollouts", default=None,
        help="write every rollout step (observations, actions, rewards, metrics) to this parquet file",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    app_launcher = AppLauncher(args)
    try:
        return _run(args, overrides, setup, app_launcher)
    except BaseException:
        traceback.print_exc()  # Kit would turn an uncaught exception into exit status 0
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _run(args, overrides, setup, app_launcher) -> int:
    if setup is not None:
        setup()
    import gymnasium as gym
    import numpy as np

    from robonuke_rl_core.config import load_from_args
    from robonuke_rl_core.envs.build import build_env, prepare_task, wrap_for_skrl
    from robonuke_rl_core.evaluation import run_eval

    from my_project.families import observes_obstacle
    from my_project.models.diffusion_policy import ChunkedBase

    cfg = load_from_args(args, overrides)
    if not observes_obstacle(cfg.task_name):
        raise ValueError(f"{cfg.task_name} does not observe the front obstacle; use a ResiP task config (63-wide obs)")
    prepare_task(cfg, cfg.task_name, cfg.task_cfg)
    raw = gym.make(cfg.task_name, cfg=cfg.task_cfg)
    env = wrap_for_skrl(build_env(cfg, raw, cfg.task_name), cfg.experiment.seed)
    unwrapped = env.unwrapped
    obs_w, act_w = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))
    if (obs_w, act_w) != (PORT_OBS_DIM, ACTION_DIM):
        raise ValueError(f"env obs {obs_w} / action {act_w}; this adapter needs {PORT_OBS_DIM} / {ACTION_DIM}")

    policy, adapter, info = load_resip_policy(args.checkpoint, unwrapped.device, args.inference_steps)
    runner = ChunkedBase(policy, unwrapped.num_envs)
    print(f"[resip_ckpt] {args.checkpoint}: {info}", flush=True)

    class Policy:
        def __call__(self, observations, states):
            fresh = unwrapped.episode_length_buf == 0  # a new episode: fresh chunk + warm-start buffer
            return adapter.action(runner.actions(adapter.obs(observations), fresh))

    steps = int(unwrapped.max_episode_length)
    result = run_eval(
        env=env, policy=Policy(), num_rollouts=cfg.eval.num_rollouts, max_episode_length=steps,
        seed=cfg.experiment.seed, save_state=args.save_rollouts is not None, device=env.device,
        progress=True,  # steps out of num_rollouts x episode length; early ends add their rest
    )
    if args.save_rollouts is not None:
        print(f"[resip_ckpt] rollouts: {result.state.write_parquet(os.path.expanduser(args.save_rollouts))}", flush=True)
    s = result.summary
    if "episode/success/mean" not in s:
        raise KeyError(f"rollouts produced no 'episode/success' metric; summary keys: {sorted(s)}")
    rate, n = float(s["episode/success/mean"]), int(s["episodes"])
    print(f"[resip_ckpt] ===== {round(rate * n)}/{n} succeed ({100 * rate:.1f}%), {steps}-step episodes", flush=True)
    env.close()
    app_launcher.app.close()
    return 0
