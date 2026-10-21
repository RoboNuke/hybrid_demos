"""Roll out ResiP's released BC checkpoint from the initial states of ResiP's own demos.

    python scripts/eval_resip_from_demos.py --config configs/experiments/pivot_demos.yaml \\
        --checkpoint ~/resip_ckpt/bc_one_leg_low.pt --zarr ~/resip_data/one_leg_teleop_low/success.zarr \\
        --trials 2 --headless [task.cfg.furniture.<field>=... overrides]

The starts are exactly the ones the policy was trained on, so a low score here is the port's
physics, not the reset distribution. Each env gets one demo (``trials`` envs per demo) and is put
into that demo's first recorded state the way Robonuke_Furniture_Bench's ``replay_abs.py --zarr``
does it -- the zarr holds no joint positions:

* the arm driven to the demo's first EE pose by the env's own IK (``--settle`` steps of that pose as
  the absolute target, the first gripper command), the EE z moved into this port's base frame
  (ResiP's base is 15 mm lower: ``frames.RESIP_BASE_Z_TO_PORT``);
* the fingers set to the recorded width, the three obstacles where the demo had them (the front
  one's recorded tag-frame position, the sides at ResiP's fixed offsets), the parts written at their
  recorded tag-frame poses (in the env's USD mesh frame), at rest;

then the policy acts as ``eval_resip_checkpoint.py`` acts it (``ResipObsAdapter``, 4 DDIM steps, warm
start) for ``--steps`` steps. Success is the env's soft assembled check, latched; a part leaving the
upstream position limits (the eval's ``part_out_of_limits`` termination) ends the episode as a
failure. ``--save_rollouts`` writes the steps in ``run_eval``'s state-table layout (``round`` = the
trial, ``env`` = the demo).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np
import torch

__all__ = ["main"]


def main(argv=None, setup=None) -> int:
    from robonuke_rl_core.config import add_config_args

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True, help="ResiP's actor_chkpt.pt (robust-rearrangement bc.py)")
    parser.add_argument("--zarr", required=True, help="ResiP's processed .zarr (the demos the checkpoint was trained on)")
    parser.add_argument("--trials", type=int, default=2, help="rollouts per demo start (envs = demos x trials)")
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--steps", type=int, default=700, help="policy steps per rollout (their eval's cap)")
    parser.add_argument("--settle", type=int, default=30, help="IK steps driving the arm to the demo's first EE pose")
    parser.add_argument("--inference_steps", type=int, default=4, help="DDIM steps (their evaluate_model.py: 4)")
    parser.add_argument("--save_rollouts", default=None, help="write every step to this parquet file")
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


def _starts(path: str, max_demos):
    """Per successful zarr episode: the first EE pose (port base frame), gripper command and width,
    tag-frame part poses (35) and front-obstacle position (tag frame)."""
    import zarr

    from furniture_lab.tasks.furniture.frames import RESIP_BASE_Z_TO_PORT

    z = zarr.open(os.path.expanduser(path), mode="r")
    ends = np.asarray(z["episode_ends"][:], dtype=np.int64)
    rs = np.asarray(z["robot_state"][:], dtype=np.float64)
    pp = np.asarray(z["parts_poses"][:], dtype=np.float64)
    act = np.asarray(z["action/pos"][:], dtype=np.float64)
    if pp.shape[1] != 42:
        raise ValueError(f"{path}: parts_poses is {pp.shape[1]} wide; expected ResiP's 42 (5 parts + the front obstacle)")
    success = np.asarray(z["success"][:], dtype=np.int64) if "success" in z else np.ones(len(ends), np.int64)
    out, lo = [], 0
    for e, hi in enumerate(ends):
        if success[e]:
            ee = rs[lo, 0:9].copy()
            ee[2] += RESIP_BASE_Z_TO_PORT
            out.append(dict(episode=e, ee=ee, grip=float(act[lo, 9]), width=float(rs[lo, 15]), parts=pp[lo, :35], front=pp[lo, 35:38]))
        lo = int(hi)
        if max_demos and len(out) >= max_demos:
            break
    return out


def _set_starts(env, starts, settle: int) -> None:
    """Every env into its demo's first state (see the module docstring), all envs at once."""
    from furniture_lab.demos import _parts_to_world
    from furniture_lab.tasks.assembly.mdp.state import tracker
    from furniture_lab.tasks.furniture.frames import TAG_POS_W, TAG_ROT_W
    from furniture_lab.tasks.furniture.square_table import MESH_FRAMES

    dev, n = env.device, env.num_envs
    target = torch.tensor(np.stack([np.concatenate([s["ee"], [s["grip"]]]) for s in starts]), device=dev, dtype=torch.float32)
    for _ in range(settle):
        env.step(target)
    robot = env.scene["robot"]
    fingers = robot.find_joints("panda_finger_joint.*")[0]
    q = robot.data.joint_pos.clone()
    q[:, fingers] = torch.tensor([s["width"] / 2 for s in starts], device=dev, dtype=q.dtype)[:, None]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    origins = env.scene.env_origins
    # the obstacles: the front one at the recorded tag-frame position, the sides at ResiP's offsets
    front = np.asarray(TAG_POS_W) + np.stack([s["front"] for s in starts]) @ np.asarray(TAG_ROT_W).T  # (n, 3) base frame
    for name, off in (("obstacle_front", (0.0, 0.0)), ("obstacle_right", (-0.075, -0.175)), ("obstacle_left", (-0.075, 0.175))):
        asset = env.scene[name]
        pose = asset.data.default_root_state[:, :7].clone()
        pose[:, :3] = torch.tensor(front + np.array([off[0], off[1], 0.0]), device=dev, dtype=pose.dtype) + origins
        asset.write_root_pose_to_sim(pose)
    R_mesh = MESH_FRAMES[env.cfg.furniture.mesh_frame]
    poses = [_parts_to_world(np.asarray(s["parts"]), R_mesh) for s in starts]
    for k, name in enumerate(env.cfg.furniture.part_names):
        pos = torch.tensor(np.stack([p[0][k] for p in poses]), device=dev, dtype=torch.float32) + origins
        quat = torch.tensor(np.stack([p[1][k] for p in poses]), device=dev, dtype=torch.float32)
        part = env.scene[name]
        part.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        part.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    tracker(env).reset()


def _run(args, overrides, setup, app_launcher) -> int:
    if setup is not None:
        setup()
    import gymnasium as gym

    from robonuke_rl_core.config import load_from_args
    from robonuke_rl_core.envs.build import prepare_task

    from furniture_lab.tasks.assembly.mdp.state import tracker
    from furniture_lab.tasks.assembly.mdp.terminations import part_out_of_limits

    from my_project.bc.resip_checkpoint import PORT_OBS_DIM, load_resip_policy
    from my_project.families import observes_obstacle
    from my_project.models.diffusion_policy import ChunkedBase

    cfg = load_from_args(args, overrides)
    if not observes_obstacle(cfg.task_name):
        raise ValueError(f"{cfg.task_name} does not observe the front obstacle; use a ResiP task config (63-wide obs)")
    starts = _starts(args.zarr, args.max_demos)
    n_demos = len(starts)
    tc = cfg.task_cfg
    tc.scene.num_envs = n_demos * int(args.trials)
    tc.episode_length_s = 1e6  # the rollout's length is --steps; no time-out reset in between
    tc.terminations.success = None  # success is read from the tracker's latch, as the eval's termination does
    tc.terminations.part_out_of_limits = None  # checked here: it ends the episode as a failure
    prepare_task(cfg, cfg.task_name, tc)
    env = gym.make(cfg.task_name, cfg=tc).unwrapped
    f = env.cfg.furniture
    print(f"[demo_starts] {n_demos} demos x {args.trials} trials = {env.num_envs} envs; parts {f.part_assets} "
          f"({f.mesh_frame}), gripper {f.gripper_mode}, friction obstacles {f.obstacle_friction} robot {f.robot_friction}, "
          f"ee_vel_point {f.ee_vel_point}", flush=True)
    env.reset()
    per_env = [starts[i % n_demos] for i in range(env.num_envs)]
    _set_starts(env, per_env, args.settle)
    obs = env.observation_manager.compute()["policy"]
    if obs.shape[-1] != PORT_OBS_DIM:
        raise ValueError(f"env obs {obs.shape[-1]}; the adapter needs {PORT_OBS_DIM}")

    policy, adapter, info = load_resip_policy(args.checkpoint, env.device, args.inference_steps)
    runner = ChunkedBase(policy, env.num_envs)
    print(f"[demo_starts] {args.checkpoint}: {info}", flush=True)

    n = env.num_envs
    success = torch.zeros(n, dtype=torch.bool, device=env.device)
    ended = torch.zeros(n, dtype=torch.bool, device=env.device)
    end_step = torch.full((n,), args.steps, dtype=torch.long, device=env.device)
    fresh = torch.ones(n, dtype=torch.bool, device=env.device)
    rows = [] if args.save_rollouts else None
    for t in range(args.steps):
        action = adapter.action(runner.actions(adapter.obs(obs), fresh))
        fresh[:] = False
        if rows is not None:
            live = (~ended).nonzero(as_tuple=False).squeeze(-1).tolist()
            o, a = obs.detach().cpu().numpy(), action.detach().cpu().numpy()
            rows += [(i // n_demos, i % n_demos, t, o[i], a[i]) for i in live]
        obs_dict, *_ = env.step(action)
        obs = obs_dict["policy"]
        done_now = ~ended & tracker(env).all_assembled()
        out_now = ~ended & ~done_now & part_out_of_limits(env)
        success |= done_now
        end_step[done_now | out_now] = t + 1
        ended |= done_now | out_now
        if bool(ended.all()):
            break

    s = success.cpu().numpy()
    per_demo = s.reshape(int(args.trials), n_demos).mean(0)
    print(f"[demo_starts] per demo (trials {args.trials}): "
          + " ".join(f"{st['episode']}:{int(round(r * args.trials))}" for st, r in zip(starts, per_demo)), flush=True)
    print(f"[demo_starts] demos solved at least once: {int((per_demo > 0).sum())}/{n_demos}; "
          f"always: {int((per_demo == 1).sum())}/{n_demos}", flush=True)
    print(f"[demo_starts] ===== {int(s.sum())}/{n} succeed ({100 * s.mean():.1f}%) from ResiP's demo starts, "
          f"{args.steps}-step episodes", flush=True)
    if rows is not None:
        import pandas as pd

        succ = {(i // n_demos, i % n_demos): bool(s[i]) for i in range(n)}
        table = {"round": [r[0] for r in rows], "env": [r[1] for r in rows], "step": [r[2] for r in rows]}
        O = np.stack([r[3] for r in rows])
        A = np.stack([r[4] for r in rows])
        table.update({f"observations_{k}": O[:, k] for k in range(O.shape[1])})
        table.update({f"actions_{k}": A[:, k] for k in range(A.shape[1])})
        table["episode_metrics/episode/success"] = [float(succ[(r[0], r[1])]) for r in rows]
        out = os.path.expanduser(args.save_rollouts)
        pd.DataFrame(table).to_parquet(out, index=False)
        print(f"[demo_starts] rollouts: {out}", flush=True)
    env.close()
    app_launcher.app.close()
    return 0
