"""Replay the forge-v1 NutThread demos in our env and keep the ones that succeed.

    python scripts/import_forge_v1.py --config configs/demos/forge_v1_nut_replay.yaml --headless

One env, ``task.*`` (the fixed-dynamics NutThread: bolt clamped, nut centred in the gripper),
bare ``gym.make`` plus the policy wrappers (world-pose actions), as the recorder. Per unique
forge-v1 episode (``my_project/demos/forge_v1.py`` recovers its bolt axis, nut offset and path):

* **start**: the reset puts the fingertip at the episode's start relative to the bolt tip, in
  its recorded orientation (the reset's IK target is replaced; everything else is the env's own
  reset). The path is the nut-centred one, since our nut sits centred in the gripper.
* **each step** sends the world pose ``[x, y, z, yaw]``: the recorded next position relative to
  our bolt tip and the recorded next yaw. From the recorded contact step on, z is instead the
  current height minus ``recorded force_z / kp_z``: the hand pushes down with the recorded force
  rather than tracking a height the dataset only knows to a few mm.
* the episode is **saved** (the recorder's columns, ``observation.*`` from our env, ``action`` =
  the world pose sent, bolt-relative with ``action.bolt_relative``) once the env's success check
  (rotation included) has held for
  ``forge_v1.success_hold_steps``; at the end of the recorded path it is dropped.

Stops at ``forge_v1.num_demos`` saved or when every episode has been tried. Prints one
``[forge_v1] progress`` line per episode.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback

from robonuke_rl_core.config import add_config_args, dump, load_from_args


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    app_launcher = AppLauncher(args)
    try:
        return _run(args, overrides, setup, app_launcher)
    except BaseException:
        # Kit's shutdown turns an uncaught exception into exit status 0; report and fail
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _run(args, overrides, setup, app_launcher) -> int:
    if setup is not None:
        setup()

    import gymnasium as gym
    import numpy as np
    import torch

    from my_project.demos import forge_v1
    from my_project.demos.record import step_signals
    from my_project.demos.state import restore, snapshot
    from my_project.demos.teleop import world_pose_to_forge_action
    from my_project.demos.writer import DemoWriter
    from my_project.wrappers import apply_policy_wrappers

    cfg = load_from_args(args, overrides)
    fv = cfg.forge_v1
    if not fv.source_dir or not fv.dataset_dir:
        raise ValueError("forge_v1.source_dir and forge_v1.dataset_dir are required")
    if cfg.task_cfg.scene.num_envs != 1:
        raise ValueError(f"the replay drives one env; task.cfg.scene.num_envs is {cfg.task_cfg.scene.num_envs}")
    if not cfg.action.world_pose:
        raise ValueError("the replay sends world poses: action.world_pose must be true")
    if cfg.task_cfg.task.name != "nut_thread":
        raise ValueError(f"forge-v1 replay is for nut threading; task is {cfg.task_cfg.task.name!r}")
    if any(float(v) != 0.0 for v in cfg.task_cfg.task.held_asset_pos_noise):
        raise ValueError("the replay follows the nut-centred path: task.cfg.task.held_asset_pos_noise must be 0")

    source = os.path.expanduser(fv.source_dir)
    episodes = forge_v1.unique_nut_episodes(source)
    height_at_contact = forge_v1.contact_height(source, episodes, fv.contact_force)
    print(f"[forge_v1] {len(episodes)} unique nut-threading episodes in {source}; fingertip "
          f"{height_at_contact * 1000:.2f} mm above the bolt tip at first contact", flush=True)

    env = gym.make(cfg.task_name, cfg=cfg.task_cfg)
    u = env.unwrapped
    control_dt = float(u.cfg.sim.dt * u.cfg.decimation)
    writer = DemoWriter(
        fv.dataset_dir,
        fps=1.0 / control_dt,
        info={
            "task": cfg.task_name,
            "control_dt": control_dt,
            "physics_dt": float(u.cfg.sim.dt),
            "decimation": int(u.cfg.decimation),
            "robot_type": "franka_panda",
            "source": "forge-v1 NutThread replay",
        },
    )
    done_sources = {e["source_episode"] for e in writer.episodes}
    session = f"{time.strftime('%Y%m%d_%H%M%S')}_seed{cfg.experiment.seed}"
    dump(cfg, writer.meta_dir / "sessions" / f"{session}.yaml", u.cfg)
    env = apply_policy_wrappers(cfg, env)

    # the reset's IK target, replaced per episode (Factory randomize_initial_state calls
    # self.set_pos_inverse_kinematics with its random pose; we hand it the demo's instead)
    original_ik = type(u).set_pos_inverse_kinematics
    target = {}

    def demo_ik(ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids):
        target["calls"] += 1
        if target["calls"] > 5:
            raise RuntimeError(f"the reset's IK did not reach episode {target['index']}'s start pose")
        pos = target["pos"].expand_as(ctrl_target_fingertip_midpoint_pos).clone()
        quat = target["quat"].expand_as(ctrl_target_fingertip_midpoint_quat).clone()
        return original_ik(u, pos, quat, env_ids)

    env.reset()  # the env's own reset: places the clamped bolt, so its tip can be read
    u.set_pos_inverse_kinematics = demo_ik

    def success() -> bool:
        return bool(u._get_curr_successes(u.cfg_task.success_threshold, check_rot=True)[0])

    host = lambda row: {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in row.items()}
    device = u.device
    bounds = torch.tensor(u.cfg.ctrl.pos_action_bounds, device=device)
    tip = u.fixed_pos_obs_frame[0].clone()  # the clamped bolt's tip (it never moves)
    yaw_low, yaw_high = -math.pi, math.pi / 2
    saved, tried, wall = writer.num_episodes, 0, time.perf_counter()

    for index in episodes:
        if saved >= fv.num_demos:
            break
        if index in done_sources:
            continue
        tried += 1
        outcome, first_success, steps = "fail", None, 0
        try:
            demo = forge_v1.reconstruct(forge_v1.load_raw(source, index), index, fv.contact_force, height_at_contact)
        except ValueError as error:
            outcome = f"unusable ({error})"
            demo = None
        if demo is not None:
            path = torch.tensor(demo.path, dtype=torch.float32, device=device)
            yaw = torch.tensor(demo.yaw, dtype=torch.float32, device=device).clamp(yaw_low, yaw_high)
            push = torch.tensor(demo.force[:, 2], dtype=torch.float32, device=device).clamp(min=0.0)
            target.update(index=index, calls=0, pos=(tip + path[0]).unsqueeze(0),
                          quat=torch.tensor(demo.quat[0], dtype=torch.float32, device=device).unsqueeze(0))
            env.reset()
            tip = u.fixed_pos_obs_frame[0].clone()
            start_error = float((u.fingertip_midpoint_pos[0] - target["pos"][0]).norm())
            start_state = snapshot(env)
            obs = restore(env, start_state)
            kp_z = float(u.task_prop_gains[0, 2])
            writer.start()
            run = 0
            for t in range(len(path) - 1):
                pose = torch.empty(4, device=device)
                pose[0:3] = tip + path[t + 1]
                if t + 1 >= demo.contact_step:
                    pose[2] = u.fingertip_midpoint_pos[0, 2] - push[t + 1] / kp_z
                pose[3] = yaw[t + 1]
                forge_action = world_pose_to_forge_action(pose, u.fixed_pos_obs_frame[0] + u.init_fixed_pos_obs_noise[0], bounds)
                action = env.policy_action(pose.unsqueeze(0))[0]  # bolt-relative with action.bolt_relative
                row = step_signals(u, obs, success())
                row.update({"action": action, "action.forge": forge_action})
                obs, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                ended = bool(terminated[0]) or bool(truncated[0])
                row["action.applied"] = torch.full((7,), math.nan) if ended else u.actions[0]
                writer.add(host(row))
                steps = t + 1
                if ended:
                    break
                if success():
                    run += 1
                    first_success = writer.length - 1 if first_success is None else first_success
                else:
                    run = 0
                if run >= fv.success_hold_steps:
                    record = writer.save(
                        {
                            "source": "forge-v1",
                            "source_episode": index,
                            "success": True,
                            "first_success_frame": first_success,
                            "ended_by": "success",
                            "contact_step": demo.contact_step,
                            "nut_offset_mm": (demo.nut_offset * 1000).round(3).tolist(),
                            "fit_residual_mm": round(demo.fit_residual * 1000, 3),
                            "start_error_mm": round(start_error * 1000, 3),
                            "height_at_contact_mm": round(height_at_contact * 1000, 3),
                            "seed": cfg.experiment.seed,
                            "session": session,
                            "input": "forge_v1_replay",
                        },
                        start_state,
                    )
                    saved += 1
                    outcome = f"saved as episode {record['episode_index']}"
                    break
            if not outcome.startswith("saved"):
                writer.discard()
                outcome = f"fail (start error {start_error * 1000:.2f} mm)"
        print(f"[forge_v1] progress: source episode {index} -> {outcome} after {steps} steps | "
              f"successful {saved}, tried {tried}, total {len(episodes)} | "
              f"{time.perf_counter() - wall:.0f} s", flush=True)

    print(f"[forge_v1] done: successful {saved}, tried {tried}, total {len(episodes)}; dataset {writer.root}",
          flush=True)
    sys.stdout.flush()
    os._exit(0)
