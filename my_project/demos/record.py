"""Record teleoperated demonstrations in the bare task env.

    python scripts/record_demos.py --config configs/demos/forge_demo_collection.yaml \\
        record.dataset_dir=~/demos/peg_insert

One env, the GUI, an Xbox-style controller (layout in ``my_project/demos/gamepad.py``). The env
is exactly ``task.*``, made with ``gym.make``, plus only the project's observation additions
(``observation.*``, ``my_project/wrappers.py``): no package wrappers, no controller (``wrappers.*`` and ``controller.*`` are not read; an enabled
controller raises, because it would change what an action means). Between the operator and the
env sit only the teleop conversion (``my_project/demos/teleop.py``: stick velocity -> Forge
action) and the writer (``my_project/demos/writer.py``).

Per control step it records, before the step: the policy observation (noisy, what training
uses) and Forge's noisy inputs to it (fingertip pose, force, hole estimate); the env's noise-free
state vector; the true fingertip pose and velocities, the wrench, the joints, the hole and peg poses, the
force threshold and the env's success flag; the action sent and, after the step, the smoothed
action the env applied; and the teleop command, speed scale and setpoint. An episode ends when
the success check has held for ``record.success_hold_steps`` (saved), on A (saved, success as the
check says), on B (discarded), at the env's time limit (discarded unless
``record.save_timeouts``) or on START (quit; the unsaved episode is discarded).

Every session draws a fresh random ``experiment.seed`` (so sessions never replay one another's
holes and start poses); passing ``experiment.seed=N`` uses that seed instead. Each episode's seed
is in ``meta/episodes.jsonl``, but replay does not need it: it restores the saved start state.
"""

from __future__ import annotations

import argparse
import math
import os
import secrets
import sys
import time
import traceback
from typing import Any, Dict

from robonuke_rl_core.config import add_config_args, dump, load_from_args


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    if not any(str(o).startswith("experiment.seed=") for o in overrides):
        overrides = [f"experiment.seed={secrets.randbelow(2**31)}", *overrides]
    app_launcher = AppLauncher(args)
    try:
        return _record(args, overrides, setup, app_launcher)
    except BaseException:
        # Kit's shutdown turns an uncaught exception into exit status 0; report and fail
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def step_signals(u: Any, obs: Dict[str, Any], success: bool) -> Dict[str, Any]:
    """One env's (env 0) columns for a step, read before the step: ``observation.state`` is what
    the policy sees (Forge's noisy observation); the ``noisy_*`` columns are Forge's noisy inputs
    to it; everything else is the true value."""
    from my_project.demos.teleop import fingertip_yaw

    return {
        "observation.state": obs["policy"][0],
        "observation.critic": obs["critic"][0],
        "observation.noisy_fingertip_pos": u.noisy_fingertip_pos[0],
        "observation.noisy_fingertip_quat": u.noisy_fingertip_quat[0],
        "observation.noisy_force": u.noisy_force[0],
        # what Forge's actions are relative to (noisy: Forge adds a per-episode offset)
        "observation.noisy_hole_pos": u.fixed_pos_obs_frame[0] + u.init_fixed_pos_obs_noise[0],
        "observation.fingertip_pos": u.fingertip_midpoint_pos[0],
        "observation.fingertip_quat": u.fingertip_midpoint_quat[0],
        "observation.fingertip_linvel": u.fingertip_midpoint_linvel[0],
        "observation.fingertip_angvel": u.fingertip_midpoint_angvel[0],
        "observation.fingertip_yaw": fingertip_yaw(u.fingertip_midpoint_quat[0]),
        "observation.wrench": u.force_sensor_smooth[0],
        "observation.joint_pos": u.joint_pos[0, 0:7],
        "observation.joint_vel": u.joint_vel[0, 0:7],
        "observation.hole_pos": u.fixed_pos_obs_frame[0],
        "observation.hole_quat": u.fixed_quat[0],
        "observation.peg_pos": u.held_pos[0],
        "observation.peg_quat": u.held_quat[0],
        "observation.force_threshold": u.contact_penalty_thresholds[0],
        "observation.success": float(success),
    }


def _record(args, overrides, setup, app_launcher) -> int:
    if setup is not None:
        setup()

    import gymnasium as gym
    import torch

    from my_project.demos.gamepad import Gamepad
    from my_project.demos.state import restore, snapshot
    from my_project.demos.teleop import SetpointTeleop, TeleopLimits, fingertip_yaw, world_pose_to_forge_action
    from my_project.demos.writer import DemoWriter
    from my_project.wrappers import apply_policy_wrappers

    cfg = load_from_args(args, overrides)
    rec = cfg.record
    if not rec.dataset_dir:
        raise ValueError("record.dataset_dir is required: where the demos are written")
    if cfg.task_cfg.scene.num_envs != 1:
        raise ValueError(f"the recorder drives one env; task.cfg.scene.num_envs is {cfg.task_cfg.scene.num_envs}")
    controller = getattr(cfg, "controller", None)
    if controller is not None and controller.enabled:
        raise ValueError("controller.enabled changes what an action means; the recorder runs the bare env")
    if rec.input == "gamepad" and args.headless:
        raise ValueError("record.input: gamepad needs the GUI (drop --headless), or use record.input=expert")

    env = gym.make(cfg.task_name, cfg=cfg.task_cfg)
    u = env.unwrapped
    control_dt = float(u.cfg.sim.dt * u.cfg.decimation)
    seed = cfg.experiment.seed

    writer = DemoWriter(
        rec.dataset_dir,
        fps=1.0 / control_dt,
        info={
            "task": cfg.task_name,
            "control_dt": control_dt,
            "physics_dt": float(u.cfg.sim.dt),
            "decimation": int(u.cfg.decimation),
            "robot_type": "franka_panda",
        },
    )
    if seed in writer.used_seeds():
        raise ValueError(
            f"experiment.seed={seed} was already used in {writer.root}: it would replay that "
            "session's initial conditions. Leave experiment.seed out to draw a fresh one."
        )
    session = f"{time.strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    config_path = dump(cfg, writer.meta_dir / "sessions" / f"{session}.yaml", u.cfg)
    env = apply_policy_wrappers(cfg, env)
    world_pose_actions = bool(cfg.action.world_pose)
    bounds = torch.tensor(u.cfg.ctrl.pos_action_bounds, device=u.device)

    device = u.device
    teleop = SetpointTeleop(
        TeleopLimits(
            max_speed_pos=rec.max_speed_pos,
            max_speed_yaw=rec.max_speed_yaw,
            leash_pos=rec.leash_pos,
            leash_yaw=rec.leash_yaw,
            pos_action_bounds=list(u.cfg.ctrl.pos_action_bounds),
        ),
        dt=control_dt,
        device=device,
    )
    if rec.input == "gamepad":
        pad = Gamepad(rec.stick_dead_zone, rec.axis_signs)
    elif rec.input == "search_expert":
        from my_project.demos.search_expert import SearchExpert

        pad = SearchExpert(rec.search_expert, max_speed_pos=rec.max_speed_pos, dt=control_dt)
    else:
        from my_project.demos.expert import Expert

        generator = torch.Generator().manual_seed(seed)  # the session seed: reproducible choices
        pad = Expert(rec.expert, max_speed_pos=rec.max_speed_pos, dt=control_dt, generator=generator)
    success_threshold = u.cfg_task.success_threshold

    def success() -> bool:
        # the env's own rule (Factory/Forge _get_rewards): nut threading also needs the wrist turned
        return bool(u._get_curr_successes(success_threshold, check_rot=u.cfg_task.name == "nut_thread")[0])

    def hole_estimate():
        # what Forge's actions are relative to (noisy: Forge adds a per-episode offset)
        return u.fixed_pos_obs_frame[0] + u.init_fixed_pos_obs_noise[0]

    signals = lambda obs: step_signals(u, obs, success())

    host = lambda row: {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in row.items()}

    print(f"[record] {cfg.task_name}: {1 / control_dt:.1f} Hz, dataset {writer.root} "
          f"({writer.num_episodes} episodes so far), session {session}, input {pad.name}", flush=True)
    if rec.input == "gamepad":
        print("[record] left stick: move | right stick: up/down, yaw | RB: fine | A: save | "
              "B: discard | START: quit", flush=True)

    saved, quit_ = 0, False
    slow_steps = 0
    try:
        while not quit_ and (rec.num_episodes == 0 or saved < rec.num_episodes):
            env.reset()
            # replay resets and then restores this snapshot; doing the same here gives the
            # recording and every replay an identical start, solver history included
            start_state = snapshot(env)
            obs = restore(env, start_state)
            teleop.reset(u.fingertip_midpoint_pos[0], u.fingertip_midpoint_quat[0])
            pad.reset()
            writer.start()
            start = {
                "hole_pos": u.fixed_pos_obs_frame[0].tolist(),
                "start_fingertip_pos": u.fingertip_midpoint_pos[0].tolist(),
                "start_yaw": float(fingertip_yaw(u.fingertip_midpoint_quat[0])),
                "force_threshold": float(u.contact_penalty_thresholds[0]),
                "ema_factor": float(u.ema_factor.reshape(-1)[0]),
                "seed": seed,
                "session": session,
                "input": rec.input,
            }
            run, first_success, wall = 0, None, time.perf_counter()
            while True:
                tick = time.perf_counter()
                if pad.take("quit"):
                    quit_ = True
                    writer.discard()
                    print("[record] quit", flush=True)
                    break
                if pad.take("discard"):
                    print(f"[record] discarded ({writer.discard()} steps)", flush=True)
                    break

                here, quat, hole = u.fingertip_midpoint_pos[0], u.fingertip_midpoint_quat[0], hole_estimate()
                # what a driver may look at: the true state (the oracle steers with it), the
                # noisy observations (the search expert's) and its own setpoint; the gamepad
                # ignores all of it
                command = pad.command(
                    {
                        "fingertip_pos": here,
                        "peg_pos": u.held_pos[0],
                        "hole_pos": u.fixed_pos_obs_frame[0],
                        "force": u.force_sensor_smooth[0, 0:3],
                        "setpoint_pos": teleop.pos,
                        "setpoint_yaw": teleop.yaw,
                        # the policy's view: what the search expert steers with
                        "noisy_fingertip_pos": u.noisy_fingertip_pos[0],
                        "noisy_hole_pos": hole,
                        "noisy_force": u.noisy_force[0],
                    }
                ).to(device)
                scale = rec.fine_scale if pad.held("fine") else 1.0
                pose = teleop.step(command, scale, here, quat, hole)  # world [x, y, z, yaw]
                forge_action = world_pose_to_forge_action(pose, hole, bounds)
                # what the policy will output: the world pose (bolt-relative with
                # action.bolt_relative), or Forge's own action
                action = env.policy_action(pose) if world_pose_actions else forge_action
                row = signals(obs)
                row.update(
                    {
                        "action": action,
                        "action.forge": forge_action,
                        "teleop.command": command,
                        "teleop.scale": scale,
                        "teleop.setpoint": torch.cat([teleop.pos, teleop.yaw.reshape(1)]),
                    }
                )

                obs, _, terminated, truncated, _ = env.step(action.unsqueeze(0))
                done = bool(terminated[0]) or bool(truncated[0])
                # after a time-limit step the env has already reset itself: nothing it holds
                # now belongs to this episode
                row["action.applied"] = torch.full((7,), math.nan) if done else u.actions[0]
                writer.add(host(row))

                if not done and success():
                    run += 1
                    first_success = writer.length - 1 if first_success is None else first_success
                else:
                    run = 0
                meta = dict(start, success=first_success is not None, first_success_frame=first_success,
                            duration_s=round(time.perf_counter() - wall, 2), **pad.episode_meta())
                if pad.take("save"):
                    record = writer.save(dict(meta, ended_by="button"), start_state)
                elif rec.success_hold_steps and run >= rec.success_hold_steps:
                    record = writer.save(dict(meta, ended_by="success"), start_state)
                elif done:
                    if rec.save_timeouts:
                        record = writer.save(dict(meta, ended_by="timeout", success=False), start_state)
                    else:
                        print(f"[record] time limit: discarded ({writer.discard()} steps) {pad.episode_meta()}", flush=True)
                        break
                else:
                    record = None
                if record is not None:
                    saved += 1
                    print(f"[record] saved episode {record['episode_index']}: {record['length']} steps, "
                          f"success={record['success']} ({record['ended_by']})", flush=True)
                    break

                if rec.realtime:
                    spare = control_dt - (time.perf_counter() - tick)
                    if spare > 0:
                        time.sleep(spare)
                    else:
                        slow_steps += 1
                        if slow_steps == 30:
                            print("[record] warning: the sim is slower than real time", flush=True)
    except KeyboardInterrupt:
        print(f"[record] interrupted; the unsaved episode ({writer.discard()} steps) is dropped", flush=True)
    finally:
        pad.close()

    print(f"[record] session {session}: saved {saved}; dataset now {writer.num_episodes} episodes, "
          f"{writer.num_frames} steps; config {config_path}", flush=True)
    env.close()
    app_launcher.app.close()
    return 0
