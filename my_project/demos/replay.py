"""Replay recorded demos: put the env back exactly as each episode began, re-send its actions.

    # watch in the GUI, in real time
    python scripts/replay_demos.py --dataset ~/demos/peg_insert --episodes 0,3,7

    # headless, one mp4 per episode (to <dataset>/videos/ unless --out is given)
    python scripts/replay_demos.py --dataset ~/demos/peg_insert --video --headless

The env is rebuilt from the recording session's own resolved config
(``meta/sessions/<session>.yaml``), made bare like the recorder makes it, reset, and then
restored to the episode's saved starting state (``states/``, see ``my_project/demos/state.py``).
Then the recorded ``action`` column is sent step by step: exactly what the recorder sent.

Every episode reports the true fingertip / peg position error right after the restore (should be 0) and the largest
difference from the recording over the episode, and whether the env's success check
agrees. The recorder starts every episode through the same reset-then-restore path, so a replay
on the recording's physics device reproduces it exactly (0.00 mm). Replaying on the other device
(``task.cfg.sim.device=cuda:0`` for a CPU recording) shows how CPU and GPU PhysX differ: they
do not agree once the peg is in contact.

Episodes from several sessions replay in one run only if their sessions resolved to the same
task config (one Isaac env per process); otherwise replay them a session at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List


def add_replay_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", required=True, help="a dataset written by scripts/record_demos.py")
    parser.add_argument(
        "--episodes", default="all", help="'all' or a comma list of episode indices, e.g. 0,3,7"
    )
    parser.add_argument("--video", action="store_true", help="write one mp4 per episode")
    parser.add_argument("--out", default=None, help="video directory (default <dataset>/videos)")
    parser.add_argument(
        "--fast", action="store_true", help="GUI only: do not pace the replay to real time"
    )
    return parser


def main(argv=None, setup=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_replay_args(parser)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    if args.video:
        args.enable_cameras = True  # the recorder camera renders headless
    app_launcher = AppLauncher(args)
    try:
        return _replay(args, overrides, setup, app_launcher)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def _select(episodes: List[dict], spec: str) -> List[dict]:
    if spec == "all":
        return episodes
    wanted = [int(part) for part in spec.split(",") if part.strip()]
    by_index = {e["episode_index"]: e for e in episodes}
    missing = [i for i in wanted if i not in by_index]
    if missing:
        raise KeyError(f"episodes {missing} are not in the dataset (it has 0..{len(episodes) - 1})")
    return [by_index[i] for i in wanted]


def _replay(args, overrides, setup, app_launcher) -> int:
    if setup is not None:
        setup()

    import gymnasium as gym
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from omegaconf import OmegaConf

    from robonuke_rl_core.config import load_from_run
    from robonuke_rl_core.recording import (
        VideoWriter,
        install_recorder_camera,
        read_camera_rgb,
        recorder_camera,
    )

    from my_project.demos.state import restore
    from my_project.wrappers import apply_policy_wrappers

    root = Path(args.dataset).expanduser()
    lines = (root / "meta" / "episodes.jsonl").read_text().splitlines()
    episodes = _select([json.loads(line) for line in lines if line], args.episodes)
    if not episodes:
        raise ValueError(f"{root} has no episodes to replay")

    sessions = sorted({e["session"] for e in episodes})
    configs = {s: root / "meta" / "sessions" / f"{s}.yaml" for s in sessions}
    task_cfgs = {s: OmegaConf.to_container(OmegaConf.load(p)["task"], resolve=True) for s, p in configs.items()}
    for s in sessions[1:]:
        if task_cfgs[s] != task_cfgs[sessions[0]]:
            raise ValueError(
                f"sessions {sessions[0]} and {s} ran different task configs; replay them separately "
                "(--episodes from one session at a time)"
            )
    extra = list(overrides)
    if args.video:
        extra.append("task.cfg.scene.clone_in_fabric=false")  # the camera needs real prims
    cfg = load_from_run(configs[sessions[0]], extra)
    if args.video:
        install_recorder_camera(
            cfg.task_name, cfg.task_cfg, width=cfg.eval.video_width, height=cfg.eval.video_height
        )
    env = apply_policy_wrappers(cfg, gym.make(cfg.task_name, cfg=cfg.task_cfg))
    u = env.unwrapped
    control_dt = float(u.cfg.sim.dt * u.cfg.decimation)
    camera = recorder_camera(env) if args.video else None
    out_dir = Path(args.out).expanduser() if args.out else root / "videos"
    realtime = not args.headless and not args.video and not args.fast
    print(f"[replay] {len(episodes)} episode(s) from {root}, {1 / control_dt:.1f} Hz"
          + (f", videos to {out_dir}" if args.video else ""), flush=True)

    for episode in episodes:
        index = episode["episode_index"]
        table = pq.read_table(root / episode["file"])
        actions = torch.from_numpy(
            table.column("action").combine_chunks().flatten().to_numpy().reshape(table.num_rows, -1)
        ).to(u.device)
        def column(name, width):
            return table.column(name).combine_chunks().flatten().to_numpy().reshape(table.num_rows, width)

        # the true (noise-free) poses: noise is redrawn on replay, the physics must not differ
        recorded_pos = column("observation.fingertip_pos", 3)
        recorded_peg = column("observation.peg_pos", 3)
        state = torch.load(root / episode["state_file"], map_location="cpu", weights_only=False)

        env.reset()
        restore(env, state)
        writer = VideoWriter(out_dir / f"episode_{index:06d}.mp4", fps=round(1 / control_dt)) if args.video else None
        drift, worst_step, ever_success, steps = 0.0, 0, False, 0
        def pose_error(t):
            fingertip = np.abs(u.fingertip_midpoint_pos[0].detach().cpu().numpy() - recorded_pos[t]).max()
            peg = np.abs(u.held_pos[0].detach().cpu().numpy() - recorded_peg[t]).max()
            return float(max(fingertip, peg))

        start_error = pose_error(0)
        try:
            for t in range(actions.shape[0]):
                tick = time.perf_counter()
                error = pose_error(t)
                if error > drift:
                    drift, worst_step = error, t
                _, _, terminated, truncated, _ = env.step(actions[t : t + 1])
                steps += 1
                if writer is not None:
                    writer.append(read_camera_rgb(camera)[0].numpy())
                if bool(terminated[0]) or bool(truncated[0]):
                    break  # the recording's last step hit the time limit
                # the env's own rule: nut threading also needs the wrist turned
                check_rot = u.cfg_task.name == "nut_thread"
                ever_success |= bool(u._get_curr_successes(u.cfg_task.success_threshold, check_rot=check_rot)[0])
                if realtime:
                    spare = control_dt - (time.perf_counter() - tick)
                    if spare > 0:
                        time.sleep(spare)
        finally:
            if writer is not None:
                writer.close()
        agree = "agrees" if ever_success == bool(episode["success"]) else "DISAGREES"
        print(
            f"[replay] episode {index}: {steps} steps, start error {start_error * 1000:.2f} mm, "
            f"max drift (fingertip, peg) {drift * 1000:.2f} mm (step {worst_step}), "
            f"success {ever_success} (recorded {episode['success']}, {agree})"
            + (f", {writer.path}" if writer is not None else ""),
            flush=True,
        )

    env.close()
    app_launcher.app.close()
    return 0
