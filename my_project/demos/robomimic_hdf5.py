"""robomimic-format demo hdf5s -> the recorder's dataset (``my_project/demos/writer.py``), for ResFiT's robosuite tasks
(``my_project/tasks/robosuite_tasks.py``).

    observation.state  the task's observation (its cfg's ``obs_keys``)
    action             the recorded action, as the task's controller takes it
    next.reward        1.0 on and after the first frame whose step succeeded, else 0

Two sources:

* :func:`import_robomimic` -- robomimic's low-dim files (Can, Square: ``low_dim_v15.hdf5``), whose ``obs`` group holds
  every key the task observes (``object`` is robosuite's ``object-state``); no sim. Their ``rewards`` mark success.
* :func:`import_replayed` -- DexMimicGen's generated files, whose ``obs`` lack the objects' state: every frame's
  recorded MuJoCo state is set in the task's own env (the demo's model XML loaded first) and the observation read as
  the env reads it (:func:`~my_project.tasks.robosuite_tasks.obs_row`), success by the env's ``_check_success``. In
  parallel worker processes.

Every frame of every demo is kept (BC trains on them all); ``meta/episodes.jsonl`` records each demo's
``success_frame``: the first frame whose step succeeded, where the env (``terminate_on_success``) would end the episode.
``states/`` holds the demo's first MuJoCo state and its model XML, for replay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["import_robomimic", "import_replayed", "task_cfg_for"]


def task_cfg_for(task_name: str):
    """The task's cfg object, from its gym registration (``my_project/tasks/__init__.py``)."""
    import importlib

    import gymnasium as gym

    import my_project.tasks  # noqa: F401  -- gym.register

    module, attr = gym.spec(task_name).kwargs["env_cfg_entry_point"].split(":")
    return getattr(importlib.import_module(module), attr)()


def _writer(out: str, task_name: str, cfg, source: str):
    from my_project.demos.writer import DemoWriter

    writer = DemoWriter(
        out,
        fps=float(cfg.control_freq),
        info={"task": task_name, "control_dt": 1.0 / float(cfg.control_freq), "source": source,
              "observation.state": list(cfg.obs_keys), "action": f"{cfg.env_name}'s controller input ({cfg.action_space})"},
    )
    if writer.num_episodes:
        raise FileExistsError(f"{out} already holds {writer.num_episodes} episodes; import into a new directory")
    return writer


def _save(writer, rows: np.ndarray, actions: np.ndarray, success_frame: int, source: str, start: Dict[str, Any]) -> None:
    writer.start()
    for t in range(actions.shape[0]):
        writer.add({"observation.state": rows[t], "action": actions[t], "next.reward": float(t >= success_frame)})
    writer.save({"success": True, "success_frame": int(success_frame), "source": source, "seed": None,
                 "reward_sum": float(actions.shape[0] - success_frame)}, start)


def _summary(lengths: List[int], success_frames: List[int], out: str) -> Dict[str, Any]:
    return {"episodes": len(lengths), "frames": int(sum(lengths)), "length_mean": float(np.mean(lengths)),
            "success_frame_mean": float(np.mean(success_frames)), "out": str(Path(out).expanduser())}


def import_robomimic(hdf5: str, out: str, task_name: str, max_demos: Optional[int] = None) -> Dict[str, Any]:
    import h5py

    cfg = task_cfg_for(task_name)
    source = Path(hdf5).expanduser()
    lengths, success_frames = [], []
    with h5py.File(source, "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"])
        writer = _writer(out, task_name, cfg, f"robomimic {source.name} ({env_args['env_name']}, robosuite {env_args.get('env_version')})")
        names = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))[: max_demos]
        for name in names:
            demo = data[name]
            obs = {k: demo["obs"][k][:] for k in demo["obs"].keys()}
            missing = [k for k in cfg.obs_keys if k not in obs]
            if missing:
                raise KeyError(f"{source}:{name} has no obs {missing}; {task_name} observes {list(cfg.obs_keys)}")
            rows = np.concatenate([obs[k].reshape(obs[k].shape[0], -1) for k in cfg.obs_keys], axis=1).astype(np.float32)
            hits = np.flatnonzero(demo["rewards"][:] >= 1.0)
            if hits.size == 0:
                raise ValueError(f"{source}:{name} never succeeds")
            actions = demo["actions"][:]
            _save(writer, rows, actions, int(hits[0]), f"{source}:{name}",
                  {"mujoco_state": demo["states"][0], "model_file": demo.attrs["model_file"]})
            lengths.append(actions.shape[0])
            success_frames.append(int(hits[0]))
    return _summary(lengths, success_frames, out)


def _replay_worker(job) -> List[Dict[str, Any]]:
    """Replay ``names`` of ``hdf5`` in the task's env: each frame's observation row and the demo's success frame."""
    import h5py

    from my_project.tasks.robosuite_tasks import make_robosuite_env, obs_row

    hdf5, task_name, names = job
    cfg = task_cfg_for(task_name)
    env = make_robosuite_env(cfg)
    out = []
    with h5py.File(hdf5, "r") as f:
        for name in names:
            demo = f["data"][name]
            states, actions = demo["states"][:], demo["actions"][:]
            env.reset()
            env.reset_from_xml_string(env.edit_model_xml(demo.attrs["model_file"]))
            env.sim.reset()
            rows, success_frame = [], None
            for t in range(states.shape[0]):
                env.sim.set_state_from_flattened(states[t])
                env.sim.forward()
                rows.append(obs_row(env, env._get_observations(force_update=True), cfg.obs_keys))
                if success_frame is None and t > 0 and env._check_success():
                    success_frame = t - 1  # the step from frame t-1 succeeded
            if success_frame is None:
                # the last frame's step: run it from the last state
                env.step(actions[-1])
                if env._check_success():
                    success_frame = states.shape[0] - 1
            out.append({"name": name, "rows": np.stack(rows), "actions": actions, "success_frame": success_frame,
                        "start": {"mujoco_state": states[0], "model_file": demo.attrs["model_file"]}})
    return out


def import_replayed(hdf5: str, out: str, task_name: str, max_demos: Optional[int] = None, workers: int = 8) -> Dict[str, Any]:
    import multiprocessing as mp

    import h5py

    cfg = task_cfg_for(task_name)
    source = Path(hdf5).expanduser()
    with h5py.File(source, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        if env_args["env_name"] != cfg.env_name:
            raise ValueError(f"{source} is {env_args['env_name']}; {task_name} is {cfg.env_name}")
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))[: max_demos]
    writer = _writer(out, task_name, cfg, f"{source.name} ({env_args['env_name']}, robosuite {env_args.get('env_version')}), "
                                          "observations replayed from its MuJoCo states")
    chunks = [names[i::workers] for i in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = {r["name"]: r for part in pool.map(_replay_worker, [(str(source), task_name, c) for c in chunks if c]) for r in part}
    lengths, success_frames, failed = [], [], []
    for name in names:
        r = results[name]
        if r["success_frame"] is None:
            failed.append(name)  # a generated demo the env does not count as successful: left out
            continue
        _save(writer, r["rows"], r["actions"], r["success_frame"], f"{source}:{name}", r["start"])
        lengths.append(r["actions"].shape[0])
        success_frames.append(r["success_frame"])
    summary = _summary(lengths, success_frames, out)
    summary["left_out_unsuccessful"] = failed
    return summary
