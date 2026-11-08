"""Raw demo episodes for the residual, offline, on the CPU, no sim (``scripts/make_raw_demos.py``).

The file holds only what is general about the demos -- nothing the residual's own settings shape (its observation,
action map, routing, reward weights, normalization): those are applied when the learner loads the file, with the
env's own code (``my_project/demo_loader.py``, the package's ``register_demo_loader``). Per recorded demo episode (the
base run's ``bc.dataset``: LeRobot parquet, ``observation.state_abs`` and ``action.abs``; the env's own per-step data
from the episode's source pkl, ``scripts/collect_scripted.py`` in Robonuke_Furniture_Bench), frames
``0..success_frame`` (the step after which the training env's ``success`` termination holds, :func:`success_frame`):

* ``states`` ``(T, 63)``: the raw env observations;
* ``actions`` ``(T, 10)``: the absolute targets the env executed (:func:`executed_abs_actions`: a DART collection's
  perturbed commands during its bursts);
* ``base_actions`` ``(T, 10)``: the frozen base's action at each frame, as the env runs it -- a plan (its own
  observation history, its DDIM steps, cold) at frames ``0, H, 2H, ...`` (``H`` its executed ``action_horizon``), each
  frame taking the plan's next action. The base is stochastic, so its sample is stored, not recomputed;
* ``infos`` ``{path: (T, ...)}``: the env's infos published with each frame's observation (frame 0 the collection's
  settle step, frame ``t`` the infos of step ``t - 1``), flattened to ``"<key>/<name>"`` paths (``eval_state/phase/id``,
  the phase tag), numeric values only (:func:`frame_infos`);
* ``reward_terms`` ``{name: (T,)}``: every env reward term's own value at step ``t`` (unweighted: the loader applies
  the training config's weights);
* ``termination_terms`` ``{name: (T,)}`` bool: every training-env termination term at step ``t``.

``meta["fingerprint"]`` (:func:`raw_fingerprint`) is what the stored data depends on: the dataset, the base run and
checkpoint, its planning settings, the task. The loader refuses a file whose fingerprint differs from the run's.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch

__all__ = ["success_frame", "frame_infos", "executed_abs_actions", "raw_fingerprint", "base_plan_actions", "RAW_FORMAT",
           "ENV_DATA", "main"]

#: ``data["format"]`` of a raw demo file
RAW_FORMAT = "raw_episodes_v2"
#: the env's own per-step data a source pkl must carry (``collect_scripted.py``)
ENV_DATA = ("infos", "infos_initial", "reward_terms", "termination_terms")


def success_frame(data: Dict[str, Any]) -> int:
    """The step after which the training env's ``success`` termination first holds (the pkl's ``termination_terms``):
    the demo's last transition; nothing after it is used."""
    import numpy as np

    hits = np.nonzero(np.asarray(data["termination_terms"]["success"], dtype=bool))[0]
    if hits.size == 0:
        raise ValueError("the episode never reaches the env's success termination")
    return int(hits[0])


def _flatten(value: Any, prefix: str, out: Dict[str, Any]) -> None:
    import numpy as np

    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten(sub, f"{prefix}/{key}" if prefix else str(key), out)
        return
    array = np.asarray(value) if isinstance(value, (bool, int, float, np.ndarray, np.generic)) else None
    if array is not None and array.dtype.kind in "biuf":
        out[prefix] = array


def frame_infos(data: Dict[str, Any], T: int) -> Dict[str, torch.Tensor]:
    """The infos published with each of the first ``T`` frames' observations (``infos_initial``, then step ``t - 1``'s),
    flattened to ``"<key>/<name>"`` paths, ``{path: (T, ...)}``: the numeric values every frame has with one shape."""
    import numpy as np

    frames = [data["infos_initial"]] + list(data["infos"][: T - 1])
    flat = []
    for frame in frames:
        out: Dict[str, Any] = {}
        _flatten(frame, "", out)
        flat.append(out)
    common = set(flat[0])
    for out in flat[1:]:
        common &= set(out)
    stacked = {}
    for path in sorted(common):
        values = [out[path] for out in flat]
        if len({v.shape for v in values}) == 1:
            stacked[path] = torch.as_tensor(np.stack(values))
    return stacked


def executed_abs_actions(source: Path, states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The ``action.abs`` actions the env was given, ``(T, 10)``, from the episode's source pkl. A DART collection
    sent perturbed commands during its bursts (``actions_executed``) while the dataset keeps the expert's own (the BC
    labels); the next observations come from the executed ones, so a transition must carry those. Converted with the
    dataset import's own ``fb_abs_frames``; the pkl's labels and states must be the dataset's (``labels``, ``states``),
    so an episode that does not line up is refused. A pkl with no ``actions_executed`` gives its labels."""
    import numpy as np

    from furniture_lab.demos import fb_abs_frames

    from my_project.demos.furniture import load_pkl

    data = load_pkl(Path(source))
    obstacle = states.shape[1] == 63  # the ResiP layout (robot-base parts, the front obstacle)
    T = labels.shape[0]
    states_pkl, labels_pkl = fb_abs_frames(data, obstacle=obstacle)
    if labels_pkl.shape[0] != T or not np.allclose(labels_pkl, labels.numpy(), atol=1e-5) or not np.allclose(
        states_pkl[:T], states.numpy(), atol=1e-5
    ):
        raise ValueError(f"{source}: its actions / states are not the dataset episode's; rebuild the dataset or check episodes.jsonl")
    _, executed = fb_abs_frames(data, obstacle=obstacle, executed=True)
    return torch.tensor(np.asarray(executed), dtype=torch.float32)


def raw_fingerprint(rc: Any, policy, dataset: Path, task_name: str, entity: Any = None) -> Dict[str, Any]:
    """What a raw demo file's data depends on, as plain data: the dataset, the base run and checkpoint, the base's
    effective planning settings (``policy`` as :func:`~my_project.residual.override_base` left it) and the task. One
    definition for the builder and the loader's check."""
    return {
        "dataset": str(Path(dataset).expanduser().resolve()),
        "base_run": str(rc.base_run) if rc.base_run else f"{rc.base_entity or entity}/{rc.base_project}/{rc.base_run_name}",
        "base_checkpoint": str(rc.base_checkpoint),
        "base_action_horizon": int(policy.action_horizon),
        "base_inference_steps": int(policy.cfg.inference_steps),
        "base_warm_start": bool(policy.cfg.warm_start),
        "base_n_obs_steps": int(policy.n_obs_steps),
        "task_name": str(task_name),
    }


def base_plan_actions(policy, states: torch.Tensor) -> torch.Tensor:
    """The base's action at each frame ``(T, 10)``, as the env runs it: a cold plan at frames ``0, H, 2H, ...`` from its
    own observation history (the first frame repeated before the episode), each frame taking the plan's next action."""
    T = states.shape[0]
    horizon = int(policy.action_horizon)
    norm = policy.normalizer
    plan_starts = list(range(0, T, horizon))
    history = torch.stack([states[[max(t - j, 0) for j in reversed(range(policy.n_obs_steps))]] for t in plan_starts])
    with torch.no_grad():
        nobs = norm.normalize(history, "obs").clamp(-policy.cfg.obs_clip, policy.cfg.obs_clip)
        plans = norm.denormalize(policy._denoise(nobs, None), "action")
    return torch.stack([plans[t // horizon, t % horizon] for t in range(T)])


def _config(path: str):
    """The experiment's residual section under its defaults, its seed and task name -- plain YAML, no sim."""
    from omegaconf import OmegaConf

    from robonuke_rl_core.config import load_file_chain

    from my_project.cfg import ResidualCfg

    layers = [layer for _, layer in load_file_chain(Path(path))]
    merged = OmegaConf.merge(*layers)
    rc = OmegaConf.merge(OmegaConf.structured(ResidualCfg), merged.get("residual", {}))
    entity = OmegaConf.select(merged, "wandb.entity")
    if rc.base_entity is None and entity is not None:
        rc.base_entity = entity
    return SimpleNamespace(residual=rc, wandb=SimpleNamespace(entity=entity),
                           seed=int(OmegaConf.select(merged, "experiment.seed", default=0)),
                           task_name=str(OmegaConf.select(merged, "task.name")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the residual experiment (its base run and base settings)")
    parser.add_argument("--dataset", default=None, help="the demo dataset (default: the base run's bc.dataset)")
    parser.add_argument("--out", required=True, help="the raw demo file to write (the experiment's demo_transitions)")
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.residual import base_run_dir, load_base_policy, override_base

    cfg = _config(args.config)
    rc = cfg.residual
    base_dir = base_run_dir(cfg)
    dataset = Path(args.dataset).expanduser() if args.dataset else Path(OmegaConf.load(resolved_config_path(base_dir)).bc.dataset).expanduser()
    policy = load_base_policy(base_dir, rc.base_checkpoint, "cpu")
    override_base(policy, SimpleNamespace(base_action_horizon=rc.base_action_horizon, base_inference_steps=rc.base_inference_steps,
                                          steps_per_chunk=rc.steps_per_chunk, base_warm_start=rc.get("base_warm_start")))
    if policy.cfg.warm_start:
        raise ValueError("the raw demos plan the base cold (every chunk from pure noise, at the env's replan frames); the env's "
                         "base warm-starts each plan from the last one's tail: set residual.base_warm_start: false")
    torch.manual_seed(cfg.seed)
    sources = {}
    with open(dataset / "meta" / "episodes.jsonl") as f:
        for line in f:
            episode = json.loads(line)
            sources[str((dataset / episode["file"]).resolve())] = episode.get("source")
    from my_project.demos.furniture import load_pkl

    episodes, skipped, dart_rows = [], [], 0
    for path in sorted(glob.glob(str(dataset / "data" / "**" / "*.parquet"), recursive=True)):
        frame = pd.read_parquet(path)
        states = torch.tensor(np.stack(frame["observation.state_abs"].values), dtype=torch.float32)
        labels = torch.tensor(np.stack(frame["action.abs"].values), dtype=torch.float32)
        source = sources.get(str(Path(path).resolve()))
        if not source:
            raise ValueError(f"{path}: meta/episodes.jsonl names no source pkl for it; the env's data and the executed "
                             "actions are there")
        data = load_pkl(Path(source))
        missing = [k for k in ENV_DATA if k not in data]
        if missing:
            raise ValueError(f"{source} carries no {missing}: collect the demos with Robonuke_Furniture_Bench's current "
                             "scripts/collect_scripted.py (it records the env's infos, reward and termination terms)")
        if not any(data["termination_terms"]["success"]):
            skipped.append((Path(path).name, "no success"))
            continue
        actions = executed_abs_actions(Path(source), states, labels)
        last = success_frame(data)
        T = last + 1
        dart_rows += int((actions[:T] - labels[:T]).abs().amax(dim=-1).gt(1e-6).sum())
        episodes.append({
            "states": states[:T], "actions": actions[:T], "base_actions": base_plan_actions(policy, states[:T]),
            "success_frame": int(last), "file": Path(path).name, "infos": frame_infos(data, T),
            "reward_terms": {k: torch.tensor(v[:T], dtype=torch.float32) for k, v in data["reward_terms"].items()},
            "termination_terms": {k: torch.tensor(v[:T], dtype=torch.bool) for k, v in data["termination_terms"].items()},
        })
    meta = {"fingerprint": raw_fingerprint(rc, policy, dataset, cfg.task_name, cfg.wandb.entity), "episodes": len(episodes), "skipped": skipped,
            "dart_rows": dart_rows, "seed": cfg.seed}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": RAW_FORMAT, "episodes": episodes, "meta": meta}, out)
    frames = sum(int(e["states"].shape[0]) for e in episodes)
    print(f"[raw demos] {len(episodes)} episodes, {frames} frames ({dart_rows} with a DART-perturbed executed action), "
          f"skipped {skipped} -> {out}")
    print(f"[raw demos] fingerprint {meta['fingerprint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
