"""Compare a trained diffusion policy's chunks against its own training data, offline.

    python scripts/analyze_bc_chunks.py --run runs/hybrid_demo_pih/diffusion_bc/diffusion_bc_a0 \\
        --checkpoint actor_chkpt_best_success_rate.pt

No simulator. The run's resolved config gives the architecture, the dataset and the exact
train / test episode split (``bc.num_demos``, ``bc.test_split``, ``experiment.seed``); the
checkpoint gives the weights and normalizer. For random windows of the demos it samples chunks
(cold start, ``--samples`` per window) from the window's observation and compares them with
the demo's real next ``horizon`` actions:

* position error along the chunk (mm, at steps 0, ``action_horizon - 1`` and the last);
* motion quality, predicted vs. demo: median step size (mm), share of consecutive steps that
  reverse direction (cosine < -0.5), and net / path length over the executed part (1 = straight);
* sample spread: how far repeated samples from the same observation are apart (mm);
* an input-reliance test: how much the predicted executed part moves when one input group is
  swapped for another window's (same noise): the fingertip estimate, and the last 4 dims, which
  are the previous action when the run observed it (a copycat test) and otherwise the force and
  force threshold.

On the furniture absolute interface (``Isaac-Furniture-*-FBAbs-*``) the swap groups are the
EE position, the table top's pose and the leg's (``parts_poses`` slots 0 and 4), and the
position error is also reported over only the windows where the leg is within 12 cm of the
top (aligning, inserting, screwing: where the rollouts fail).

It also writes ``chunks_<split>.png``: x, y, z over the chunk for a few windows, the demo and
each sample. If the predictions zigzag here, on the training observations themselves, the model
never learned the demos' fine motion; if they are smooth here but not in rollouts, the problem
is in the closed loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def _motion(chunks: np.ndarray, executed: int) -> Dict[str, float]:
    """``chunks`` (N, H, 3) in metres -> step size, reversals, straightness of the executed part."""
    steps = np.diff(chunks, axis=1)
    size = np.linalg.norm(steps, axis=-1)
    unit = steps / np.maximum(size[..., None], 1e-12)
    cos = np.sum(unit[:, 1:] * unit[:, :-1], axis=-1)
    head = chunks[:, :executed]
    net = np.linalg.norm(head[:, -1] - head[:, 0], axis=-1)
    path = np.sum(np.linalg.norm(np.diff(head, axis=1), axis=-1), axis=-1)
    return {
        "step_mm": float(np.median(size) * 1000),
        "reversals": float(np.mean(cos < -0.5)),
        "net_over_path": float(np.median(net / np.maximum(path, 1e-12))),
    }


#: furniture FBAbs observation (61): 16 robot dims, then 5 parts x (pos 3, rot6d 6), the table
#: top first and one_leg's leg (square_table_leg4) last
FURNITURE_ABS_GROUPS: List[Tuple[str, slice]] = [
    ("ee_pos", slice(0, 3)),
    ("top_pose", slice(16, 25)),
    ("leg_pose", slice(52, 61)),
]
FURNITURE_NEAR_TOP_M = 0.12


def analyze(args) -> Dict[str, Dict[str, float]]:
    from omegaconf import OmegaConf

    from robonuke_rl_core.evaluation import resolved_config_path

    from my_project.bc.dataset import WindowedSplit, load_lerobot, split_episodes
    from my_project.models.cfg import actor_cfg_from
    from my_project.models.diffusion_policy import DiffusionPolicy

    run_dir = Path(args.run).expanduser()
    resolved = OmegaConf.load(resolved_config_path(run_dir))
    actor_cfg = actor_cfg_from(resolved.model)
    if args.inference_steps:
        actor_cfg.inference_steps = args.inference_steps
    device = torch.device(args.device)
    state = torch.load(run_dir / "checkpoints" / args.checkpoint, map_location=device, weights_only=False)
    policy = DiffusionPolicy(actor_cfg, state["obs_dim"], state["action_dim"]).to(device)
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()

    bc = resolved.bc
    demos = load_lerobot(args.dataset or bc.dataset, bc.obs_key, bc.action_key)
    train_eps, test_eps = split_episodes(demos.num_episodes, bc.num_demos, bc.test_split, int(resolved.experiment.seed))
    window = dict(n_obs_steps=policy.n_obs_steps, horizon=policy.horizon, action_horizon=policy.action_horizon, device=device)
    out_dir = Path(args.out).expanduser() if args.out else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    executed = policy.action_horizon
    report: Dict[str, Dict[str, float]] = {}
    from my_project.families import is_abs

    furniture_abs = is_abs(str(resolved.task.name))
    if furniture_abs and state["obs_dim"] != 61:
        raise ValueError(f"furniture FBAbs run with obs_dim {state['obs_dim']}, expected 61")

    for split, episodes in (("train", train_eps), ("test", test_eps)):
        data = WindowedSplit(demos, episodes, policy.normalizer, **window)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        index = torch.randperm(len(data), generator=generator)[: args.windows].to(device)
        batch = data.windows(index)
        nobs = batch["obs"]
        truth = policy.normalizer.denormalize(batch["action"], "action")[..., 0:3].cpu().numpy()

        samples = []
        for k in range(args.samples):
            torch.manual_seed(args.seed + k)
            samples.append(policy.predict_chunk(nobs)[..., 0:3].cpu().numpy())
        samples = np.stack(samples)  # (S, N, H, 3)

        # copycat test: same noise, one input group swapped for another window's
        def predict_with(obs):
            torch.manual_seed(args.seed)
            return policy.predict_chunk(obs)[..., 0:3].cpu().numpy()

        permuted = torch.roll(torch.arange(nobs.shape[0], device=device), 1)
        base = samples[0]
        effects = {}
        # the previous action is the last 4 observation dims only when the run observed it
        previous = bool(resolved.get("action", {}).get("world_pose", False)) and bool(
            resolved.get("action", {}).get("observe_previous_action", True)
        )
        if furniture_abs:
            groups = list(FURNITURE_ABS_GROUPS)
        else:
            groups = [("fingertip_estimate", slice(0, 3))]
            groups.append(("previous_action", slice(-4, None)) if previous else ("force_and_threshold", slice(-4, None)))
        for name, dims in groups:
            swapped = nobs.clone()
            swapped[..., dims] = nobs[permuted][..., dims]
            moved = predict_with(swapped)
            effects[name] = float(np.median(np.linalg.norm(moved[:, :executed] - base[:, :executed], axis=-1)) * 1000)

        per_window = np.linalg.norm(samples - truth[None], axis=-1).mean(axis=0) * 1000  # (N, H) mm
        error = per_window.mean(axis=0)  # per horizon step
        near = {}
        if furniture_abs:
            raw = policy.normalizer.denormalize(nobs[:, -1], "obs").cpu().numpy()
            close = np.linalg.norm(raw[:, 52:55] - raw[:, 16:19], axis=-1) < FURNITURE_NEAR_TOP_M
            near["near_top_windows"] = int(close.sum())
            if close.any():
                near["near_top_error_mm_step0"] = float(per_window[close, 0].mean())
                near[f"near_top_error_mm_step{executed - 1}"] = float(per_window[close, executed - 1].mean())
        spread = np.linalg.norm(samples - samples.mean(axis=0, keepdims=True), axis=-1).mean() * 1000
        predicted = _motion(samples.reshape(-1, *samples.shape[2:]), executed)
        demo = _motion(truth, executed)
        report[split] = {
            "windows": int(nobs.shape[0]),
            "error_mm_step0": float(error[0]),
            f"error_mm_step{executed - 1}": float(error[executed - 1]),
            f"error_mm_step{len(error) - 1}": float(error[-1]),
            "sample_spread_mm": float(spread),
            **{f"pred_{k}": v for k, v in predicted.items()},
            **{f"demo_{k}": v for k, v in demo.items()},
            **{f"swap_{k}_moves_mm": v for k, v in effects.items()},
            **near,
        }
        _plot(out_dir / f"chunks_{split}.png", truth, samples, executed, args.plots)

    (out_dir / "chunk_report.json").write_text(json.dumps(report, indent=2))
    _print(report, executed, out_dir)
    return report


def _plot(path: Path, truth: np.ndarray, samples: np.ndarray, executed: int, count: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(count, truth.shape[0])
    fig, axes = plt.subplots(count, 3, figsize=(12, 2.4 * count), squeeze=False)
    steps = np.arange(truth.shape[1])
    for row in range(count):
        for axis, label in enumerate("xyz"):
            ax = axes[row, axis]
            for s in range(samples.shape[0]):
                ax.plot(steps, samples[s, row, :, axis] * 1000, color="tab:red", alpha=0.6, lw=1, label="policy" if s == 0 else None)
            ax.plot(steps, truth[row, :, axis] * 1000, color="black", lw=2, label="demo")
            ax.axvline(executed - 0.5, color="gray", ls="--", lw=0.8)
            ax.set_title(f"window {row}: {label} (mm)", fontsize=9)
            if row == 0 and axis == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _print(report, executed: int, out_dir: Path) -> None:
    for split, r in report.items():
        print(f"== {split} ({r['windows']} windows)")
        print(f"  position error (mm): step 0 {r['error_mm_step0']:.2f}, step {executed - 1} "
              f"{r[f'error_mm_step{executed - 1}']:.2f}, last {[v for k, v in r.items() if k.startswith('error_mm_step')][-1]:.2f}")
        if "near_top_windows" in r:
            if r["near_top_windows"]:
                print(f"  leg within 12 cm of the top ({r['near_top_windows']} windows): error step 0 "
                      f"{r['near_top_error_mm_step0']:.2f}, step {executed - 1} {r[f'near_top_error_mm_step{executed - 1}']:.2f}")
            else:
                print("  leg within 12 cm of the top: no windows")
        print(f"  step size (mm)       policy {r['pred_step_mm']:.2f}   demo {r['demo_step_mm']:.2f}")
        print(f"  reversals            policy {r['pred_reversals']:.2f}   demo {r['demo_reversals']:.2f}")
        print(f"  net / path (exec.)   policy {r['pred_net_over_path']:.2f}   demo {r['demo_net_over_path']:.2f}")
        print(f"  sample spread (mm)   {r['sample_spread_mm']:.2f}")
        swaps = "; ".join(
            f"swap {k[len('swap_'):-len('_moves_mm')].replace('_', ' ')} -> {v:.2f} mm"
            for k, v in r.items() if k.startswith("swap_")
        )
        print(f"  executed part moves when one input is swapped with another window's: {swaps}")
    print(f"plots and chunk_report.json in {out_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="the BC run directory (holds checkpoints/)")
    parser.add_argument("--checkpoint", default="actor_chkpt_best_success_rate.pt")
    parser.add_argument("--dataset", default=None, help="override the run's bc.dataset")
    parser.add_argument("--windows", type=int, default=256, help="windows per split")
    parser.add_argument("--samples", type=int, default=3, help="chunks sampled per window")
    parser.add_argument("--plots", type=int, default=4, help="windows drawn per split")
    parser.add_argument("--inference_steps", type=int, default=0, help="override DDIM steps (0: the run's)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None, help="output directory (default <run>/analysis)")
    analyze(parser.parse_args(argv))
    return 0
