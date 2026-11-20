"""Fit ``residual_noise.mlp``'s std MLP on a residual experiment's demos (``scripts/fit_noise_std.py``).

    python scripts/fit_noise_std.py --config configs/experiments/can_compare/can_add.yaml --headless \\
        --dataset base --out data/resfit_can/noise_std_residual.pt [--input residual|base] [--phase insert]

``--rows_out <file>`` in place of ``--out`` writes the rows the fit would take (each frame's input, target and base
action) and stops before fitting.

The demos: ``--dataset <dir>`` (or ``base``, the base run's own ``bc.dataset``) builds the raw demo episodes from a demo
dataset in-process, the base's action at every frame planned as the env runs it (``my_project/raw_demos.py:
build_raw_demos``, what ``scripts/make_raw_demos.py`` writes; ``--save_raw`` keeps them); ``--demos <file>`` reads a raw
demo file already made; neither: the learner's ``demo_transitions``.

Logged to wandb (``--wandb_project``, default ``var_mlps``; the config's ``wandb.entity``; group the config's stem, the
run named ``<stem>_<input>[_<phase>]`` unless ``--name``), one point per epoch: ``train/nll``, ``train/grad_norm``,
``train/lr``, ``val/nll``, ``val/nll_constant`` (the best constant std's), ``val/nll_gain`` (constant - MLP: > 0 the input
helps), ``val/calibration`` (``E[c^2 / s^2]``: 1 when the predicted variance matches), ``val/pred_std_mean``, and each of
the last four per dim under ``val_dims/``. The summary holds the best epoch's report and the linear reference, a table
per dim, and the fitted file is uploaded as a plain run file.

The run's env is built (with its residual wrappers, as training builds it) only to run the demo frames through exactly
the conversions the learner's demo loader uses (``my_project/wrappers/residual_policy/demos.py: convert_episodes``):

* input: each frame's agent observation (``--input residual``, what the learner acts on) or the observation the base
  policy was given (``--input base``, the recorded env observation);
* target: the demo's correction in ``residual_noise``'s units (``noise_target``: the executed action minus the base's for
  the additive residual, the sent action's units; ``r - 1`` for the scaled one, r units).

The MLP is fitted by the zero-mean Gaussian NLL (``noise_scale.fit_std_mlp``), so its ``s(x)^2`` estimates
``E[c^2 | x]``. The report prints per dim the validation NLL against the best constant std, the targets' RMS and the
mean predicted std, and a linear reference (``a1 * |base| + a0`` fitted by the same NLL) for ``mode: linear``. The file
records the input, the residual, the base's fingerprint (checked when a run loads it) and this report.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None, setup=None) -> int:
    from robonuke_rl_core.config import add_config_args, load_from_args, simulator_of, start_simulator

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--out", default=None, help="the fitted file to write (residual_noise.mlp.path)")
    parser.add_argument("--rows_out", default=None,
                        help="write the fit's rows (each demo frame's input, target and base action, exactly as the fit "
                             "takes them) to this file and stop, no fit (the demo residual analysis notebook's MLP line)")
    parser.add_argument("--input", choices=("residual", "base"), default="residual",
                        help="the MLP's input: the agent's observation (residual) or the base policy's (base)")
    parser.add_argument("--dataset", default=None,
                        help="a demo dataset (LeRobot dir) to build the raw demos from, or 'base' for the base run's bc.dataset")
    parser.add_argument("--save_raw", default=None, help="with --dataset: also write the raw demo file built here")
    parser.add_argument("--demos", default=None, help="a raw demo file (default: the learner's demo_transitions)")
    parser.add_argument("--phase", default=None, help="fit on this phase's frames only (the learner's std_label_names)")
    parser.add_argument("--hidden", default="256,256", help="hidden widths, comma-separated")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20, help="epochs without a better validation NLL before stopping")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--floor", type=float, default=1e-4, help="the smallest std the MLP outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit_device", default=None, help="the fit's device (default: cuda when available)")
    parser.add_argument("--wandb_project", default="var_mlps")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--name", default=None, help="the wandb run's name (default: <config stem>_<input>[_<phase>])")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)
    app = start_simulator(args, simulator_of(args, overrides))
    if setup is not None:
        setup()

    import gymnasium as gym
    import torch

    from robonuke_rl_core.envs.build import build_env, prepare_task

    from my_project.cfg import q_section
    from my_project.raw_demos import RAW_FORMAT
    from my_project.wrappers.residual_policy import noise_scale as ns
    from my_project.wrappers.residual_policy.combine import find_combine
    from my_project.wrappers.residual_policy.demos import _cpu, convert_episodes

    cfg = load_from_args(args, overrides)
    if not cfg.base_policy.enabled:
        raise ValueError("fit_noise_std fits a residual's noise: the config needs base_policy.enabled")
    sec = q_section(cfg)
    if args.dataset and args.demos:
        raise ValueError("pass --dataset (build the raw demos here) or --demos (a raw demo file), not both")
    if (args.out is None) == (args.rows_out is None):
        raise ValueError("pass --out (fit and write the MLP) or --rows_out (write the fit's rows only), one of them")
    if args.save_raw and not args.dataset:
        raise ValueError("--save_raw writes the raw demos --dataset builds: pass --dataset")
    device = args.fit_device or ("cuda" if torch.cuda.is_available() else "cpu")

    prepare_task(cfg, cfg.task_name, cfg.task_cfg)
    env = build_env(cfg, gym.make(cfg.task_name, cfg=cfg.task_cfg), cfg.task_name)
    combine = find_combine(env)
    if args.dataset:
        from types import SimpleNamespace

        from my_project.raw_demos import build_raw_demos

        raw_cfg = SimpleNamespace(base_policy=cfg.base_policy, wandb=SimpleNamespace(entity=cfg.wandb.entity),
                                  seed=int(cfg.experiment.seed), task_name=cfg.task_name)
        data = build_raw_demos(raw_cfg, None if args.dataset == "base" else args.dataset, device=device)
        demos = data["meta"]["fingerprint"]["dataset"]
        print(f"[fit_noise_std] built {data['meta']['episodes']} raw demo episodes from {demos} "
              f"(skipped {data['meta']['skipped']})", flush=True)
        if args.save_raw:
            Path(args.save_raw).parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, args.save_raw)
            print(f"[fit_noise_std] wrote the raw demos to {args.save_raw}", flush=True)
    else:
        demos = args.demos or getattr(sec, "demo_transitions", None)
        if not demos:
            raise ValueError(f"no demos: pass --dataset, --demos or set {cfg.trainer.learner}.demo_transitions")
        data = torch.load(demos, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or data.get("format") != RAW_FORMAT:
        raise ValueError(f"{demos} is not a raw demo file ({RAW_FORMAT}): build it with scripts/make_raw_demos.py")
    made = (data.get("meta") or {}).get("fingerprint") or {}
    ns.check_raw_fingerprint(made, ns.base_fingerprint(cfg, combine), f"the demo file {demos}",
                             "rebuild it with scripts/make_raw_demos.py")

    episodes = [_cpu(e) for e in data["episodes"]]
    names = list(getattr(sec, "std_label_names", None) or [])
    needs_phase = args.phase is not None or combine.uses_phase_rows()
    phases = [None] * len(episodes)
    if needs_phase:
        label_path = sec.std_by_label
        if args.phase is not None and args.phase not in names:
            raise ValueError(f"--phase {args.phase!r} is not one of the learner's std_label_names {names}")
        for i, e in enumerate(episodes):
            infos = e.get("infos") or {}
            if not label_path or label_path not in infos:
                raise ValueError(f"the demos record no phase channel {label_path!r} (the learner's std_by_label)")
            phases[i] = infos[label_path].reshape(-1).round().long()
    steps = convert_episodes(combine, episodes, phases, combine.device)

    xs, ys, bs = [], [], []
    width = combine.action_dim
    for e, step, phase in zip(episodes, steps, phases):
        x = step["policy"] if args.input == "residual" else e["states"].float()
        # convert_episodes returns CPU tensors; the residual's scale lives on the env's device
        y = combine.noise_target(step["r"].to(combine.device), step["side"].to(combine.device)).cpu()
        b = step["side"][..., :width]
        if args.phase is not None:
            keep = phase == names.index(args.phase)
            x, y, b = x[keep], y[keep], b[keep]
        xs.append(x)
        ys.append(y)
        bs.append(b)
    inputs, targets, base = torch.cat(xs).float(), torch.cat(ys).float(), torch.cat(bs).float()
    if inputs.shape[0] == 0:
        raise ValueError(f"no demo frame in phase {args.phase!r}")
    units = "sent" if combine.noise_kind == "additive" else "r"
    print(f"[fit_noise_std] {inputs.shape[0]} frames from {len(episodes)} episodes"
          + (f" (phase {args.phase})" if args.phase else "") + f"; input {args.input} ({inputs.shape[1]} columns), "
          f"target the {combine.noise_kind} residual's correction ({width} dims, {units} units)", flush=True)

    if args.rows_out:
        Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"inputs": inputs, "targets": targets, "base": base, "input": args.input,
                    "residual": combine.noise_kind, "units": units, "phase": args.phase,
                    "demos": str(Path(demos).resolve()), "config": str(args.config or args.from_run)}, args.rows_out)
        print(f"[fit_noise_std] wrote the fit's {inputs.shape[0]} rows to {args.rows_out} (no fit)", flush=True)
        env.close()
        if app is not None:
            app.app.close()
        return 0

    hidden = [int(h) for h in str(args.hidden).split(",") if h.strip()]
    source = {"demos": str(Path(demos).resolve()), "built_from_dataset": bool(args.dataset), "phase": args.phase,
              "frames": int(inputs.shape[0]), "episodes": len(episodes), "hidden": hidden, "seed": args.seed,
              "units": units, "config": str(args.config or args.from_run)}
    run = _start_wandb(args, cfg, combine, source, inputs.shape[1], width)
    out = ns.fit_std_mlp(inputs, targets, hidden=hidden, floor=args.floor, epochs=args.epochs, lr=args.lr,
                         batch_size=args.batch_size, weight_decay=args.weight_decay, val_fraction=args.val_fraction,
                         patience=args.patience, seed=args.seed, device=device,
                         on_epoch=(lambda epoch, metrics: run.log(metrics, step=epoch)) if run is not None else None)
    report = out["report"]
    report["linear_reference"] = ns.fit_linear(base, targets, device=device)
    report["slice_linear"] = ns.slice_linear_fit(base.numpy(), targets.numpy())
    _print_report(report, units)
    _print_slice_fit(report["slice_linear"], units)
    ns.save_std_mlp(args.out, out["model"], input=args.input, residual=combine.noise_kind,
                    fingerprint=ns.base_fingerprint(cfg, combine), report=report, source=source)
    print(f"[fit_noise_std] wrote {args.out}: set residual_noise.mode: mlp, residual_noise.mlp.path: {args.out}, "
          f"residual_noise.mlp.input: {args.input}", flush=True)
    if run is not None:
        _finish_wandb(run, report, args.out)
    env.close()
    if app is not None:
        app.app.close()
    return 0


def _start_wandb(args, cfg, combine, source, in_dim: int, width: int):
    """The fit's wandb run in ``--wandb_project`` (None with ``--wandb_mode disabled``)."""
    if args.wandb_mode == "disabled":
        return None
    import wandb

    stem = Path(str(args.config or args.from_run)).stem
    name = args.name or "_".join([stem, args.input] + ([args.phase] if args.phase else []))
    config = {"input": args.input, "residual": combine.noise_kind, "in_dim": in_dim, "action_dim": width,
              "epochs": args.epochs, "patience": args.patience, "lr": args.lr, "weight_decay": args.weight_decay,
              "batch_size": args.batch_size, "val_fraction": args.val_fraction, "floor": args.floor, **source,
              "base_policy": {k: getattr(cfg.base_policy, k) for k in ("base_run", "base_project", "base_run_name",
                                                                        "base_checkpoint")}}
    run = wandb.init(project=args.wandb_project, entity=cfg.wandb.entity, group=stem, name=name, mode=args.wandb_mode,
                     config=config, reinit="create_new", job_type="noise_std_mlp")
    run.define_metric("epoch")
    print(f"[fit_noise_std] wandb: {args.wandb_project}/{name} ({args.wandb_mode})", flush=True)
    return run


def _finish_wandb(run, report, out_path) -> None:
    """The best epoch's report in the summary, a per-dim table, and the fitted file as a plain run file."""
    import wandb

    lin = report["linear_reference"]
    run.summary.update({"best/val_nll": report["best_val_nll"], "best/epoch": report["best_epoch"],
                        "epochs_run": report["epochs"],
                        "best/val_nll_constant": sum(report["val_nll_constant_per_dim"]) / len(report["val_nll_constant_per_dim"]),
                        "linear/val_nll": sum(lin["nll_per_dim"]) / len(lin["nll_per_dim"])})
    rows = [[d, report["val_nll_per_dim"][d], report["val_nll_constant_per_dim"][d], lin["nll_per_dim"][d],
             report["target_rms_per_dim"][d], report["predicted_std_mean_per_dim"][d],
             report["predicted_rms_std_per_dim"][d], lin["a1"][d], lin["a0"][d]]
            for d in range(len(report["val_nll_per_dim"]))]
    sl = report.get("slice_linear")
    if sl is not None:
        rows = [row + [sl["a1"][d], sl["a0"][d], sl["r2"][d]] for d, row in enumerate(rows)]
    run.log({"report/per_dim": wandb.Table(columns=["dim", "nll_mlp", "nll_constant", "nll_linear", "target_rms",
                                                    "pred_std_mean", "pred_rms_std", "linear_a1", "linear_a0"]
                                           + (["slice_a1", "slice_a0", "slice_r2"] if sl is not None else []), data=rows)})
    run.save(str(Path(out_path).resolve()), base_path=str(Path(out_path).resolve().parent), policy="now")
    run.finish()


def _print_slice_fit(fit, units: str) -> None:
    """The notebook's slice fit per dim, then ready to paste as ``residual_noise.linear``; a negative or missing
    coefficient flagged (the linear mode takes numbers >= 0)."""
    print(f"[fit_noise_std] the notebook's slice fit (std of the residual per slice of |base|, weighted), {units} units:",
          flush=True)
    print(f"  {'dim':>3} {'a1':>10} {'a0':>10} {'R2':>7} {'slices':>6}", flush=True)
    bad = []
    for d, (a1, a0, r2, n) in enumerate(zip(fit["a1"], fit["a0"], fit["r2"], fit["slices"])):
        flag = "  <-- NEGATIVE / MISSING" if not (a1 >= 0 and a0 >= 0) else ""
        if flag:
            bad.append(d)
        print(f"  {d:>3} {a1:>10.6f} {a0:>10.6f} {r2:>7.3f} {n:>6}{flag}", flush=True)
    print("residual_noise:\n  linear:\n    a1: [" + ", ".join(f"{v:.6g}" for v in fit["a1"]) + "]\n    a0: ["
          + ", ".join(f"{v:.6g}" for v in fit["a0"]) + "]", flush=True)
    if bad:
        print(f"[fit_noise_std] WARNING: dims {bad} have a negative or missing slice-fit coefficient", flush=True)


def _print_report(report, units: str) -> None:
    lin = report["linear_reference"]
    print(f"[fit_noise_std] {report['epochs']} epochs, best validation NLL {report['best_val_nll']:.5f} "
          f"({report['rows']['train']} train / {report['rows']['val']} validation rows); per dim ({units} units):", flush=True)
    print(f"  {'dim':>3} {'NLL mlp':>9} {'NLL const':>9} {'NLL lin':>9} {'rms c':>9} {'mean s':>9} {'rms s':>9} "
          f"{'lin a1':>9} {'lin a0':>9}", flush=True)
    for d in range(len(report["val_nll_per_dim"])):
        print(f"  {d:>3} {report['val_nll_per_dim'][d]:>9.4f} {report['val_nll_constant_per_dim'][d]:>9.4f} "
              f"{lin['nll_per_dim'][d]:>9.4f} {report['target_rms_per_dim'][d]:>9.5f} "
              f"{report['predicted_std_mean_per_dim'][d]:>9.5f} {report['predicted_rms_std_per_dim'][d]:>9.5f} "
              f"{lin['a1'][d]:>9.5f} {lin['a0'][d]:>9.5f}", flush=True)
