"""Submit one SLURM job per training config, running `scripts/watch_residual_eval.py` (the
`hpc.watch_script` of configs/base/hpc.yaml) over that training's run directory. Thin caller;
the logic lives in the package.

    python launchers/launch_watch.py configs/experiments/scaled_ablation/ \\
        --project one_leg_scaled_ablations --group_prefix abs \\
        --eval_config configs/eval/residual_watch_ablation.yaml [--follow] [--dry_run]

The same --project and --group_prefix as the launch_train that started the runs: the run
directory is runs/<project>/<group_prefix>_<config stem>. By default the watcher evaluates every
checkpoint on disk and exits (--once); --follow keeps it polling beside a live training.
"""

if __name__ == "__main__":
    from robonuke_rl_core.hpc.launch_watch import main

    raise SystemExit(main())
