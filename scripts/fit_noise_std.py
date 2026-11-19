"""Fit the residual's noise-std MLP (``residual_noise.mlp``) on an experiment's raw demos: my_project/noise_fit.py.

    python scripts/fit_noise_std.py --config configs/experiments/can_compare/can_add.yaml --headless \\
        --out data/resfit_can/noise_std_residual.pt [--input residual|base]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

if __name__ == "__main__":
    from my_project.noise_fit import main
    from train import setup  # scripts/train.py's project registrations (sections, tasks, hooks)

    raise SystemExit(main(setup=setup))
