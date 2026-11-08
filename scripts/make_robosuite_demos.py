"""A robosuite task's recorded demos as one whole-task residual's demo transitions, offline, no sim stepped:
delegates to :func:`my_project.robosuite_demos.main`, with make_demo_transitions.py's ``setup`` (the full config,
for the file's fingerprint).

    python scripts/make_robosuite_demos.py --config configs/experiments/resfit_can/residual_resfit_can.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

if __name__ == "__main__":
    from make_demo_transitions import setup

    from my_project.robosuite_demos import main

    raise SystemExit(main(setup=setup))
