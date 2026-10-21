"""Roll out ResiP's released one_leg BC checkpoint from the initial states of ResiP's own demos.

    python scripts/eval_resip_from_demos.py --config configs/experiments/resip_replica/bc_resip_replica.yaml \
        --checkpoint ~/resip_ckpt/bc_one_leg_low.pt --zarr ~/resip_data/one_leg_teleop_low/success.zarr \
        --trials 2 --headless

Thin caller; the start-state setting and the rollout are in my_project/bc/resip_demo_starts.py.
The config gives the task and its overrides; scene.num_envs becomes demos x trials.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

# the same setup() hook as train_bc.py (every project registration)
_spec = importlib.util.spec_from_file_location("_train_bc_setup", Path(__file__).with_name("train_bc.py"))
_train_bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_bc)

if __name__ == "__main__":
    from my_project.bc.resip_demo_starts import main

    raise SystemExit(main(setup=_train_bc.setup))
