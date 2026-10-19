"""Roll out ResiP's released one_leg BC checkpoint in this port's ResiP task.

    aws s3 cp --no-sign-request \\
        s3://iai-robust-rearrangement/checkpoints/bc/one_leg/low/actor_chkpt.pt ~/resip_ckpt/bc_one_leg_low.pt
    python scripts/eval_resip_checkpoint.py --config configs/experiments/pivot_demos.yaml \\
        --checkpoint ~/resip_ckpt/bc_one_leg_low.pt --headless

Thin caller; the loader, the observation adapter and the rollout are in
``my_project/bc/resip_checkpoint.py``. The config gives the task, the env count, the episode
length and ``eval.num_rollouts``; its ``model`` / ``bc`` sections are not used.
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
    from my_project.bc.resip_checkpoint import main

    raise SystemExit(main(setup=_train_bc.setup))
