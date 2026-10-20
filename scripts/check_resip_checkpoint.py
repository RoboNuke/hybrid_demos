"""Check ResiP's released BC checkpoint loads correctly, offline (no sim): run it on its own
training demos and compare its chunks with the recorded actions.

    python scripts/check_resip_checkpoint.py --checkpoint ~/resip_ckpt/bc_one_leg_low.pt \\
        --zarr ~/resip_data/one_leg_teleop_low/success.zarr

Thin caller; the check is ``my_project/bc/resip_offline_check.py``. Needs ``pip install zarr``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

from my_project.bc.resip_offline_check import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
