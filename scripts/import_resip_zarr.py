"""ResiP's processed one_leg demos (robust-rearrangement ``.zarr``) -> the recorder's dataset,
absolute rot6d interface (no Isaac Sim needed).

    aws s3 sync --no-sign-request \\
        s3://iai-robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/low/ ~/resip_data/one_leg_teleop_low/
    python scripts/import_resip_zarr.py --zarr ~/resip_data/one_leg_teleop_low/<name>.zarr \\
        --out data/furniture_one_leg_resip_teleop

Columns and conventions: ``my_project/demos/resip_zarr.py``. Needs ``pip install zarr``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

from my_project.demos.resip_zarr import import_resip_zarr  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zarr", required=True, help="the processed .zarr directory")
    parser.add_argument("--out", required=True, help="dataset directory to create or append to")
    parser.add_argument("--task_name", default="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0")
    parser.add_argument("--control_hz", type=float, default=10.0, help="ResiP's FurnitureRLSimEnv rate")
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--include_failures", action="store_true", help="also import episodes with success=0")
    args = parser.parse_args(argv)
    r = import_resip_zarr(
        Path(args.zarr), Path(args.out), args.task_name, args.control_hz, args.max_demos,
        successes_only=not args.include_failures,
    )
    print(
        f"[import] {r['root']}: +{r['saved']} episodes ({r['skipped']} already there, {r['failures_dropped']} "
        f"failures dropped); now {r['episodes']} episodes, {r['frames']} frames",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
