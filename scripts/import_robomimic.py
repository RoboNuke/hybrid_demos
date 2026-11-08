"""robomimic-format demo hdf5s -> the recorder's dataset, for ResFiT's robosuite tasks (no Isaac Sim).

    curl -L -o data/robomimic_can_mh/low_dim_v15.hdf5 \\
        https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/can/mh/low_dim_v15.hdf5
    python scripts/import_robomimic.py --hdf5 data/robomimic_can_mh/low_dim_v15.hdf5 --out data/robomimic_can_mh_300
    curl -L -o data/dexmimicgen/two_arm_coffee.hdf5 \
        https://huggingface.co/datasets/MimicGen/dexmimicgen_datasets/resolve/main/generated/two_arm_coffee.hdf5
    python scripts/import_robomimic.py --hdf5 data/dexmimicgen/two_arm_coffee.hdf5 --out data/dexmg_coffee_1000 \
        --task_name Robosuite-Coffee-LowDim-v0 --replay --max_demos 1000

Columns and conventions: ``my_project/demos/robomimic_hdf5.py``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

from my_project.demos.robomimic_hdf5 import import_replayed, import_robomimic  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", required=True, help="the robomimic low_dim hdf5")
    parser.add_argument("--out", required=True, help="dataset directory to create")
    parser.add_argument("--task_name", default="Robosuite-Can-LowDim-v0")
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--replay", action="store_true",
                        help="read the observations by replaying the recorded MuJoCo states (DexMimicGen files)")
    parser.add_argument("--workers", type=int, default=8, help="--replay's worker processes")
    args = parser.parse_args(argv)
    if args.replay:
        print(import_replayed(args.hdf5, args.out, args.task_name, args.max_demos, args.workers))
    else:
        print(import_robomimic(args.hdf5, args.out, args.task_name, args.max_demos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
