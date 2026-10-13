"""FurnitureBench-format demos -> the recorder's dataset (no Isaac Sim needed).

    python scripts/import_furniture_demos.py --src ~/Robonuke_Furniture_Bench/demos --out data/furniture_one_leg
    python scripts/import_furniture_demos.py --src ~/fb_demos --out data/furniture_one_leg_all --include_failures

Reads ``<src>/<task>/<stamp>/<stamp>.pkl`` (and ``<src>/<task>/fails/*.pkl`` with
``--include_failures``) as ``scripts/collect_scripted.py`` writes them, and appends each one
not already in the dataset. Columns and layout: ``my_project/demos/furniture.py``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

from my_project.demos.furniture import import_demos  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="the collector's --out_dir (holds <task>/<stamp>/<stamp>.pkl)")
    parser.add_argument("--out", required=True, help="dataset directory to create or append to")
    parser.add_argument("--include_failures", action="store_true", help="also import <task>/fails/*.pkl (success=false)")
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--task_name", default="Isaac-Furniture-OneLeg-v0")
    args = parser.parse_args(argv)
    result = import_demos(Path(args.src), Path(args.out), args.include_failures, args.task_name, args.max_demos)
    print(
        f"[import] {result['root']}: +{result['saved']} episodes ({result['skipped']} already there); "
        f"now {result['episodes']} episodes, {result['frames']} frames",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
