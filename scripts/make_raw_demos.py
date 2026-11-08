"""Raw demo episodes for the residual (my_project/raw_demos.py): observations, executed actions and the base's actions,
offline, no sim; the residual's transitions are built from them when the learner loads the file.

    python scripts/make_raw_demos.py --config configs/experiments/scaled_residual/residual_scaled_residual_v3.yaml \
        --dataset data/physic_aligned_tcp_300 --out data/scaled_residual_v3/raw_demos.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

if __name__ == "__main__":
    from my_project.raw_demos import main

    raise SystemExit(main())
