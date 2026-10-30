"""Per-phase demo transitions for a routed residual (sac.route_labels), offline, CPU only, no sim:
delegates to :func:`my_project.phase_demos.main`.

    python scripts/make_phase_demos.py --config configs/experiments/parts_pd/residual_parts_pd.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

if __name__ == "__main__":
    from my_project.phase_demos import main

    raise SystemExit(main())
