"""Add a residual run's sampled-action eval results to its deterministic eval wandb run:
delegates to :func:`my_project.residual_eval.publish_stochastic`. No simulator.

    python scripts/publish_stochastic_eval.py runs/hybrid_demo_pih/residual_sac_all/residual_sac_all_a0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

if __name__ == "__main__":
    from my_project.residual_eval import publish_stochastic

    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/publish_stochastic_eval.py <residual run agent dir>")
    publish_stochastic(sys.argv[1])
