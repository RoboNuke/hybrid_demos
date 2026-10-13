"""Compare a trained diffusion policy's chunks against its training data (no simulator):
delegates to :func:`my_project.bc.analyze.main`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make my_project importable

if __name__ == "__main__":
    from my_project.bc.analyze import main

    raise SystemExit(main())
