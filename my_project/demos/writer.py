"""Write teleoperated demos as a LeRobot-style parquet dataset, one file per episode.

    <root>/data/chunk-000/episode_000123.parquet   one row per control step
    <root>/states/episode_000123.pt                the env's full state at the episode's start
    <root>/meta/info.json                          fps, control / physics dt, every column's width
    <root>/meta/episodes.jsonl                     one line per saved episode
    <root>/meta/sessions/<session>.yaml            the resolved config of each recording session

The BC loader (``my_project/bc/dataset.py``) reads ``data/**/*.parquet`` and needs only
``episode_index``, ``frame_index`` and the obs / action columns it is pointed at; everything else
here is raw material for preprocessing (absolute positions, other observation layouts) so a
format change never needs a re-collection.

An episode file is written once, complete, when the episode is saved: a crash loses at most the
episode in progress. A session appends to an existing dataset; the columns must match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["DemoWriter"]

#: rows of an episode that are scalars, not vectors
SCALARS = ("episode_index", "frame_index", "index", "task_index", "timestamp")


class DemoWriter:
    def __init__(self, root: str | Path, *, fps: float, info: Dict[str, Any]) -> None:
        self.root = Path(root).expanduser()
        self.fps = float(fps)
        self.data_dir = self.root / "data" / "chunk-000"
        self.meta_dir = self.root / "meta"
        self.states_dir = self.root / "states"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)
        (self.meta_dir / "sessions").mkdir(parents=True, exist_ok=True)
        self.info_path = self.meta_dir / "info.json"
        self.episodes_path = self.meta_dir / "episodes.jsonl"
        self._info = dict(info, fps=self.fps)
        self.episodes: List[Dict[str, Any]] = []
        if self.episodes_path.exists():
            self.episodes = [json.loads(line) for line in self.episodes_path.read_text().splitlines() if line]
        if self.info_path.exists():
            existing = json.loads(self.info_path.read_text())
            for key in ("fps", "control_dt", "physics_dt", "decimation", "task"):
                if key in self._info and existing.get(key) != self._info[key]:
                    raise ValueError(
                        f"{self.root}: this session's {key}={self._info[key]!r} differs from the "
                        f"dataset's {existing.get(key)!r}; record into a new dataset_dir"
                    )
            self._features: Optional[Dict[str, int]] = existing.get("features")
        else:
            self._features = None
        self._rows: List[Dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------ sessions
    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_frames(self) -> int:
        return sum(e["length"] for e in self.episodes)

    def sessions(self) -> List[str]:
        return sorted(p.stem for p in (self.meta_dir / "sessions").glob("*.yaml"))

    def used_seeds(self) -> set:
        return {e["seed"] for e in self.episodes}

    def add_session(self, name: str, resolved_config: Path) -> None:
        target = self.meta_dir / "sessions" / f"{name}.yaml"
        if target.exists():
            raise FileExistsError(f"session {name} already recorded in {self.root}")
        target.write_text(Path(resolved_config).read_text())

    # ------------------------------------------------------------------ an episode
    def start(self) -> None:
        self._rows = []

    def add(self, row: Dict[str, Any]) -> None:
        """One control step: column name -> 1-d array (or a scalar for a 1-wide column)."""
        self._rows.append({k: np.atleast_1d(np.asarray(v, dtype=np.float32)) for k, v in row.items()})

    @property
    def length(self) -> int:
        return len(self._rows)

    def save(self, meta: Dict[str, Any], start_state: Dict[str, Any]) -> Dict[str, Any]:
        """Write the episode in progress; ``meta`` lands in ``meta/episodes.jsonl`` and
        ``start_state`` (``my_project.demos.state.snapshot``) in ``states/``, for replay."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        import torch

        if not self._rows:
            raise ValueError("cannot save an empty episode")
        index = self.num_episodes
        start = self.num_frames
        n = len(self._rows)
        features = {k: int(v.shape[0]) for k, v in self._rows[0].items()}
        if self._features is not None and features != self._features:
            raise ValueError(
                f"{self.root}: columns {features} differ from the dataset's {self._features}"
            )
        columns = {
            "episode_index": pa.array(np.full(n, index), pa.int64()),
            "frame_index": pa.array(np.arange(n), pa.int64()),
            "index": pa.array(np.arange(start, start + n), pa.int64()),
            "task_index": pa.array(np.zeros(n), pa.int64()),
            "timestamp": pa.array(np.arange(n) / self.fps, pa.float32()),
        }
        for name, width in features.items():
            values = np.stack([row[name] for row in self._rows]).astype(np.float32)
            if width == 1:
                columns[name] = pa.array(values[:, 0], pa.float32())
            else:
                columns[name] = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), width)
        path = self.data_dir / f"episode_{index:06d}.parquet"
        if path.exists():
            raise FileExistsError(f"{path} exists but meta/episodes.jsonl does not list it")
        state_path = self.states_dir / f"episode_{index:06d}.pt"
        torch.save(start_state, state_path)
        pq.write_table(pa.table(columns), path)

        record = {
            "episode_index": index,
            "length": n,
            "file": str(path.relative_to(self.root)),
            "state_file": str(state_path.relative_to(self.root)),
            **meta,
        }
        with self.episodes_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.episodes.append(record)
        self._features = features
        self._info.update(
            features=features,
            total_episodes=self.num_episodes,
            total_frames=self.num_frames,
            codebase_version="v3.0-style (one file per episode)",
        )
        self.info_path.write_text(json.dumps(self._info, indent=2))
        self._rows = []
        return record

    def discard(self) -> int:
        dropped, self._rows = len(self._rows), []
        return dropped
