"""Demonstrations for behaviour cloning: a LeRobot parquet dataset, windowed for diffusion.

**The format the trainer consumes** (any preprocessing must keep it): a LeRobot-style dataset
directory, as ``scripts/record_demos.py`` writes it, whose ``data/**/*.parquet`` rows carry ``episode_index``, ``frame_index``, an observation
column (``bc.obs_key``, a fixed-length float vector) and an action column (``bc.action_key``).
Every column the policy sees is one flat vector per frame; nothing else is read.

Windows follow robust-rearrangement's ``StateDataset`` (predicting future actions only):
a window is ``n_obs_steps + horizon - 1`` frames; the first ``n_obs_steps`` frames are the
observation history and the ``horizon`` actions start at the last observed frame. Episodes
are padded by repeating their first frame (``n_obs_steps - 1`` before) and last frame
(``action_horizon - 1`` after), so every frame gets to be "now".

``bc.num_demos`` training episodes are drawn at random, and the test set is drawn from the
episodes left over (see :func:`split_episodes`). Changed from the reference: the train / test
split is **by episode**. The reference split the windows, so neighbouring windows of one
demonstration landed on both sides and the test loss partly measured memorization.
Normalization stats are fitted on the selected train and test episodes, as the reference
fits them on everything it loaded.

Everything lives on the training device; a batch is two gathers, no DataLoader workers.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

__all__ = ["Demonstrations", "load_lerobot", "WindowedSplit", "split_episodes"]


@dataclass
class Demonstrations:
    """Every frame, episode-contiguous: ``obs`` ``(N, obs_dim)``, ``action`` ``(N, A)``."""

    obs: torch.Tensor
    action: torch.Tensor
    #: ``(E,)`` index of each episode's first frame, and its length
    episode_starts: torch.Tensor
    episode_lengths: torch.Tensor
    source: str

    @property
    def num_episodes(self) -> int:
        return int(self.episode_starts.numel())

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    def frames_of(self, episodes: np.ndarray) -> torch.Tensor:
        """Every frame index of ``episodes``, for fitting stats on a subset."""
        return torch.cat(
            [
                torch.arange(int(self.episode_starts[e]), int(self.episode_starts[e] + self.episode_lengths[e]))
                for e in episodes
            ]
        )


def _parquet_files(directory: str) -> Tuple[list, str]:
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"bc.dataset={directory!r} is not a directory")
    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"bc.dataset={directory!r}: no data/**/*.parquet under {root}; expected a dataset "
            "written by scripts/record_demos.py"
        )
    return files, str(root)


def load_lerobot(directory: str, obs_key: str, action_key: str) -> Demonstrations:
    import pyarrow.parquet as pq

    files, root = _parquet_files(directory)
    columns = ["episode_index", "frame_index", obs_key, action_key]
    tables = []
    for path in files:
        names = pq.read_schema(path).names
        missing = [c for c in columns if c not in names]
        if missing:
            raise KeyError(
                f"{path} has no column(s) {missing} (bc.obs_key={obs_key!r}, "
                f"bc.action_key={action_key!r}); columns: {names}"
            )
        tables.append(pq.read_table(path, columns=columns))

    episode = np.concatenate([t.column("episode_index").to_numpy() for t in tables])
    frame = np.concatenate([t.column("frame_index").to_numpy() for t in tables])
    obs = np.concatenate([_stack(t, obs_key) for t in tables]).astype(np.float32)
    action = np.concatenate([_stack(t, action_key) for t in tables]).astype(np.float32)

    order = np.lexsort((frame, episode))  # episode-major, frames in order
    episode, frame, obs, action = episode[order], frame[order], obs[order], action[order]
    ids, starts, lengths = np.unique(episode, return_index=True, return_counts=True)
    expected = np.concatenate([np.arange(n) for n in lengths])
    if not np.array_equal(frame, expected):
        bad = int(ids[np.searchsorted(starts, np.flatnonzero(frame != expected)[0], "right") - 1])
        raise ValueError(f"{root}: episode {bad} has missing or duplicate frame_index values")
    if not (np.isfinite(obs).all() and np.isfinite(action).all()):
        raise ValueError(f"{root}: non-finite values in {obs_key!r} or {action_key!r}")

    return Demonstrations(
        obs=torch.from_numpy(obs),
        action=torch.from_numpy(action),
        episode_starts=torch.from_numpy(starts.astype(np.int64)),
        episode_lengths=torch.from_numpy(lengths.astype(np.int64)),
        source=root,
    )


def _stack(table, key: str) -> np.ndarray:
    column = table.column(key).combine_chunks()
    if hasattr(column, "flatten"):  # fixed_size_list<float>[n]
        width = column.type.list_size
        return column.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)
    return np.stack(column.to_numpy(zero_copy_only=False))


def split_episodes(
    num_episodes: int, num_demos: Optional[int], test_split: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Seeded ``(train, test)`` episode indices.

    * ``num_demos`` < the dataset: ``num_demos`` training episodes drawn at random, and
      ``ceil(test_split * num_demos)`` test episodes drawn from the ones left over;
    * ``num_demos`` null or equal to the dataset: every episode is used, and
      ``ceil(test_split * num_episodes)`` of them are held out for the test set.

    Asking for more demos than the dataset has, or leaving too few for the test set, raises.
    """
    order = np.random.default_rng(seed).permutation(num_episodes)
    if num_demos is None or num_demos == num_episodes:
        num_test = math.ceil(test_split * num_episodes)
        if not 1 <= num_test < num_episodes:
            raise ValueError(
                f"{num_episodes} episodes cannot hold out {num_test} test episodes and keep "
                "some for training"
            )
        return np.sort(order[num_test:]), np.sort(order[:num_test])
    if num_demos > num_episodes:
        raise ValueError(f"bc.num_demos={num_demos}, but the dataset has {num_episodes} episodes")
    num_test = math.ceil(test_split * num_demos)
    if num_demos + num_test > num_episodes:
        raise ValueError(
            f"bc.num_demos={num_demos} leaves {num_episodes - num_demos} unused episodes, fewer "
            f"than the {num_test} test episodes ceil(bc.test_split * num_demos) asks for"
        )
    return np.sort(order[:num_demos]), np.sort(order[num_demos : num_demos + num_test])


class WindowedSplit:
    """The windows of a subset of episodes, as normalized tensors on ``device``.

    A window is ``(episode, offset)``: frame ``j`` of the window is episode frame
    ``clamp(offset + j, 0, length - 1)``, which is exactly the reference's edge padding.
    """

    def __init__(
        self,
        demos: Demonstrations,
        episodes: np.ndarray,
        normalizer,
        *,
        n_obs_steps: int,
        horizon: int,
        action_horizon: int,
        device,
    ) -> None:
        if len(episodes) == 0:
            raise ValueError("a split needs at least one episode")
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.sequence_length = n_obs_steps + horizon - 1
        self.device = torch.device(device)
        self.num_episodes = len(episodes)

        pad_before, pad_after = n_obs_steps - 1, action_horizon - 1
        starts, lengths, offsets = [], [], []
        for e in episodes:
            start = int(demos.episode_starts[e])
            length = int(demos.episode_lengths[e])
            first, last = -pad_before, length - self.sequence_length + pad_after
            count = last - first + 1
            if count < 1:
                raise ValueError(
                    f"episode {int(e)} has {length} frames, too short for a window of "
                    f"{self.sequence_length} with {pad_after} frames of end padding"
                )
            starts.append(np.full(count, start))
            lengths.append(np.full(count, length))
            offsets.append(np.arange(first, last + 1))
        as_tensor = lambda parts: torch.from_numpy(np.concatenate(parts)).to(self.device)
        self._start, self._length, self._offset = as_tensor(starts), as_tensor(lengths), as_tensor(offsets)

        with torch.no_grad():
            self._obs = normalizer.normalize(demos.obs.to(self.device), "obs")
            self._action = normalizer.normalize(demos.action.to(self.device), "action")
        self._steps = torch.arange(self.sequence_length, device=self.device)
        self._order: Optional[torch.Tensor] = None
        self._cursor = 0

    def __len__(self) -> int:
        return int(self._start.numel())

    def windows(self, index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``obs`` ``(B, n_obs_steps, obs_dim)`` and ``action`` ``(B, horizon, A)``, normalized."""
        within = (self._offset[index, None] + self._steps).clamp(min=0)
        within = torch.minimum(within, self._length[index, None] - 1)
        frames = self._start[index, None] + within
        first_action = self.n_obs_steps - 1
        return {
            "obs": self._obs[frames[:, : self.n_obs_steps]],
            "action": self._action[frames[:, first_action : first_action + self.horizon]],
        }

    def batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """The next ``batch_size`` windows of a shuffled pass; reshuffles when one ends."""
        picked = []
        need = batch_size
        while need:
            if self._order is None or self._cursor >= len(self):
                self._order = torch.randperm(len(self), device=self.device)
                self._cursor = 0
            take = self._order[self._cursor : self._cursor + need]
            self._cursor += take.numel()
            need -= take.numel()
            picked.append(take)
        return self.windows(torch.cat(picked))
