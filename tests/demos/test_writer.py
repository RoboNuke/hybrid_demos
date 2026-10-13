"""DemoWriter: what it writes is what the BC loader reads; appends are checked, never silent."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from my_project.bc.dataset import load_lerobot
from my_project.demos.writer import DemoWriter

INFO = {"task": "T", "control_dt": 1 / 15, "physics_dt": 1 / 120, "decimation": 8}


def episode(writer, n, offset=0.0, seed=1, extra=None):
    writer.start()
    for t in range(n):
        row = {"observation.state": np.full(4, offset + t), "action": np.full(7, -t), "observation.success": 0.0}
        row.update(extra or {})
        writer.add(row)
    return writer.save({"seed": seed, "success": True}, {"scene": {"x": torch.ones(1)}, "attrs": {}})


def test_episodes_round_trip_through_the_bc_loader(tmp_path):
    writer = DemoWriter(tmp_path, fps=15.0, info=INFO)
    episode(writer, 5, offset=0.0)
    episode(writer, 3, offset=100.0)
    demos = load_lerobot(str(tmp_path), "observation.state", "action")
    assert demos.num_episodes == 2 and demos.obs_dim == 4 and demos.action_dim == 7
    assert demos.episode_lengths.tolist() == [5, 3]
    assert demos.obs[5, 0] == 100.0  # episode 1 starts after episode 0's five rows

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["fps"] == 15.0 and info["total_episodes"] == 2 and info["features"]["action"] == 7
    lines = (tmp_path / "meta" / "episodes.jsonl").read_text().splitlines()
    assert [json.loads(l)["length"] for l in lines] == [5, 3]
    assert torch.load(tmp_path / json.loads(lines[1])["state_file"])["scene"]["x"].item() == 1.0


def test_a_new_session_appends_and_numbers_on(tmp_path):
    episode(DemoWriter(tmp_path, fps=15.0, info=INFO), 4, seed=1)
    writer = DemoWriter(tmp_path, fps=15.0, info=INFO)
    assert writer.num_episodes == 1 and writer.used_seeds() == {1}
    assert episode(writer, 2, seed=2)["episode_index"] == 1
    assert (tmp_path / "data" / "chunk-000" / "episode_000001.parquet").exists()


def test_mismatches_raise(tmp_path):
    episode(DemoWriter(tmp_path, fps=15.0, info=INFO), 2)
    with pytest.raises(ValueError, match="fps"):
        DemoWriter(tmp_path, fps=20.0, info=INFO)
    writer = DemoWriter(tmp_path, fps=15.0, info=INFO)
    with pytest.raises(ValueError, match="columns"):
        episode(writer, 2, extra={"observation.extra": np.zeros(2)})
    writer.start()
    with pytest.raises(ValueError, match="empty"):
        writer.save({}, {})


def test_discard_drops_the_episode_in_progress(tmp_path):
    writer = DemoWriter(tmp_path, fps=15.0, info=INFO)
    writer.start()
    writer.add({"action": np.zeros(7)})
    assert writer.discard() == 1 and writer.length == 0 and writer.num_episodes == 0
