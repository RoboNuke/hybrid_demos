"""ResiP's processed zarr -> our FBAbs dataset: layout, parts as rot6d, episode split."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from my_project.bc.dataset import load_lerobot  # noqa: E402
from my_project.demos.resip_zarr import import_resip_zarr, zarr_states_abs  # noqa: E402


def write_zarr(path, lengths, success):
    rng = np.random.default_rng(0)
    T = sum(lengths)
    z = zarr.open_group(str(path), mode="w", zarr_format=2)
    z.create_array("robot_state", data=rng.normal(size=(T, 16)).astype(np.float32))
    parts = rng.normal(size=(T, 5, 7)).astype(np.float32)
    parts[..., 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quats (xyzw)
    parts[0, 0, 3:7] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90 deg about z
    z.create_array("parts_poses", data=parts.reshape(T, 35))
    z.create_group("action").create_array("pos", data=rng.normal(size=(T, 10)).astype(np.float32))
    z.create_array("episode_ends", data=np.cumsum(lengths).astype(np.int64))
    z.create_array("reward", data=np.zeros(T, np.float32))
    z.create_array("skill", data=np.zeros(T, np.float32))
    z.create_array("success", data=np.asarray(success, np.int64))
    return z


def test_parts_become_rot6d_rows():
    rs = np.zeros((1, 16), np.float32)
    parts = np.zeros((1, 35), np.float32)
    parts[0, 3:7] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90 deg about z
    for i in range(1, 5):
        parts[0, 7 * i + 6] = 1.0
    s = zarr_states_abs(rs, parts)
    assert s.shape == (1, 61)
    # R_z(90) = [[0,-1,0],[1,0,0],[0,0,1]]: first two rows
    np.testing.assert_allclose(s[0, 19:25], [0, -1, 0, 1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(s[0, 28:34], [1, 0, 0, 0, 1, 0], atol=1e-6)  # identity


def test_import_splits_episodes_drops_failures_and_trains(tmp_path):
    write_zarr(tmp_path / "d.zarr", [5, 4, 6], [1, 0, 1])
    r = import_resip_zarr(tmp_path / "d.zarr", tmp_path / "out")
    assert r["saved"] == 2 and r["failures_dropped"] == 1 and r["frames"] == 11
    demos = load_lerobot(str(tmp_path / "out"), "observation.state_abs", "action.abs")
    assert demos.num_episodes == 2
    again = import_resip_zarr(tmp_path / "d.zarr", tmp_path / "out")
    assert again["saved"] == 0 and again["skipped"] == 2  # appending is idempotent
