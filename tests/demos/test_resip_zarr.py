"""ResiP's processed zarr -> our FBAbs dataset: layout, parts as rot6d, episode split."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from my_project.bc.dataset import load_lerobot  # noqa: E402
from my_project.demos.resip_zarr import import_resip_zarr, zarr_states_abs  # noqa: E402


def write_zarr(path, lengths, success, robot_state=None, actions=None):
    rng = np.random.default_rng(0)
    T = sum(lengths)
    z = zarr.open_group(str(path), mode="w", zarr_format=2)
    rs = rng.normal(size=(T, 16)).astype(np.float32) if robot_state is None else robot_state
    z.create_array("robot_state", data=rs)
    parts = rng.normal(size=(T, 5, 7)).astype(np.float32)
    parts[..., 3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quats (xyzw)
    parts[0, 0, 3:7] = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]  # 90 deg about z
    obstacle = np.tile(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32), (T, 1))
    z.create_array("parts_poses", data=np.concatenate([parts.reshape(T, 35), obstacle], axis=1))  # ResiP's 42
    action = rng.normal(size=(T, 10)).astype(np.float32)
    action[:, 3:9] = rs[:, 3:9]  # zero rotation deltas: under ResiP's 0.35 rad clip, kept as is
    z.create_group("action").create_array("pos", data=action if actions is None else actions)
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


def test_ee_and_target_z_move_into_the_ports_base_frame(tmp_path):
    z = write_zarr(tmp_path / "d.zarr", [3], [1])
    r = import_resip_zarr(tmp_path / "d.zarr", tmp_path / "out")
    assert r["saved"] == 1
    demos = load_lerobot(str(tmp_path / "out"), "observation.state_abs", "action.abs")
    import pyarrow.parquet as pq, glob

    t = pq.read_table(glob.glob(str(tmp_path / "out" / "data" / "*" / "*.parquet"))[0]).to_pydict()
    np.testing.assert_allclose(np.array(t["observation.state_abs"])[:, 2], z["robot_state"][:, 2] - 0.015, atol=1e-6)
    np.testing.assert_allclose(np.array(t["action.abs"])[:, 2], z["action"]["pos"][:, 2] - 0.015, atol=1e-6)
    np.testing.assert_allclose(np.array(t["observation.state_abs"])[:, 0], z["robot_state"][:, 0], atol=1e-6)  # x as is


def test_import_scales_resips_shrunk_rotation_deltas_back_up(tmp_path):
    """process_pickles shrinks an episode's rotation deltas by one factor to an episode norm of
    0.35 rad; the importer writes the targets ResiP's sim was sent (20 deg steps recovered)."""
    from scipy.spatial.transform import Rotation
    from furniture_lab.demos import mat_to_rot6d

    T, rng = 30, np.random.default_rng(3)
    raw = Rotation.random(T, random_state=4).as_rotvec()
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True) * np.deg2rad(rng.uniform(1, 15, T))[:, None]
    raw[:8] *= np.deg2rad(20.0) / np.linalg.norm(raw[:8], axis=1, keepdims=True)  # the teleop bound
    f = np.linalg.norm(raw) / 0.35
    R = Rotation.random(T, random_state=5).as_matrix()
    rs = rng.normal(size=(T, 16)).astype(np.float32)
    rs[:, 3:9] = [mat_to_rot6d(m) for m in R]
    act = rng.normal(size=(T, 10)).astype(np.float32)
    act[:, 3:9] = [mat_to_rot6d(m) for m in R @ Rotation.from_rotvec(raw / f).as_matrix()]
    write_zarr(tmp_path / "d.zarr", [T], [1], robot_state=rs, actions=act)
    import_resip_zarr(tmp_path / "d.zarr", tmp_path / "out")
    import glob, json, pyarrow.parquet as pq

    t = pq.read_table(glob.glob(str(tmp_path / "out" / "data" / "*" / "*.parquet"))[0]).to_pydict()
    want = np.stack([mat_to_rot6d(m) for m in R @ Rotation.from_rotvec(raw).as_matrix()])
    np.testing.assert_allclose(np.array(t["action.abs"])[:, 3:9], want, atol=1e-3)
