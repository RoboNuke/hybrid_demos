"""FurnitureBench pkl -> dataset: columns, widths, episode bookkeeping, and that the BC
loader reads what was written."""

import pickle

import numpy as np
import pytest

from my_project.demos.furniture import ACTION_DIM, STATE_DIM, find_demos, import_demos, state_vector


def _demo(T: int, success: bool, seed: int):
    rng = np.random.default_rng(seed)

    def obs():
        return {
            "robot_state": {
                "ee_pos": rng.normal(size=3).astype(np.float32),
                "ee_quat": np.array([0, 0, 0, 1], dtype=np.float32),
                "ee_pos_vel": rng.normal(size=3).astype(np.float32),
                "ee_ori_vel": rng.normal(size=3).astype(np.float32),
                "gripper_width": np.array([0.065], dtype=np.float32),
                "joint_positions": rng.normal(size=7).astype(np.float32),
                "joint_velocities": np.zeros(7, np.float32),
                "joint_torques": np.zeros(7, np.float32),
            },
            "parts_poses": rng.normal(size=35).astype(np.float32),
        }

    return {
        "observations": [obs() for _ in range(T + 1)],  # the collector appends the final observation
        "actions": [rng.normal(size=8).astype(np.float32) for _ in range(T)],
        "rewards": [0.0] * (T - 1) + [1.0 if success else 0.0],
        "skills": [0.0] * T,
        "success": success,
        "furniture": "one_leg",
        "error": False,
        "error_description": "",
    }


def _write(root, name, demo, fail=False):
    d = root / "one_leg" / ("fails" if fail else name)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(demo, f)
    return p


def test_state_vector_is_furniturebench_default_state_obs():
    v = state_vector(_demo(2, True, 0)["observations"][0])
    assert v.shape == (STATE_DIM,) and v.dtype == np.float32
    assert STATE_DIM == 49


def test_import_writes_the_dataset_and_the_loader_reads_it(tmp_path):
    src = tmp_path / "demos"
    _write(src, "2026-10-13-16:00:00", _demo(12, True, 1))
    _write(src, "2026-10-13-16:05:00", _demo(7, True, 2))
    _write(src, "fail1", _demo(5, False, 3), fail=True)
    assert len(find_demos(src)) == 2 and len(find_demos(src, include_failures=True)) == 3

    out = tmp_path / "dataset"
    r = import_demos(src, out)
    assert (r["saved"], r["episodes"], r["frames"]) == (2, 2, 19)
    # appending is idempotent per source file; failures come in only when asked
    r = import_demos(src, out)
    assert (r["saved"], r["skipped"]) == (0, 2)
    r = import_demos(src, out, include_failures=True)
    assert (r["saved"], r["episodes"], r["frames"]) == (1, 3, 24)

    import json

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["fps"] == 20.0 and info["task"] == "Isaac-Furniture-OneLeg-v0"
    assert info["features"]["observation.state"] == STATE_DIM and info["features"]["action"] == ACTION_DIM
    assert info["features"]["observation.next_state"] == STATE_DIM
    assert info["features"]["observation.state_abs"] == 61 and info["features"]["action.abs"] == 10
    assert info["features"]["observation.next_state_abs"] == 61
    from my_project.bc.dataset import load_lerobot

    abs_demos = load_lerobot(str(out), "observation.state_abs", "action.abs")
    assert abs_demos.obs_dim == 61 and abs_demos.action_dim == 10
    episodes = [json.loads(l) for l in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["success"] for e in episodes] == [True, True, False]
    assert [e["length"] for e in episodes] == [12, 7, 5]
    assert episodes[0]["reward_sum"] == 1.0 and episodes[2]["ended_by"] == "timeout"

    pytest.importorskip("pyarrow")
    pytest.importorskip("torch")
    from my_project.bc.dataset import load_lerobot

    demos = load_lerobot(str(out), "observation.state", "action")
    assert demos.num_episodes == 3 and demos.obs_dim == STATE_DIM and demos.action_dim == ACTION_DIM
    # next_state is the following frame's state
    import pyarrow.parquet as pq

    t = pq.read_table(str(out / "data" / "chunk-000" / "episode_000000.parquet")).to_pandas()
    s = np.stack(t["observation.state"].to_numpy())
    n = np.stack(t["observation.next_state"].to_numpy())
    assert np.allclose(n[:-1], s[1:])


def test_the_resip_task_appends_the_obstacle_pose(tmp_path):
    """Isaac-Furniture-*-FBAbs-ResiP: state_abs is 70, the pkl's recorded obstacle last."""
    import json

    from furniture_lab.demos import obstacle_tag_rot6d
    from furniture_lab.tasks.furniture import frames

    src, out = tmp_path / "src", tmp_path / "out"
    demo = _demo(5, True, 0)
    moved = np.asarray(frames.OBSTACLE_FRONT_POS) + np.array([0.01, 0.02, 0.0])
    demo["obstacles_robot"] = {"obstacle_front": moved}
    _write(src, "a", demo)
    import_demos(src, out, task_name="Isaac-Furniture-OneLeg-FBAbs-ResiP-v0")
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["features"]["observation.state_abs"] == 70
    import glob

    import pyarrow.parquet as pq

    t = pq.read_table(glob.glob(str(out / "data" / "*" / "*.parquet"))[0]).to_pydict()
    np.testing.assert_allclose(np.array(t["observation.state_abs"])[:, 61:], np.tile(obstacle_tag_rot6d(moved), (5, 1)), atol=1e-5)
