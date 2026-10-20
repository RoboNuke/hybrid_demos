"""ResiP's released checkpoint in this port (no sim): the rot6d -> quaternion conversion (their
sim's), the
observation adapter as the exact inverse of the zarr importer's conversion, and the loader on
a synthetic robust-rearrangement checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from furniture_lab.demos import mat_to_rot6d, quat_to_mat, quat_xyzw_to_wxyz
from furniture_lab.tasks.furniture.frames import RESIP_BASE_Z_TO_PORT

from my_project.bc import resip_checkpoint as R
from my_project.models.cfg import DiffusionUnetActorCfg
from my_project.models.diffusion_policy import DiffusionPolicy


def random_quats(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rot6d_of(q_xyzw):
    return mat_to_rot6d(quat_to_mat(quat_xyzw_to_wxyz(q_xyzw)))


def canonical(q):
    """ResiP's sim writes every quaternion with w >= 0 (matrix_to_quaternion_xyzw)."""
    return q * np.where(q[:, 3:4] < 0, -1.0, 1.0)


def test_rot6d_to_quat_is_their_sims_quaternion():
    rng = np.random.default_rng(0)
    for q in canonical(random_quats(rng, 200)):
        got = R.rot6d_to_quat_xyzw(torch.tensor(rot6d_of(q), dtype=torch.float64)).numpy()
        assert np.allclose(got, q, atol=1e-9)


def resip_obs(rng, T):
    """ResiP-layout observations (T, 58), quaternions as their sim writes them; the 6th pose is
    the front obstacle: slid in x, y only (fixed z), rotated 90 deg about z as their sim builds it."""
    from furniture_lab.tasks.furniture.frames import OBSTACLE_FRONT_POS, TAG_POS_W, TAG_ROT_W

    robot = rng.normal(size=(T, 16)).astype(np.float32)
    robot[:, 3:9] = np.stack([rot6d_of(q) for q in random_quats(rng, T)])
    poses = []
    for i in range(5):
        q = canonical(random_quats(rng, T))
        poses.append(np.concatenate([rng.normal(size=(T, 3)), q], axis=1))
    front_b = np.asarray(OBSTACLE_FRONT_POS) + np.concatenate([rng.uniform(-0.02, 0.02, size=(T, 2)), np.zeros((T, 1))], axis=1)
    front_t = (front_b - np.asarray(TAG_POS_W)) @ np.asarray(TAG_ROT_W)
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # their obstacle: 90 deg about z
    q_ob = R.mat_to_quat_xyzw(torch.tensor(np.asarray(TAG_ROT_W).T @ Rz, dtype=torch.float64)).numpy()
    poses.append(np.concatenate([front_t, np.tile(q_ob, (T, 1))], axis=1))
    return np.concatenate([robot] + poses, axis=1).astype(np.float32)


def port_obs(theirs):
    """What the port observes for the same state: the importer's conversion (and z shift)."""
    from furniture_lab.demos import resip_abs_states
    from furniture_lab.tasks.furniture.frames import TAG_POS_W, TAG_ROT_W

    robot = theirs[:, :16].copy()
    robot[:, 2] += RESIP_BASE_Z_TO_PORT
    front_robot = np.asarray(TAG_POS_W) + theirs[:, 16 + 35 : 16 + 38] @ np.asarray(TAG_ROT_W).T
    return resip_abs_states(robot, theirs[:, 16 : 16 + 35], front_robot)


def test_the_adapter_inverts_the_importer():
    rng = np.random.default_rng(1)
    theirs = resip_obs(rng, 64)
    adapter = R.ResipObsAdapter(RESIP_BASE_Z_TO_PORT)
    ours = port_obs(theirs)
    assert ours.shape == (64, R.PORT_OBS_DIM)
    back = adapter.obs(torch.tensor(ours)).numpy()
    assert back.shape == (64, R.RESIP_OBS_DIM)
    assert np.allclose(back, theirs, atol=1e-4)  # float32 states


def test_the_action_goes_back_into_the_ports_base_frame():
    adapter = R.ResipObsAdapter(RESIP_BASE_Z_TO_PORT)
    a = torch.arange(10, dtype=torch.float32)[None]
    out = adapter.action(a)
    assert torch.isclose(out[0, 2], a[0, 2] + RESIP_BASE_Z_TO_PORT)
    assert torch.equal(out[0, [0, 1, *range(3, 10)]], a[0, [0, 1, *range(3, 10)]])


def fake_checkpoint(tmp_path):
    """A robust-rearrangement bc.py save: the actor's state dict (model + normalizer) + config."""
    cfg = DiffusionUnetActorCfg(down_dims=[16, 32], inference_steps=4)
    net = DiffusionPolicy(cfg, R.RESIP_OBS_DIM, R.ACTION_DIM).model
    sd = {f"model.{k}": v for k, v in net.state_dict().items()}
    for key, dim in (("robot_state", 16), ("parts_poses", 42), ("action", 10)):
        sd[f"normalizer.stats.{key}.min"] = -torch.ones(dim) * 2
        sd[f"normalizer.stats.{key}.max"] = torch.ones(dim) * 3
    config = {
        "observation_type": "state",
        "task": "one_leg",
        "randomness": "low",
        "control": {"control_mode": "pos", "act_rot_repr": "rot_6d", "controller": "diffik"},
        "actor": {
            "name": "diffusion", "pred_horizon": 32, "obs_horizon": 1, "action_horizon": 8,
            "num_diffusion_iters": 100, "beta_schedule": "squaredcos_cap_v2", "prediction_type": "epsilon",
            "clip_sample": True, "inference_steps": 16, "predict_past_actions": False,
            "diffusion_model": {"name": "unet", "down_dims": [16, 32]},
        },
    }
    path = tmp_path / "actor_chkpt.pt"
    torch.save({"model_state_dict": sd, "config": config, "epoch": 7, "best_success_rate": 0.5}, path)
    return path, net


def test_the_loader_reproduces_their_network_and_normalizer(tmp_path):
    path, net = fake_checkpoint(tmp_path)
    policy, adapter, info = R.load_resip_policy(str(path), "cpu")
    assert info["inference_steps"] == 4 and policy.cfg.warm_start and policy.cfg.warm_start_timestep == 50
    x, c, k = torch.randn(2, 32, 10), torch.randn(2, 58), torch.tensor([3, 40])
    net.eval()
    assert torch.allclose(policy.model(x, k, global_cond=c), net(x, k, global_cond=c))
    assert torch.equal(policy.normalizer.obs_min, -2 * torch.ones(58))
    assert torch.equal(policy.normalizer.action_max, 3 * torch.ones(10))


def test_the_loader_refuses_an_unmapped_state_dict_entry(tmp_path):
    path, _ = fake_checkpoint(tmp_path)
    ckpt = torch.load(path, weights_only=False)
    ckpt["model_state_dict"]["encoder1.weight"] = torch.zeros(1)
    torch.save(ckpt, path)
    with pytest.raises(KeyError, match="encoder1.weight"):
        R.load_resip_policy(str(path), "cpu")


def test_offline_check_windows_and_metrics(tmp_path):
    """The zarr windows pad with the episode's last action, and the check runs end to end on
    a synthetic checkpoint + zarr (an untrained net: only the plumbing is tested)."""
    zarr = pytest.importorskip("zarr")
    from my_project.bc import resip_offline_check as C

    rs = np.arange(10 * 16, dtype=np.float32).reshape(10, 16)
    parts = np.zeros((10, 42), np.float32)
    acts = np.arange(10 * 10, dtype=np.float32).reshape(10, 10)
    ends = np.array([4, 10])
    obs, tgt, t = C.windows_from_zarr(rs, parts, acts, ends, 6, 50, np.random.default_rng(0))
    for k in range(50):
        e_end = 4 if t[k] < 4 else 10
        want = acts[np.minimum(t[k] + np.arange(6), e_end - 1)]
        assert np.array_equal(tgt[k], want) and np.array_equal(obs[k, :16], rs[t[k]])

    path, _ = fake_checkpoint(tmp_path)
    z = zarr.open_group(str(tmp_path / "d.zarr"), mode="w", zarr_format=2)
    rng = np.random.default_rng(0)
    z.create_array("robot_state", data=rng.normal(size=(40, 16)).astype(np.float32))
    z.create_array("parts_poses", data=rng.normal(size=(40, 42)).astype(np.float32))
    a = rng.normal(size=(40, 10)).astype(np.float32)
    a[:, 3:9] = [1, 0, 0, 0, 1, 0]
    z.create_group("action").create_array("pos", data=a)
    z.create_array("episode_ends", data=np.array([20, 40], np.int64))
    r = C.check(str(path), str(tmp_path / "d.zarr"), windows=16, samples=1)
    assert set(r["pos_mm"]) == {"step0", "step7", "step31"} and 0.0 <= r["gripper_agreement"] <= 1.0


def test_the_obstacle_quaternion_is_their_rotated_bodys_or_the_checkpoints_constant(tmp_path):
    """Their front obstacle is Rz(90) in the base frame (ours is unrotated): in the tag frame a
    180 deg turn, not R_tag^T; and a checkpoint that saw it constant gets exactly its value."""
    from furniture_lab.tasks.furniture.frames import TAG_ROT_W

    q = R.ResipObsAdapter(RESIP_BASE_Z_TO_PORT).obstacle_quat.numpy()
    assert abs(abs(q[0]) - 1.0) < 1e-5 and np.abs(q[1:]).max() < 1e-5  # (+-1, 0, 0, 0)
    wrong = R.mat_to_quat_xyzw(torch.tensor(np.asarray(TAG_ROT_W).T, dtype=torch.float32)).numpy()
    assert np.abs(q - wrong).max() > 0.5  # the old "never rotated" value
    path, _ = fake_checkpoint(tmp_path)
    ckpt = torch.load(path, weights_only=False)
    v = torch.tensor([-1.0, 1.7e-4, 0.0, 0.0])
    ckpt["model_state_dict"]["normalizer.stats.parts_poses.min"][38:42] = v - 1
    ckpt["model_state_dict"]["normalizer.stats.parts_poses.max"][38:42] = v + 1
    torch.save(ckpt, path)
    _, adapter, info = R.load_resip_policy(str(path), "cpu")
    assert torch.allclose(adapter.obstacle_quat, v) and "constant" in info["obstacle_quat_source"]
    out = adapter.obs(torch.zeros(3, R.PORT_OBS_DIM))
    assert torch.allclose(out[:, 16 + 35 + 3 : 16 + 42], v.expand(3, 4))
