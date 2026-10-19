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
from my_project.demos.resip_zarr import zarr_states_abs
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
    """ResiP-layout observations (T, 58), quaternions as their sim writes them."""
    robot = rng.normal(size=(T, 16)).astype(np.float32)
    robot[:, 3:9] = np.stack([rot6d_of(q) for q in random_quats(rng, T)])
    poses = []
    for i in range(R.N_POSES):
        q = canonical(random_quats(rng, T))
        poses.append(np.concatenate([rng.normal(size=(T, 3)), q], axis=1))
    return np.concatenate([robot] + poses, axis=1).astype(np.float32)


def port_obs(theirs):
    """What the port observes for the same state: the importer's conversion (and z shift)."""
    T = theirs.shape[0]
    robot = theirs[:, :16].copy()
    robot[:, 2] += RESIP_BASE_Z_TO_PORT
    ours = zarr_states_abs(robot, theirs[:, 16 : 16 + 35])
    ob = theirs[:, 16 + 35 :]
    obstacle = np.concatenate([ob[:, :3], np.stack([rot6d_of(q) for q in ob[:, 3:7]])], axis=1)
    return np.concatenate([ours, obstacle], axis=1).astype(np.float32)


def test_the_adapter_inverts_the_importer():
    rng = np.random.default_rng(1)
    theirs = resip_obs(rng, 64)
    adapter = R.ResipObsAdapter(RESIP_BASE_Z_TO_PORT)
    back = adapter.obs(torch.tensor(port_obs(theirs))).numpy()
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
