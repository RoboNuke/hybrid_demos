"""model.actor.action_frame: ee_at_obs -- every step of the chunk relative to the EE pose at the observation."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from my_project.bc.dataset import WindowedSplit, load_lerobot
from my_project.models import action_frame as af
from my_project.models.cfg import DiffusionTransformerActorCfg
from my_project.models.diffusion_policy import ChunkedBase, DiffusionPolicy, MinMaxNormalizer

B, H = 16, 6


def poses(n, seed=0, max_deg=180.0):
    """``(n, 9)`` FBAbs-style observation heads ``[ee_pos 3, ee_rot6d 6]`` with real rotations."""
    g = torch.Generator().manual_seed(seed)
    axis = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=torch.float64), dim=-1)
    angle = torch.rand(n, 1, generator=g, dtype=torch.float64) * math.radians(max_deg)
    R = af.axis_angle_to_mat(axis * angle)
    return torch.cat([torch.rand(n, 3, generator=g, dtype=torch.float64), af.mat_to_rot6d(R)], dim=-1)


def abs_targets(obs, seed=1, max_deg=170.0):
    """``(B, H, 10)`` absolute targets near each observation's EE: a few cm, up to ``max_deg`` away."""
    g = torch.Generator().manual_seed(seed)
    p, R = af.ee_pose(obs)
    axis = torch.nn.functional.normalize(torch.randn(len(obs), H, 3, generator=g, dtype=torch.float64), dim=-1)
    angle = torch.rand(len(obs), H, 1, generator=g, dtype=torch.float64) * math.radians(max_deg)
    R_goal = R[:, None] @ af.axis_angle_to_mat(axis * angle)
    goal = p[:, None] + torch.randn(len(obs), H, 3, generator=g, dtype=torch.float64) * 0.1
    grip = torch.sign(torch.randn(len(obs), H, 1, generator=g, dtype=torch.float64))
    return torch.cat([goal, af.mat_to_rot6d(R_goal), grip], dim=-1)


def same_rot6d(a, b):
    return torch.allclose(af.rot6d_to_mat(a), af.rot6d_to_mat(b), atol=1e-9)


def test_the_ee_frame_round_trips_the_absolute_targets():
    obs = poses(B)
    target = abs_targets(obs)
    p, R = af.ee_pose(obs)
    rel = af.to_ee_frame(p, R, target)
    assert rel.shape == (B, H, af.NET_DIM)
    back = af.from_ee_frame(p, R, rel)
    assert torch.allclose(back[..., 0:3], target[..., 0:3], atol=1e-12)
    assert same_rot6d(back[..., 3:9], target[..., 3:9]) and torch.equal(back[..., 9], target[..., 9])
    # unscaled: dpos in m, drot in rad
    assert torch.allclose(rel[..., 0:3].norm(dim=-1), (target[..., 0:3] - p[:, None]).norm(dim=-1), atol=1e-12)


def test_zero_is_the_pose_at_the_observation_and_every_step_shares_it():
    obs = poses(B)
    p, R = af.ee_pose(obs)
    zeros = torch.zeros(B, H, af.NET_DIM, dtype=torch.float64)
    out = af.from_ee_frame(p, R, zeros)
    assert torch.allclose(out[..., 0:3], p[:, None].expand(B, H, 3), atol=1e-12)
    assert same_rot6d(out[..., 3:9], obs[:, None, 3:9].expand(B, H, 6))
    # a step 5 cm along the EE's x is 5 cm along it from the observation's pose, whatever step it is
    step = zeros.clone()
    step[..., 0] = 0.05
    assert torch.allclose(af.from_ee_frame(p, R, step)[..., 0:3], p[:, None] + 0.05 * R[:, None, :, 0], atol=1e-12)


def test_reframing_a_chunk_keeps_its_absolute_targets():
    old, new = poses(B, seed=2), poses(B, seed=3)
    target = abs_targets(old, max_deg=60.0)
    moved = af.reframe(af.to_ee_frame(*af.ee_pose(old), target), old, new)
    want = af.to_ee_frame(*af.ee_pose(new), target)
    assert torch.allclose(moved[..., 0:3], want[..., 0:3], atol=1e-10)
    assert torch.allclose(af.axis_angle_to_mat(moved[..., 3:6]), af.axis_angle_to_mat(want[..., 3:6]), atol=1e-9)
    same = af.reframe(want, new, new)  # the same pose: unchanged
    assert torch.allclose(same, want, atol=1e-9)


# ------------------------------------------------------------------ the training labels
def write_fbabs_dataset(root, lengths):
    obs = poses(sum(lengths), seed=4, max_deg=60.0).float().numpy()
    act = abs_targets(poses(sum(lengths), seed=4, max_deg=60.0), seed=5, max_deg=40.0)[:, 0].float().numpy()
    table = pa.table({
        "episode_index": pa.array(np.repeat(np.arange(len(lengths)), lengths), pa.int64()),
        "frame_index": pa.array(np.concatenate([np.arange(n) for n in lengths]), pa.int64()),
        "observation.state_abs": pa.array(obs.tolist(), pa.list_(pa.float32(), 9)),
        "action.abs": pa.array(act.tolist(), pa.list_(pa.float32(), 10)),
    })
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
    return obs, act


def test_each_windows_labels_are_its_absolute_targets_in_its_observations_ee_frame(tmp_path):
    lengths = [9, 7]
    write_fbabs_dataset(tmp_path, lengths)
    demos = load_lerobot(str(tmp_path), "observation.state_abs", "action.abs")
    norm = MinMaxNormalizer(9, af.NET_DIM)  # the [-1, 1] default: identity
    kw = dict(n_obs_steps=2, horizon=4, action_horizon=2, device="cpu")
    rel = WindowedSplit(demos, np.arange(2), norm, action_frame=af.EE_AT_OBS, **kw)
    ab = WindowedSplit(demos, np.arange(2), MinMaxNormalizer(9, 10), **kw)
    index = torch.arange(len(rel))
    got, ref = rel.windows(index), ab.windows(index)  # ab: identity stats, raw absolute actions
    assert torch.equal(got["obs"], ref["obs"]) and got["action"].shape == (len(rel), 4, af.NET_DIM)
    last_obs = ref["obs"][:, -1].double()  # the window's last observed frame
    want = af.to_ee_frame(*af.ee_pose(last_obs), ref["action"].double())
    assert torch.allclose(got["action"].double(), want, atol=1e-5)


# ------------------------------------------------------------------ acting
def make_policy(**overrides) -> DiffusionPolicy:
    cfg = DiffusionTransformerActorCfg(
        horizon=H, n_obs_steps=1, action_horizon=2, n_layer=1, n_head=1, n_emb=8, inference_steps=2,
        num_diffusion_iters=10, warm_start_timestep=5, action_frame=af.EE_AT_OBS, clip_sample=False,
    )
    torch.manual_seed(0)
    policy = DiffusionPolicy(dataclasses.replace(cfg, **overrides), 9, 10)
    policy.normalizer.fit("obs", poses(50, seed=6).float())  # action stats: the [-1, 1] default
    return policy.eval()


def test_the_network_works_on_the_ee_frame_action_and_acts_with_absolute_targets(monkeypatch):
    policy = make_policy(warm_start=False)
    assert (policy.action_dim, policy.net_action_dim) == (10, af.NET_DIM)
    assert policy.normalizer.action_min.shape == (af.NET_DIM,)
    chunk = torch.zeros(B, H, af.NET_DIM)
    chunk[..., 2] = 0.03  # every step: 3 cm along the observation's EE z
    monkeypatch.setattr(policy, "_denoise", lambda nobs, start: chunk[: nobs.shape[0]])
    obs = poses(B, seed=7).float()
    p, R = af.ee_pose(obs)
    for _ in range(2):  # both executed steps share the observation's pose
        a = policy.act(obs)
        assert a.shape == (B, 10)
        assert torch.allclose(a[:, 0:3], p + 0.03 * R[:, :, 2], atol=1e-5)
        assert torch.allclose(af.rot6d_to_mat(a[:, 3:9]), R, atol=1e-5)
    runner = ChunkedBase(policy, B)
    a = runner.actions(obs, torch.ones(B, dtype=torch.bool))
    assert torch.allclose(a[:, 0:3], p + 0.03 * R[:, :, 2], atol=1e-5)


def test_the_warm_start_tail_is_re_expressed_at_the_next_observation(monkeypatch):
    policy = make_policy()
    planned = af.to_ee_frame(*af.ee_pose(poses(B, seed=8)), abs_targets(poses(B, seed=8), max_deg=30.0)).float()
    starts = []

    def denoise(nobs, start):
        starts.append(None if start is None else start.clone())
        return planned

    monkeypatch.setattr(policy, "_denoise", denoise)
    runner = ChunkedBase(policy, B)
    obs0, obs1 = poses(B, seed=8).float(), poses(B, seed=9, max_deg=40.0).float()
    runner.actions(obs0, torch.ones(B, dtype=torch.bool))
    assert torch.equal(starts[0], torch.zeros(B, H, af.NET_DIM))  # a fresh episode: zeros, the pose itself
    runner.actions(obs0, torch.zeros(B, dtype=torch.bool))  # the second executed step: no new chunk
    runner.actions(obs1, torch.zeros(B, dtype=torch.bool))  # a new chunk at a new observation
    tail = torch.cat([planned[:, 2:], torch.zeros(B, 2, af.NET_DIM)], dim=1)  # the unexecuted tail, then zeros
    assert torch.allclose(starts[1], af.reframe(tail.double(), obs0.double(), obs1.double()).float(), atol=1e-4)


def test_the_absolute_frame_is_untouched():
    policy = DiffusionPolicy(DiffusionTransformerActorCfg(n_layer=1, n_head=1, n_emb=8), 9, 10)
    assert policy.net_action_dim == policy.action_dim == 10 and policy.action_frame == af.ABSOLUTE
    x = torch.randn(B, H, 10)
    assert policy.reframe(x, torch.zeros(B, 9), torch.ones(B, 9)) is x
    assert policy.to_env(x, torch.zeros(B, 9)) is x
