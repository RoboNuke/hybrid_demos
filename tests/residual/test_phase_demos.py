"""my_project/phase_demos.py: the target map's inverse, and a demo episode cut into per-phase chunk transitions."""

from __future__ import annotations

import math

import pytest
import torch

from my_project.models.action_frame import mat_to_rot6d
from my_project.models.diffusion_policy import MinMaxNormalizer
from my_project.phase_demos import episode_chunks, invert_target
from my_project.residual_ee import exp_so3
from my_project.residual_target import EERelative, TargetResidual

NAMES = ["corner", "grasp", "insert", "screw", "success"]


def targets(n, seed=0):
    torch.manual_seed(seed)
    return torch.cat((torch.randn(n, 3) * 0.2, mat_to_rot6d(exp_so3(torch.randn(n, 3))), torch.rand(n, 1)), -1)


def test_the_inverse_reproduces_the_demo_through_the_target_map():
    scales = (0.01, math.radians(6.0), 0.1)
    base = targets(20)
    r = torch.cat((torch.randn(20, 3) * 2, torch.randn(20, 3) * 0.5, torch.randn(20, 1)), -1)  # some beyond 1
    demo = TargetResidual(scales[0], 6.0, scales[2]).step(base, r)
    torch.testing.assert_close(invert_target(base, demo, *scales), r, atol=1e-4, rtol=1e-4)


def a_normalizer():
    norm = MinMaxNormalizer(63, 10)
    norm.fit("obs", torch.randn(64, 63))
    norm.fit("action", targets(64, 3) * 1.5)
    return norm


def test_an_episode_is_cut_into_chunks_per_phase_with_local_rewards_and_ends():
    k, T = 4, 24
    states = torch.randn(T, 63)
    states[:, 3:9] = mat_to_rot6d(exp_so3(torch.randn(T, 3)))
    actions = targets(T, 1)
    # frames 0-7 grasp (1), 8-15 insert (2), 16-23 screw (3); success on frame 22
    phases = torch.tensor([1] * 8 + [2] * 8 + [3] * 8)
    starts = list(range(0, 23, k))
    base_chunks = targets(len(starts) * k, 2).reshape(len(starts), k, 10)
    norm = a_normalizer()
    rows = episode_chunks(states, actions, phases, 22, base_chunks, k=k, names=NAMES, routes=["insert", "screw"],
                          scales=(0.01, math.radians(6.0), 0.1), ee_rel=EERelative(0.1, 0.1, norm), normalizer=norm)
    insert, screw = rows["insert"], rows["screw"]
    assert insert["actions"].shape == (2, 28) and screw["actions"].shape == (2, 28)  # chunks at 8, 12 and 16, 20
    assert insert["rewards"].reshape(-1).tolist() == [0.0, 1.0]  # the chunk from 12 hands over to screw at 16
    assert insert["terminated"].reshape(-1).tolist() == [False, True]
    assert screw["rewards"].reshape(-1).tolist() == [0.0, 1.0]  # success on frame 22, inside the chunk from 20
    assert screw["terminated"].reshape(-1).tolist() == [False, True]
    # consecutive rows hand over: the next observation is the next row's observation
    torch.testing.assert_close(insert["next_observations"][0], insert["observations"][1])
    assert insert["observations"].shape[1] == 63 + 28 and insert["residual_base"].shape[1] == 40 + 9
    assert torch.equal(insert["states"][0], states[8])


def test_the_inverse_reproduces_the_demo_through_the_base_axes_map():
    scales = (0.02, math.radians(6.0), 0.1)
    base = targets(20, 1)
    r = torch.cat((torch.randn(20, 3) * 2, torch.randn(20, 3) * 0.5, torch.randn(20, 1)), -1)
    demo = TargetResidual(scales[0], 6.0, scales[2], axes="base").step(base, r)
    torch.testing.assert_close(invert_target(base, demo, *scales, axes="base"), r, atol=1e-4, rtol=1e-4)


def test_the_absolute_form_uses_the_env_wrappers_observation_and_critic_state():
    from my_project.residual import critic_state, residual_observation

    k, T = 4, 16
    states = torch.randn(T, 63)
    actions = targets(T, 2)
    phases = torch.tensor([2] * 8 + [3] * 8)
    base_chunks = targets(T // k * k, 3).reshape(T // k, k, 10)
    norm = a_normalizer()

    def encode(state, chunk):
        obs, _ = residual_observation(norm, state, chunk, False, True)
        return obs[0], critic_state(norm, state, chunk)[0], None

    out = episode_chunks(states, actions, phases, T - 1, base_chunks, k=k, names=NAMES, routes=["insert", "screw"],
                         scales=(0.02, math.radians(6.0), 0.1), ee_rel=None, normalizer=norm, encode=encode, axes="base")
    rows = out["insert"]
    assert "residual_base" not in rows  # no side input: the critic state carries the base chunk
    obs0, _ = residual_observation(norm, states[0:1], base_chunks[0:1], False, True)
    torch.testing.assert_close(rows["observations"][0], obs0[0])
    torch.testing.assert_close(rows["states"][0], critic_state(norm, states[0:1], base_chunks[0:1])[0])
    assert rows["actions"].shape[-1] == k * 7


def test_a_shaped_route_adds_gamma_phi_next_minus_phi_and_keeps_its_success_flag():
    k, T = 4, 24
    states = torch.randn(T, 63)
    states[:, 3:9] = mat_to_rot6d(exp_so3(torch.randn(T, 3)))
    actions = targets(T, 1)
    phases = torch.tensor([1] * 8 + [2] * 8 + [3] * 8)
    base_chunks = targets(len(range(0, 23, k)) * k, 2).reshape(-1, k, 10)
    norm = a_normalizer()
    phi = -torch.arange(T, dtype=torch.float32).flip(0)  # Phi = -(frames left): closer every frame
    rows = episode_chunks(states, actions, phases, 22, base_chunks, k=k, names=NAMES, routes=["insert", "screw"],
                          scales=(0.01, math.radians(6.0), 0.1), ee_rel=EERelative(0.1, 0.1, norm), normalizer=norm,
                          potentials={"insert": phi}, gamma=0.9)
    # insert chunks at 8 and 12: 0.9 Phi(12) - Phi(8), then 1 + 0.9 Phi(16) - Phi(12)
    expected = [0.9 * float(phi[12]) - float(phi[8]), 1.0 + 0.9 * float(phi[16]) - float(phi[12])]
    assert rows["insert"]["rewards"].reshape(-1).tolist() == pytest.approx(expected)
    assert rows["insert"]["success"].reshape(-1).tolist() == [0.0, 1.0]
    assert rows["screw"]["rewards"].reshape(-1).tolist() == [0.0, 1.0]  # no entry: unshaped


def test_the_success_is_the_step_that_assembles_not_the_rewarded_no_op_after_it():
    from my_project.phase_demos import assembling_frame

    # the collector rewards frame 7, the first already-assembled one (its action the hold-and-open no-op)
    assert assembling_frame([0.0] * 7 + [1.0, 0.0, 0.0]) == 6
    with pytest.raises(ValueError, match="no reward"):
        assembling_frame([0.0] * 5)
    with pytest.raises(ValueError, match="first frame"):
        assembling_frame([1.0, 0.0])


def test_a_gripperless_residual_keeps_the_pose_of_the_demo_residual():
    k, T = 1, 24
    states = torch.randn(T, 63)
    states[:, 3:9] = mat_to_rot6d(exp_so3(torch.randn(T, 3)))
    actions = targets(T, 1)
    phases = torch.tensor([1] * 8 + [2] * 8 + [3] * 8)
    base_chunks = targets(len(range(0, 23, k)) * k, 2).reshape(-1, k, 10)
    norm = a_normalizer()
    kw = dict(k=k, names=NAMES, routes=["insert", "screw"], scales=(0.02, math.radians(6.0), 0.1),
              ee_rel=EERelative(0.1, 0.1, norm), normalizer=norm, axes="base")
    full = episode_chunks(states, actions, phases, 22, base_chunks, **kw)
    pose = episode_chunks(states, actions, phases, 22, base_chunks, residual_gripper=False, **kw)
    assert pose["insert"]["actions"].shape[1] == 6
    torch.testing.assert_close(pose["insert"]["actions"], full["insert"]["actions"][:, :6])
