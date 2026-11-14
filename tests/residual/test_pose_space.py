"""``residual.action_space: pose`` (my_project/pose_residual.py, my_project/action_spaces.py PoseSpace): for every valid
combination of frame, rotation, parametrisation, combine and target, the neutral r sends the base's own targets, the
critics' rebuilt action is what was sent (the base's own where the phase has no residual), the chunk frame and the
chained target restart where the base replans, the widths follow the parametrisation, and the refusals."""

import math

import pytest
import torch

from my_project.action_spaces import PoseSpace
from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat
from my_project.pose_residual import SIDE_WIDTH, PoseResidual, ee_frame, exp_so3
from tests.residual.test_action_spaces import POSES, H, N, T, observations, scale_for


def neutral(pose, n):
    return torch.full((n, pose.width), 1.0 if pose.combine == "multiply" else 0.0)


def episode(space, r_of, routed_of=None):
    obs, base = observations(seed=4)
    state = space.initial_state(N, "cpu")
    rows = []
    for t in range(T):
        routed = torch.ones(N, dtype=torch.bool) if routed_of is None else routed_of(t)
        step, state = space.observe(obs[t], base[t], state, fresh=torch.full((N,), t == 0),
                                    new_chunk=torch.full((N,), t % H == 0), routed=routed)
        r = r_of(t)
        sent, state = space.send(r, step, state)
        rows.append((obs[t], base[t], step, r, sent, dict(state)))
    return rows


@pytest.mark.parametrize("name", list(POSES))
def test_the_neutral_r_sends_the_bases_own_targets(name):
    pose = PoseResidual(**POSES[name])
    space = PoseSpace(pose, 10, critic_mode="observation")
    for _, base, _, _, sent, _ in episode(space, lambda t: neutral(pose, N)):
        torch.testing.assert_close(sent[:, 0:3], base[:, 0:3], atol=1e-5, rtol=0)
        torch.testing.assert_close(sent[:, 3:9], base[:, 3:9], atol=1e-5, rtol=0)
        torch.testing.assert_close(sent[:, 9:], base[:, 9:])  # the gripper is always the base's


@pytest.mark.parametrize("name", list(POSES))
def test_the_critics_action_is_the_sent_target_and_the_base_where_the_phase_has_no_residual(name):
    pose = PoseResidual(**POSES[name])
    space = PoseSpace(pose, 10, critic_mode="observation")
    g = torch.Generator().manual_seed(7)

    def r_of(t):
        r = (torch.rand(N, pose.width, generator=g) * 2 - 1) * 0.5
        return r + 1.0 if pose.combine == "multiply" else r

    routed_of = lambda t: torch.tensor([True, t % 3 != 1, False, True])  # env 1 leaves its routed phase, env 2 never in one
    for obs, base, step, r, sent, _ in episode(space, r_of, routed_of):
        assert step["side"].shape == (N, SIDE_WIDTH)
        got = pose.critic_action(step["side"], r)
        torch.testing.assert_close(got, torch.cat((ee_frame(sent, obs[:, 0:9]), sent[:, 9:]), dim=-1), atol=1e-5, rtol=0)
        unrouted = ~step["routed"]
        torch.testing.assert_close(sent[unrouted], base[unrouted])  # an unrouted env sends the base itself


def test_the_chunk_frame_is_the_ee_where_the_chunk_began_and_moves_only_at_a_chunk_start():
    pose = PoseResidual(action_frame="chunk", action_rotation="relative", observation_frame="chunk",
                        observation_rotation="relative", step_scale=scale_for("rotvec"))
    space = PoseSpace(pose, 10, critic_mode="observation")
    rows = episode(space, lambda t: neutral(pose, N))
    for t, (obs, _, _, _, _, state) in enumerate(rows):
        start = (t // H) * H
        torch.testing.assert_close(state["ee_chunk"], rows[start][0][:, 0:9])


def test_the_chained_target_starts_at_the_ee_and_holds_when_the_multiplier_is_zero():
    pose = PoseResidual(combine="multiply", action_target="target")
    space = PoseSpace(pose, 10, critic_mode="observation")
    rows = episode(space, lambda t: torch.zeros(N, pose.width))
    for t, (obs, _, _, _, sent, _) in enumerate(rows):
        chunk_ee = rows[(t // H) * H][0][:, 0:9]
        torch.testing.assert_close(sent[:, 0:9], chunk_ee, atol=1e-5, rtol=0)  # m = 0: the target stays at the chunk's EE


def test_the_actor_observes_the_motion_and_the_current_target_and_the_critics_the_target_too():
    for param, width in (("rotvec", 6), ("euler", 6), ("rot6d", 9)):
        pose = PoseResidual(rotation_param=param, action_target="target", step_scale=scale_for(param))
        assert pose.width == width and pose.observed_width == width + 1 and pose.current_width == width
        space = PoseSpace(pose, 10, critic_mode="observation")
        obs, base, step, _, _, _ = episode(space, lambda t: neutral(pose, N))[1]
        assert step["policy"].shape == (N, 63 + width + 1 + width)
        assert step["critic"].shape == (N, 63 + width)
        torch.testing.assert_close(step["critic"][:, 63:], step["policy"][:, -width:])
    plain = PoseResidual(step_scale=scale_for("rotvec"))
    assert plain.current_width == 0
    hidden = PoseResidual(action_target="target", critic_observes_target=False, step_scale=scale_for("rotvec"))
    space = PoseSpace(hidden, 10, critic_mode="observation")
    obs, base, step, _, _, _ = episode(space, lambda t: neutral(hidden, N))[1]
    assert hidden.critic_current_width == 0 and step["policy"].shape == (N, 63 + 6 + 1 + 6)
    assert step["critic"].shape == (N, 63)  # the agent still sees the current target, the critics do not


def test_the_adaptive_factor_is_the_bases_motion_in_r_units_read_from_the_agents_observation():
    pose = PoseResidual(action_frame="relative", action_rotation="relative",
                        step_scale=scale_for("rotvec", 0.02, math.radians(4.0)))
    space = PoseSpace(pose, 10, critic_mode="observation")
    _, _, step, _, _, _ = episode(space, lambda t: neutral(pose, N))[2]
    expected = step["d"].abs() / torch.tensor([0.02] * 3 + [math.radians(4.0)] * 3)
    torch.testing.assert_close(space.motion_factor(step["policy"], 63), expected)  # d right after the 63 kept columns
    history = torch.cat((step["policy"], torch.randn(N, 20)), dim=-1)  # the history wrapper appends after: same factor
    torch.testing.assert_close(space.motion_factor(history, 63), expected)
    mult = PoseResidual(combine="multiply", action_target="target")
    mspace = PoseSpace(mult, 10, critic_mode="observation")
    _, _, mstep, _, _, _ = episode(mspace, lambda t: neutral(mult, N))[2]
    torch.testing.assert_close(mspace.motion_factor(mstep["policy"], 63), torch.ones(N, 6))  # r = 1: the base's motion


def test_multiplying_a_pose_is_refused():
    with pytest.raises(ValueError, match="scales a motion"):
        PoseResidual(combine="multiply")  # the absolute frame without a target: a pose
    with pytest.raises(ValueError, match="scales a motion"):
        PoseResidual(combine="multiply", action_frame="relative", action_rotation="absolute")
    with pytest.raises(ValueError, match="must be one of"):
        PoseResidual(action_frame="ee", step_scale=scale_for("rotvec"))


def test_the_step_scale_is_one_number_per_r_dim_for_add_and_none_for_multiply():
    pose = PoseResidual(step_scale=[0.01, 0.02, 0.03, 0.1, 0.2, 0.3])  # per axis, m and rad
    d = torch.zeros(1, 6)
    torch.testing.assert_close(pose.step(d, torch.ones(1, 6)), torch.tensor([[0.01, 0.02, 0.03, 0.1, 0.2, 0.3]]))
    for bad in (None, [0.01] * 5, [0.01] * 3 + [0.0] * 3):
        with pytest.raises(ValueError, match="step_scale"):
            PoseResidual(step_scale=bad)
    assert PoseResidual(rotation_param="rot6d", step_scale=[0.01] * 9).width == 9
    with pytest.raises(ValueError, match="multiply has none"):
        PoseResidual(combine="multiply", action_target="target", step_scale=[0.01] * 6)


def test_multiplied_rot6d_turns_the_vectors_and_a_collapsed_vector_is_the_bases():
    pose = PoseResidual(combine="multiply", action_target="target", rotation_param="rot6d")
    g = torch.Generator().manual_seed(3)
    R = exp_so3(torch.randn(N, 3, generator=g) * 0.3)
    d = torch.cat((torch.randn(N, 3, generator=g), mat_to_rot6d(R)), dim=-1)
    r = 1.0 + 0.3 * torch.randn(N, 9, generator=g)  # each number its own multiplier
    turned = pose.step(d, r)
    torch.testing.assert_close(turned, r * d)  # no collapse: exactly the per-number product
    zero = torch.ones(N, 9)
    zero[:, 3:6] = 0.0  # the first vector shrunk to nothing
    torch.testing.assert_close(pose.step(d, zero)[:, 3:6], d[:, 3:6])
    ratio = torch.ones(N, 9)
    ratio[:, 6:9] = d[:, 3:6] / d[:, 6:9]  # the second vector made the first (parallel to it)
    torch.testing.assert_close(pose.step(d, ratio)[:, 6:9], d[:, 6:9])
    R_out = rot6d_to_mat(pose.step(d, zero)[:, 3:9])
    torch.testing.assert_close(R_out @ R_out.transpose(-1, -2), torch.eye(3).expand(N, 3, 3), atol=1e-5, rtol=0)


def test_ee_relative_actions_observes_the_bases_target_relative_to_the_ee():
    pose = PoseResidual(action_target="target", step_scale=scale_for("rotvec"), ee_relative_actions=True)
    assert pose.observed_width == 7
    space = PoseSpace(pose, 10, critic_mode="observation")
    obs, base = observations(seed=4)
    step, _ = space.observe(obs[0], base[0], space.initial_state(N, "cpu"), fresh=torch.ones(N, dtype=torch.bool),
                            new_chunk=torch.ones(N, dtype=torch.bool), routed=torch.ones(N, dtype=torch.bool))
    torch.testing.assert_close(step["policy"][:, 63:69], ee_frame(base[0], obs[0][:, 0:9]))
    torch.testing.assert_close(step["policy"][:, 69:70], base[0][:, 9:10])
    assert step["policy"].shape == (N, 63 + 7 + 6)  # then the current target
    with pytest.raises(NotImplementedError, match="ee_relative_actions"):
        space.motion_factor(step["policy"], 63)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for the sync-debug mode")
@pytest.mark.parametrize("combine, target, param", [("add", "target", "rotvec"), ("multiply", "target", "rotvec"),
                                                   ("add", "none", "rotvec"), ("multiply", "target", "rot6d")])
def test_a_pose_step_never_synchronizes_with_the_host(combine, target, param):
    """The per-step path (observe, send, the critics' rebuilt action, the exploration scale) stays on the device: no
    tensor from a Python list, no host read (the critics' action also runs inside a captured CUDA graph)."""
    pose = PoseResidual(combine=combine, action_target=target, observation_frame="relative", observation_rotation="relative",
                        action_frame="absolute" if target == "target" else "relative",
                        action_rotation="absolute" if target == "target" else "relative", rotation_param=param,
                        step_scale=scale_for(param) if combine == "add" else None)
    space = PoseSpace(pose, 10, critic_mode="observation")
    obs, base = (x.cuda() for x in observations(seed=5))
    state = {k: v.cuda() for k, v in space.initial_state(N, "cpu").items()}
    routed = torch.ones(N, dtype=torch.bool, device="cuda")
    fresh = torch.zeros(N, dtype=torch.bool, device="cuda")
    r = neutral(pose, N).cuda()
    for t in range(3):  # the first step makes the device constants; the rest must not touch the host
        if t == 1:
            torch.cuda.set_sync_debug_mode("error")
        try:
            step, state = space.observe(obs[t], base[t], state, fresh=fresh, new_chunk=fresh, routed=routed)
            sent, state = space.send(r, step, state)
            pose.critic_action(step["side"], r)
            space.motion_factor(step["policy"], 63).add_(0.0)
        finally:
            torch.cuda.set_sync_debug_mode("default")
