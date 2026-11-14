"""``residual.action_space: pose`` (my_project/pose_residual.py, my_project/action_spaces.py PoseSpace): for every valid
combination of frame, rotation, parametrisation, combine and target, the neutral r sends the base's own targets, the
critics' rebuilt action is what was sent (the base's own where the phase has no residual), the chunk frame and the
chained target restart where the base replans, the widths follow the parametrisation, and the refusals."""

import math

import pytest
import torch

from my_project.action_spaces import PoseSpace
from my_project.pose_residual import SIDE_WIDTH, PoseResidual, ee_frame
from tests.residual.test_action_spaces import POSES, H, N, T, observations


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
                        observation_rotation="relative")
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
        pose = PoseResidual(rotation_param=param, action_target="target")
        assert pose.width == width and pose.observed_width == width + 1 and pose.current_width == width
        space = PoseSpace(pose, 10, critic_mode="observation")
        obs, base, step, _, _, _ = episode(space, lambda t: neutral(pose, N))[1]
        assert step["policy"].shape == (N, 63 + width + 1 + width)
        assert step["critic"].shape == (N, 63 + width)
        torch.testing.assert_close(step["critic"][:, 63:], step["policy"][:, -width:])
    plain = PoseResidual()
    assert plain.current_width == 0
    hidden = PoseResidual(action_target="target", critic_observes_target=False)
    space = PoseSpace(hidden, 10, critic_mode="observation")
    obs, base, step, _, _, _ = episode(space, lambda t: neutral(hidden, N))[1]
    assert hidden.critic_current_width == 0 and step["policy"].shape == (N, 63 + 6 + 1 + 6)
    assert step["critic"].shape == (N, 63)  # the agent still sees the current target, the critics do not


def test_the_adaptive_scale_is_the_bases_motion_in_r_units():
    pose = PoseResidual(action_frame="relative", action_rotation="relative", step_scale_pos=0.02,
                        step_scale_rot=math.radians(4.0))
    space = PoseSpace(pose, 10, critic_mode="observation")
    _, _, step, _, _, _ = episode(space, lambda t: neutral(pose, N))[2]
    expected = step["d"].abs() / torch.tensor([0.02] * 3 + [math.radians(4.0)] * 3)
    torch.testing.assert_close(step["motion_scale"], expected)
    mult = PoseResidual(combine="multiply", action_target="target")
    _, _, mstep, _, _, _ = episode(PoseSpace(mult, 10, critic_mode="observation"), lambda t: neutral(mult, N))[2]
    torch.testing.assert_close(mstep["motion_scale"], torch.ones(N, 6))  # r = 1 is the base's own motion


def test_multiplying_a_pose_or_rot6d_numbers_is_refused():
    with pytest.raises(ValueError, match="scales a motion"):
        PoseResidual(combine="multiply")  # the absolute frame without a target: a pose
    with pytest.raises(ValueError, match="scales a motion"):
        PoseResidual(combine="multiply", action_frame="relative", action_rotation="absolute")
    with pytest.raises(ValueError, match="rot6d"):
        PoseResidual(combine="multiply", action_target="target", rotation_param="rot6d")
    with pytest.raises(ValueError, match="must be one of"):
        PoseResidual(action_frame="ee")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device for the sync-debug mode")
@pytest.mark.parametrize("combine, target", [("add", "target"), ("multiply", "target"), ("add", "none")])
def test_a_pose_step_never_synchronizes_with_the_host(combine, target):
    """The per-step path (observe, send, the critics' rebuilt action, the exploration scale) stays on the device: no
    tensor from a Python list, no host read (the critics' action also runs inside a captured CUDA graph)."""
    pose = PoseResidual(combine=combine, action_target=target, observation_frame="relative", observation_rotation="relative",
                        action_frame="absolute" if target == "target" else "relative",
                        action_rotation="absolute" if target == "target" else "relative")
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
            step["motion_scale"].add_(0.0)
        finally:
            torch.cuda.set_sync_debug_mode("default")
