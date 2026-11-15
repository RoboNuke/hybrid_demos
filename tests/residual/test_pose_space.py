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


PHASES = ["corner", "grasp", "insert", "screw", "success"]
#: a per-phase step_scale (m, rad): insert's and screw's own, far enough apart that a wrong row shows
PER_PHASE = {"insert": [0.004, 0.003, 0.006, 0.014, 0.020, 0.024], "screw": [0.002, 0.0022, 0.0027, 0.006, 0.011, 0.062]}


def test_a_per_phase_step_scale_steps_each_env_by_its_phases_scale_and_the_critics_and_demos_rebuild_it():
    from my_project.demo_loader import episode_steps

    pose = PoseResidual(action_target="target", step_scale=PER_PHASE, phase_names=PHASES)
    space = PoseSpace(pose, 10, critic_mode="observation")
    assert pose.side_width == SIDE_WIDTH + 6  # the rows' scale ends the side input
    obs, base = observations(seed=5)
    phase_of = lambda t: torch.tensor([2, 3, 2 if t < T // 2 else 3, 3])  # env 2 goes from insert to screw
    g = torch.Generator().manual_seed(11)
    state = space.initial_state(N, "cpu")
    executed, rs = [], []
    on = torch.ones(N, dtype=torch.bool)
    for t in range(T):
        rows = space.scale_rows_for(phase_of(t))
        torch.testing.assert_close(rows, torch.tensor([PER_PHASE[PHASES[p]] for p in phase_of(t).tolist()]))
        step, state = space.observe(obs[t], base[t], state, fresh=torch.full((N,), t == 0),
                                    new_chunk=torch.full((N,), t % H == 0), routed=on, scale_rows=rows)
        r = (torch.rand(N, 6, generator=g) * 2 - 1) * 2.0
        sent, state = space.send(r, step, state)
        # each env steps d + its own phase's scale * r
        torch.testing.assert_close(pose.step(step["d"], r, rows), step["d"] + rows * r)
        torch.testing.assert_close(sent, pose.sent(base[t], step["d"], step["anchor"], step["A"], r, on, rows))
        # the critics rebuild what was sent from the side input alone
        got = pose.critic_action(step["side"], r)
        torch.testing.assert_close(got, torch.cat((ee_frame(sent, obs[t][:, 0:9]), sent[:, 9:]), dim=-1), atol=1e-5, rtol=0)
        executed.append(sent), rs.append(r)
    # the demo loader inverts each frame with its own phase's scale
    episodes = [{"states": obs[:, i], "base_actions": base[:, i], "actions": torch.stack(executed)[:, i],
                 "phase": torch.stack([phase_of(t)[i] for t in range(T)])} for i in range(N)]
    out = episode_steps(space, episodes, H, lambda phase: torch.ones(phase.shape[0], dtype=torch.bool), "cpu")
    for i in range(N):
        torch.testing.assert_close(out[i]["r"], torch.stack(rs)[:, i], atol=2e-3, rtol=0)


def test_a_per_phase_step_scale_needs_each_rows_scale_and_known_phases():
    pose = PoseResidual(action_target="target", step_scale=PER_PHASE, phase_names=PHASES)
    with pytest.raises(ValueError, match="pass each row's scale"):
        pose.step(torch.zeros(1, 6), torch.ones(1, 6))
    space = PoseSpace(pose, 10, critic_mode="observation")
    obs, base = observations(seed=5)
    with pytest.raises(ValueError, match="pass each row's scale"):
        space.observe(obs[0], base[0], space.initial_state(N, "cpu"), fresh=torch.ones(N, dtype=torch.bool),
                      new_chunk=torch.ones(N, dtype=torch.bool), routed=torch.ones(N, dtype=torch.bool))
    with pytest.raises(ValueError, match="not phase labels"):
        PoseResidual(step_scale={"lift": PER_PHASE["insert"]}, phase_names=PHASES)
    with pytest.raises(ValueError, match=r"step_scale\.screw"):
        PoseResidual(step_scale={"insert": PER_PHASE["insert"], "screw": [0.01] * 5})
    with pytest.raises(ValueError, match="multiply has none"):
        PoseResidual(combine="multiply", action_target="target", step_scale=PER_PHASE)
    # one scale for every phase: no rows, the side input as before
    assert PoseResidual(step_scale=PER_PHASE["insert"]).scale_rows(torch.tensor([2])) is None


def test_the_adaptive_factor_divides_by_the_routed_learners_own_phases_scale():
    from my_project.residual import MotionFactor

    pose = PoseResidual(action_target="target", step_scale=PER_PHASE, phase_names=PHASES)
    space = PoseSpace(pose, 10, critic_mode="observation")
    obs, base = observations(seed=6)
    state = space.initial_state(N, "cpu")
    phase = torch.tensor([2, 2, 3, 3])
    for t in range(2):
        step, state = space.observe(obs[t], base[t], state, fresh=torch.full((N,), t == 0),
                                    new_chunk=torch.full((N,), t % H == 0), routed=torch.ones(N, dtype=torch.bool),
                                    scale_rows=space.scale_rows_for(phase))
        _, state = space.send(torch.zeros(N, 6), step, state)
    factor = MotionFactor(space, 63)
    for route in ("insert", "screw"):
        torch.testing.assert_close(factor.for_route(route)(step["policy"]), step["d"].abs() / torch.tensor(PER_PHASE[route]))
    with pytest.raises(NotImplementedError, match="one residual per phase"):
        factor(step["policy"])  # one learner for every phase: its rows' phases are not in its observations
    with pytest.raises(ValueError, match="no scale for phase"):
        factor.for_route("corner")(step["policy"])


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
                                                   ("add", "none", "rotvec"), ("multiply", "target", "rot6d"),
                                                   ("add", "target", "per_phase")])
def test_a_pose_step_never_synchronizes_with_the_host(combine, target, param):
    """The per-step path (observe, send, the critics' rebuilt action, the exploration scale) stays on the device: no
    tensor from a Python list, no host read (the critics' action also runs inside a captured CUDA graph)."""
    per_phase = param == "per_phase"  # a per-phase step_scale: the rows' scale from the phase labels, also on the device
    param = "rotvec" if per_phase else param
    pose = PoseResidual(combine=combine, action_target=target, observation_frame="relative", observation_rotation="relative",
                        action_frame="absolute" if target == "target" else "relative",
                        action_rotation="absolute" if target == "target" else "relative", rotation_param=param,
                        step_scale=(PER_PHASE if per_phase else scale_for(param)) if combine == "add" else None,
                        phase_names=PHASES)
    space = PoseSpace(pose, 10, critic_mode="observation")
    phase = torch.tensor([2, 3, 2, 3], device="cuda") if torch.cuda.is_available() else None
    obs, base = (x.cuda() for x in observations(seed=5))
    state = {k: v.cuda() for k, v in space.initial_state(N, "cpu").items()}
    routed = torch.ones(N, dtype=torch.bool, device="cuda")
    fresh = torch.zeros(N, dtype=torch.bool, device="cuda")
    r = neutral(pose, N).cuda()
    for t in range(3):  # the first step makes the device constants; the rest must not touch the host
        if t == 1:
            torch.cuda.set_sync_debug_mode("error")
        try:
            step, state = space.observe(obs[t], base[t], state, fresh=fresh, new_chunk=fresh, routed=routed,
                                        scale_rows=space.scale_rows_for(phase))
            sent, state = space.send(r, step, state)
            pose.critic_action(step["side"], r)
            space.motion_factor(step["policy"], 63, "insert" if per_phase else None).add_(0.0)
        finally:
            torch.cuda.set_sync_debug_mode("default")


def test_the_config_refuses_a_per_phase_step_scale_missing_a_phase_the_residual_acts_in():
    from types import SimpleNamespace

    from my_project.cfg import ResidualCfg

    rc = ResidualCfg(enabled=True, action_space="pose", action_target="target", step_scale={"insert": PER_PHASE["insert"]},
                     critic_state="observation", critic_action="full", observe_base_action=True, action_scale=None)
    routed = SimpleNamespace(trainer=SimpleNamespace(learner="ddpg"),
                             ddpg=SimpleNamespace(std_label_names=PHASES, route_labels=["insert", "screw"]))
    with pytest.raises(ValueError, match=r"no scale for the residual's phases \['screw'\]"):
        rc._validate_pose(routed)
    one = SimpleNamespace(trainer=SimpleNamespace(learner="ddpg"), ddpg=SimpleNamespace(std_label_names=PHASES, route_labels=[]))
    with pytest.raises(ValueError, match="no scale for the residual's phases"):
        rc._validate_pose(one)  # one residual for every phase: every phase needs a scale
