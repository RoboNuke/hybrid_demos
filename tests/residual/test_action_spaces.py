"""my_project/action_spaces.py: every action space's ``invert`` is its ``send``'s inverse over an episode (the same ``r``
and the same state after every step), and the demo loader's pieces (my_project/demo_loader.py): each frame's inputs and
``r`` through the space, the env's rewards (recorded terms x the run's weights), its terminations and the phase-switch
terminal."""

from types import SimpleNamespace

import pytest
import torch

import itertools

from my_project.action_spaces import PerDimSpace, PoseSpace
from my_project.demo_loader import env_rewards, env_terminations, episode_steps, episode_transitions
from my_project.models.action_frame import mat_to_rot6d
from my_project.pose_residual import FRAMES, PARAMS, ROTATIONS, PoseResidual, exp_so3

N, T, H = 4, 10, 4


def observations(seed=0):
    """Raw FBAbs ResiP observations ``(T, N, 63)`` (EE pose first, parts' rotations valid) and base targets ``(T, N, 10)``."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(T, N, 63, generator=g) * 0.05
    obs[..., 0:3] += torch.tensor([0.5, 0.0, 0.1])
    obs[..., 3:9] = mat_to_rot6d(exp_so3(torch.randn(T, N, 3, generator=g) * 0.3))
    for p in range(5):
        obs[..., 16 + 9 * p + 3 : 16 + 9 * p + 9] = mat_to_rot6d(exp_so3(torch.randn(T, N, 3, generator=g) * 0.3))
    base = torch.cat((obs[..., 0:3] + torch.randn(T, N, 3, generator=g) * 0.01,
                      mat_to_rot6d(exp_so3(torch.randn(T, N, 3, generator=g) * 0.05) @ exp_so3(torch.randn(T, N, 3, generator=g) * 0.3)),
                      torch.rand(T, N, 1, generator=g) * 2 - 1), dim=-1)
    return obs, base


def pose_settings():
    """Every valid pose configuration: frame x rotation x parametrisation x combine x target."""
    out = {}
    for frame, rotation, param, combine, target in itertools.product(FRAMES, ROTATIONS, PARAMS, ("add", "multiply"), ("none", "target")):
        kw = dict(action_frame=frame, action_rotation=rotation, observation_frame=frame, observation_rotation=rotation,
                  rotation_param=param, combine=combine, action_target=target, step_scale_pos=0.01, step_scale_rot=0.05)
        try:
            PoseResidual(**kw)
        except ValueError:
            continue
        out[f"pose_{frame}_{rotation}_{param}_{combine}_{target}"] = kw
    return out


POSES = pose_settings()
SPACES = {
    "per_dim": lambda: PerDimSpace(torch.full((10,), 0.05)),
    "per_dim_clip": lambda: PerDimSpace(torch.full((10,), 0.05), clip=(torch.full((10,), -5.0), torch.full((10,), 5.0))),
    **{name: (lambda kw=kw: PoseSpace(PoseResidual(**kw), 10, critic_mode="observation")) for name, kw in POSES.items()},
}


def test_every_axis_combination_is_buildable_except_the_refused_ones():
    # multiply needs a motion (a target, or a relative frame and rotation) and a zero-at-identity parametrisation
    multiply = [n for n in POSES if "_multiply_" in n]
    assert all("_rot6d_" not in n for n in multiply)
    assert all(n.endswith("_target") or ("_absolute_" not in n.split("pose_")[1][:9] and "_relative_" in n) for n in multiply)
    assert len([n for n in POSES if "_add_" in n]) == 3 * 2 * 3 * 2  # every add combination


def r_draw(space, g):
    if isinstance(space, PoseSpace) and space.pose.combine == "multiply":
        return 1.0 + (torch.rand(N, space.width, generator=g) * 2 - 1) * 0.6  # multipliers about 1
    return (torch.rand(N, space.width, generator=g) * 2 - 1) * 0.8


@pytest.mark.parametrize("name", list(SPACES))
def test_invert_recovers_r_and_the_state_send_produced_over_an_episode(name):
    space = SPACES[name]()
    obs, base = observations()
    g = torch.Generator().manual_seed(1)
    online = space.initial_state(N, "cpu")
    offline = space.initial_state(N, "cpu")
    routed = torch.ones(N, dtype=torch.bool)
    for t in range(T):
        kw = dict(fresh=torch.full((N,), t == 0), new_chunk=torch.full((N,), t % H == 0), routed=routed)
        step, online = space.observe(obs[t], base[t], online, **kw)
        r = r_draw(space, g)
        sent, online = space.send(r, step, online)
        step_off, offline = space.observe(obs[t], base[t], offline, **kw)
        for key in ("policy", "side"):
            torch.testing.assert_close(step_off[key], step[key])
        before = offline
        r_back, offline = space.invert(sent, step_off, offline)
        if isinstance(space, PoseSpace) and space.pose.param == "rot6d":
            # rot6d: 6 numbers for 3 rotational degrees of freedom, Gram-Schmidt absorbs the rest -- the inverted r is one
            # that sends the same target (a right inverse), not necessarily the r drawn
            again, _ = space.send(r_back, step_off, before)
            torch.testing.assert_close(again, sent, atol=1e-5, rtol=0)
        elif isinstance(space, PoseSpace) and space.pose.combine == "multiply":
            # a dim whose base motion is under the inversion's floor reads as m = 1
            pose = space.pose
            big = step_off["d"].abs() >= torch.tensor([pose.eps_pos] * 3 + [pose.eps_rot] * (pose.width - 3))
            torch.testing.assert_close(r_back[big], r[big], atol=2e-3, rtol=0)
        else:
            torch.testing.assert_close(r_back, r, atol=2e-3, rtol=0)
        for key in online:
            torch.testing.assert_close(offline[key], online[key], atol=2e-4, rtol=0)


def test_episode_steps_rebuild_each_frame_and_r_through_the_space():
    space = PoseSpace(PoseResidual(observation_frame="relative", observation_rotation="relative", combine="multiply",
                                   action_target="target"), 10, critic_mode="observation")
    obs, base = observations(seed=3)
    g = torch.Generator().manual_seed(2)
    state = space.initial_state(N, "cpu")
    r_true, executed, steps = [], [], []
    for t in range(T):
        step, state = space.observe(obs[t], base[t], state, fresh=torch.full((N,), t == 0),
                                    new_chunk=torch.full((N,), t % H == 0), routed=torch.ones(N, dtype=torch.bool))
        r = 1.0 + (torch.rand(N, 6, generator=g) * 1.2 - 0.6)
        sent, state = space.send(r, step, state)
        r_true.append(r), executed.append(sent), steps.append(step)
    episodes = [{"states": obs[:, i], "base_actions": base[:, i], "actions": torch.stack(executed)[:, i],
                 "phase": torch.full((T,), 2)} for i in range(N)]
    out = episode_steps(space, episodes, H, lambda phase: torch.ones(phase.shape[0], dtype=torch.bool), "cpu")
    for i in range(N):
        torch.testing.assert_close(out[i]["policy"], torch.stack([s["policy"][i] for s in steps]))
        d = torch.stack([s["d"][i] for s in steps])
        big = d.abs() >= torch.tensor([space.pose.eps_pos] * 3 + [space.pose.eps_rot] * 3)
        torch.testing.assert_close(out[i]["r"][big], torch.stack(r_true)[:, i][big], atol=2e-3, rtol=0)


def _terms(**weights):
    return SimpleNamespace(**{k: SimpleNamespace(func=print, weight=w, time_out=False) for k, w in weights.items()})


def test_rewards_are_the_recorded_terms_times_the_runs_weights_and_unused_terms_stay_out():
    recorded = {"assembled": torch.tensor([0.0, 0.0, 1.0]), "insert_done": torch.tensor([0.0, 1.0, 0.0]),
                "step": torch.ones(3)}
    rewards = env_rewards(recorded, _terms(assembled=1.0, insert_done=0.5, step=0.0))
    assert rewards.tolist() == [0.0, 0.5, 1.0]
    with pytest.raises(ValueError, match="no reward term"):
        env_rewards(recorded, _terms(assembled=1.0, grasp_done=1.0))


def test_terminations_are_the_runs_terms_without_its_time_out():
    recorded = {"success": torch.tensor([False, False, True]), "time_out": torch.tensor([True, False, False]),
                "part_out_of_limits": torch.tensor([False, False, False])}
    section = SimpleNamespace(success=SimpleNamespace(func=print, time_out=False), time_out=SimpleNamespace(func=print, time_out=True),
                              part_out_of_limits=SimpleNamespace(func=print, time_out=False), off=None)
    assert env_terminations(recorded, section).tolist() == [False, False, True]


def test_a_routed_phase_keeps_its_steps_with_the_envs_reward_and_ends_at_the_phase_switch():
    T_ = 6
    step = {k: torch.arange(T_, dtype=torch.float32).reshape(T_, 1) for k in ("policy", "critic", "side", "r")}
    phase = torch.tensor([1, 2, 2, 2, 3, 3])  # grasp, insert x3, screw x2
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0])  # the env's insert_done and assembled terms
    terminated = torch.tensor([False, False, False, False, False, True])
    insert = episode_transitions(step, phase, rewards, terminated, index=2)
    screw = episode_transitions(step, phase, rewards, terminated, index=3)
    assert [float(v) for v in insert["rewards"]] == [0.0, 0.0, 1.0]  # copied, never recomputed
    assert [bool(v) for v in insert["terminated"]] == [False, False, True]  # the step into screw
    assert [float(v) for v in insert["next_observations"]] == [2.0, 3.0, 3.0]
    assert [bool(v) for v in screw["terminated"]] == [False, True]  # the env's success
    every = episode_transitions(step, phase, rewards, terminated, index=None)
    assert len(every["rewards"]) == T_ and sum(bool(v) for v in every["terminated"]) == 1  # no phase switch unrouted


def test_the_observation_form_check_takes_raw_episode_files_routed_or_not(tmp_path):
    from my_project.cfg import check_observation_form_demos
    from my_project.raw_demos import RAW_FORMAT

    raw = tmp_path / "raw.pt"
    torch.save({"format": RAW_FORMAT, "episodes": [], "meta": {}}, raw)
    check_observation_form_demos(str(raw))  # the demo loader builds the observation form at load
    ready = tmp_path / "ready.pt"
    torch.save({"transitions": {}, "meta": {"form": "observation"}}, ready)
    check_observation_form_demos(str(ready))
    other = tmp_path / "other.pt"
    torch.save({"transitions": {}, "meta": {"form": "residual"}}, other)
    with pytest.raises(ValueError, match="observation form"):
        check_observation_form_demos(str(other))
    check_observation_form_demos(str(tmp_path / "missing.pt"))  # not a file yet: left to the learner
