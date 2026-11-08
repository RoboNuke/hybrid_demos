"""residual.action_frame: multiply (my_project/residual_multiply.py): the multiplier's range and neutral point, the
increment chain (the current target, the EE at a chunk start), the critics' resulting-target action, the demo
inversion, the EE-relative observation, the raw demo episodes' load-time rows (my_project/demo_loader.py) and the
config's refusals."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from my_project.cfg import ResidualCfg
from my_project.models.action_frame import mat_to_rot6d, rot6d_to_mat
from my_project.residual import SIDE_INPUT, MultiplySideInput
from my_project.residual_ee import exp_so3
from my_project.residual_multiply import MultiplyResidual

NAMES = ["corner", "grasp", "insert", "screw", "success"]


def rot6d(v):
    return mat_to_rot6d(exp_so3(v))


def poses(n, seed=0):
    """An EE pose ``(n, 9)`` and base targets ``(n, 10)`` near it."""
    g = torch.Generator().manual_seed(seed)
    ee = torch.cat((torch.rand(n, 3, generator=g) * 0.2, rot6d(torch.randn(n, 3, generator=g) * 0.5)), dim=-1)
    base = torch.cat((ee[:, 0:3] + torch.randn(n, 3, generator=g) * 0.02,
                      mat_to_rot6d(exp_so3(torch.randn(n, 3, generator=g) * 0.1) @ exp_so3(torch.randn(n, 3, generator=g) * 0.5)),
                      torch.rand(n, 1, generator=g) * 2 - 1), dim=-1)
    return ee, base


def trajectory(T, seed=0):
    """A base chunk of ``T`` targets along a smooth path, and the EE where the chunk starts."""
    g = torch.Generator().manual_seed(seed)
    ee0 = torch.cat((torch.tensor([[0.5, 0.0, 0.1]]), rot6d(torch.randn(1, 3, generator=g) * 0.3)), dim=-1)
    steps = torch.cumsum(torch.randn(T, 3, generator=g) * 0.004, dim=0)
    turns = torch.cumsum(torch.randn(T, 3, generator=g) * 0.03, dim=0)
    R0 = rot6d_to_mat(ee0[:, 3:9])
    base = torch.cat((ee0[:, 0:3] + 0.01 + steps, mat_to_rot6d(exp_so3(turns) @ R0), torch.full((T, 1), 0.4)), dim=-1)
    return ee0, base


def mult(reference="increment"):
    return MultiplyResidual(-0.05, 1.25, reference)


def chain(m, base, ee0, r, routed=None):
    """The wrapper's increment chain over one chunk (one env, the EE held at the chunk's start pose): the sent targets
    and the per-step inputs."""
    T = base.shape[0]
    routed = torch.ones(T, dtype=torch.bool) if routed is None else routed
    prev_sent, prev_base = torch.cat((ee0, base[:1, 9:]), -1), base[:1]
    obs = torch.zeros(1, 63)
    obs[:, 0:9] = ee0
    sent_rows, inputs_rows = [], []
    for t in range(T):
        inputs = m.step_inputs(obs, base[t : t + 1], None, routed[t : t + 1], prev_sent=prev_sent, prev_base=prev_base,
                               chunk_start=torch.tensor([t == 0]))
        sent = m.sent(base[t : t + 1], inputs["d"], inputs["ref"], r[t : t + 1], routed[t : t + 1])
        prev_sent, prev_base = sent, base[t : t + 1]
        sent_rows.append(sent)
        inputs_rows.append(inputs)
    return torch.cat(sent_rows), inputs_rows


def test_r_spans_the_range_and_the_neutral_r_is_the_base():
    m = mult()
    assert float(m.multiplier(torch.tensor(-1.0))) == pytest.approx(-0.05)
    assert float(m.multiplier(torch.tensor(1.0))) == pytest.approx(1.25)
    assert float(m.multiplier(torch.tensor(m.neutral))) == pytest.approx(1.0)
    ee0, base = trajectory(8)
    sent, _ = chain(m, base, ee0, torch.full((8, 6), m.neutral))
    torch.testing.assert_close(sent, base, atol=1e-5, rtol=0)  # m = 1 every step: the base's own trajectory
    ee, b = poses(6, seed=1)
    e = mult("ee")
    d, ref = e.motion(b, ee)
    torch.testing.assert_close(e.sent(b, d, ref, torch.full((6, 6), e.neutral), torch.ones(6, dtype=torch.bool)), b, atol=1e-5, rtol=0)


def test_m_zero_holds_the_current_target_and_a_held_step_never_grows_the_next():
    m = mult()
    ee0, base = trajectory(8, seed=2)
    held, _ = chain(m, base, ee0, torch.full((8, 6), m.unit(0.0)))
    torch.testing.assert_close(held[:, 0:9], ee0.expand(8, 9), atol=1e-5, rtol=0)  # never leaves the EE
    r = torch.full((8, 6), m.neutral)
    r[1:4] = m.unit(0.0)
    sent, _ = chain(m, base, ee0, r)
    torch.testing.assert_close(sent[4, 0:3] - sent[3, 0:3], base[4, 0:3] - base[3, 0:3], atol=1e-5, rtol=0)
    # an env whose phase has no residual sends the base whatever r is
    off, _ = chain(m, base, ee0, torch.full((8, 6), -1.0), routed=torch.zeros(8, dtype=torch.bool))
    assert torch.equal(off, base)


def test_the_current_target_is_zero_at_a_chunk_start_and_the_last_sent_target_after():
    m = mult()
    ee0, base = trajectory(4, seed=3)
    r = torch.rand(4, 6) * 2 - 1
    sent, inputs = chain(m, base, ee0, r)
    width = 63  # no columns dropped here
    current = [i["policy"][:, width + 7:] for i in inputs]
    torch.testing.assert_close(current[0], torch.zeros(1, 6), atol=1e-6, rtol=0)
    torch.testing.assert_close(current[2], m.ee_frame(sent[1:2], ee0), atol=1e-6, rtol=0)
    assert inputs[0]["policy"].shape == (1, width + 7 + 6) and inputs[0]["critic"].shape == (1, width)


def test_the_critics_action_is_the_resulting_target_relative_to_the_ee():
    routed = torch.tensor([True, True, False, True, True])
    r = torch.rand(5, 6) * 2 - 1
    # increment: the chain's side inputs
    m = mult()
    ee0, base = trajectory(5, seed=4)
    sent, inputs = chain(m, base, ee0, r, routed)
    cases = [(m, torch.cat([i["side"] for i in inputs]), sent, ee0.expand(5, 9))]
    # ee: one step from the EE
    e = mult("ee")
    ee, b = poses(5, seed=5)
    obs = torch.zeros(5, 63)
    obs[:, 0:9] = ee
    ins = e.step_inputs(obs, b, None, routed)
    cases.append((e, ins["side"], e.sent(b, ins["d"], ins["ref"], r, routed), ee))
    for mm, side, sent_rows, ee_rows in cases:
        transform = MultiplySideInput(mm)
        assert transform.critic_action_width == 7 and transform.side_inputs == {SIDE_INPUT: mm.side_width}
        rg = r.clone().requires_grad_(True)
        out = transform(None, rg, {SIDE_INPUT: side})
        expected = torch.cat((mm.ee_frame(sent_rows, ee_rows), sent_rows[:, 9:10]), dim=-1)
        torch.testing.assert_close(out, expected, atol=1e-4, rtol=0)
        out.sum().backward()
        assert torch.isfinite(rg.grad).all() and torch.all(rg.grad[~routed] == 0)


def test_the_demo_inversion_gives_back_the_multipliers():
    m = mult()
    ee0, base = trajectory(8, seed=6)
    r = torch.rand(8, 6) * 1.6 - 0.8
    demo, inputs = chain(m, base, ee0, r)
    d, ref = torch.cat([i["d"] for i in inputs]), torch.cat([i["ref"] for i in inputs])
    back = m.invert(d, demo, ref)
    big = d.abs() >= torch.tensor([m.eps_pos] * 3 + [m.eps_rot] * 3)
    torch.testing.assert_close(back[big], r[big], atol=3e-3, rtol=0)
    assert torch.allclose(back[~big], torch.full_like(back[~big], m.neutral))  # no base motion: m = 1


def test_the_observation_is_fbrels_relative_to_the_ee_with_the_ee_pose_kept():
    from furniture_lab.demos import ee_relative_states

    g = torch.Generator().manual_seed(4)
    n = 5
    obs = torch.randn(n, 63, generator=g) * 0.1
    obs[:, 3:9] = rot6d(torch.randn(n, 3, generator=g))
    for p in range(5):
        obs[:, 16 + 9 * p + 3 : 16 + 9 * p + 9] = rot6d(torch.randn(n, 3, generator=g))
    rel = MultiplyResidual.observation(obs)
    fbrel = torch.tensor(ee_relative_states(obs.numpy().astype(np.float64)))
    assert rel.shape == obs.shape
    torch.testing.assert_close(rel[:, 0:9], obs[:, 0:9])
    torch.testing.assert_close(rel[:, 9:61], fbrel[:, 0:52], atol=1e-5, rtol=0)
    torch.testing.assert_close(rel[:, 61:63], obs[:, 61:63])


def test_the_raw_demo_rows_follow_the_chain_and_the_routes_rewards():
    from my_project.demo_loader import episode_rows

    m = mult()
    T, H = 8, 4
    ee0, base = trajectory(T, seed=7)
    labels = torch.tensor([1, 2, 2, 2, 2, 3, 3, 3])  # grasp, insert x4, screw x3 (assembled on the last frame)
    routed = torch.tensor([NAMES[int(p)] in ("insert", "screw") for p in labels])
    r_true = torch.rand(T, 6) * 1.4 - 0.7
    # the demo: the chain run with r_true where routed, every chunk (frames 0, 4) from the EE at that frame
    states = torch.zeros(T, 63)
    states[:, 0:9] = ee0
    demo = torch.zeros(T, 10)
    prev_sent, prev_base = torch.cat((ee0, base[:1, 9:]), -1), base[:1]
    for t in range(T):
        ins = m.step_inputs(states[t : t + 1], base[t : t + 1], None, routed[t : t + 1], prev_sent=prev_sent,
                            prev_base=prev_base, chunk_start=torch.tensor([t % H == 0]))
        demo[t] = m.sent(base[t : t + 1], ins["d"], ins["ref"], r_true[t : t + 1], routed[t : t + 1])[0]
        prev_sent, prev_base = demo[t : t + 1], base[t : t + 1]
    episode = {"states": states, "actions": demo, "base_actions": base, "success_frame": T - 1}
    insert = episode_rows(m, episode, labels, NAMES, ["insert", "screw"], "insert", H, None)
    screw = episode_rows(m, episode, labels, NAMES, ["insert", "screw"], "screw", H, None)
    assert insert["actions"].shape[0] == 4 and screw["actions"].shape[0] == 3
    d = torch.cat([insert["residual_base"][:, 9:15]])
    big = d.abs() >= torch.tensor([m.eps_pos] * 3 + [m.eps_rot] * 3)
    torch.testing.assert_close(insert["actions"][big], r_true[1:5][big], atol=3e-3, rtol=0)
    assert insert["rewards"].reshape(-1).tolist() == [0.0, 0.0, 0.0, 1.0]  # insert passed into screw on its last frame
    assert insert["terminated"].reshape(-1).tolist() == [False, False, False, True]
    assert screw["rewards"].reshape(-1).tolist() == [0.0, 0.0, 1.0] and bool(screw["terminated"][-1])  # assembled
    torch.testing.assert_close(insert["next_observations"][0], insert["observations"][1])


def _cfg(**over):
    half = 0.65
    ddpg = SimpleNamespace(actor_action_scale=1.0, action_clip=1.0, random_action_mode="actor_gaussian",
                           random_action_scale=0.05 / half, exploration_std=0.025 / half, exploration_std_end=None,
                           warmup_episodes=0, route_labels=["insert", "screw"])
    actor = SimpleNamespace(output_init_scale=0.0, output_bias=math.atanh(1.05 / half - 1))
    for k, v in over.items():
        setattr(actor if k.startswith("output") else ddpg, k, v)
    return SimpleNamespace(trainer=SimpleNamespace(learner="ddpg"), ddpg=ddpg, model=SimpleNamespace(actor=actor))


def _rc(**over):
    kw = dict(enabled=True, action_frame="multiply", multiply_reference="increment", critic_state="observation",
              critic_action="full", observation_drop=[[0, 9], [25, 52], [61, 63]])
    kw.update(over)
    return ResidualCfg(**kw)


def test_the_config_holds_the_learner_to_the_multipliers_units():
    _rc()._validate_multiply(_cfg())
    with pytest.raises(ValueError, match="exploration_std"):
        _rc()._validate_multiply(_cfg(exploration_std=0.025))
    with pytest.raises(ValueError, match="output_bias"):
        _rc()._validate_multiply(_cfg(output_bias=None))
    with pytest.raises(ValueError, match="random_action_mode"):
        _rc()._validate_multiply(_cfg(random_action_mode="uniform"))
    with pytest.raises(ValueError, match="obstacle"):
        _rc(observation_drop=[[25, 52]])._validate_multiply(None)
    with pytest.raises(ValueError, match="phase_scales"):
        _rc(phase_scales=True, phase_stats="x.pt")._validate_multiply(None)
    with pytest.raises(ValueError, match="hold 1"):
        _rc(multiply_min=1.0)._validate_multiply(None)
    with pytest.raises(ValueError, match="multiply_reference"):
        _rc(multiply_reference="target")._validate_multiply(None)
