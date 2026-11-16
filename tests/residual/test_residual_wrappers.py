"""The residual wrappers' conversions and rules (``my_project/wrappers``), without an env: each step's observation goes
through the conversions as the wrappers run them, the agent's action back down through them, and a demo's executed
action through their inverses -- against the old code's fixtures (``tests/residual/fixtures``: 5 envs x 12 steps, chunk
starts every 4 steps with one env offset, an episode reset mid-run, steps in phases without a residual):

* ``multiply_increment_smol2.pt`` -- smol2_1_base's multiplied chain: drop columns (legs 2-4, the obstacle, the gripper
  action) + target tracking (the EE's pose, reset at every chunk) + the EE-relative observation + the multiply residual;
* ``absolute.pt`` -- the per-dim additive residual with a clamp on the sent action (phase_v7's rule) over dropped
  observation columns.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from my_project.wrappers.action_spaces.drop_columns import DropColumns
from my_project.wrappers.action_spaces.target_tracking import Target, TargetTracking
from my_project.wrappers.observation_spaces.ee_relative import EERelative
from my_project.wrappers.pose_math import exp_so3, log_so3, mat_to_rot, rot_to_mat
from my_project.wrappers.residual_policy.additive import AdditiveCriticAction, AdditiveResidual
from my_project.wrappers.residual_policy.multiply import MultiplyCriticAction, MultiplyResidual
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
SIDE = "residual_base"


def close(a, b):
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def additive(width, scale, clip=None):
    w = object.__new__(AdditiveResidual)
    w.action_dim, w.scale, w.clip, w.table, w.phase_names = width, scale, clip, None, []
    return w


def multiply(width, floor, r_range):
    w = object.__new__(MultiplyResidual)
    w.action_dim, w.floor, w.r_range = width, torch.as_tensor(floor, dtype=torch.float32), r_range
    return w


def build(conversions, obs_width, action_dim):
    critic = obs_width
    for c in conversions:
        c.build(obs_width, critic, action_dim)
        obs_width, critic, action_dim = c.obs_width, c.critic_width, c.action_dim
    return obs_width, critic, action_dim


def run(conversions, combine, data, r):
    """Every step online (observe, combine, act down) and offline (observe, invert up), each with its own states."""
    T, N = data["obs"].shape[:2]
    online = [c.initial_state(N, "cpu") for c in conversions]
    offline = [c.initial_state(N, "cpu") for c in conversions]
    out = {k: [] for k in ("policy", "critic", "base", "side", "sent", "invert_r")}
    for t in range(T):
        flags = dict(fresh=data["fresh"][t], new_chunk=data["new_chunk"][t])
        obs, base = data["obs"][t], data["base"][t]
        policy, critic, ctxs = obs, obs, []
        for k, c in enumerate(conversions):
            policy, critic, base_k, ctx, online[k] = c.observe(policy, critic, base, online[k], **flags)
            base = base_k
            ctxs.append(ctx)
        side = combine.side_of(base, None)
        action = combine.combine(r[t], side)
        for k in reversed(range(len(conversions))):
            action, online[k] = conversions[k].act(action, ctxs[k], online[k])
        # offline: the executed action up through the inverses
        policy_o, critic_o, base_o, octxs = obs, obs, data["base"][t], []
        for k, c in enumerate(conversions):
            policy_o, critic_o, base_o, octx, offline[k] = c.observe(policy_o, critic_o, base_o, offline[k], **flags)
            octxs.append(octx)
        executed = data["executed"][t]
        for k, c in enumerate(conversions):
            executed, offline[k] = c.invert(executed, octxs[k], offline[k])
        out["invert_r"].append(combine.invert(executed, combine.side_of(base_o, None)))
        for name, value in (("policy", policy), ("critic", critic), ("base", base), ("side", side), ("sent", action)):
            out[name].append(value)
    return {k: torch.stack(v) for k, v in out.items()}


def smol2_stack(data):
    conversions = [
        DropColumns([[25, 52], [61, 63]], [9]),
        TargetTracking([Target([0, 3], [0, 3], [3, 9], "rot6d", [3, 9], "rot6d")], reset_at_chunk=True,
                       target_in_policy=True, target_in_critic=False),
        EERelative(([0, 3], [3, 9], "rot6d"), [([16, 19], [19, 25], "rot6d"), ([25, 28], [28, 34], "rot6d")],
                   [[9, 12], [12, 15]]),
    ]
    widths = build(conversions, 63, 10)
    floor = [5e-4] * 3 + [math.radians(0.5)] * 3
    return conversions, multiply(6, floor, (data["low"], data["high"])), widths


def test_smol2s_chain_is_reproduced():
    data = torch.load(FIXTURES / "multiply_increment_smol2.pt", weights_only=False)
    conversions, combine, (obs_w, critic_w, act_w) = smol2_stack(data)
    assert (obs_w, critic_w, act_w) == (31, 25, 6)  # 25 kept columns + the current target; the 6 changes
    routed = data["routed"]
    # a phase without a residual gets the neutral action from the learner: the multiplier 1
    r = torch.where(routed.unsqueeze(-1), data["multiplier"], torch.ones_like(data["multiplier"]))
    got = run(conversions, combine, data, r)
    # the old wrapper sent the base itself where the phase had no residual; the chain with r = 1 sends the base's own
    # step from the current target, the same once the chain restarts at a chunk: compare the steps whose chunk saw no
    # step without a residual
    T, N = routed.shape
    clean = torch.zeros(T, N, dtype=torch.bool)
    dirty = torch.zeros(N, dtype=torch.bool)
    for t in range(T):
        start = data["fresh"][t] | data["new_chunk"][t]
        dirty = torch.where(start, torch.zeros_like(dirty), dirty)
        clean[t] = routed[t] & ~dirty
        dirty = dirty | ~routed[t]
    assert clean.sum() > 20
    close(got["sent"][clean], data["sent"][clean])
    close(got["invert_r"][clean], data["invert_multiplier"][clean])
    # the agent's observation: the old [kept 25, the base's step 6, its gripper 1, the current target 6] is now
    # [kept 25, the current target 6] then the base's step (the gripper is dropped: the base keeps it)
    close(got["policy"][..., :25], data["policy"][..., :25])
    close(got["policy"][..., 25:31][clean], data["policy"][..., 32:38][clean])
    close(got["base"][clean], data["policy"][..., 25:31][clean])
    close(got["critic"], data["critic"])


def test_the_additive_per_dim_rule_with_its_clamp_is_reproduced():
    data = torch.load(FIXTURES / "absolute.pt", weights_only=False)
    keep = data["keep"].tolist()
    drop = [[c, c + 1] for c in range(63) if c not in keep]
    conversions = [DropColumns(drop, [])]
    build(conversions, 63, 10)
    combine = additive(10, data["scale"], (data["clip_low"], data["clip_high"]))
    got = run(conversions, combine, data, data["r"])
    close(got["policy"], data["policy"][..., :25])
    close(got["base"], data["policy"][..., 25:])
    close(got["critic"], data["critic"])
    close(got["sent"], data["sent"])
    close(got["invert_r"], data["invert_r"])
    crit = AdditiveCriticAction(10, data["scale"])
    close(crit(None, data["r"].reshape(-1, 10), {SIDE: got["side"].reshape(-1, 10)}), data["critic_action"].reshape(-1, 10))


def test_dropped_action_dims_are_the_bases():
    c = DropColumns([[1, 3]], [0, 4])
    c.build(6, 6, 5)
    obs, base = torch.arange(12.0).reshape(2, 6), torch.arange(10.0).reshape(2, 5) + 100
    policy, critic, b, ctx, _ = c.observe(obs, obs, base, {}, fresh=None, new_chunk=None)
    assert policy.tolist() == [[0, 3, 4, 5], [6, 9, 10, 11]] and critic.tolist() == policy.tolist()
    assert b.tolist() == [[101, 102, 103], [106, 107, 108]]
    inner, _ = c.act(torch.zeros(2, 3), ctx, {})
    assert inner.tolist() == [[100, 0, 0, 0, 104], [105, 0, 0, 0, 109]]
    assert c.invert(base, ctx, {})[0].tolist() == b.tolist()


def test_target_tracking_starts_at_the_body_and_moves_by_the_bases_change_per_target():
    """Two arms (an axis-angle pose each, absolute IK targets) and a finger joint group, as a multi-arm env has them."""
    torch.manual_seed(0)
    n = 4
    targets = [Target([0, 3], [0, 3], [3, 6], "rotvec", [3, 7], "quat_xyzw"),
               Target([6, 9], [7, 10], [9, 12], "rotvec", [10, 14], "quat_xyzw"),
               Target([12, 14], [14, 16])]
    c = TargetTracking(targets, reset_at_chunk=False, target_in_policy=True, target_in_critic=True)
    c.build(16, 16, 15)
    assert c.passthrough == [14] and c.action_dim == 6 + 6 + 2 + 1 and c.obs_width == 16 + 14
    obs = torch.randn(n, 16)
    for q in (slice(3, 7), slice(10, 14)):
        obs[:, q] = obs[:, q] / obs[:, q].norm(dim=-1, keepdim=True)
    base = torch.randn(n, 15) * 0.3
    state = c.initial_state(n, "cpu")
    yes, no = torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    policy, critic, d, ctx, state = c.observe(obs, obs, base, state, fresh=yes, new_chunk=no)
    # at a start the change is the base's target against the body itself
    torch.testing.assert_close(d[:, 0:3], base[:, 0:3] - obs[:, 0:3])
    R_body = rot_to_mat(obs[:, 3:7], "quat_xyzw")
    torch.testing.assert_close(exp_so3(d[:, 3:6]) @ R_body, exp_so3(base[:, 3:6]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(d[:, 12:14], base[:, 12:14] - obs[:, 14:16])
    torch.testing.assert_close(d[:, 14], base[:, 14])  # the pass-through dim as it is
    # the current target relative to its body is zero at a start
    torch.testing.assert_close(policy[:, 16:], torch.zeros(n, 14), atol=1e-6, rtol=0)
    # sending the base's own change sends the base's own targets
    sent, state = c.act(d, ctx, state)
    torch.testing.assert_close(sent, base, atol=1e-5, rtol=1e-5)
    # the next step goes on from the sent target: the change is the base's own step, and a zero change holds the target
    base2 = base + 0.05
    _, _, d2, ctx2, state = c.observe(obs, obs, base2, state, fresh=no, new_chunk=no)
    torch.testing.assert_close(d2[:, 0:3], torch.full((n, 3), 0.05))
    held, _ = c.act(torch.zeros_like(d2), ctx2, state)
    torch.testing.assert_close(held[:, :14], base[:, :14], atol=1e-5, rtol=1e-5)
    # invert recovers the change that sends a given target
    torch.testing.assert_close(c.invert(base2, ctx2, state)[0], d2, atol=1e-5, rtol=1e-5)


def test_the_ee_relative_observation_drops_the_ee_and_expresses_poses_and_vectors_in_its_frame():
    torch.manual_seed(1)
    n = 3
    obs = torch.randn(n, 20)
    obs[:, 3:7] = obs[:, 3:7] / obs[:, 3:7].norm(dim=-1, keepdim=True)
    obs[:, 10:14] = obs[:, 10:14] / obs[:, 10:14].norm(dim=-1, keepdim=True)
    c = EERelative(([0, 3], [3, 7], "quat_xyzw"), [([7, 10], [10, 14], "quat_xyzw")], [[14, 17]])
    c.build(20, 20, 7)
    assert c.obs_width == 13
    out, _, _, _, _ = c.observe(obs, obs, torch.zeros(n, 7), {}, fresh=None, new_chunk=None)
    R = rot_to_mat(obs[:, 3:7], "quat_xyzw")
    Rt = R.transpose(-1, -2)
    torch.testing.assert_close(out[:, 0:3], (Rt @ (obs[:, 7:10] - obs[:, 0:3]).unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(rot_to_mat(out[:, 3:7], "quat_xyzw"), Rt @ rot_to_mat(obs[:, 10:14], "quat_xyzw"),
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[:, 7:10], (Rt @ obs[:, 14:17].unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(out[:, 10:13], obs[:, 17:20])


def test_the_multiply_rule_and_its_critic_action():
    combine = multiply(3, [1e-3], (-1.0, 3.0))
    base = torch.tensor([[0.5, -0.2, 0.0]])
    r = torch.tensor([[2.0, 0.5, 3.0]])
    sent = combine.combine(r, base)
    torch.testing.assert_close(sent, torch.tensor([[1.0, -0.1, 0.0]]))
    # a dim with no base motion inverts to 1; a ratio past the range is clamped
    torch.testing.assert_close(combine.invert(torch.tensor([[1.0, -0.1, 0.3]]), base), torch.tensor([[2.0, 0.5, 1.0]]))
    assert combine.invert(torch.tensor([[5.0, -0.1, 0.0]]), base)[0, 0].item() == 3.0
    torch.testing.assert_close(MultiplyCriticAction(3)(None, r, {SIDE: base}), sent)


def test_the_additive_noise_factor_reads_the_base_after_the_observation_columns():
    w = additive(2, torch.tensor([0.5, 0.0]))
    w.obs_width = 3
    factor = w.noise_factor()
    observations = torch.tensor([[9.0, 9.0, 9.0, -1.0, 4.0, 7.0, 7.0]])  # 3 columns, the base (2), then a history (2)
    assert factor(observations).tolist() == [[2.0, 0.0]]


def test_a_per_phase_additive_scale_steps_each_env_with_its_phase_and_binds_each_routed_learners_noise():
    w = additive(2, None)
    w.phase_names = ["corner", "grasp", "insert", "screw"]
    w.table = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.5, 0.1], [0.2, 0.4]])
    w.obs_width = 1
    base = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    side = w.side_of(base, torch.tensor([2, 3]))  # env 0 in insert, env 1 in screw
    r = torch.ones(2, 2)
    torch.testing.assert_close(w.combine(r, side), torch.tensor([[1.5, 1.1], [1.2, 1.4]]))
    torch.testing.assert_close(w.invert(torch.tensor([[1.5, 1.1], [1.2, 1.4]]), side), r)
    crit = w.critic_action()
    torch.testing.assert_close(crit(None, r, {SIDE: side}), torch.tensor([[1.5, 1.1], [1.2, 1.4]]))
    factor = w.noise_factor()
    observations = torch.tensor([[9.0, 0.4, -0.2]])
    with pytest.raises(NotImplementedError, match="one residual per phase"):
        factor(observations)  # unbound: one learner for every phase cannot tell a row's phase
    torch.testing.assert_close(factor.for_route("insert")(observations), torch.tensor([[0.8, 2.0]]))
    torch.testing.assert_close(factor.for_route("screw")(observations), torch.tensor([[2.0, 0.5]]))
