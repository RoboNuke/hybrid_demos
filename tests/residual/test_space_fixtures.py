"""The rewritten action spaces against the old ones, recorded before the old ones were removed
(``tests/residual/fixtures/*.pt``: 5 envs x 12 steps, chunk starts every 4 steps with one env offset, an episode reset
mid-run, steps in unrouted phases). ``multiply_increment_smol2.pt`` is the old ``action_frame: multiply``,
``multiply_reference: increment`` with smol2_1_base's range ``[-1.5, 3.5]`` and observation columns; the old r was a unit
r in [-1, 1] mapped onto that range, the new r is the multiplier itself. ``absolute.pt`` is the old ``action_frame:
absolute`` with an executed clip. Observe / send / invert / the critics' action must agree to 1e-5; the one intended
change is that the critics now also observe the current target (``action_target: target``)."""

from pathlib import Path

import pytest
import torch

from my_project.action_spaces import PerDimSpace, PoseSpace
from my_project.pose_residual import PoseResidual

FIXTURES = Path(__file__).parent / "fixtures"


def run(space, data, r, critic_action):
    T, N = data["obs"].shape[:2]
    online, offline = space.initial_state(N, "cpu"), space.initial_state(N, "cpu")
    out = {k: [] for k in ("policy", "critic", "sent", "invert_r", "critic_action", "critic_action_invert")}
    for t in range(T):
        kw = dict(fresh=data["fresh"][t], new_chunk=data["new_chunk"][t], routed=data["routed"][t])
        step, online = space.observe(data["obs"][t], data["base"][t], online, **kw)
        sent, online = space.send(r[t], step, online)
        ostep, offline = space.observe(data["obs"][t], data["base"][t], offline, **kw)
        inv, offline = space.invert(data["executed"][t], ostep, offline)
        out["policy"].append(step["policy"])
        out["critic"].append(step["critic"])
        out["sent"].append(sent)
        out["invert_r"].append(inv)
        out["critic_action"].append(critic_action(step, r[t]))
        out["critic_action_invert"].append(critic_action(ostep, inv))
    return {k: torch.stack(v) for k, v in out.items()}


def close(a, b):
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_the_pose_space_reproduces_smol2s_multiply_chain():
    data = torch.load(FIXTURES / "multiply_increment_smol2.pt", weights_only=False)
    pose = PoseResidual(action_frame="absolute", action_rotation="absolute", observation_frame="relative",
                        observation_rotation="relative", rotation_param="rotvec", combine="multiply",
                        action_target="target", r_range=(data["low"], data["high"]))
    space = PoseSpace(pose, 10, keep=data["keep"], observe_base=True, critic_mode="observation")
    got = run(space, data, data["multiplier"], lambda step, r: pose.critic_action(step["side"], r))
    close(got["policy"], data["policy"])
    close(got["sent"], data["sent"])
    close(got["invert_r"], data["invert_multiplier"])
    close(got["critic_action"], data["critic_action"])
    close(got["critic_action_invert"], data["critic_action_invert"])
    old = data["critic"].shape[-1]
    close(got["critic"][..., :old], data["critic"])
    close(got["critic"][..., old:], data["policy"][..., -6:])  # the current target, the agent's last 6 columns


def test_the_per_dim_space_reproduces_the_absolute_space():
    data = torch.load(FIXTURES / "absolute.pt", weights_only=False)
    space = PerDimSpace(data["scale"], clip=(data["clip_low"], data["clip_high"]), keep=data["keep"], observe_base=True,
                        critic_mode="observation")
    got = run(space, data, data["r"], lambda step, r: step["side"] + space.scale * r)
    for name in ("policy", "critic", "sent", "invert_r", "critic_action", "critic_action_invert"):
        close(got[name], data[name])
