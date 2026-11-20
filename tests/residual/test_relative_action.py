"""``relative_action`` and the noise pieces that must agree with ``data_analysis/demo_residual_analysis.ipynb``: the
relative action is the notebook's ``relative_to_ee`` (poses against the hand's observed pose with ``obs_rot_offset``,
fingers against the previous executed command), the slice fit is its last cell's, and the scaled residual's
``scaled_units: sent`` noise lands in the sent action unscaled. The notebook's code is copied here as it is, and the
package's is checked against it."""

import math

import numpy as np
import pytest
import torch

from my_project.wrappers.action_spaces.relative_action import RelativeAction, RelativePose
from my_project.wrappers.pose_math import mat_to_rot, rot_to_mat
from my_project.wrappers.residual_policy import noise_scale as ns

OFFSET = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]  # dexmg_coffee's obs_rot_offset (models.yaml)
COFFEE = {  # dexmg_coffee's relative_to_ee, reduced to one hand and three fingers on a 10-wide action
    "poses": [{"action_pos": [0, 3], "action_rot": [3, 6], "action_rot_param": "rotvec", "obs_pos": [0, 3],
               "obs_rot": [3, 7], "obs_rot_param": "quat_xyzw", "obs_rot_offset": OFFSET}],
    "fingers": [6, 7, 8],
}


def notebook_relative(base, executed, states, lengths, relative_to_ee):
    """``demo_residual_analysis.ipynb``'s cell 4, its RELATIVE_TO_EE branch, as written there (non-rot6d poses)."""
    base, executed, states = base.clone().double(), executed.clone().double(), states.double()
    episodes_actions = list(torch.split(executed.clone(), lengths))
    for pose in relative_to_ee.get("poses") or []:
        action_pos, obs_pos = slice(*pose["action_pos"]), slice(*pose["obs_pos"])
        base[:, action_pos] -= states[:, obs_pos]
        executed[:, action_pos] -= states[:, obs_pos]
        if pose.get("action_rot") is None:
            continue
        hand_R = rot_to_mat(states[:, slice(*pose["obs_rot"])], pose["obs_rot_param"])
        if pose.get("obs_rot_offset") is not None:
            hand_R = hand_R @ torch.tensor(pose["obs_rot_offset"], dtype=hand_R.dtype)
        action_rot, param = slice(*pose["action_rot"]), pose["action_rot_param"]
        for a in (base, executed):
            a[:, action_rot] = mat_to_rot(rot_to_mat(a[:, action_rot], param) @ hand_R.transpose(-1, -2), param)
    fingers = [int(c) for c in relative_to_ee.get("fingers") or []]
    if fingers:
        held = torch.cat([torch.cat([torch.zeros_like(e[:1]), e[:-1]]) for e in episodes_actions]).double()
        base[:, fingers] -= held[:, fingers]
        executed[:, fingers] -= held[:, fingers]
    return base, executed


def random_quat(n, g):
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def episodes(g, lengths, width=10):
    T = sum(lengths)
    states = torch.cat([torch.randn(T, 3, generator=g, dtype=torch.float64), random_quat(T, g),
                        torch.randn(T, 5, generator=g, dtype=torch.float64)], dim=1)
    absolute = torch.randn(T, width, generator=g, dtype=torch.float64)
    absolute[:, 3:6] = mat_to_rot(rot_to_mat(random_quat(T, g), "quat_xyzw"), "rotvec")  # valid targets
    return states, absolute


def conversion():
    poses = [RelativePose(p["action_pos"], p["obs_pos"], p["action_rot"], p["action_rot_param"], p["obs_rot"],
                          p["obs_rot_param"], p["obs_rot_offset"]) for p in COFFEE["poses"]]
    c = RelativeAction(poses, COFFEE["fingers"])
    c.build(12, 12, 10)
    return c


def run_episodes(c, states, base, executed, lengths):
    """Each episode stepped in order, as convert_episodes does (one env): the base's relative action at each frame and
    the executed action's, the state carried from the executed one."""
    out_b, out_e, t0 = [], [], 0
    for length in lengths:
        state = c.initial_state(1, "cpu")
        for t in range(t0, t0 + length):
            fresh = torch.tensor([t == t0])
            _, _, b_rel, ctx, state = c.observe(states[t:t + 1], states[t:t + 1], base[t:t + 1], state, fresh=fresh,
                                                new_chunk=fresh)
            e_rel, state = c.invert(executed[t:t + 1], ctx, state)
            out_b.append(b_rel)
            out_e.append(e_rel)
        t0 += length
    return torch.cat(out_b), torch.cat(out_e)


def test_the_relative_action_is_the_notebooks_relative_to_ee():
    g = torch.Generator().manual_seed(0)
    lengths = [7, 5, 9]
    states, executed = episodes(g, lengths)
    _, base = episodes(g, lengths)
    nb_base, nb_exec = notebook_relative(base, executed, states, lengths, COFFEE)
    b_rel, e_rel = run_episodes(conversion(), states, base, executed, lengths)
    torch.testing.assert_close(b_rel, nb_base, atol=1e-9, rtol=0)
    torch.testing.assert_close(e_rel, nb_exec, atol=1e-9, rtol=0)
    # and so the residual the noise is fitted on: executed - base, relative
    torch.testing.assert_close(e_rel - b_rel, nb_exec - nb_base, atol=1e-9, rtol=0)


def test_acting_maps_back_to_the_absolute_action_and_keeps_the_executed_finger_command():
    g = torch.Generator().manual_seed(1)
    states, absolute = episodes(g, [2])
    c = conversion()
    state = c.initial_state(1, "cpu")
    fresh = torch.tensor([True])
    _, _, b_rel, ctx, state = c.observe(states[:1], states[:1], absolute[:1], state, fresh=fresh, new_chunk=fresh)
    sent, state = c.act(b_rel, ctx, state)
    torch.testing.assert_close(sent, absolute[:1], atol=1e-9, rtol=0)  # the base's relative action sends the base's own
    torch.testing.assert_close(state["held"], absolute[:1, 6:9])
    # the next step's fingers are relative to that command; a new episode starts from 0
    _, _, b2, ctx2, _ = c.observe(states[1:2], states[1:2], absolute[1:2], state, fresh=torch.tensor([False]),
                                  new_chunk=torch.tensor([False]))
    torch.testing.assert_close(b2[:, 6:9], absolute[1:2, 6:9] - absolute[:1, 6:9])
    _, _, b3, _, _ = c.observe(states[1:2], states[1:2], absolute[1:2], state, fresh=torch.tensor([True]),
                               new_chunk=torch.tensor([True]))
    torch.testing.assert_close(b3[:, 6:9], absolute[1:2, 6:9])
    # an executed clamp bounds the absolute action, and the fingers keep the clamped command
    c.executed_clip = (torch.full((10,), -0.1, dtype=torch.float64), torch.full((10,), 0.1, dtype=torch.float64))
    big = torch.full((1, 10), 5.0, dtype=torch.float64)
    sent, state = c.act(big, ctx, c.initial_state(1, "cpu"))
    assert float(sent.abs().max()) <= 0.1 + 1e-12
    torch.testing.assert_close(state["held"], sent[:, 6:9])


def notebook_fit_slices(centres, values, counts):
    """The notebook's fit_slices with fit_degree 1, FIT_WEIGHTED True."""
    x, y = np.asarray(centres), np.asarray(values)
    w = np.asarray(counts, dtype=np.float64)
    powers = [1, 0]
    sw = np.sqrt(w)
    solved, *_ = np.linalg.lstsq(np.stack([x ** p for p in powers], axis=1) * sw[:, None], y * sw, rcond=None)
    mean = np.average(y, weights=w)
    ss_res = np.sum(w * (y - np.polyval(solved, x)) ** 2)
    ss_tot = np.sum(w * (y - mean) ** 2)
    return solved, 1.0 - ss_res / ss_tot


def notebook_spread(x, y, NUM_BINS=50, X_RANGE_QUANTILE=0.995, MIN_POINTS=5):
    """The notebook's spread_panel's slices with X_MODE "magnitude", STAT "std", FIT_TRIM_BINS (0, 0)."""
    lo, hi = 0.0, np.quantile(x, X_RANGE_QUANTILE)
    edges = np.linspace(lo, hi, NUM_BINS + 1)
    slice_of = np.digitize(x, edges[1:-1])
    inside = (x >= edges[0]) & (x <= edges[-1])
    centres, values, counts = [], [], []
    for k in range(len(edges) - 1):
        ys = y[inside & (slice_of == k)]
        if len(ys) >= MIN_POINTS:
            centres.append((edges[k] + edges[k + 1]) / 2)
            values.append(ys.std())
            counts.append(len(ys))
    return notebook_fit_slices(centres, values, counts)


def test_the_slice_fit_is_the_notebooks_last_cell():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20000, 3)) * [1.0, 0.1, 0.01]
    residual = (0.3 * np.abs(base) + [0.05, 0.01, 0.002]) * rng.normal(size=(20000, 3))
    fit = ns.slice_linear_fit(base, residual)
    for d in range(3):
        (a1, a0), r2 = notebook_spread(np.abs(base[:, d]), residual[:, d])
        assert fit["a1"][d] == pytest.approx(a1, rel=1e-12) and fit["a0"][d] == pytest.approx(a0, rel=1e-12)
        assert fit["r2"][d] == pytest.approx(r2, rel=1e-12)
    assert fit["a1"] == pytest.approx([0.3] * 3, rel=0.1)  # and it recovers the generating line
    # the plot scale changes a0 only: the notebook's a0 in mm is ours in m times 1000
    (a1_mm, a0_mm), _ = notebook_spread(np.abs(base[:, 0]) * 1000, residual[:, 0] * 1000)
    assert a1_mm == pytest.approx(fit["a1"][0], rel=1e-9) and a0_mm == pytest.approx(fit["a0"][0] * 1000, rel=1e-9)


def test_the_scaled_residuals_sent_unit_noise_is_added_to_r_times_base_unscaled():
    s = ns.NoiseScale([ns.LinearSource(3, 5, torch.tensor([0.0, 0.0]), torch.tensor([0.2, 0.2]))], [0, 0],
                      residual="multiply", warmup_scale=2.0, scaled_units="sent", base_cols=(3, 5))
    obs = torch.tensor([[9.0, 9.0, 9.0, 0.5, -0.1]])
    r_scale = s(obs, "train")
    torch.testing.assert_close(r_scale, torch.tensor([[0.4, 2.0]]))  # 0.2 / |base|
    torch.testing.assert_close(r_scale * obs[:, 3:5].abs(), torch.tensor([[0.2, 0.2]]))  # in the sent action: s itself
    torch.testing.assert_close(s(obs, "warmup"), 2.0 * r_scale)
    # a draw: (r_det + e / base) * base = r_det * base + e, e ~ N(0, s), whatever the base's size or sign
    torch.manual_seed(0)
    base = torch.tensor([[0.5, -0.1]]).expand(20000, 2)
    e = torch.randn(20000, 2) * s(torch.cat([torch.zeros(20000, 3), base], dim=1), "train")
    sent = (1.0 + e) * base
    assert (sent - base).std(dim=0).tolist() == pytest.approx([0.2, 0.2], rel=0.05)
    with pytest.raises(ValueError, match="scaled residual"):
        ns.NoiseScale([ns.LinearSource(3, 5, torch.zeros(2), torch.zeros(2))], [0, 0], residual="additive",
                      warmup_scale=1.0, scaled_units="sent", base_cols=(3, 5))
