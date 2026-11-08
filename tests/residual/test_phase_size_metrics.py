"""The residual's correction sizes per task phase (``residual/<size>_<phase>``) when the env publishes its phases."""

from __future__ import annotations

import math

import torch

from my_project.residual import PHASE_NAMES, ResidualPolicyWrapper


def a_wrapper(n):
    w = object.__new__(ResidualPolicyWrapper)
    w.phase = torch.zeros(n, dtype=torch.long)
    w.obs_phase = None
    return w


def infos(phase, position, rot):
    return {"eval_state": {"phase/id": torch.tensor(phase, dtype=torch.float32)},
            "metrics_to_log": {"residual/position_mm": torch.tensor(position), "residual/rot_deg": torch.tensor(rot),
                               "other/thing": torch.tensor([1.0, 2.0, 3.0])}}


def test_each_step_is_logged_under_the_phase_its_action_was_chosen_in():
    w = a_wrapper(3)
    first = infos([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    w._phase_metrics(first, None)  # no phase known yet for the acting observation: nothing split
    assert not any(name.startswith("residual/position_mm_") for name in first["metrics_to_log"])
    assert w.obs_phase.tolist() == [1, 2, 3]
    # this step's actions were chosen in grasp, insert, screw; env 2's episode ends (code 4): its next phase is corner
    step = infos([2.0, 2.0, 4.0], [5.0, 6.0, 7.0], [0.5, 0.6, 0.7])
    w._phase_metrics(step, w.obs_phase)
    m = step["metrics_to_log"]
    nan = math.nan
    def same(name, expected):
        got = m[name].tolist()
        assert all((math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-6 for a, b in zip(got, expected)), (name, got)
    same("residual/position_mm_corner", [nan, nan, nan])
    same("residual/position_mm_grasp", [5.0, nan, nan])
    same("residual/position_mm_insert", [nan, 6.0, nan])
    same("residual/position_mm_screw", [nan, nan, 7.0])
    same("residual/rot_deg_insert", [nan, 0.6, nan])
    assert not any(name.startswith("other/thing_") for name in m)  # only the correction sizes are split
    assert w.obs_phase.tolist() == [2, 2, 0]
    assert {f"residual/position_mm_{p}" for p in PHASE_NAMES} <= set(m)


def test_without_published_phases_nothing_is_added():
    w = a_wrapper(2)
    plain = {"metrics_to_log": {"residual/position_mm": torch.tensor([1.0, 2.0])}}
    w._phase_metrics(plain, torch.tensor([2, 3]))
    assert set(plain["metrics_to_log"]) == {"residual/position_mm"} and w.obs_phase is None
