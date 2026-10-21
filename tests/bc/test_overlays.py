"""The one_leg video overlay and the waypoint colours, on synthetic frames (no sim).

Pinned: the overlay refuses an env without the phase channel and a frame it was not handed the
episode for; each phase's tests read measurement/goal (green once passed); a passed phase's box
freezes at the step it passed while the next one opens below it; the timeline marks every
phase's start from the first frame; the middle of the frame is left alone; the waypoints are
yellow (now), grey (done), green (still to come) and red (never executed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robonuke_rl_core.recording import HudOverlay, StepData

from my_project.bc.eval_policy import CURRENT, DONE, PLANNED, UPCOMING, waypoint_roles, waypoint_roles_base
from my_project.overlays import GREEN, PHASE_COLORS, WHITE, OneLegOverlay, passed_steps, phase_lines, phase_starts

H, W = 360, 480

GOALS = {
    "phase_goal/top_grip_mm": 20.0, "phase_goal/lift_mm": 30.0, "phase_goal/held_mm": 30.0,
    "phase_goal/hole_mm": 13.0, "phase_goal/tilt_deg": 16.0, "phase_goal/open_mm": 50.0,
    "phase_goal/hold_steps": 10.0, "phase_goal/force_N": 50.0, "phase/max_steps": 100.0,
}


def state(phase: int, **values) -> dict:
    base = {
        "phase/id": float(phase), "phase/corner_mm": 4.0, "phase/top_held": 0.0, "phase/top_grip_mm": 65.0,
        "phase/lift_mm": 12.0, "phase/held_mm": 8.0, "phase/hole_mm": 3.0, "phase/tilt_deg": 20.0,
        "phase/open_mm": 70.0, "phase/hold_steps": 4.0, "phase/height_mm": 9.0, "phase/ori_deg": 2.0,
        "phase/turned_deg": 95.0, "phase/track_mm": 3.2, "phase/force_N": 20.0,
    }
    base.update({f"phase/{k}": v for k, v in values.items()})
    return {**base, **GOALS}


def step(t: int, phase: int, env: int = 0, **values) -> StepData:
    return StepData(env=env, step=t, reward=0.0, total_reward=0.0, valid=True, terminated=False, truncated=False,
                    state=state(phase, **values))


def frame() -> np.ndarray:
    return np.full((H, W, 3), 100, dtype=np.uint8)


def episode(phases, env: int = 0, **values) -> list:
    return [step(t, p, env, lift_mm=float(t), **values) for t, p in enumerate(phases)]


def test_the_overlay_needs_the_phase_channel_and_the_episode():
    bare = StepData(env=0, step=0, reward=0, total_reward=0, valid=True, terminated=False, truncated=False)
    with pytest.raises(RuntimeError, match="track_phases"):
        OneLegOverlay().start_episode([bare])
    with pytest.raises(RuntimeError, match="start_episode"):
        OneLegOverlay().apply(frame(), step(0, 0))  # never handed the episode
    overlay = OneLegOverlay()
    overlay.start_episode(episode([0, 0]))
    with pytest.raises(RuntimeError, match="start_episode"):
        overlay.apply(frame(), step(0, 0, env=1))  # another env's frame


def test_each_phase_shows_its_tests_as_measurement_over_goal_green_once_passed():
    corner = phase_lines(state(0), 0)
    assert corner == [("grip top 65.0/20 mm", WHITE), ("open 70/50 mm", WHITE), ("corner 4.0 mm", WHITE)]
    held = phase_lines(state(0, top_held=1.0, top_grip_mm=6.7), 0)
    assert held[:2] == [("grip top 6.7/20 mm", GREEN), ("open 70/50 mm", GREEN)]  # let go of the top
    grasp = dict(phase_lines(state(1), 1))
    assert grasp["lift 1.2/3.0 cm"] == WHITE and grasp["hand-leg 0.8/3.0 cm"] == GREEN
    insert = dict(phase_lines(state(2), 2))
    assert insert["tilt 20.0/16 deg"] == WHITE and insert["open 70/50 mm"] == GREEN and insert["hold 4/10"] == WHITE
    # Screw: the seated height is 0, the turn has no goal, no orientation line
    assert phase_lines(state(3), 3) == [("height 9.0/0 mm", WHITE), ("turned 95 deg", WHITE)]


def test_phase_starts_and_the_steps_phases_were_passed_on():
    phases = [0, 0, 1, 1, 3, 3, 4]  # Grasp -> Screw in one step (Insert passed with it), then success
    assert phase_starts(phases) == [(0, 0), (2, 1), (4, 3), (6, 4)]
    assert passed_steps(phases) == {0: 2, 1: 4, 2: 4, 3: 6}
    assert passed_steps([0, 1, 2, 2, 6]) == {0: 1, 1: 2}  # timed out in Insert: Insert never passed


def test_a_passed_phase_freezes_and_the_next_opens_below_it():
    overlay = OneLegOverlay()
    steps = episode([0, 0, 1, 1, 1, 4])
    overlay.start_episode(steps)
    titles = lambda t: [lines[0][0] for lines in overlay.boxes(steps[t])]  # noqa: E731
    assert titles(1) == ["CORNER"]
    assert titles(3) == ["CORNER (t=2)", "GRASP"]
    assert titles(5) == ["CORNER (t=2)", "GRASP (t=5)", "INSERT (t=5)", "SCREW (t=5)", "SUCCESS"]
    # the Grasp box at step 4 shows step 4's lift; Corner's is frozen at step 2's values
    grasp = overlay.boxes(steps[4])[1]
    assert grasp[1][0] == "lift 0.4/3.0 cm"


def test_the_timeline_marks_every_phase_start_from_the_first_frame():
    overlay = OneLegOverlay()
    phases = [0] * 20 + [1] * 40 + [2] * 40
    overlay.start_episode(episode(phases))
    out = overlay.apply(frame(), step(0, 0))
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert (out[H // 2, W // 2] == 100).all()  # the scene itself
    assert (out[: HudOverlay.banner_height, :] == 100).all()  # the hud banner is the hud's
    x = lambda t: int(round(t / 100 * (W - 1)))  # noqa: E731
    assert tuple(out[H - 2, x(20)]) == WHITE and tuple(out[H - 2, x(60)]) == WHITE  # both starts, already
    assert tuple(out[H - 2, x(40)]) != PHASE_COLORS[1]  # not filled before it happens
    later = overlay.apply(frame(), step(50, 1))
    assert tuple(later[H - 2, x(40)]) == PHASE_COLORS[1] and tuple(later[H - 2, x(10)]) == PHASE_COLORS[0]


def test_waypoints_are_yellow_now_grey_done_green_to_come_red_never():
    base = waypoint_roles_base(6, 3)
    assert base.tolist() == [UPCOMING] * 3 + [PLANNED] * 3
    # actions() has moved index past the waypoint it returned: env 0 just returned waypoint 0,
    # env 1 waypoint 2 (the last executed one)
    roles = waypoint_roles(base, torch.tensor([1, 3]))
    assert roles[0].tolist() == [CURRENT, UPCOMING, UPCOMING, PLANNED, PLANNED, PLANNED]
    assert roles[1].tolist() == [DONE, DONE, CURRENT, PLANNED, PLANNED, PLANNED]
