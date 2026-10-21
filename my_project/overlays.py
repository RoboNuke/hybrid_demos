"""The ``one_leg`` video overlay (``eval.overlays: [hud, one_leg]``): the task phases, what each is
judged by, a force bar and the episode's phase timeline.

It reads the furniture env's phase channel, ``furniture.track_phases``
(Robonuke_Furniture_Bench ``furniture_lab/tasks/assembly/phases.py``), which arrives per env as
``step.state`` (the env's ``infos["eval_state"]``); without it the overlay raises. It is a
``whole_episode`` overlay: the recorder hands it the episode's steps before its first frame
(:meth:`OneLegOverlay.start_episode`), so the timeline can mark phases still to come.

* Top left, under the ``hud`` banner: one box per phase reached so far (Corner, Grasp, Insert,
  Screw), each with what the phase is judged by as ``{measurement}/{goal}`` -- green once that
  test passes. A phase's box freezes on the step it passes, showing the values it passed with
  (``(t=<step>)`` in its title), and the next phase's box opens below it; on the step the
  episode ends, a last box says how (success / failure / timeout).
* Bottom left: what is always measured -- the tracking error (the action's target position
  against the TCP).
* Right edge: the estimated EE force's magnitude, a bar full at ``furniture.phase_force_scale``.
* Bottom: the timeline of the whole episode (``max_episode_length`` steps) with a line at every
  phase's start from the first frame on, filled in the phase colours up to now, and a marker at
  the current step.

Laid out for the one_leg_phases eval config's 480x360; smaller frames crowd it.

Registered at import (``@register_overlay``): the entry scripts import this in ``setup()``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from robonuke_rl_core.recording import HudOverlay, Overlay, StepData, register_overlay

__all__ = ["OneLegOverlay", "PHASE_NAMES", "PHASE_COLORS", "phase_lines", "phase_starts", "passed_steps"]

#: phase ids as furniture_lab.tasks.assembly.phases publishes them: 0-3 the task phases, then
#: how the episode ended (on its last step only)
PHASE_NAMES = ("corner", "grasp", "insert", "screw", "success", "failure", "timeout")
TASK_PHASES, SUCCESS = 4, 4
PHASE_COLORS = (
    (80, 140, 255),  # corner
    (255, 170, 40),  # grasp
    (190, 100, 255),  # insert
    (40, 210, 210),  # screw
    (60, 220, 60),  # success
    (235, 70, 60),  # failure
    (230, 200, 60),  # timeout
)
WHITE, GREEN, RED = (235, 235, 235), (90, 230, 90), (240, 80, 70)


def _v(state: Dict[str, Any], key: str) -> float:
    value = state.get(key, math.nan)
    return float(value) if not hasattr(value, "__len__") else math.nan


def _phase(step: StepData) -> int:
    if "phase/id" not in step.state:
        raise RuntimeError(
            "the one_leg overlay reads the env's phase channel, but eval_state has no 'phase/id': "
            "set task.cfg.furniture.track_phases: true (configs/eval/one_leg_phases.yaml)"
        )
    phase = int(round(_v(step.state, "phase/id")))
    if not 0 <= phase < len(PHASE_NAMES):
        raise ValueError(f"phase/id {phase} is not one of {PHASE_NAMES}")
    return phase


def phase_lines(state: Dict[str, Any], phase: int) -> List[Tuple[str, Tuple[int, int, int]]]:
    """``(text, colour)`` lines for task phase ``phase`` (0-3): its tests as
    ``{measurement}/{goal}``, green once passed, and what it only measures (white)."""
    s = lambda key: _v(state, f"phase/{key}")  # noqa: E731
    g = lambda key: _v(state, f"phase_goal/{key}")  # noqa: E731

    def below(text, key):
        return text, GREEN if s(key) <= g(key) else WHITE

    def above(text, key):
        return text, GREEN if s(key) >= g(key) else WHITE

    if phase == 0:
        held = s("top_held") >= 0.5
        return [
            (f"grip top {s('top_grip_mm'):.1f}/{g('top_grip_mm'):.0f} mm", GREEN if held else WHITE),
            (f"open {s('open_mm'):.0f}/{g('open_mm'):.0f} mm", GREEN if held and s("open_mm") >= g("open_mm") else WHITE),
            (f"corner {s('corner_mm'):.1f} mm", WHITE),
        ]
    if phase == 1:
        return [
            above(f"lift {s('lift_mm') / 10:.1f}/{g('lift_mm') / 10:.1f} cm", "lift_mm"),
            below(f"hand-leg {s('held_mm') / 10:.1f}/{g('held_mm') / 10:.1f} cm", "held_mm"),
        ]
    if phase == 2:
        return [
            below(f"hole {s('hole_mm'):.1f}/{g('hole_mm'):.0f} mm", "hole_mm"),
            below(f"tilt {s('tilt_deg'):.1f}/{g('tilt_deg'):.0f} deg", "tilt_deg"),
            above(f"open {s('open_mm'):.0f}/{g('open_mm'):.0f} mm", "open_mm"),
            above(f"hold {s('hold_steps'):.0f}/{g('hold_steps'):.0f}", "hold_steps"),
        ]
    if phase == 3:
        # the leg's height above its seat (0 is seated) and its turn since Screw began, positive
        # screwing in: neither has a pass test of its own -- Screw ends with the task's success
        return [
            (f"height {s('height_mm'):.1f}/0 mm", WHITE),
            (f"turned {s('turned_deg'):.0f} deg", WHITE),
        ]
    raise ValueError(f"phase {phase} is not a task phase (0-3)")


def phase_starts(phases: List[int]) -> List[Tuple[int, int]]:
    """``[(step, phase id)]`` at the episode's first step and at every change after it."""
    starts: List[Tuple[int, int]] = []
    for step, phase in enumerate(phases):
        if not starts or starts[-1][1] != phase:
            starts.append((step, phase))
    return starts


def passed_steps(phases: List[int]) -> Dict[int, int]:
    """``{task phase: the step it was passed on}``: the first step showing a later task phase, or
    success (which passes Screw and everything before it)."""
    passed: Dict[int, int] = {}
    for step, phase in enumerate(phases):
        for p in range(TASK_PHASES):
            if p not in passed and (p < phase < TASK_PHASES or phase == SUCCESS):
                passed[p] = step
    return passed


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: one small bitmap font
        return ImageFont.load_default()


@register_overlay
class OneLegOverlay(Overlay):
    """See the module docstring. Holds the episode being drawn (:meth:`start_episode`)."""

    name = "one_leg"
    whole_episode = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._env: Optional[int] = None
        self._phases: List[int] = []
        self._states: List[Dict[str, Any]] = []
        self._starts: List[Tuple[int, int]] = []
        self._passed: Dict[int, int] = {}
        self._reached: List[int] = []

    def start_episode(self, steps: List[StepData]) -> None:
        self._env = steps[0].env if steps else None
        self._phases = [_phase(step) for step in steps]
        self._states = [step.state for step in steps]
        self._starts = phase_starts(self._phases)
        self._passed = passed_steps(self._phases)
        # the furthest task phase by each step (Screw on success; the one it ended in otherwise)
        self._reached, furthest = [], 0
        for phase in self._phases:
            furthest = TASK_PHASES - 1 if phase == SUCCESS else (max(furthest, phase) if phase < TASK_PHASES else furthest)
            self._reached.append(furthest)

    def boxes(self, step: StepData) -> List[List[Tuple[str, Tuple[int, int, int]]]]:
        """The top-left boxes at ``step``: one per phase reached, its title then its lines -- a
        passed phase's frozen at the step it passed -- and, on the last step, how it ended."""
        now, phase = step.step, _phase(step)
        out = []
        for p in range(self._reached[now] + 1):
            done_at = self._passed.get(p)
            frozen = done_at is not None and done_at <= now
            values = self._states[done_at] if frozen else step.state
            title = PHASE_NAMES[p].upper() + (f" (t={done_at})" if frozen else "")
            out.append([(title, PHASE_COLORS[p])] + phase_lines(values, p))
        if phase >= TASK_PHASES:
            out.append([(PHASE_NAMES[phase].upper(), PHASE_COLORS[phase])])
        return out

    def apply(self, frame: Any, step: StepData) -> Any:
        import numpy as np
        from PIL import Image, ImageDraw

        if step.env != self._env or step.step >= len(self._phases):
            raise RuntimeError(
                "the one_leg overlay draws with the whole episode known: the recorder calls "
                "start_episode(steps) for each (round, env) before its frames (whole_episode)"
            )
        now = step.step

        image = Image.fromarray(np.ascontiguousarray(frame)).convert("RGBA")
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        W, H = image.size
        size = max(10, H // 28)
        font = _font(size)
        pad = max(2, size // 4)
        bar_w = max(6, W // 40)
        timeline_h = max(6, H // 30)
        line_h = size + pad

        def box(top: float, lines: List[Tuple[str, Tuple[int, int, int]]], left: float = 0) -> float:
            width = max(draw.textlength(text, font=font) for text, _ in lines) + 2 * pad
            bottom = top + len(lines) * line_h + pad
            draw.rectangle([(left, top), (left + width, bottom)], fill=(0, 0, 0, 150))
            for k, (text, color) in enumerate(lines):
                draw.text((left + pad, top + pad + k * line_h), text, font=font, fill=color + (255,))
            return bottom

        # -- one box per phase reached, top left under the hud banner; a passed one frozen
        top = HudOverlay.banner_height + pad
        for lines in self.boxes(step):
            top = box(top, lines) + pad

        # -- what is always measured, bottom left above the timeline
        t0 = H - timeline_h
        track = _v(step.state, "phase/track_mm")
        if not math.isnan(track):
            box(t0 - 2 * pad - 1 - (line_h + pad), [(f"tracking error {track:.1f} mm", WHITE)])

        # -- the force bar along the right edge
        force, scale = _v(step.state, "phase/force_N"), _v(step.state, "phase_goal/force_N")
        x1, x0 = W - pad, W - pad - bar_w
        y0, y1 = HudOverlay.banner_height + pad + line_h, t0 - 2 * pad
        draw.rectangle([(x0 - 1, y0 - 1), (x1 + 1, y1 + 1)], fill=(0, 0, 0, 150), outline=(200, 200, 200, 255))
        if not math.isnan(force):
            frac = min(max(force / scale, 0.0), 1.0)
            color = GREEN if frac < 0.5 else ((240, 210, 60) if frac < 0.8 else RED)
            draw.rectangle([(x0, y1 - frac * (y1 - y0)), (x1, y1)], fill=color + (255,))
            label = f"{force:.0f}N"
            draw.text((x1 - draw.textlength(label, font=font), HudOverlay.banner_height + pad), label, font=font,
                      fill=(RED if force > scale else WHITE) + (255,))

        # -- the timeline along the bottom: every phase's start marked from the first frame,
        #    the episode so far in phase colours, and now
        steps = max(_v(step.state, "phase/max_steps"), len(self._phases))
        x_of = lambda t: int(round(t / steps * (W - 1)))  # noqa: E731
        draw.rectangle([(0, t0), (W - 1, H - 1)], fill=(40, 40, 40, 220))
        for k, (start, p) in enumerate(self._starts):
            if start > now:
                break
            end = min(self._starts[k + 1][0] if k + 1 < len(self._starts) else now + 1, now + 1)
            draw.rectangle([(x_of(start), t0), (max(x_of(end) - 1, x_of(start)), H - 1)], fill=PHASE_COLORS[p] + (255,))
        for start, _ in self._starts[1:]:
            draw.line([(x_of(start), t0 - pad), (x_of(start), H - 1)], fill=WHITE + (255,), width=1)
        x = x_of(now)
        draw.polygon([(x - pad, t0 - 2 * pad), (x + pad, t0 - 2 * pad), (x, t0)], fill=WHITE + (255,))

        return np.asarray(Image.alpha_composite(image, layer).convert("RGB"))
