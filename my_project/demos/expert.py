"""A scripted peg-insertion expert that drives the recorder like an operator would.

It steers with **ground truth** (the true peg, fingertip and hole positions and the true
force), so the peg's offset in the gripper and the noise on the hole estimate do not matter: it
lines up the peg itself with the hole. What gets recorded is unchanged: the noisy policy
observation, the true values beside it, and the action, through the same
velocity -> setpoint -> Forge action path as the gamepad. Per episode:

1. ``approach``: pick a random point on the hole block's top surface, ``surface_radius_min``
   to ``surface_radius_max`` from the hole axis (far enough that the peg lands on the surface,
   not in the hole), and fly the peg's bottom ``approach_clearance`` above it.
2. ``descend``: lower at ``descend_speed`` until the force passes ``contact_force``: the peg
   is on the surface.
3. ``slide``: move the peg toward the hole axis at ``slide_speed`` while pushing down: the
   height target is always ``push_depth`` below the fingertip, so the peg drops in when it
   reaches the opening.
4. ``insert``: once the peg's bottom is ``drop_depth`` below the hole top, keep it on the axis
   and keep pushing, by ``insert_push_depth`` (default ``push_depth``): with the action
   smoothing's lag it sets the insertion speed, and it can be larger than the sliding push,
   which must stay gentle or friction stops the slide short of the hole; the recorder saves on
   the env's success check (peg seated).

Positions are the env frame: the peg's bottom is ``held_pos``, the hole top
``fixed_pos_obs_frame`` (Factory's success check uses the same points). Yaw is held. Randomness
comes from a generator the recorder seeds per session.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch

__all__ = ["Expert", "PHASES"]

PHASES = ("approach", "descend", "slide", "insert")


class Expert:
    def __init__(self, cfg: Any, *, max_speed_pos: float, dt: float, generator: torch.Generator) -> None:
        self.cfg = cfg
        self.max_speed = float(max_speed_pos)
        self.dt = float(dt)
        self.generator = generator
        self.name = "expert"
        self.reset()

    # ------------------------------------------------------------------ the driver interface
    def reset(self) -> None:
        """A new episode: a new surface point, back to ``approach``."""
        cfg = self.cfg
        u = torch.rand(2, generator=self.generator).tolist()
        self.radius = cfg.surface_radius_min + u[0] * (cfg.surface_radius_max - cfg.surface_radius_min)
        self.angle = 2 * math.pi * u[1]
        self.phase = "approach"
        self.phase_steps: Dict[str, int] = {name: 0 for name in PHASES}
        self.contact_height: Optional[float] = None
        self.slide_xy: Optional[torch.Tensor] = None
        self.last: Dict[str, float] = {}

    def held(self, button: str) -> bool:
        return False

    def take(self, button: str) -> bool:
        return False

    def close(self) -> None:
        pass

    def episode_meta(self) -> Dict[str, Any]:
        return {
            "expert_surface_radius": self.radius,
            "expert_surface_angle": self.angle,
            "expert_contact_height": self.contact_height,
            "expert_final_phase": self.phase,
            "expert_phase_steps": dict(self.phase_steps),
            "expert_last": dict(self.last),
        }

    def command(self, view: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(vx, vy, vz, yaw rate) in [-1, 1] from the true state and the setpoint."""
        cfg = self.cfg
        f64 = lambda name: view[name].detach().cpu().double()
        fingertip, peg, hole = f64("fingertip_pos"), f64("peg_pos"), f64("hole_pos")
        setpoint = f64("setpoint_pos")
        force = float(f64("force").norm())
        grip = fingertip - peg  # where the fingertip sits relative to the peg's bottom
        surface_xy = hole[:2] + self.radius * torch.tensor(
            [math.cos(self.angle), math.sin(self.angle)], dtype=torch.float64
        )
        push = torch.tensor([fingertip[2] - cfg.push_depth])
        self.last = {
            "force": round(force, 3),
            "peg_above_hole_mm": round(float(peg[2] - hole[2]) * 1000, 2),
            "peg_off_axis_mm": round(float((peg[:2] - hole[:2]).norm()) * 1000, 2),
        }

        if self.phase == "approach":
            peg_target = torch.cat([surface_xy, (hole[2] + cfg.approach_clearance).reshape(1)])
            target = peg_target + grip
            if float((peg_target - peg).abs().max()) < 0.002:
                self.phase = "descend"
        if self.phase == "descend":
            target = torch.cat([surface_xy + grip[:2], (setpoint[2] - cfg.descend_speed * self.dt).reshape(1)])
            if force > cfg.contact_force:
                self.phase, self.contact_height = "slide", float(peg[2] - hole[2])
                self.slide_xy = peg[:2].clone()
        if self.phase == "slide" and peg[2] < hole[2] - cfg.drop_depth:
            self.phase = "insert"
        if self.phase == "slide":
            to_hole = hole[:2] - self.slide_xy
            step = cfg.slide_speed * self.dt
            distance = float(to_hole.norm())
            self.slide_xy = hole[:2].clone() if distance <= step else self.slide_xy + to_hole / distance * step
            target = torch.cat([self.slide_xy + grip[:2], push])
        if self.phase == "insert":
            push_in = cfg.insert_push_depth if cfg.insert_push_depth is not None else cfg.push_depth
            target = torch.cat([hole[:2] + grip[:2], torch.tensor([fingertip[2] - push_in])])
        self.phase_steps[self.phase] += 1

        velocity = (target - setpoint) / (self.max_speed * self.dt)
        return torch.cat([velocity.clamp(-1.0, 1.0), torch.zeros(1, dtype=torch.float64)]).float()
