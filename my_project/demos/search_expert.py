"""A scripted peg-insertion expert that sees only what the policy sees, and searches.

Unlike the oracle (``expert.py``), it steers with the **noisy** observations alone: Forge's noisy
fingertip estimate, noisy hole estimate and noisy force. It assumes the peg is centred in the
gripper (it cannot see the offset), so it has to find the hole by touch. Per episode:

1. ``approach``: fly the fingertip ``approach_height`` above the believed hole (the noisy hole
   estimate), directly over it.
2. ``descend``: lower at ``descend_speed`` until the 3-step mean of the noisy force's magnitude
   passes ``contact_force``: the peg is on the surface (or the hole's rim).
3. ``search``: an Archimedean spiral around the believed hole centre (turns ``search_pitch``
   apart, walked at ``search_speed``, restarting past ``search_radius``), while pushing down:
   the height target stays ``push_depth`` below the fingertip, so the peg drops in when it
   passes over the opening.
4. ``insert``: once the fingertip is ``drop_depth`` below the contact height, follow the
   fingertip horizontally (no forcing against the hole wall) and keep pushing by
   ``insert_push_depth``; the recorder saves on the env's success check.

The fingertip estimate is averaged over 3 steps against its per-step noise; the hole estimate's
per-episode offset cannot be averaged away, which is what the spiral is for. Yaw is held.
Positions are the env (robot base) frame.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Optional

import torch

__all__ = ["SearchExpert", "PHASES"]

PHASES = ("approach", "descend", "search", "insert")


class SearchExpert:
    def __init__(self, cfg: Any, *, max_speed_pos: float, dt: float) -> None:
        self.cfg = cfg
        self.max_speed = float(max_speed_pos)
        self.dt = float(dt)
        self.name = "search_expert"
        self.reset()

    # ------------------------------------------------------------------ the driver interface
    def reset(self) -> None:
        self.phase = "approach"
        self.phase_steps: Dict[str, int] = {name: 0 for name in PHASES}
        self.contact_height: Optional[float] = None
        self.spiral_angle = 0.0
        self.spiral_laps = 0
        self.insert_xy: Optional[torch.Tensor] = None
        self.forces: deque = deque(maxlen=3)
        self.tips: deque = deque(maxlen=3)

    def held(self, button: str) -> bool:
        return False

    def take(self, button: str) -> bool:
        return False

    def close(self) -> None:
        pass

    def episode_meta(self) -> Dict[str, Any]:
        return {
            "expert_contact_height": self.contact_height,
            "expert_final_phase": self.phase,
            "expert_phase_steps": dict(self.phase_steps),
            "expert_spiral_restarts": self.spiral_laps,
        }

    def command(self, view: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(vx, vy, vz, yaw rate) in [-1, 1] from the noisy observations and the setpoint."""
        cfg = self.cfg
        f64 = lambda name: view[name].detach().cpu().double()
        self.tips.append(f64("noisy_fingertip_pos"))
        self.forces.append(float(f64("noisy_force").norm()))
        tip = torch.stack(list(self.tips)).mean(dim=0)
        force = sum(self.forces) / len(self.forces)
        hole = f64("noisy_hole_pos")
        setpoint = f64("setpoint_pos")
        push = torch.tensor([tip[2] - cfg.push_depth])

        if self.phase == "approach":
            target = torch.cat([hole[:2], (hole[2] + cfg.approach_height).reshape(1)])
            if float((target - tip).abs().max()) < 0.002:
                self.phase = "descend"
        if self.phase == "descend":
            target = torch.cat([hole[:2], (setpoint[2] - cfg.descend_speed * self.dt).reshape(1)])
            if len(self.forces) == self.forces.maxlen and force > cfg.contact_force:
                self.phase, self.contact_height = "search", float(tip[2])
        if self.phase == "search" and float(tip[2]) < self.contact_height - cfg.drop_depth:
            self.phase, self.insert_xy = "insert", tip[:2].clone()
        if self.phase == "search":
            r = cfg.search_pitch * self.spiral_angle / (2 * math.pi)
            if r > cfg.search_radius:
                self.spiral_angle, r = 0.0, 0.0
                self.spiral_laps += 1
            # arc length per step = search_speed * dt at radius r (a small floor keeps the
            # centre from spinning in place)
            self.spiral_angle += cfg.search_speed * self.dt / max(r, cfg.search_pitch / 2)
            offset = torch.tensor([r * math.cos(self.spiral_angle), r * math.sin(self.spiral_angle)], dtype=torch.float64)
            target = torch.cat([hole[:2] + offset, push])
        if self.phase == "insert":
            # follow the fingertip sideways: the hole guides the peg, the expert only pushes down
            target = torch.cat([tip[:2], torch.tensor([tip[2] - cfg.insert_push_depth])])
        self.phase_steps[self.phase] += 1

        velocity = (target - setpoint) / (self.max_speed * self.dt)
        return torch.cat([velocity.clamp(-1.0, 1.0), torch.zeros(1, dtype=torch.float64)]).float()
