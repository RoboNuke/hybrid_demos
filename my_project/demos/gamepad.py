"""An Xbox-style gamepad through Kit's carb input interface (needs the GUI app window).

Same subscription Isaac Lab's ``Se3Gamepad`` uses. Kit reports each stick as four half-axes in
[0, 1] (left stick up, left stick down, ...) and each button as 0 / 1; this keeps the latest
value of every input, and turns button presses into one-shot flags.

Default layout, chosen for the recorder's camera (it looks at the hole from in front of the
robot):

    left stick        move in the horizontal plane: up = away from you, sideways = sideways
    right stick       up / down = raise / lower; left / right = yaw
    RB (hold)         fine mode: every speed times record.fine_scale
    A                 save the episode now (marked successful or not by the env's check)
    B                 discard the episode and reset
    START             quit (the current, unsaved episode is discarded)
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

__all__ = ["Gamepad", "BUTTONS"]

#: button name -> carb.input.GamepadInput member name
BUTTONS = {"save": "A", "discard": "B", "fine": "RIGHT_SHOULDER", "quit": "MENU2"}

_STICKS = (
    "LEFT_STICK_UP",
    "LEFT_STICK_DOWN",
    "LEFT_STICK_LEFT",
    "LEFT_STICK_RIGHT",
    "RIGHT_STICK_UP",
    "RIGHT_STICK_DOWN",
    "RIGHT_STICK_LEFT",
    "RIGHT_STICK_RIGHT",
)


class Gamepad:
    """Latest stick values and button presses of gamepad 0."""

    def __init__(self, dead_zone: float, axis_signs: Sequence[float]) -> None:
        import carb
        import omni.appwindow

        self._carb = carb
        self.dead_zone = float(dead_zone)
        self.axis_signs = torch.tensor(list(axis_signs), dtype=torch.float32)
        # Kit flies the viewport camera with the gamepad unless this is off
        carb.settings.get_settings().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        self._input = carb.input.acquire_input_interface()
        self._pad = omni.appwindow.get_default_app_window().get_gamepad(0)
        if self._pad is None:
            raise RuntimeError("no gamepad 0: plug the controller in before starting the recorder")
        self.name = self._input.get_gamepad_name(self._pad)
        self._values: Dict[str, float] = {name: 0.0 for name in _STICKS}
        self._held: Dict[str, bool] = {name: False for name in BUTTONS.values()}
        self._pressed: Dict[str, bool] = {name: False for name in BUTTONS}
        self._by_input = {getattr(carb.input.GamepadInput, n): n for n in (*_STICKS, *BUTTONS.values())}
        self._sub = self._input.subscribe_to_gamepad_events(self._pad, self._on_event)

    def _on_event(self, event, *args, **kwargs) -> bool:
        name = self._by_input.get(event.input)
        if name is None:
            return True
        value = float(event.value)
        if name in self._values:
            self._values[name] = value if value >= self.dead_zone else 0.0
            return True
        down = value > 0.5
        if down and not self._held[name]:
            for button, member in BUTTONS.items():
                if member == name:
                    self._pressed[button] = True
        self._held[name] = down
        return True

    def reset(self) -> None:
        """Nothing carries over between episodes."""

    def episode_meta(self) -> dict:
        return {}

    def command(self, view=None) -> torch.Tensor:
        """(vx, vy, vz, yaw rate) in [-1, 1], in the env frame for the default camera.

        ``view`` (the observations the recorder offers a driver) is not used: the operator
        looks at the screen."""
        v = self._values
        # the camera looks along -x, so "away from you" is -x and screen right is +y
        raw = torch.tensor(
            [
                v["LEFT_STICK_DOWN"] - v["LEFT_STICK_UP"],
                v["LEFT_STICK_RIGHT"] - v["LEFT_STICK_LEFT"],
                v["RIGHT_STICK_UP"] - v["RIGHT_STICK_DOWN"],
                v["RIGHT_STICK_LEFT"] - v["RIGHT_STICK_RIGHT"],
            ],
            dtype=torch.float32,
        )
        return raw * self.axis_signs

    def held(self, button: str) -> bool:
        return self._held[BUTTONS[button]]

    def take(self, button: str) -> bool:
        """True once per press."""
        pressed, self._pressed[button] = self._pressed[button], False
        return pressed

    def close(self) -> None:
        self._input.unsubscribe_to_gamepad_events(self._pad, self._sub)
