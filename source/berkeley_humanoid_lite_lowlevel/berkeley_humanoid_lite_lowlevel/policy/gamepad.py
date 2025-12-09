# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Gamepad Controller Module for Berkeley Humanoid Lite

This module implements UDP-based controllers for the Berkeley Humanoid Lite robot,
supporting both gamepad and keyboard input devices. It handles command broadcasting
over UDP for robot control modes and movement velocities.
"""

import threading
from typing import Dict

from inputs import get_gamepad


class XInputEntry:
    """
    Constants for gamepad button and axis mappings.

    This class defines the standard mapping for various gamepad controls,
    including analog sticks, triggers, d-pad, and buttons.
    """
    AXIS_X_L = "ABS_X"
    AXIS_Y_L = "ABS_Y"
    AXIS_TRIGGER_L = "ABS_Z"
    AXIS_X_R = "ABS_RX"
    AXIS_Y_R = "ABS_RY"
    AXIS_TRIGGER_R = "ABS_RZ"

    BTN_HAT_X = "ABS_HAT0X"
    BTN_HAT_Y = "ABS_HAT0Y"

    BTN_A = "BTN_SOUTH"
    BTN_B = "BTN_EAST"
    BTN_X = "BTN_NORTH"
    BTN_Y = "BTN_WEST"
    BTN_BUMPER_L = "BTN_TL"
    BTN_BUMPER_R = "BTN_TR"
    BTN_THUMB_L = "BTN_THUMBL"
    BTN_THUMB_R = "BTN_THUMBR"
    BTN_BACK = "BTN_SELECT"
    BTN_START = "BTN_START"


class Se2Gamepad:
    def __init__(self,
                 stick_sensitivity: float = 1.0,
                 dead_zone: float = 0.01,
                 ) -> None:
        self.stick_sensitivity = stick_sensitivity
        self.dead_zone = dead_zone

        self._stopped = threading.Event()
        self._run_forever_thread = None

        self.reset()

        self.commands = {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_yaw": 0.0,
            "mode_switch": 0,
        }

    def reset(self) -> None:
        self._states = {key: 0 for key in XInputEntry.__dict__.values()}

    def stop(self) -> None:
        print("Gamepad stopping...")
        self._stopped.set()
        # self._run_forever_thread.join()

    def run(self) -> None:
        self._run_forever_thread = threading.Thread(target=self.run_forever)
        self._run_forever_thread.start()

    def run_forever(self) -> None:
        while not self._stopped.is_set():
            self.advance()

    def advance(self) -> None:
        events = get_gamepad()

        # update all events from the joystick
        for event in events:
            self._states[event.code] = event.state

        self._update_command_buffer()

    def _update_command_buffer(self) -> Dict[str, float]:
        velocity_x = self._states.get(XInputEntry.AXIS_Y_L)
        velocity_y = self._states.get(XInputEntry.AXIS_X_R)
        velocity_yaw = self._states.get(XInputEntry.AXIS_X_L)

        if velocity_x is not None:
            self.commands["velocity_x"] = velocity_x / -32768.0
        if velocity_y is not None:
            self.commands["velocity_y"] = velocity_y / -32768.0
        if velocity_yaw is not None:
            self.commands["velocity_yaw"] = velocity_yaw / -32768.0

        mode_switch = 0

        # Enter RL control mode (A + Right Bumper)
        if self._states.get(XInputEntry.BTN_A) and self._states.get(XInputEntry.BTN_BUMPER_R):
            mode_switch = 3

        # Enter init mode (A + Left Bumper)
        if self._states.get(XInputEntry.BTN_A) and self._states.get(XInputEntry.BTN_BUMPER_L):
            mode_switch = 2

        # Enter idle mode (B or Left/Right Thumbstick)
        if self._states.get(XInputEntry.BTN_X) or self._states.get(XInputEntry.BTN_THUMB_L) or self._states.get(XInputEntry.BTN_THUMB_R):
            mode_switch = 1

        self.commands["mode_switch"] = mode_switch


if __name__ == "__main__":
    command_controller = Se2Gamepad()
    command_controller.run()

    try:
        while True:
            print(f"""{command_controller.commands.get("velocity_x"):.2f}, {command_controller.commands.get("velocity_y"):.2f}, {command_controller.commands.get("velocity_yaw"):.2f}""")
            pass
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    command_controller.stop()
