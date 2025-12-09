"""
calibrate_joints.py

Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

Run this script after each power cycle to calibrate the encoder offset of each joint.
"""

import time

import numpy as np
import yaml

from berkeley_humanoid_lite_lowlevel.robot import Humanoid


robot = Humanoid()

joint_axis_directions = np.array([
    -1, +1, -1,
    -1,
    -1, +1,
    -1, +1, +1,
    +1,
    +1, +1
])


ideal_values = np.array([
    np.deg2rad(-(10)),
    np.deg2rad(+(33.75)),
    np.deg2rad(+(56.25)),
    np.deg2rad(+(0)),
    np.deg2rad(-(45)),
    np.deg2rad(-(15)),

    np.deg2rad(+(10)),
    np.deg2rad(-(33.75)),
    np.deg2rad(+(56.25)),
    np.deg2rad(+(0)),
    np.deg2rad(-(45)),
    np.deg2rad(+(15)),
])


print("initial readings:")
limit_readings = np.array([joint[0].read_position_measured(joint[1]) for joint in robot.joints]) * joint_axis_directions
print([f"{reading:.2f}" for reading in limit_readings])

while robot.command_controller.commands.get("mode_switch") != 1:
    joint_readings = np.array([joint[0].read_position_measured(joint[1]) for joint in robot.joints]) * joint_axis_directions

    limit_readings[0] = min(limit_readings[0], joint_readings[0])
    limit_readings[1] = max(limit_readings[1], joint_readings[1])
    limit_readings[2] = max(limit_readings[2], joint_readings[2])
    limit_readings[3] = min(limit_readings[3], joint_readings[3])
    limit_readings[4] = min(limit_readings[4], joint_readings[4])
    limit_readings[5] = min(limit_readings[5], joint_readings[5])

    limit_readings[6] = max(limit_readings[6], joint_readings[6])
    limit_readings[7] = min(limit_readings[7], joint_readings[7])
    limit_readings[8] = max(limit_readings[8], joint_readings[8])
    limit_readings[9] = min(limit_readings[9], joint_readings[9])
    limit_readings[10] = min(limit_readings[10], joint_readings[10])
    limit_readings[11] = max(limit_readings[11], joint_readings[11])

    print(time.time(), [f"{reading:.2f}" for reading in limit_readings])

    time.sleep(0.05)


print("final readings at the limits:")
print([f"{limit:.4f}" for limit in limit_readings])

print("offsets:")
print([f"{offset:.4f}" for offset in (limit_readings - ideal_values)])

calibration_data = {
    "position_offsets": [float(offset) for offset in (limit_readings - ideal_values)],
}

with open("calibration.yaml", "w") as f:
    yaml.dump(calibration_data, f)

robot.stop()
