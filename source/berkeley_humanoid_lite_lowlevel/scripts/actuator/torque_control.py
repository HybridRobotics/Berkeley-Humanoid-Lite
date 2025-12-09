#!/usr/bin/env python3
"""
Torque Control Script for Berkeley Humanoid Lite Actuators

This script demonstrates gravity compensation and torque control modes.
Useful for compliant interaction and force control.

Usage:
    python torque_control.py [--motor-id ID] [--mode MODE]

Example:
    python torque_control.py --motor-id 0 --mode gravity_comp
"""

import argparse
import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot.bimanual import Bimanual


def gravity_compensation(motor_id: int = None, duration: float = 30.0):
    """
    Run motors in gravity compensation mode (very low gains).
    This allows manual manipulation while providing some support.

    Args:
        motor_id: Motor to control (None = all motors)
        duration: Duration to run in seconds
    """
    np.set_printoptions(precision=3, suppress=True)

    rate = RateLimiter(100)  # 100 Hz control loop
    robot = Bimanual()

    # Very low gains for compliant behavior
    robot.start(kp=5.0, kd=0.5, torque_limit=0.3)

    actions = np.zeros((12,), dtype=np.float32)

    print("Running in gravity compensation mode")
    print("The robot should be compliant and easy to move by hand")
    if motor_id is not None:
        print(f"Controlling motor {motor_id}")
    else:
        print("Controlling all motors")
    print(f"Duration: {duration:.1f}s")
    print("\nPress Ctrl+C to stop")
    print("You can now manually move the robot")
    print("-" * 60)

    try:
        t = 0.0
        iteration = 0
        while t < duration:
            # Get current observation
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions

            # Update actions to track current position (soft hold)
            if motor_id is not None:
                actions[motor_id] = obs[motor_id]
            else:
                actions[:] = obs[:]

            # Print status every 50 iterations (2 Hz)
            if iteration % 50 == 0:
                if motor_id is not None:
                    print(f"t={t:.2f}s | Motor {motor_id}: pos={obs[motor_id]:.3f} rad")
                else:
                    print(f"t={t:.2f}s | All positions: {obs}")

            iteration += 1
            t += rate.period()
            rate.sleep()

    except KeyboardInterrupt:
        print("\nStopping robot due to keyboard interrupt.")
    finally:
        robot.stop()
        print("Robot stopped.")


def zero_torque(duration: float = 5.0):
    """
    Run with zero torque commands (free motion).
    WARNING: Robot will collapse under gravity!

    Args:
        duration: Duration to run in seconds
    """
    np.set_printoptions(precision=3, suppress=True)

    rate = RateLimiter(100)
    robot = Bimanual()

    # Minimal gains
    robot.start(kp=0.1, kd=0.1, torque_limit=0.1)

    actions = np.zeros((12,), dtype=np.float32)

    print("WARNING: Running in zero torque mode")
    print("The robot will NOT resist gravity!")
    print("Make sure the robot is supported or in a safe position")
    print(f"Duration: {duration:.1f}s")
    print("\nPress Ctrl+C to stop")
    print("-" * 60)

    try:
        t = 0.0
        iteration = 0
        while t < duration:
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions

            if iteration % 50 == 0:
                print(f"t={t:.2f}s | Positions: {obs}")

            iteration += 1
            t += rate.period()
            rate.sleep()

    except KeyboardInterrupt:
        print("\nStopping robot due to keyboard interrupt.")
    finally:
        robot.stop()
        print("Robot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torque control modes for motors")
    parser.add_argument("--mode", type=str, default="gravity_comp",
                        choices=['gravity_comp', 'zero_torque'],
                        help="Control mode")
    parser.add_argument("--motor-id", type=int, default=None,
                        help="Motor ID (None = all motors)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration to run in seconds")

    args = parser.parse_args()

    if args.motor_id is not None and (args.motor_id < 0 or args.motor_id > 11):
        print("Error: Motor ID must be between 0 and 11")
        exit(1)

    if args.mode == "gravity_comp":
        gravity_compensation(motor_id=args.motor_id, duration=args.duration)
    elif args.mode == "zero_torque":
        if args.motor_id is not None:
            print("Warning: zero_torque mode controls all motors, ignoring --motor-id")
        zero_torque(duration=args.duration)
