#!/usr/bin/env python3
"""
Position Control Script for Berkeley Humanoid Lite Actuators

This script demonstrates precise position control of individual motors
with configurable PID gains and torque limits.

Usage:
    python position_control.py [--motor-id ID] [--position POS] [--kp KP] [--kd KD]

Example:
    python position_control.py --motor-id 0 --position 0.5 --kp 50 --kd 2
"""

import argparse
import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot.bimanual import Bimanual


def position_control(motor_id: int, target_position: float, kp: float = 50.0,
                     kd: float = 2.0, torque_limit: float = 1.0, duration: float = 5.0):
    """
    Control a single motor to a target position.

    Args:
        motor_id: Motor index (0-11 for bimanual robot)
        target_position: Target position in radians
        kp: Position proportional gain
        kd: Velocity damping gain
        torque_limit: Maximum torque in Nm
        duration: Duration to hold position in seconds
    """
    np.set_printoptions(precision=3, suppress=True)

    rate = RateLimiter(100)  # 100 Hz control loop
    robot = Bimanual()

    # Start robot with specified gains
    robot.start(kp=kp, kd=kd, torque_limit=torque_limit)

    # Create action array - zeros for all motors except target
    actions = np.zeros((12,), dtype=np.float32)
    actions[motor_id] = target_position

    print(f"Moving motor {motor_id} to position {target_position:.3f} rad")
    print(f"Control gains: kp={kp}, kd={kd}, torque_limit={torque_limit} Nm")
    print(f"Duration: {duration:.1f}s")
    print("\nMotor states (position, velocity, torque):")
    print("-" * 60)

    try:
        start_time = 0
        iteration = 0
        while start_time < duration:
            # Apply action and get observation
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions

            # Print status every 10 iterations (10 Hz)
            if iteration % 10 == 0:
                print(f"t={start_time:.2f}s | Motor {motor_id}: "
                      f"pos={obs[motor_id]:.3f} rad, "
                      f"error={target_position - obs[motor_id]:.3f} rad")

            iteration += 1
            start_time += rate.period()
            rate.sleep()

    except KeyboardInterrupt:
        print("\nStopping robot due to keyboard interrupt.")
    finally:
        robot.stop()
        print("Robot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Position control for individual motors")
    parser.add_argument("--motor-id", type=int, default=0,
                        help="Motor ID (0-11)")
    parser.add_argument("--position", type=float, default=0.0,
                        help="Target position in radians")
    parser.add_argument("--kp", type=float, default=50.0,
                        help="Position proportional gain")
    parser.add_argument("--kd", type=float, default=2.0,
                        help="Velocity damping gain")
    parser.add_argument("--torque-limit", type=float, default=1.0,
                        help="Maximum torque in Nm")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration to hold position in seconds")

    args = parser.parse_args()

    if args.motor_id < 0 or args.motor_id > 11:
        print("Error: Motor ID must be between 0 and 11")
        exit(1)

    position_control(
        motor_id=args.motor_id,
        target_position=args.position,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit,
        duration=args.duration
    )
