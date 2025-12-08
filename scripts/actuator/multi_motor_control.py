#!/usr/bin/env python3
"""
Multi-Motor Coordinated Control Script

This script demonstrates controlling multiple motors simultaneously
with coordinated trajectories.

Usage:
    python multi_motor_control.py [--pattern PATTERN]

Example:
    python multi_motor_control.py --pattern wave
"""

import argparse
import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot.bimanual import Bimanual


def wave_pattern(t: float, motor_count: int = 10, amplitude: float = 0.3,
                 frequency: float = 0.5, phase_offset: float = 0.5) -> np.ndarray:
    """
    Generate a wave pattern across motors.

    Args:
        t: Current time
        motor_count: Number of motors
        amplitude: Wave amplitude
        frequency: Wave frequency
        phase_offset: Phase offset between adjacent motors
    """
    positions = np.zeros(motor_count)
    for i in range(motor_count):
        phase = 2 * np.pi * (frequency * t + i * phase_offset / motor_count)
        positions[i] = amplitude * np.sin(phase)
    return positions


def mirror_pattern(t: float, amplitude: float = 0.5, frequency: float = 0.3) -> np.ndarray:
    """
    Generate mirrored motion between left and right arms.

    Args:
        t: Current time
        amplitude: Motion amplitude
        frequency: Motion frequency
    """
    positions = np.zeros(10)

    # Shoulder and elbow joints
    shoulder_pos = amplitude * np.sin(2 * np.pi * frequency * t)
    elbow_pos = amplitude * np.cos(2 * np.pi * frequency * t)

    # Left arm (motors 0-4)
    positions[0] = shoulder_pos
    positions[1] = elbow_pos

    # Right arm (motors 5-9) - mirrored
    positions[5] = shoulder_pos
    positions[6] = elbow_pos

    return positions


def synchronized_pattern(t: float, amplitude: float = 0.4, frequency: float = 0.4) -> np.ndarray:
    """
    All motors move in synchronization.

    Args:
        t: Current time
        amplitude: Motion amplitude
        frequency: Motion frequency
    """
    position = amplitude * np.sin(2 * np.pi * frequency * t)
    return np.full(10, position)


def multi_motor_control(pattern: str, duration: float = 15.0,
                        kp: float = 50.0, kd: float = 2.0, torque_limit: float = 1.0):
    """
    Control multiple motors with coordinated patterns.

    Args:
        pattern: Pattern type ('wave', 'mirror', 'sync')
        duration: Duration to run in seconds
        kp: Position proportional gain
        kd: Velocity damping gain
        torque_limit: Maximum torque in Nm
    """
    pattern_funcs = {
        'wave': lambda t: wave_pattern(t),
        'mirror': lambda t: mirror_pattern(t),
        'sync': lambda t: synchronized_pattern(t)
    }

    if pattern not in pattern_funcs:
        print(f"Error: Unknown pattern '{pattern}'")
        print(f"Available patterns: {list(pattern_funcs.keys())}")
        return

    pattern_func = pattern_funcs[pattern]

    np.set_printoptions(precision=3, suppress=True)

    rate = RateLimiter(100)  # 100 Hz control loop
    robot = Bimanual()

    robot.start(kp=kp, kd=kd, torque_limit=torque_limit)

    actions = np.zeros((12,), dtype=np.float32)

    print(f"Running '{pattern}' pattern on multiple motors")
    print(f"Control gains: kp={kp}, kd={kd}, torque_limit={torque_limit} Nm")
    print(f"Duration: {duration:.1f}s")
    print("\nPress Ctrl+C to stop")
    print("-" * 60)

    try:
        t = 0.0
        iteration = 0
        while t < duration:
            # Generate target positions from pattern
            target_positions = pattern_func(t)
            actions[0:10] = target_positions

            # Apply action and get observation
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions

            # Print status every 25 iterations (4 Hz)
            if iteration % 25 == 0:
                print(f"t={t:.2f}s")
                print(f"  Targets:  {target_positions}")
                print(f"  Actual:   {obs[0:10]}")
                errors = target_positions - obs[0:10]
                print(f"  Errors:   {errors}")
                print()

            iteration += 1
            t += rate.period()
            rate.sleep()

    except KeyboardInterrupt:
        print("\nStopping robot due to keyboard interrupt.")
    finally:
        robot.stop()
        print("Robot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-motor coordinated control")
    parser.add_argument("--pattern", type=str, default="wave",
                        choices=['wave', 'mirror', 'sync'],
                        help="Motion pattern")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration to run in seconds")
    parser.add_argument("--kp", type=float, default=50.0,
                        help="Position proportional gain")
    parser.add_argument("--kd", type=float, default=2.0,
                        help="Velocity damping gain")
    parser.add_argument("--torque-limit", type=float, default=1.0,
                        help="Maximum torque in Nm")

    args = parser.parse_args()

    multi_motor_control(
        pattern=args.pattern,
        duration=args.duration,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit
    )
