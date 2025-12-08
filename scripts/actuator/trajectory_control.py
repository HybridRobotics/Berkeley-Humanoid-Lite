#!/usr/bin/env python3
"""
Trajectory Control Script for Berkeley Humanoid Lite Actuators

This script enables smooth trajectory following for motors using
sinusoidal, linear, or custom trajectories.

Usage:
    python trajectory_control.py [--motor-id ID] [--trajectory TYPE] [--amplitude AMP]

Example:
    python trajectory_control.py --motor-id 0 --trajectory sine --amplitude 0.5 --frequency 0.5
"""

import argparse
import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot.bimanual import Bimanual


def sine_trajectory(t: float, amplitude: float, frequency: float, offset: float = 0.0) -> float:
    """Generate sinusoidal trajectory."""
    return offset + amplitude * np.sin(2 * np.pi * frequency * t)


def triangle_trajectory(t: float, amplitude: float, frequency: float, offset: float = 0.0) -> float:
    """Generate triangular wave trajectory."""
    period = 1.0 / frequency
    phase = (t % period) / period
    if phase < 0.5:
        return offset + 4 * amplitude * (phase - 0.25)
    else:
        return offset + 4 * amplitude * (0.75 - phase)


def square_trajectory(t: float, amplitude: float, frequency: float, offset: float = 0.0) -> float:
    """Generate square wave trajectory."""
    period = 1.0 / frequency
    phase = (t % period) / period
    return offset + amplitude if phase < 0.5 else offset - amplitude


def trajectory_control(motor_id: int, trajectory_type: str, amplitude: float = 0.5,
                       frequency: float = 0.5, offset: float = 0.0, duration: float = 10.0,
                       kp: float = 50.0, kd: float = 2.0, torque_limit: float = 1.0):
    """
    Follow a trajectory with a single motor.

    Args:
        motor_id: Motor index (0-11 for bimanual robot)
        trajectory_type: Type of trajectory ('sine', 'triangle', 'square')
        amplitude: Amplitude of motion in radians
        frequency: Frequency of motion in Hz
        offset: Center position offset in radians
        duration: Duration to run trajectory in seconds
        kp: Position proportional gain
        kd: Velocity damping gain
        torque_limit: Maximum torque in Nm
    """
    trajectory_funcs = {
        'sine': sine_trajectory,
        'triangle': triangle_trajectory,
        'square': square_trajectory
    }

    if trajectory_type not in trajectory_funcs:
        print(f"Error: Unknown trajectory type '{trajectory_type}'")
        print(f"Available types: {list(trajectory_funcs.keys())}")
        return

    traj_func = trajectory_funcs[trajectory_type]

    np.set_printoptions(precision=3, suppress=True)

    rate = RateLimiter(100)  # 100 Hz control loop
    robot = Bimanual()

    robot.start(kp=kp, kd=kd, torque_limit=torque_limit)

    actions = np.zeros((12,), dtype=np.float32)

    print(f"Running {trajectory_type} trajectory on motor {motor_id}")
    print(f"Amplitude: {amplitude:.3f} rad, Frequency: {frequency:.3f} Hz, Offset: {offset:.3f} rad")
    print(f"Control gains: kp={kp}, kd={kd}, torque_limit={torque_limit} Nm")
    print(f"Duration: {duration:.1f}s")
    print("\nPress Ctrl+C to stop")
    print("-" * 60)

    try:
        t = 0.0
        iteration = 0
        while t < duration:
            # Generate target position from trajectory
            target_pos = traj_func(t, amplitude, frequency, offset)
            actions[motor_id] = target_pos

            # Apply action and get observation
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions

            # Print status every 10 iterations (10 Hz)
            if iteration % 10 == 0:
                error = target_pos - obs[motor_id]
                print(f"t={t:.2f}s | Motor {motor_id}: "
                      f"target={target_pos:.3f}, actual={obs[motor_id]:.3f}, "
                      f"error={error:.3f} rad")

            iteration += 1
            t += rate.period()
            rate.sleep()

    except KeyboardInterrupt:
        print("\nStopping robot due to keyboard interrupt.")
    finally:
        robot.stop()
        print("Robot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory control for individual motors")
    parser.add_argument("--motor-id", type=int, default=0,
                        help="Motor ID (0-11)")
    parser.add_argument("--trajectory", type=str, default="sine",
                        choices=['sine', 'triangle', 'square'],
                        help="Trajectory type")
    parser.add_argument("--amplitude", type=float, default=0.5,
                        help="Trajectory amplitude in radians")
    parser.add_argument("--frequency", type=float, default=0.5,
                        help="Trajectory frequency in Hz")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Center position offset in radians")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration to run in seconds")
    parser.add_argument("--kp", type=float, default=50.0,
                        help="Position proportional gain")
    parser.add_argument("--kd", type=float, default=2.0,
                        help="Velocity damping gain")
    parser.add_argument("--torque-limit", type=float, default=1.0,
                        help="Maximum torque in Nm")

    args = parser.parse_args()

    if args.motor_id < 0 or args.motor_id > 11:
        print("Error: Motor ID must be between 0 and 11")
        exit(1)

    trajectory_control(
        motor_id=args.motor_id,
        trajectory_type=args.trajectory,
        amplitude=args.amplitude,
        frequency=args.frequency,
        offset=args.offset,
        duration=args.duration,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit
    )
