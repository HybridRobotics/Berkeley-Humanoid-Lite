#!/usr/bin/env python3
"""
Motor State Monitor

This script monitors and logs motor states including position, velocity,
and estimated torque. Useful for debugging and performance analysis.

Usage:
    python monitor_motors.py [--motor-id ID] [--rate RATE] [--log LOG_FILE]

Example:
    python monitor_motors.py --motor-id 0 --rate 50
    python monitor_motors.py --log motor_data.csv
"""

import argparse
import time
import csv
import numpy as np
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot.bimanual import Bimanual


def monitor_motors(motor_id: int = None, update_rate: float = 10.0,
                   log_file: str = None, duration: float = None):
    """
    Monitor motor states in real-time.

    Args:
        motor_id: Specific motor to monitor (None = all motors)
        update_rate: Display update rate in Hz
        log_file: Optional CSV file to log data
        duration: Duration to monitor (None = infinite until Ctrl+C)
    """
    np.set_printoptions(precision=3, suppress=True)

    control_rate = RateLimiter(100)  # 100 Hz control

    robot = Bimanual()
    robot.start(kp=20, kd=2, torque_limit=0.5)

    actions = np.zeros((12,), dtype=np.float32)

    print("Motor State Monitor")
    print("=" * 80)
    if motor_id is not None:
        print(f"Monitoring motor {motor_id}")
    else:
        print("Monitoring all motors (0-11)")
    print(f"Update rate: {update_rate:.1f} Hz")
    if log_file:
        print(f"Logging to: {log_file}")
    print("\nPress Ctrl+C to stop")
    print("-" * 80)

    # CSV logging setup
    csv_writer = None
    csv_file = None
    if log_file:
        csv_file = open(log_file, 'w', newline='')
        fieldnames = ['timestamp', 'motor_id', 'position', 'velocity', 'current']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    try:
        start_time = time.perf_counter()
        iteration = 0
        display_interval = int(100 / update_rate)  # iterations between displays

        while True:
            # Get current state (zero torque command = passive monitoring)
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions

            current_time = time.perf_counter() - start_time

            # Check duration limit
            if duration is not None and current_time >= duration:
                break

            # Log to CSV if enabled
            if csv_writer:
                for i in range(12):
                    csv_writer.writerow({
                        'timestamp': current_time,
                        'motor_id': i,
                        'position': obs[i],
                        'velocity': 0.0,  # Velocity would need to be computed
                        'current': 0.0     # Current would need sensor data
                    })

            # Display at specified rate
            if iteration % display_interval == 0:
                print(f"\nTime: {current_time:.2f}s")

                if motor_id is not None:
                    # Single motor detailed view
                    print(f"Motor {motor_id}:")
                    print(f"  Position: {obs[motor_id]:>8.3f} rad  ({np.rad2deg(obs[motor_id]):>7.2f}°)")
                    # Note: Velocity and torque would require additional sensor data
                    print(f"  Velocity: N/A")
                    print(f"  Torque:   N/A")
                else:
                    # All motors compact view
                    print("Motor | Position (rad) | Position (deg)")
                    print("-" * 45)
                    for i in range(12):
                        print(f"  {i:2d}  | {obs[i]:>8.3f}      | {np.rad2deg(obs[i]):>7.2f}")

            iteration += 1
            control_rate.sleep()

    except KeyboardInterrupt:
        print("\n\nStopping monitor due to keyboard interrupt.")
    finally:
        robot.stop()
        if csv_file:
            csv_file.close()
            print(f"Data saved to {log_file}")
        print("Monitor stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor motor states")
    parser.add_argument("--motor-id", type=int, default=None,
                        help="Specific motor to monitor (None = all)")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Display update rate in Hz")
    parser.add_argument("--log", type=str, default=None,
                        help="CSV file to log data")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration to monitor in seconds (None = until Ctrl+C)")

    args = parser.parse_args()

    if args.motor_id is not None and (args.motor_id < 0 or args.motor_id > 11):
        print("Error: Motor ID must be between 0 and 11")
        exit(1)

    monitor_motors(
        motor_id=args.motor_id,
        update_rate=args.rate,
        log_file=args.log,
        duration=args.duration
    )
