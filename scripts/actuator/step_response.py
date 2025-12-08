#!/usr/bin/env python3
"""
Step Response Analysis Script

This script performs step response tests to analyze motor dynamics
and help tune PID parameters. Useful for system identification and
controller tuning.

Usage:
    python step_response.py [--motor-id ID] [--step-size SIZE]

Example:
    python step_response.py --motor-id 0 --step-size 0.5 --kp 50 --kd 2
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot.bimanual import Bimanual


def step_response_test(motor_id: int, step_size: float = 0.5, kp: float = 50.0,
                       kd: float = 2.0, torque_limit: float = 1.0,
                       settle_time: float = 3.0, plot_file: str = None):
    """
    Perform a step response test on a motor.

    Args:
        motor_id: Motor index (0-11)
        step_size: Step size in radians
        kp: Position proportional gain
        kd: Velocity damping gain
        torque_limit: Maximum torque in Nm
        settle_time: Time to wait for settling (seconds)
        plot_file: Optional file to save response plot
    """
    np.set_printoptions(precision=3, suppress=True)

    control_rate = RateLimiter(100)  # 100 Hz control loop
    robot = Bimanual()

    robot.start(kp=kp, kd=kd, torque_limit=torque_limit)

    actions = np.zeros((12,), dtype=np.float32)

    print("=" * 70)
    print("Step Response Test")
    print("=" * 70)
    print(f"Motor ID: {motor_id}")
    print(f"Step size: {step_size:.3f} rad ({np.rad2deg(step_size):.2f}°)")
    print(f"Control gains: kp={kp}, kd={kd}, torque_limit={torque_limit} Nm")
    print(f"Settle time: {settle_time:.1f}s")
    print("-" * 70)

    # Data collection
    time_data = []
    position_data = []
    target_data = []
    error_data = []

    try:
        # Phase 1: Initial position (zero)
        print("\nPhase 1: Moving to initial position (0.0 rad)...")
        t = 0.0
        while t < 1.0:  # 1 second to reach initial position
            actions[motor_id] = 0.0
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions
            t += control_rate.period()
            control_rate.sleep()

        initial_position = obs[motor_id]
        print(f"Initial position: {initial_position:.3f} rad")

        # Phase 2: Step input
        print(f"\nPhase 2: Applying step input to {step_size:.3f} rad...")
        step_start_time = time.perf_counter()
        t = 0.0
        iteration = 0

        # Record initial state
        time_data.append(0.0)
        position_data.append(initial_position)
        target_data.append(0.0)
        error_data.append(0.0)

        # Apply step and record response
        while t < settle_time:
            actions[motor_id] = step_size
            obs = robot.step(actions * robot.joint_axis_directions) * robot.joint_axis_directions

            # Record data
            current_time = time.perf_counter() - step_start_time
            time_data.append(current_time)
            position_data.append(obs[motor_id])
            target_data.append(step_size)
            error_data.append(step_size - obs[motor_id])

            # Print progress
            if iteration % 25 == 0:  # Print every 0.25 seconds
                error = step_size - obs[motor_id]
                error_pct = (abs(error) / abs(step_size)) * 100 if step_size != 0 else 0
                print(f"t={current_time:.2f}s | pos={obs[motor_id]:.3f} rad | "
                      f"error={error:.3f} rad ({error_pct:.1f}%)")

            iteration += 1
            t += control_rate.period()
            control_rate.sleep()

        # Calculate metrics
        final_position = position_data[-1]
        final_error = abs(step_size - final_position)
        steady_state_error = final_error

        # Find rise time (10% to 90%)
        rise_start_val = 0.1 * step_size
        rise_end_val = 0.9 * step_size
        rise_start_idx = None
        rise_end_idx = None

        for i, pos in enumerate(position_data):
            if rise_start_idx is None and pos >= rise_start_val:
                rise_start_idx = i
            if rise_end_idx is None and pos >= rise_end_val:
                rise_end_idx = i
                break

        rise_time = None
        if rise_start_idx is not None and rise_end_idx is not None:
            rise_time = time_data[rise_end_idx] - time_data[rise_start_idx]

        # Find settling time (within 2% of final value)
        settling_threshold = 0.02 * abs(step_size)
        settling_idx = None
        for i in range(len(error_data) - 1, 0, -1):
            if abs(error_data[i]) > settling_threshold:
                settling_idx = i + 1
                break
        settling_time = time_data[settling_idx] if settling_idx else time_data[-1]

        # Find overshoot
        max_position = max(position_data)
        overshoot = max(0, max_position - step_size)
        overshoot_pct = (overshoot / abs(step_size)) * 100 if step_size != 0 else 0

        # Print results
        print("\n" + "=" * 70)
        print("Step Response Analysis Results")
        print("=" * 70)
        print(f"Target position:        {step_size:.3f} rad ({np.rad2deg(step_size):.2f}°)")
        print(f"Final position:         {final_position:.3f} rad ({np.rad2deg(final_position):.2f}°)")
        print(f"Steady-state error:     {steady_state_error:.4f} rad ({np.rad2deg(steady_state_error):.3f}°)")
        print(f"Steady-state error %:   {(steady_state_error/abs(step_size)*100):.2f}%")
        if rise_time:
            print(f"Rise time (10%-90%):    {rise_time:.3f} s")
        print(f"Settling time (2%):     {settling_time:.3f} s")
        print(f"Overshoot:              {overshoot:.4f} rad ({overshoot_pct:.2f}%)")
        print(f"Max position:           {max_position:.3f} rad")
        print("=" * 70)

        # Generate plot
        if plot_file or True:  # Always generate plot
            plt.figure(figsize=(12, 8))

            # Position plot
            plt.subplot(2, 1, 1)
            plt.plot(time_data, position_data, 'b-', linewidth=2, label='Actual Position')
            plt.plot(time_data, target_data, 'r--', linewidth=2, label='Target Position')
            plt.axhline(y=step_size * 0.9, color='g', linestyle=':', alpha=0.5, label='90% Target')
            plt.axhline(y=step_size * 1.02, color='orange', linestyle=':', alpha=0.5, label='±2% Band')
            plt.axhline(y=step_size * 0.98, color='orange', linestyle=':', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time (s)')
            plt.ylabel('Position (rad)')
            plt.title(f'Step Response - Motor {motor_id} (kp={kp}, kd={kd}, torque_limit={torque_limit})')
            plt.legend()

            # Error plot
            plt.subplot(2, 1, 2)
            plt.plot(time_data, error_data, 'r-', linewidth=2, label='Position Error')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=settling_threshold, color='orange', linestyle='--', alpha=0.5, label='±2% Threshold')
            plt.axhline(y=-settling_threshold, color='orange', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)
            plt.xlabel('Time (s)')
            plt.ylabel('Error (rad)')
            plt.title('Tracking Error')
            plt.legend()

            plt.tight_layout()

            output_file = plot_file if plot_file else f'step_response_motor{motor_id}_kp{kp}_kd{kd}.png'
            plt.savefig(output_file, dpi=150)
            print(f"\nPlot saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        # Return to zero position
        print("\nReturning to zero position...")
        actions[motor_id] = 0.0
        for _ in range(50):  # 0.5 seconds
            robot.step(actions * robot.joint_axis_directions)
            control_rate.sleep()

        robot.stop()
        print("Robot stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step response analysis for motor tuning")
    parser.add_argument("--motor-id", type=int, default=0,
                        help="Motor ID (0-11)")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="Step size in radians")
    parser.add_argument("--kp", type=float, default=50.0,
                        help="Position proportional gain")
    parser.add_argument("--kd", type=float, default=2.0,
                        help="Velocity damping gain")
    parser.add_argument("--torque-limit", type=float, default=1.0,
                        help="Maximum torque in Nm")
    parser.add_argument("--settle-time", type=float, default=3.0,
                        help="Time to wait for settling (seconds)")
    parser.add_argument("--plot", type=str, default=None,
                        help="Output file for response plot (auto-generated if not specified)")

    args = parser.parse_args()

    if args.motor_id < 0 or args.motor_id > 11:
        print("Error: Motor ID must be between 0 and 11")
        exit(1)

    step_response_test(
        motor_id=args.motor_id,
        step_size=args.step_size,
        kp=args.kp,
        kd=args.kd,
        torque_limit=args.torque_limit,
        settle_time=args.settle_time,
        plot_file=args.plot
    )
