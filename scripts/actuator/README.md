# Berkeley Humanoid Lite - Actuator Control Scripts

This directory contains scripts for precise control of the Berkeley Humanoid Lite actuators. These scripts provide various control modes for testing, calibration, and precise motor manipulation.

## Prerequisites

Make sure you have:
- Initialized the `berkeley_humanoid_lite_lowlevel` submodule
- Installed required dependencies (`loop_rate_limiters`, etc.)
- Connected to the robot hardware
- Configured `motor_configuration.json` in the root directory

## Available Scripts

### 1. **position_control.py** - Single Motor Position Control

Control a single motor to a target position with configurable PID gains.

**Usage:**
```bash
python position_control.py --motor-id 0 --position 0.5 --kp 50 --kd 2
```

**Arguments:**
- `--motor-id`: Motor ID (0-11)
- `--position`: Target position in radians
- `--kp`: Position proportional gain (default: 50.0)
- `--kd`: Velocity damping gain (default: 2.0)
- `--torque-limit`: Maximum torque in Nm (default: 1.0)
- `--duration`: Duration to hold position in seconds (default: 5.0)

**Examples:**
```bash
# Move motor 0 to 0.5 radians
python position_control.py --motor-id 0 --position 0.5

# Move motor 3 to -0.3 radians with high stiffness
python position_control.py --motor-id 3 --position -0.3 --kp 80 --kd 3

# Move motor 5 with low torque limit for safety
python position_control.py --motor-id 5 --position 0.2 --torque-limit 0.5
```

---

### 2. **trajectory_control.py** - Smooth Trajectory Following

Execute smooth trajectories (sine, triangle, square waves) for precise motion testing.

**Usage:**
```bash
python trajectory_control.py --motor-id 0 --trajectory sine --amplitude 0.5 --frequency 0.5
```

**Arguments:**
- `--motor-id`: Motor ID (0-11)
- `--trajectory`: Trajectory type ('sine', 'triangle', 'square')
- `--amplitude`: Trajectory amplitude in radians (default: 0.5)
- `--frequency`: Trajectory frequency in Hz (default: 0.5)
- `--offset`: Center position offset in radians (default: 0.0)
- `--duration`: Duration to run in seconds (default: 10.0)
- `--kp`, `--kd`, `--torque-limit`: PID gains and torque limit

**Examples:**
```bash
# Sinusoidal motion at 0.5 Hz
python trajectory_control.py --motor-id 0 --trajectory sine --amplitude 0.5 --frequency 0.5

# Fast triangular motion
python trajectory_control.py --motor-id 2 --trajectory triangle --amplitude 0.3 --frequency 1.0

# Slow square wave with offset
python trajectory_control.py --motor-id 4 --trajectory square --amplitude 0.4 --frequency 0.2 --offset 0.1
```

---

### 3. **torque_control.py** - Gravity Compensation and Compliant Control

Run motors in compliant mode for manual teaching or gravity compensation.

**Usage:**
```bash
python torque_control.py --mode gravity_comp --motor-id 0
```

**Arguments:**
- `--mode`: Control mode ('gravity_comp', 'zero_torque')
- `--motor-id`: Motor ID (None = all motors)
- `--duration`: Duration to run in seconds (default: 30.0)

**Examples:**
```bash
# Gravity compensation on all motors (robot becomes manually movable)
python torque_control.py --mode gravity_comp

# Gravity compensation on single motor
python torque_control.py --mode gravity_comp --motor-id 3

# Zero torque mode (WARNING: robot will collapse!)
python torque_control.py --mode zero_torque --duration 5
```

**Note:** In gravity compensation mode, you can manually move the robot while it provides gentle resistance. This is useful for teaching positions or testing joint range of motion.

---

### 4. **step_response.py** - Step Response Analysis and PID Tuning

Perform step response tests for system identification and PID parameter tuning. Generates plots showing position tracking and error dynamics.

**Usage:**
```bash
python step_response.py --motor-id 0 --step-size 0.5 --kp 50 --kd 2
```

**Arguments:**
- `--motor-id`: Motor ID (0-11)
- `--step-size`: Step size in radians (default: 0.5)
- `--kp`: Position proportional gain (default: 50.0)
- `--kd`: Velocity damping gain (default: 2.0)
- `--torque-limit`: Maximum torque in Nm (default: 1.0)
- `--settle-time`: Time to wait for settling in seconds (default: 3.0)
- `--plot`: Output file for response plot (auto-generated if not specified)

**Examples:**
```bash
# Basic step response test
python step_response.py --motor-id 0 --step-size 0.5

# Test different PID gains
python step_response.py --motor-id 1 --kp 80 --kd 3 --step-size 0.3

# Long settling time for slow systems
python step_response.py --motor-id 2 --step-size 0.4 --settle-time 5.0

# Custom output plot file
python step_response.py --motor-id 0 --step-size 0.5 --plot my_test.png
```

**Metrics Provided:**
- Rise time (10% to 90%)
- Settling time (within 2% of target)
- Overshoot percentage
- Steady-state error
- Position and error plots

**Use Case:** This script is essential for tuning PID parameters. Run tests with different kp and kd values to find optimal gains that minimize overshoot and settling time while maintaining stability.

---

### 5. **multi_motor_control.py** - Coordinated Multi-Motor Control

Control multiple motors simultaneously with coordinated motion patterns.

**Usage:**
```bash
python multi_motor_control.py --pattern wave
```

**Arguments:**
- `--pattern`: Motion pattern ('wave', 'mirror', 'sync')
- `--duration`: Duration to run in seconds (default: 15.0)
- `--kp`, `--kd`, `--torque-limit`: PID gains and torque limit

**Examples:**
```bash
# Wave pattern across all motors
python multi_motor_control.py --pattern wave

# Mirrored motion between left and right arms
python multi_motor_control.py --pattern mirror --duration 20

# Synchronized motion of all motors
python multi_motor_control.py --pattern sync --kp 40
```

**Patterns:**
- **wave**: Creates a wave motion across motors with phase offset
- **mirror**: Mirrors motion between left and right arms
- **sync**: All motors move in synchronization

---

### 6. **monitor_motors.py** - Real-time Motor State Monitor

Monitor motor states in real-time and optionally log data to CSV.

**Usage:**
```bash
python monitor_motors.py --motor-id 0 --rate 10
```

**Arguments:**
- `--motor-id`: Specific motor to monitor (None = all motors)
- `--rate`: Display update rate in Hz (default: 10.0)
- `--log`: CSV file to log data (optional)
- `--duration`: Duration to monitor in seconds (None = until Ctrl+C)

**Examples:**
```bash
# Monitor all motors at 10 Hz
python monitor_motors.py

# Monitor single motor with high update rate
python monitor_motors.py --motor-id 0 --rate 50

# Monitor and log to CSV file
python monitor_motors.py --log motor_data.csv --duration 60

# Monitor specific motor and save data
python monitor_motors.py --motor-id 5 --log motor5_data.csv --rate 100
```

---

## Motor Configuration

The actuators use the configuration defined in `motor_configuration.json` at the repository root:

**Key Parameters:**
- **Gear Ratio**: -15.0
- **Position Control**: kp=50.0, ki=0.0
- **Velocity Control**: kp=2.0, ki=0.0
- **Torque Limit**: 1.0 Nm (configurable per script)
- **Velocity Limit**: 20.0 rad/s
- **Encoder**: 4096 counts per revolution (CPR)

## Safety Guidelines

⚠️ **Important Safety Notes:**

1. **Start with low gains**: When testing, start with lower kp and kd values and gradually increase
2. **Use torque limits**: Always specify appropriate torque limits to prevent damage
3. **Emergency stop**: Keep Ctrl+C ready to stop the robot immediately
4. **Support the robot**: When using compliant modes, ensure the robot is properly supported
5. **Test incrementally**: Test small motions before attempting large movements
6. **Check connections**: Verify motor connections before running control scripts

## Motor ID Reference

For the Bimanual configuration:
- Motors 0-4: Left arm joints
- Motors 5-9: Right arm joints
- Motors 10-11: Grippers (if equipped)

Refer to your robot's documentation for specific joint mappings.

## Common Commands Summary

```bash
# Quick position test
python position_control.py --motor-id 0 --position 0.3 --duration 3

# Smooth sine wave test
python trajectory_control.py --motor-id 1 --trajectory sine --amplitude 0.4

# Make robot compliant for manual positioning
python torque_control.py --mode gravity_comp --duration 60

# Coordinated wave motion
python multi_motor_control.py --pattern wave --duration 10

# PID tuning with step response
python step_response.py --motor-id 0 --step-size 0.5 --kp 50 --kd 2

# Monitor and record motor data
python monitor_motors.py --motor-id 0 --log test_data.csv --duration 30
```

## Troubleshooting

**Issue**: Script can't import `berkeley_humanoid_lite_lowlevel`
- **Solution**: Initialize the submodule: `git submodule update --init --recursive`

**Issue**: Motors not responding
- **Solution**: Check hardware connections and verify with `check_connection.py`

**Issue**: High tracking error
- **Solution**: Increase kp gain or reduce torque limit, check for mechanical binding

**Issue**: Oscillations or instability
- **Solution**: Reduce kp gain or increase kd damping

## Additional Resources

- Main repository: [Berkeley-Humanoid-Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)
- Documentation: [Berkeley Humanoid Lite Docs](https://berkeley-humanoid-lite.gitbook.io)
- Lowlevel code: [Berkeley-Humanoid-Lite-Lowlevel](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel)
