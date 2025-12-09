# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import berkeley_humanoid_lite_lowlevel.recoil as recoil


args = recoil.util.get_args()
bus = recoil.Bus(channel=args.channel, bitrate=1000000)

device_id = args.id


def configure_fast_frame_rate(bus: recoil.Bus, device_id: int, fast_frame_rate: int):
    print("Rate (before):\t", bus._read_parameter_u32(device_id, recoil.Parameter.FAST_FRAME_FREQUENCY))
    bus._write_parameter_u32(device_id, recoil.Parameter.FAST_FRAME_FREQUENCY, fast_frame_rate)
    print("Rate (updated):\t", bus._read_parameter_u32(device_id, recoil.Parameter.FAST_FRAME_FREQUENCY))


def configure_gear_ratio(bus: recoil.Bus, device_id: int, gear_ratio: float):
    print("Gear Ratio (before):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO))
    bus._write_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO, -gear_ratio)
    print("Gear Ratio (updated):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO))


def configure_position_pd(bus: recoil.Bus, device_id: int, kp: float, kd: float):
    print("Kp (before):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_KP))
    bus._write_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_KP, kp)
    print("Kp (updated):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_KP))

    print("Kd (before):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_VELOCITY_KP))
    bus._write_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_VELOCITY_KP, kd)
    print("Kd (updated):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_VELOCITY_KP))


def configure_torque_limit(bus: recoil.Bus, device_id: int, torque_limit: float):
    print("Torque Limit (before):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_TORQUE_LIMIT))
    bus._write_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_TORQUE_LIMIT, torque_limit)
    print("Torque Limit (updated):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_TORQUE_LIMIT))


def configure_phase_order(bus: recoil.Bus, device_id: int, phase_order: int):
    print("Phase order (before):\t", bus._read_parameter_i32(device_id, recoil.Parameter.MOTOR_PHASE_ORDER))
    bus._write_parameter_i32(device_id, recoil.Parameter.MOTOR_PHASE_ORDER, phase_order)
    print("Phase order (updated):\t", bus._read_parameter_i32(device_id, recoil.Parameter.MOTOR_PHASE_ORDER))


def configure_position_limit(bus: recoil.Bus, device_id: int, lower_limit: float, upper_limit: float):
    print("Position Lower Limit (before):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_LIMIT_LOWER))
    bus._write_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_LIMIT_LOWER, lower_limit)
    print("Position Lower Limit (updated):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_LIMIT_LOWER))

    print("Position Upper Limit (before):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_LIMIT_UPPER))
    bus._write_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_LIMIT_UPPER, upper_limit)
    print("Position Upper Limit (updated):\t", bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_POSITION_LIMIT_UPPER))


def configure_current_bandwidth(bus: recoil.Bus, device_id: int, bandwidth: float, kp: float = 0.0001, ki: float = 0.000001):
    print("Previous current kp:\t", bus._read_parameter_f32(device_id, recoil.Parameter.CURRENT_CONTROLLER_I_KP))
    print("Previous current ki:\t", bus._read_parameter_f32(device_id, recoil.Parameter.CURRENT_CONTROLLER_I_KI))

    bus.set_current_bandwidth(bandwidth, kp, ki)

    print("New current kp:\t", bus._read_parameter_f32(device_id, recoil.Parameter.CURRENT_CONTROLLER_I_KP))
    print("New current ki:\t", bus._read_parameter_f32(device_id, recoil.Parameter.CURRENT_CONTROLLER_I_KI))


def store_to_flash(bus: recoil.Bus, device_id: int):
    bus.store_setting_to_flash(device_id)
    print("Settings stored to flash!")


# configure_fast_frame_rate(motor, 0)

# configure_gear_ratio(motor, 15)

# configure_position_pd(motor, 20, 1)

# configure_torque_limit(motor, 2)

# configure_phase_order(motor, -1)

# pi = math.pi

# d90 = 0.5 * pi
# d60 = 1./3. * math.pi
# d45 = 0.25 * math.pi
# d30 = 1./6. * math.pi
# d10 = 0.5 * pi/9

# configure_position_limit(motor, -d90, d90)

# configure_current_bandwidth(motor, 100)


# store_to_flash(motor)


bus.stop()
