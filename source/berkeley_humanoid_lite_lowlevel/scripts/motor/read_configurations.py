# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import json
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil


args = recoil.util.get_args()
bus = recoil.Bus(channel=args.channel, bitrate=1000000)

device_id = args.id

print(f"Reading configuration from actuator #{args.id}")

config = {
    "position_controller": {},
    "current_controller": {},
    "powerstage": {},
    "motor": {},
    "encoder": {},
}

config["device_id"] = bus._read_parameter_u32(device_id, recoil.Parameter.DEVICE_ID)
config["firmware_version"] = hex(bus._read_parameter_u32(device_id, recoil.Parameter.FIRMWARE_VERSION))
config["watchdog_timeout"] = bus._read_parameter_u32(device_id, recoil.Parameter.WATCHDOG_TIMEOUT)
config["fast_frame_frequency"] = bus.read_fast_frame_frequency(device_id)

config["position_controller"]["gear_ratio"] = bus._read_parameter_f32(device_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
config["position_controller"]["position_kp"] = bus.read_position_kp(device_id)
config["position_controller"]["position_ki"] = bus.read_position_ki(device_id)
config["position_controller"]["velocity_kp"] = bus.read_velocity_kp(device_id)
config["position_controller"]["velocity_ki"] = bus.read_velocity_ki(device_id)
config["position_controller"]["torque_limit"] = bus.read_torque_limit(device_id)
config["position_controller"]["velocity_limit"] = bus.read_velocity_limit(device_id)
config["position_controller"]["position_limit_upper"] = bus.read_position_limit_upper(device_id)
config["position_controller"]["position_limit_lower"] = bus.read_position_limit_lower(device_id)
config["position_controller"]["position_offset"] = bus.read_position_offset(device_id)
config["position_controller"]["torque_filter_alpha"] = bus.read_torque_filter_alpha(device_id)

config["current_controller"]["i_limit"] = bus.read_current_limit(device_id)
config["current_controller"]["i_kp"] = bus.read_current_kp(device_id)
config["current_controller"]["i_ki"] = bus.read_current_ki(device_id)

config["powerstage"]["undervoltage_threshold"] = bus._read_parameter_f32(device_id, recoil.Parameter.POWERSTAGE_UNDERVOLTAGE_THRESHOLD)
config["powerstage"]["overvoltage_threshold"] = bus._read_parameter_f32(device_id, recoil.Parameter.POWERSTAGE_OVERVOLTAGE_THRESHOLD)
config["powerstage"]["bus_voltage_filter_alpha"] = bus.read_bus_voltage_filter_alpha(device_id)

config["motor"]["pole_pairs"] = bus.read_motor_pole_pairs(device_id)
config["motor"]["torque_constant"] = bus.read_motor_torque_constant(device_id)
config["motor"]["phase_order"] = bus.read_motor_phase_order(device_id)
config["motor"]["max_calibration_current"] = bus.read_motor_calibration_current(device_id)

config["encoder"]["cpr"] = bus.read_encoder_cpr(device_id)
config["encoder"]["position_offset"] = bus.read_encoder_position_offset(device_id)
config["encoder"]["velocity_filter_alpha"] = bus.read_encoder_velocity_filter_alpha(device_id)
config["encoder"]["flux_offset"] = bus.read_encoder_flux_offset(device_id)


time.sleep(0.1)


with open("motor_configuration.json", "w") as f:
    json.dump(config, f, indent=4)

bus.stop()

print("Done")
