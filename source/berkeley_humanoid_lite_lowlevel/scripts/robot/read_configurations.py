# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

import json
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot import Humanoid


robot_configuration = {}

robot = Humanoid()

robot.check_connection()

for entry in robot.joints:
    bus, joint_id, joint_name = entry

    print(f"Reading configuration for {joint_name}")

    config = {
        "position_controller": {},
        "current_controller": {},
        "powerstage": {},
        "motor": {},
        "encoder": {},
    }

    config["device_id"] = bus._read_parameter_u32(joint_id, recoil.Parameter.DEVICE_ID)
    config["firmware_version"] = hex(bus._read_parameter_u32(joint_id, recoil.Parameter.FIRMWARE_VERSION))
    config["watchdog_timeout"] = bus._read_parameter_u32(joint_id, recoil.Parameter.WATCHDOG_TIMEOUT)
    config["fast_frame_frequency"] = bus.read_fast_frame_frequency(joint_id)

    config["position_controller"]["gear_ratio"] = bus._read_parameter_f32(joint_id, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
    config["position_controller"]["position_kp"] = bus.read_position_kp(joint_id)
    config["position_controller"]["position_ki"] = bus.read_position_ki(joint_id)
    config["position_controller"]["velocity_kp"] = bus.read_velocity_kp(joint_id)
    config["position_controller"]["velocity_ki"] = bus.read_velocity_ki(joint_id)
    config["position_controller"]["torque_limit"] = bus.read_torque_limit(joint_id)
    config["position_controller"]["velocity_limit"] = bus.read_velocity_limit(joint_id)
    config["position_controller"]["position_limit_upper"] = bus.read_position_limit_upper(joint_id)
    config["position_controller"]["position_limit_lower"] = bus.read_position_limit_lower(joint_id)
    config["position_controller"]["position_offset"] = bus.read_position_offset(joint_id)
    config["position_controller"]["torque_filter_alpha"] = bus.read_torque_filter_alpha(joint_id)

    config["current_controller"]["i_limit"] = bus.read_current_limit(joint_id)
    config["current_controller"]["i_kp"] = bus.read_current_kp(joint_id)
    config["current_controller"]["i_ki"] = bus.read_current_ki(joint_id)

    config["powerstage"]["undervoltage_threshold"] = bus._read_parameter_f32(joint_id, recoil.Parameter.POWERSTAGE_UNDERVOLTAGE_THRESHOLD)
    config["powerstage"]["overvoltage_threshold"] = bus._read_parameter_f32(joint_id, recoil.Parameter.POWERSTAGE_OVERVOLTAGE_THRESHOLD)
    config["powerstage"]["bus_voltage_filter_alpha"] = bus.read_bus_voltage_filter_alpha(joint_id)

    config["motor"]["pole_pairs"] = bus.read_motor_pole_pairs(joint_id)
    config["motor"]["torque_constant"] = bus.read_motor_torque_constant(joint_id)
    config["motor"]["phase_order"] = bus.read_motor_phase_order(joint_id)
    config["motor"]["max_calibration_current"] = bus.read_motor_calibration_current(joint_id)

    config["encoder"]["cpr"] = bus.read_encoder_cpr(joint_id)
    config["encoder"]["position_offset"] = bus.read_encoder_position_offset(joint_id)
    config["encoder"]["velocity_filter_alpha"] = bus.read_encoder_velocity_filter_alpha(joint_id)
    config["encoder"]["flux_offset"] = bus.read_encoder_flux_offset(joint_id)

    robot_configuration[joint_name] = config
    time.sleep(0.1)


with open("robot_configuration.json", "w") as f:
    json.dump(robot_configuration, f, indent=4)

robot.stop(joint_id)

print("Done")
