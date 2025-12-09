// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#pragma once

#include <stdint.h>
#include <math.h>

#include "socketcan.h"
#include "motor_controller_conf.h"


#define DEVICE_ID_MSK  0b1111111
#define FUNC_ID_POS    7
#define FUNC_ID_MSK    (0b1111 << FUNC_ID_POS)


// Helper functions for CAN frames
inline uint32_t make_can_id(uint8_t device_id, uint8_t func_id) {
  return ((uint32_t)func_id << FUNC_ID_POS) | device_id;
}

inline uint8_t get_device_id(uint32_t can_id) {
  return can_id & DEVICE_ID_MSK;
}


class MotorController {
  public:
    MotorController(SocketCan *bus, size_t device_id);

    void ping();
    void feed();
    void set_mode(uint8_t mode);
    void load_settings_from_flash();
    void store_settings_to_flash();

    uint32_t read_fast_frame_frequency();
    void write_fast_frame_frequency(uint32_t frequency);
    float read_gear_ratio();
    void write_gear_ratio(float ratio);
    float read_position_kp();
    void write_position_kp(float kp);
    float read_position_kd();
    void write_position_kd(float kd);
    float read_position_ki();
    void write_position_ki(float ki);
    float read_velocity_kp();
    void write_velocity_kp(float kp);
    float read_velocity_ki();
    void write_velocity_ki(float ki);
    float read_torque_limit();
    void write_torque_limit(float torque);
    float read_velocity_limit();
    void write_velocity_limit(float limit);
    float read_position_limit_lower();
    void write_position_limit_lower(float limit);
    float read_position_limit_upper();
    void write_position_limit_upper(float limit);
    float read_position_offset();
    void write_position_offset(float offset);
    float read_torque_target();
    void write_torque_target(float torque);
    float read_torque_measured();
    float read_velocity_measured();
    float read_position_target();
    void write_position_target(float position);
    float read_position_measured();
    float read_torque_filter_alpha();
    void write_torque_filter_alpha(float alpha);
    float read_current_limit();
    void write_current_limit(float current);
    float read_current_kp();
    void write_current_kp(float kp);
    float read_current_ki();
    void write_current_ki(float ki);
    float read_bus_voltage_filter_alpha();
    void write_bus_voltage_filter_alpha(float alpha);
    float read_motor_torque_constant();
    void write_motor_torque_constant(float torque_constant);
    int read_motor_phase_order();
    void write_motor_phase_order(int order);
    float read_encoder_velocity_filter_alpha();
    void write_encoder_velocity_filter_alpha(float alpha);
    void write_pdo_2();
    void read_pdo_2();

    void set_current_bandwidth(float bandwidth_hz, float phase_resistance, float phase_inductance);
    void set_torque_bandwidth(float bandwidth_hz, float position_loop_rate);
    void set_bus_voltage_bandwidth(float bandwidth_hz, float bus_voltage_update_rate);
    void set_encoder_velocity_bandwidth(float bandwidth_hz, float encoder_update_rate);

    float get_measured_position();
    float get_measured_velocity();
    void set_target_position(float position);
    void set_target_velocity(float velocity);

private:
    SocketCan *bus;
    size_t device_id;

    float position_measured;
    float velocity_measured;

    float position_target;
    float velocity_target;

    float read_parameter_f32(Parameter param_id);
    int32_t read_parameter_i32(Parameter param_id);
    uint32_t read_parameter_u32(Parameter param_id);
    void write_parameter_f32(Parameter param_id, float value);
    void write_parameter_i32(Parameter param_id, int32_t value);
    void write_parameter_u32(Parameter param_id, uint32_t value);
};
