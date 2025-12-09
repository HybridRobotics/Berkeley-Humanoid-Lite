// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include "motor_controller.h"
#include "motor_controller_conf.h"


MotorController::MotorController(SocketCan *bus, size_t device_id) : bus(bus), device_id(device_id) {
}


void MotorController::ping() {
  printf("Pinging motor controller %d\n", (int)device_id);

  // send ping command
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_PDO_1);
  tx_frame.len = 0;

  bus->write(&tx_frame);
  
  // wait for response
  can_frame rx_frame = bus->read();

  // if (get_device_id(rx_frame.can_id) == device_id && get_func_id(rx_frame.can_id) == FUNC_PING) {
  //   printf("Received ping response from motor joint %d\n", (int)joint->device_id);
  //   break;
  // }
}

void MotorController::feed() {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_HEARTBEAT);
  tx_frame.len = 0;
  bus->write(&tx_frame);
}

void MotorController::set_mode(uint8_t mode) {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_NMT);
  tx_frame.len = 2;
  *((uint8_t *)(tx_frame.data)) = mode;
  *((uint8_t *)(tx_frame.data + 1)) = device_id;
  bus->write(&tx_frame);
}

void MotorController::load_settings_from_flash() {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_FLASH);
  tx_frame.len = 1;
  *((uint8_t *)(tx_frame.data)) = 2;
  bus->write(&tx_frame);
}

void MotorController::store_settings_to_flash() {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_FLASH);
  tx_frame.len = 1;
  *((uint8_t *)(tx_frame.data)) = 1;
  bus->write(&tx_frame);
}

float MotorController::read_parameter_f32(Parameter param_id) {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_SDO);
  tx_frame.len = 3;
  *((uint8_t *)(tx_frame.data)) = 0x02 << 5;
  *((uint16_t *)(tx_frame.data + 1)) = param_id;
  bus->write(&tx_frame);

  can_frame rx_frame = bus->read();
  if (get_device_id(rx_frame.can_id) == device_id) {
    return *((float *)(rx_frame.data));
  }
  return 0;
}

int32_t MotorController::read_parameter_i32(Parameter param_id) {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_SDO);
  tx_frame.len = 3;
  *((uint8_t *)(tx_frame.data)) = 0x02 << 5;
  *((uint16_t *)(tx_frame.data + 1)) = param_id;
  bus->write(&tx_frame);

  can_frame rx_frame = bus->read();
  if (get_device_id(rx_frame.can_id) == device_id) {
    return *((int32_t *)(rx_frame.data));
  }
  return 0;
}

uint32_t MotorController::read_parameter_u32(Parameter param_id) {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_SDO);
  tx_frame.len = 3;
  *((uint8_t *)(tx_frame.data)) = 0x02 << 5;
  *((uint16_t *)(tx_frame.data + 1)) = param_id;
  bus->write(&tx_frame);

  can_frame rx_frame = bus->read();
  if (get_device_id(rx_frame.can_id) == device_id) {
    return *((uint32_t *)(rx_frame.data));
  }
  return 0;
}

void MotorController::write_parameter_f32(Parameter param_id, float value) {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_SDO);
  tx_frame.len = 8;
  *((uint8_t *)(tx_frame.data)) = 0x01 << 5;
  *((uint16_t *)(tx_frame.data + 1)) = param_id;
  *((uint8_t *)(tx_frame.data + 3)) = 0;
  *((float *)(tx_frame.data + 4)) = value;
  bus->write(&tx_frame);
}

void MotorController::write_parameter_i32(Parameter param_id, int32_t value) {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_SDO);
  tx_frame.len = 7;
  *((uint8_t *)(tx_frame.data)) = 0x02 << 5;
  *((uint16_t *)(tx_frame.data + 1)) = param_id;
  *((uint8_t *)(tx_frame.data + 3)) = 0;
  *((int32_t *)(tx_frame.data + 4)) = value;
  bus->write(&tx_frame);
}

void MotorController::write_parameter_u32(Parameter param_id, uint32_t value) {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_SDO);
  tx_frame.len = 7;
  *((uint8_t *)(tx_frame.data)) = 0x02 << 5;
  *((uint16_t *)(tx_frame.data + 1)) = param_id;
  *((uint8_t *)(tx_frame.data + 3)) = 0;
  *((uint32_t *)(tx_frame.data + 4)) = value;
  bus->write(&tx_frame);
}

uint32_t MotorController::read_fast_frame_frequency() {
  return read_parameter_f32(PARAM_FAST_FRAME_FREQUENCY);
}

void MotorController::write_fast_frame_frequency(uint32_t frequency) {
  write_parameter_f32(PARAM_FAST_FRAME_FREQUENCY, frequency);
}

float MotorController::read_gear_ratio() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_GEAR_RATIO);
}

void MotorController::write_gear_ratio(float ratio) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_GEAR_RATIO, ratio);
}

float MotorController::read_position_kp() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_KP);
}

void MotorController::write_position_kp(float kp) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_KP, kp);
}

float MotorController::read_position_kd() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_KP);
}

void MotorController::write_position_kd(float kd) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_KP, kd);
}

float MotorController::read_position_ki() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_KI);
}

void MotorController::write_position_ki(float ki) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_KI, ki);
}

float MotorController::read_velocity_kp() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_KP);
}

void MotorController::write_velocity_kp(float kp) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_KP, kp);
}

float MotorController::read_velocity_ki() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_KI);
}

void MotorController::write_velocity_ki(float ki) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_KI, ki);
}

float MotorController::read_torque_limit() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_TORQUE_LIMIT);
}

void MotorController::write_torque_limit(float torque) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_TORQUE_LIMIT, torque);
}

float MotorController::read_velocity_limit() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_LIMIT);
}

void MotorController::write_velocity_limit(float limit) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_LIMIT, limit);
}

float MotorController::read_position_limit_lower() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_LIMIT_LOWER);
}

void MotorController::write_position_limit_lower(float limit) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_LIMIT_LOWER, limit);
}

float MotorController::read_position_limit_upper() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_LIMIT_UPPER);
}

void MotorController::write_position_limit_upper(float limit) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_LIMIT_UPPER, limit);
}

float MotorController::read_position_offset() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_OFFSET);
}

void MotorController::write_position_offset(float offset) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_OFFSET, offset);
}

float MotorController::read_torque_target() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_TORQUE_TARGET);
}

void MotorController::write_torque_target(float torque) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_TORQUE_TARGET, torque);
}

float MotorController::read_torque_measured() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_TORQUE_MEASURED);
}

float MotorController::read_velocity_measured() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_VELOCITY_MEASURED);
}

float MotorController::read_position_target() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_TARGET);
}

void MotorController::write_position_target(float position) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_TARGET, position);
}

float MotorController::read_position_measured() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_POSITION_MEASURED);
}

float MotorController::read_torque_filter_alpha() {
  return read_parameter_f32(PARAM_POSITION_CONTROLLER_TORQUE_FILTER_ALPHA);
}

void MotorController::write_torque_filter_alpha(float alpha) {
  write_parameter_f32(PARAM_POSITION_CONTROLLER_TORQUE_FILTER_ALPHA, alpha);
}

float MotorController::read_current_limit() {
  return read_parameter_f32(PARAM_CURRENT_CONTROLLER_I_LIMIT);
}

void MotorController::write_current_limit(float current) {
  write_parameter_f32(PARAM_CURRENT_CONTROLLER_I_LIMIT, current);
}

float MotorController::read_current_kp() {
  return read_parameter_f32(PARAM_CURRENT_CONTROLLER_I_KP);
}

void MotorController::write_current_kp(float kp) {
  write_parameter_f32(PARAM_CURRENT_CONTROLLER_I_KP, kp);
}

float MotorController::read_current_ki() {
  return read_parameter_f32(PARAM_CURRENT_CONTROLLER_I_KI);
}

void MotorController::write_current_ki(float ki) {
  write_parameter_f32(PARAM_CURRENT_CONTROLLER_I_KI, ki);
}

float MotorController::read_bus_voltage_filter_alpha() {
  return read_parameter_f32(PARAM_POWERSTAGE_BUS_VOLTAGE_FILTER_ALPHA);
}

void MotorController::write_bus_voltage_filter_alpha(float alpha) {
  write_parameter_f32(PARAM_POWERSTAGE_BUS_VOLTAGE_FILTER_ALPHA, alpha);
}

float MotorController::read_motor_torque_constant() {
  return read_parameter_f32(PARAM_MOTOR_TORQUE_CONSTANT);
}

void MotorController::write_motor_torque_constant(float torque_constant) {
  write_parameter_f32(PARAM_MOTOR_TORQUE_CONSTANT, torque_constant);
}

int MotorController::read_motor_phase_order() {
  return read_parameter_i32(PARAM_MOTOR_PHASE_ORDER);
}

float MotorController::read_encoder_velocity_filter_alpha() {
  return read_parameter_f32(PARAM_ENCODER_VELOCITY_FILTER_ALPHA);
}

void MotorController::write_encoder_velocity_filter_alpha(float alpha) {
  write_parameter_f32(PARAM_ENCODER_VELOCITY_FILTER_ALPHA, alpha);
}

void MotorController::set_current_bandwidth(float bandwidth_hz, float phase_resistance, float phase_inductance) {
  float kp = bandwidth_hz * 2.0 * M_PI * phase_inductance;
  float ki = phase_resistance / phase_inductance;
  write_current_kp(kp);
  write_current_ki(ki);
}

void MotorController::set_torque_bandwidth(float bandwidth_hz, float position_loop_rate) {
  float alpha = fmin(fmax(1. - exp(-2. * M_PI * (bandwidth_hz / position_loop_rate)), 0.), 1.);
  write_torque_filter_alpha(alpha);
}

void MotorController::set_bus_voltage_bandwidth(float bandwidth_hz, float bus_voltage_update_rate) {
  float alpha = fmin(fmax(1. - exp(-2. * M_PI * (bandwidth_hz / bus_voltage_update_rate)), 0.), 1.);
  write_bus_voltage_filter_alpha(alpha);
}

void MotorController::set_encoder_velocity_bandwidth(float bandwidth_hz, float encoder_update_rate) {
  float alpha = fmin(fmax(1. - exp(-2. * M_PI * (bandwidth_hz / encoder_update_rate)), 0.), 1.);
  write_encoder_velocity_filter_alpha(alpha);
}


void MotorController::read_pdo_2() {
  can_frame rx_frame = bus->read();
  if (get_device_id(rx_frame.can_id) == device_id) {
    this->position_measured = *((float *)rx_frame.data + 0);
    this->velocity_measured = *((float *)rx_frame.data + 1);
  }
}

void MotorController::write_pdo_2() {
  can_frame tx_frame;
  tx_frame.can_id = make_can_id(device_id, FUNC_RECEIVE_PDO_2);
  tx_frame.len = 8;
  *((float *)tx_frame.data + 0) = this->position_target;
  *((float *)tx_frame.data + 1) = this->velocity_target;

  bus->write(&tx_frame);
}

float MotorController::get_measured_position() {
  return this->position_measured;
}

float MotorController::get_measured_velocity() {
  return this->velocity_measured;
}

void MotorController::set_target_position(float position) {
  this->position_target = position;
}

void MotorController::set_target_velocity(float velocity) {
  this->velocity_target = velocity;
}
