// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#pragma once


/** ======== Controller Settings ======== **/

/**
 * Firmware Version:
 * The firmware version is represented as a 32-bit hexadecimal number.
 * It follows the format: (MAJOR [7:4]) . (MINOR [3:2]) . (PATCH [1:0]).
 * For example, 0x00010005 represents version 1.0.5 of the firmware.
 */
//                                        YYYYmmdd
#define FIRMWARE_VERSION                0x20250226


/** ======== Controller State Definitions ======== **/

/**
 * @brief Mode definition.
 */
typedef enum {
  // these are three safe modes
  MODE_DISABLED                   = 0x00U,
  MODE_IDLE                       = 0x01U,  // operational

  // these are special modes
  MODE_DAMPING                    = 0x02U,  // stopped
  MODE_CALIBRATION                = 0x05U,

  // these are closed-loop modes
  MODE_CURRENT                    = 0x10U,
  MODE_TORQUE                     = 0x11U,
  MODE_VELOCITY                   = 0x12U,
  MODE_POSITION                   = 0x13U,

  // these are open-loop modes
  MODE_VABC_OVERRIDE              = 0x20U,
  MODE_VALPHABETA_OVERRIDE        = 0x21U,
  MODE_VQD_OVERRIDE               = 0x22U,

  MODE_DEBUG                      = 0x80U,  // pre-operational
} Mode;

/**
 * @brief ErrorCode definition.
 */
typedef enum {
  ERROR_NO_ERROR                  = 0b0000000000000000U,
  ERROR_GENERAL                   = 0b0000000000000001U,
  ERROR_ESTOP                     = 0b0000000000000010U,
  ERROR_INITIALIZATION_ERROR      = 0b0000000000000100U,
  ERROR_CALIBRATION_ERROR         = 0b0000000000001000U,
  ERROR_POWERSTAGE_ERROR          = 0b0000000000010000U,
  ERROR_INVALID_MODE              = 0b0000000000100000U,
  ERROR_WATCHDOG_TIMEOUT          = 0b0000000001000000U,
  ERROR_OVER_VOLTAGE              = 0b0000000010000000U,
  ERROR_OVER_CURRENT              = 0b0000000100000000U,
  ERROR_OVER_TEMPERATURE          = 0b0000001000000000U,
  ERROR_CAN_RX_FAULT              = 0b0000010000000000U,
  ERROR_CAN_TX_FAULT              = 0b0000100000000000U,
  ERROR_I2C_FAULT                 = 0b0001000000000000U,
} ErrorCode;

/** ======== CAN Packet Definitions ======== **/
/**
 * @brief CAN FrameFunction definition.
 */
typedef enum {
  FUNC_NMT                      = 0b0000U,
  FUNC_SYNC_EMCY                = 0b0001U,
  FUNC_TIME                     = 0b0010U,
  FUNC_TRANSMIT_PDO_1           = 0b0011U,
  FUNC_RECEIVE_PDO_1            = 0b0100U,
  FUNC_TRANSMIT_PDO_2           = 0b0101U,
  FUNC_RECEIVE_PDO_2            = 0b0110U,
  FUNC_TRANSMIT_PDO_3           = 0b0111U,
  FUNC_RECEIVE_PDO_3            = 0b1000U,
  FUNC_TRANSMIT_PDO_4           = 0b1001U,
  FUNC_RECEIVE_PDO_4            = 0b1010U,
  FUNC_TRANSMIT_SDO             = 0b1011U,
  FUNC_RECEIVE_SDO              = 0b1100U,
  FUNC_FLASH                    = 0b1101U,
  FUNC_HEARTBEAT                = 0b1110U,
} FrameFunction;

/**
 * @brief CAN Parameter ID definition.
 */
typedef enum {
  PARAM_DEVICE_ID                                       = 0x000U,
  PARAM_FIRMWARE_VERSION                                = 0x004U,
  PARAM_WATCHDOG_TIMEOUT                                = 0x008U,
  PARAM_FAST_FRAME_FREQUENCY                            = 0x00CU,
  PARAM_MODE                                            = 0x010U,
  PARAM_ERROR                                           = 0x014U,
  PARAM_POSITION_CONTROLLER_UPDATE_COUNTER              = 0x018U,
  PARAM_POSITION_CONTROLLER_GEAR_RATIO                  = 0x01CU,
  PARAM_POSITION_CONTROLLER_POSITION_KP                 = 0x020U,
  PARAM_POSITION_CONTROLLER_POSITION_KI                 = 0x024U,
  PARAM_POSITION_CONTROLLER_VELOCITY_KP                 = 0x028U,
  PARAM_POSITION_CONTROLLER_VELOCITY_KI                 = 0x02CU,
  PARAM_POSITION_CONTROLLER_TORQUE_LIMIT                = 0x030U,
  PARAM_POSITION_CONTROLLER_VELOCITY_LIMIT              = 0x034U,
  PARAM_POSITION_CONTROLLER_POSITION_LIMIT_LOWER        = 0x038U,
  PARAM_POSITION_CONTROLLER_POSITION_LIMIT_UPPER        = 0x03CU,
  PARAM_POSITION_CONTROLLER_POSITION_OFFSET             = 0x040U,
  PARAM_POSITION_CONTROLLER_TORQUE_TARGET               = 0x044U,
  PARAM_POSITION_CONTROLLER_TORQUE_MEASURED             = 0x048U,
  PARAM_POSITION_CONTROLLER_TORQUE_SETPOINT             = 0x04CU,
  PARAM_POSITION_CONTROLLER_VELOCITY_TARGET             = 0x050U,
  PARAM_POSITION_CONTROLLER_VELOCITY_MEASURED           = 0x054U,
  PARAM_POSITION_CONTROLLER_VELOCITY_SETPOINT           = 0x058U,
  PARAM_POSITION_CONTROLLER_POSITION_TARGET             = 0x05CU,
  PARAM_POSITION_CONTROLLER_POSITION_MEASURED           = 0x060U,
  PARAM_POSITION_CONTROLLER_POSITION_SETPOINT           = 0x064U,
  PARAM_POSITION_CONTROLLER_POSITION_INTEGRATOR         = 0x068U,
  PARAM_POSITION_CONTROLLER_VELOCITY_INTEGRATOR         = 0x06CU,
  PARAM_POSITION_CONTROLLER_TORQUE_FILTER_ALPHA         = 0x070U,
  PARAM_CURRENT_CONTROLLER_I_LIMIT                      = 0x074U,
  PARAM_CURRENT_CONTROLLER_I_KP                         = 0x078U,
  PARAM_CURRENT_CONTROLLER_I_KI                         = 0x07CU,
  PARAM_CURRENT_CONTROLLER_I_A_MEASURED                 = 0x080U,
  PARAM_CURRENT_CONTROLLER_I_B_MEASURED                 = 0x084U,
  PARAM_CURRENT_CONTROLLER_I_C_MEASURED                 = 0x088U,
  PARAM_CURRENT_CONTROLLER_V_A_SETPOINT                 = 0x08CU,
  PARAM_CURRENT_CONTROLLER_V_B_SETPOINT                 = 0x090U,
  PARAM_CURRENT_CONTROLLER_V_C_SETPOINT                 = 0x094U,
  PARAM_CURRENT_CONTROLLER_I_ALPHA_MEASURED             = 0x098U,
  PARAM_CURRENT_CONTROLLER_I_BETA_MEASURED              = 0x09CU,
  PARAM_CURRENT_CONTROLLER_V_ALPHA_SETPOINT             = 0x0A0U,
  PARAM_CURRENT_CONTROLLER_V_BETA_SETPOINT              = 0x0A4U,
  PARAM_CURRENT_CONTROLLER_V_Q_TARGET                   = 0x0A8U,
  PARAM_CURRENT_CONTROLLER_V_D_TARGET                   = 0x0ACU,
  PARAM_CURRENT_CONTROLLER_V_Q_SETPOINT                 = 0x0B0U,
  PARAM_CURRENT_CONTROLLER_V_D_SETPOINT                 = 0x0B4U,
  PARAM_CURRENT_CONTROLLER_I_Q_TARGET                   = 0x0B8U,
  PARAM_CURRENT_CONTROLLER_I_D_TARGET                   = 0x0BCU,
  PARAM_CURRENT_CONTROLLER_I_Q_MEASURED                 = 0x0C0U,
  PARAM_CURRENT_CONTROLLER_I_D_MEASURED                 = 0x0C4U,
  PARAM_CURRENT_CONTROLLER_I_Q_SETPOINT                 = 0x0C8U,
  PARAM_CURRENT_CONTROLLER_I_D_SETPOINT                 = 0x0CCU,
  PARAM_CURRENT_CONTROLLER_I_Q_INTEGRATOR               = 0x0D0U,
  PARAM_CURRENT_CONTROLLER_I_D_INTEGRATOR               = 0x0D4U,
  PARAM_POWERSTAGE_HTIM                                 = 0x0D8U,
  PARAM_POWERSTAGE_HADC1                                = 0x0DCU,
  PARAM_POWERSTAGE_HADC2                                = 0x0E0U,
  PARAM_POWERSTAGE_ADC_READING_RAW                      = 0x0E4U,
  PARAM_POWERSTAGE_ADC_READING_OFFSET                   = 0x0ECU,
  PARAM_POWERSTAGE_UNDERVOLTAGE_THRESHOLD               = 0x0F4U,
  PARAM_POWERSTAGE_OVERVOLTAGE_THRESHOLD                = 0x0F8U,
  PARAM_POWERSTAGE_BUS_VOLTAGE_FILTER_ALPHA             = 0x0FCU,
  PARAM_POWERSTAGE_BUS_VOLTAGE_MEASURED                 = 0x100U,
  PARAM_MOTOR_POLE_PAIRS                                = 0x104U,
  PARAM_MOTOR_TORQUE_CONSTANT                           = 0x108U,
  PARAM_MOTOR_PHASE_ORDER                               = 0x10CU,
  PARAM_MOTOR_MAX_CALIBRATION_CURRENT                   = 0x110U,
  PARAM_ENCODER_HI2C                                    = 0x114U,
  PARAM_ENCODER_I2C_BUFFER                              = 0x118U,
  PARAM_ENCODER_I2C_UPDATE_COUNTER                      = 0x11CU,
  PARAM_ENCODER_CPR                                     = 0x120U,
  PARAM_ENCODER_POSITION_OFFSET                         = 0x124U,
  PARAM_ENCODER_VELOCITY_FILTER_ALPHA                   = 0x128U,
  PARAM_ENCODER_POSITION_RAW                            = 0x12CU,
  PARAM_ENCODER_N_ROTATIONS                             = 0x130U,
  PARAM_ENCODER_POSITION                                = 0x134U,
  PARAM_ENCODER_VELOCITY                                = 0x138U,
  PARAM_ENCODER_FLUX_OFFSET                             = 0x13CU,
  PARAM_ENCODER_FLUX_OFFSET_TABLE                       = 0x140U,
} Parameter;
