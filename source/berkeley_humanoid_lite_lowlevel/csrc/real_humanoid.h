// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#pragma once


#include <stdio.h>
#include <yaml-cpp/yaml.h>

#include "consts.h"
#include "loop_function.h"
#include "motor_controller.h"
#include "imu.h"
#include "udp.h"
#include "socketcan.h"




#define N_LOWLEVEL_STATES       (4+3+12+12+1+3)
#define N_LOWLEVEL_COMMANDS     12



static inline float deg2rad(float deg) {
    return deg * M_PI / 180.0f;
}


enum ControllerState {
  STATE_ERROR = 0,
  STATE_IDLE = 1,
  STATE_RL_INIT,
  STATE_RL_RUNNING,
  STATE_GETUP,
  STATE_HOLD,
  STATE_GETDOWN,
};


/**
 * Joint order:
 *  0: left_hip_roll
 *  1: left_hip_yaw
 *  2: left_hip_pitch
 *  3: left_knee_pitch
 *  4: left_ankle_pitch
 *  5: left_ankle_roll
 *  6: right_hip_roll
 *  7: right_hip_yaw
 *  8: right_hip_pitch
 *  9: right_knee_pitch
 *  10: right_ankle_pitch
 *  11: right_ankle_roll
 */
enum JointOrder {
  LEG_LEFT_HIP_ROLL_JOINT         = 0,
  LEG_LEFT_HIP_YAW_JOINT          = 1,
  LEG_LEFT_HIP_PITCH_JOINT        = 2,
  LEG_LEFT_KNEE_PITCH_JOINT       = 3,
  LEG_LEFT_ANKLE_PITCH_JOINT      = 4,
  LEG_LEFT_ANKLE_ROLL_JOINT       = 5,
  LEG_RIGHT_HIP_ROLL_JOINT        = 6,
  LEG_RIGHT_HIP_YAW_JOINT         = 7,
  LEG_RIGHT_HIP_PITCH_JOINT       = 8,
  LEG_RIGHT_KNEE_PITCH_JOINT      = 9,
  LEG_RIGHT_ANKLE_PITCH_JOINT     = 10,
  LEG_RIGHT_ANKLE_ROLL_JOINT      = 11,
};
// enum JointOrder {
//   ARM_LEFT_SHOULDER_PITCH_JOINT   = 0,
//   ARM_LEFT_SHOULDER_ROLL_JOINT    = 1,
//   ARM_LEFT_SHOULDER_YAW_JOINT     = 2,
//   ARM_LEFT_ELBOW_PITCH_JOINT      = 3,
//   ARM_LEFT_ELBOW_ROLL_JOINT       = 4,
//   ARM_RIGHT_SHOULDER_PITCH_JOINT  = 5,
//   ARM_RIGHT_SHOULDER_ROLL_JOINT   = 6,
//   ARM_RIGHT_SHOULDER_YAW_JOINT    = 7,
//   ARM_RIGHT_ELBOW_PITCH_JOINT     = 8,
//   ARM_RIGHT_ELBOW_ROLL_JOINT      = 9,
//   LEG_LEFT_HIP_ROLL_JOINT         = 10,
//   LEG_LEFT_HIP_YAW_JOINT          = 11,
//   LEG_LEFT_HIP_PITCH_JOINT        = 12,
//   LEG_LEFT_KNEE_PITCH_JOINT       = 13,
//   LEG_LEFT_ANKLE_PITCH_JOINT      = 14,
//   LEG_LEFT_ANKLE_ROLL_JOINT       = 15,
//   LEG_RIGHT_HIP_ROLL_JOINT        = 16,
//   LEG_RIGHT_HIP_YAW_JOINT         = 17,
//   LEG_RIGHT_HIP_PITCH_JOINT       = 18,
//   LEG_RIGHT_KNEE_PITCH_JOINT      = 19,
//   LEG_RIGHT_ANKLE_PITCH_JOINT     = 20,
//   LEG_RIGHT_ANKLE_ROLL_JOINT      = 21,
// };


class RealHumanoid {
  public:
    float position_target[N_JOINTS];
    float position_measured[N_JOINTS];
    float velocity_measured[N_JOINTS];

    float starting_positions[N_JOINTS];

    float position_offsets[N_JOINTS] = {0};

    float joint_kp[N_JOINTS] = {0};
    float joint_kd[N_JOINTS] = {0};
    float torque_limit[N_JOINTS] = {0};

    float rl_init_positions[N_JOINTS] = {
       0.0, 0.0, -0.2,
       0.4,
      -0.3, 0.0,
       0.0, 0.0, -0.2,
       0.4,
      -0.3, 0.0
    };


    float joint_axis_directions[N_JOINTS] = {
      -1, 1, -1,
      -1,
      -1, 1,
      -1, 1, 1,
       1,
       1, 1
    };

    RealHumanoid();
    ~RealHumanoid();

    /**
     * Stop the humanoid low-level control.
     *
     * This method will join all the threads. On first execution, it will set motor to damping
     * mode; on second execution, it will set motor to idle mode and exits the main loop.
     */
    void stop();

    /**
     * Start the humanoid low-level control.
     *
     * This method will initialize the low-level components of the robot, create the udp
     * communication threads, and start the control loop.
     */
    void run();

    void run_calibration();

  private:
    /* Low-level components */
    uint8_t stopped = 0;
    ControllerState state = STATE_IDLE;
    ControllerState next_state = STATE_IDLE;

    float config_control_dt_ = 0.;
    float config_policy_dt_ = 0.;

    float stick_command_velocity_x_ = 0.0;
    float stick_command_velocity_y_ = 0.0;
    float stick_command_velocity_yaw_ = 0.0;

    std::string config_udp_host_addr_;

    /* devices */
    IMU *imu;
    SocketCan left_arm_bus;
    SocketCan right_arm_bus;
    SocketCan left_leg_bus;
    SocketCan right_leg_bus;

    std::array<std::shared_ptr<MotorController>, N_JOINTS> joint_ptrs;


    uint8_t control_loop_count = 0;

    float init_percentage = 0.0;

    /* Threading */
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_udp_recv;
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_joystick;
    std::shared_ptr<LoopFunc> loop_imu;

    /* UDP stuff */
    UDP udp;
    float lowlevel_commands[N_LOWLEVEL_COMMANDS];
    float lowlevel_states[N_LOWLEVEL_STATES];

    UDP udp_joystick;

    /* Policy stuff */
    // torch::Tensor policy_observations;
    // torch::Tensor policy_actions;


    void initialize();

    /**
     * The control loop that communicates with the hardware.
     *
     * This loop will run at 100 Hz.
     *
     * On each loop iteration, it will perform the following:
     *  1. collects the action terms from action array set by either UDP communication or the policy.
     *  2. triggers an IMU update.
     *  3. receives the data from IMU.
     *  4. sends the target positions to joints and reads the meaured positions, with some delay between each joint.
     *  5. populates the observation array.
     */
    void control_loop();

    void keyboard_loop();

    void joystick_loop();
    
    void imu_loop();

    void process_actions();

    void process_observations();

    void policy_forward();

    void udp_recv();

    void update_imu();

    void update_joints();
};
