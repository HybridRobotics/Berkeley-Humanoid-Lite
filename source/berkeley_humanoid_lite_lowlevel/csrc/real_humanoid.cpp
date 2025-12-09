// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include <sys/ioctl.h>

#include "real_humanoid.h"
#include "motor_controller_conf.h"


static float linear_interpolate(float start, float end, float percentage) {
  float target;
  percentage = std::fmin(std::fmax(percentage, 0.0f), 1.0f);
  target = start * (1.f - percentage) + end * percentage;
  return target;
}


RealHumanoid::RealHumanoid() {
  imu = nullptr;
  state = STATE_IDLE;
  next_state = STATE_IDLE;

  for (size_t i=0; i<N_JOINTS; i+=1) {
    position_target[i] = 0;
    position_measured[i] = 0;
    velocity_measured[i] = 0;
  }

  for (size_t i=0; i<N_LOWLEVEL_COMMANDS; i+=1) {
    lowlevel_commands[i] = 0;
  }
  for (size_t i=0; i<N_LOWLEVEL_STATES; i+=1) {
    lowlevel_states[i] = 0;
  }


  const std::string config_path = "calibration.yaml";
  YAML::Node config = YAML::LoadFile(config_path);
  for (size_t i=0; i<N_JOINTS; i+=1) {
    position_offsets[i] = config["position_offsets"][i].as<float>();
  }

  printf("loaded joint offsets: ");
  for (size_t i=0; i<N_JOINTS; i+=1) {
    printf("%.3f ", position_offsets[i]);
  }
  printf("\n");

  const std::string policy_config_path = POLICY_CONFIG_PATH;
  YAML::Node policy_config = YAML::LoadFile(policy_config_path);
  for (size_t i=0; i<N_JOINTS; i+=1) {
    joint_kp[i] = policy_config["joint_kp"][i].as<float>();
    joint_kd[i] = policy_config["joint_kd"][i].as<float>();
    torque_limit[i] = policy_config["effort_limits"][i].as<float>();
  }
  config_control_dt_ = policy_config["control_dt"].as<float>();
  config_policy_dt_ = policy_config["policy_dt"].as<float>();
}

RealHumanoid::~RealHumanoid() {
  delete imu;
}


void RealHumanoid::control_loop() {
  switch (state) {
    case STATE_IDLE:
      /* In this state, the motor positions are held at the current measured positions */
      /* When Y is pressed, the robot will switch to RL initialization mode */

      for (int i = 0; i < N_JOINTS; i += 1) {
        position_target[i] = position_measured[i];
      }
      if (next_state == STATE_RL_INIT) {
        printf("Switching to RL initialization mode\n");
        state = next_state;

        for (int i = 0; i < N_JOINTS; i += 1) {
          usleep(5);
          joint_ptrs[i]->feed();
          joint_ptrs[i]->set_mode(MODE_POSITION);
          starting_positions[i] = position_target[i];
        }
        init_percentage = 0.0;
      }
      break;

    case STATE_RL_INIT:
      /* In this state, the robot will hold the getup position */
      // printf("init: %.3f\n", init_percentage);

      if (init_percentage < 1.0) {
        init_percentage += 1 / 200.0;
        init_percentage = init_percentage < 1.0 ? init_percentage : 1.0;

        for (size_t i=0; i<12; i+=1) {
          position_target[i] = linear_interpolate(starting_positions[i], rl_init_positions[i], init_percentage);
        }
      }
      else {
        if (next_state == STATE_RL_RUNNING) {
          printf("Switching to RL running mode\n");
          state = next_state;
        }
        if (next_state == STATE_IDLE) {
          printf("Switching to idle mode\n");
          state = next_state;

          for (int i = 0; i < N_JOINTS; i += 1) {
            usleep(5);
            joint_ptrs[i]->set_mode(MODE_DAMPING);
          }
        }
      }
      break;

    case STATE_RL_RUNNING:
      /* In this state, the robot will follow the policy */
      for (int i = 0; i < N_LOWLEVEL_COMMANDS; i += 1) {
        position_target[i] = lowlevel_commands[i];
      }

      if (next_state == STATE_IDLE) {
        printf("Switching to idle mode\n");
        state = next_state;

        for (int i = 0; i < N_JOINTS; i += 1) {
          usleep(5);
          joint_ptrs[i]->set_mode(MODE_DAMPING);
        }
      }

      break;
  }

  #if DEBUG_DISABLE_TRANSPORTS == 0
    update_joints();
  #endif

  #if DEBUG_JOINT_DATA_LOGGING == 1
    printf("%d %.2f \t%.2f \t%.2f \t- %.2f \t- %.2f \t%.2f \t",
      control_loop_count,
      position_measured[0], position_measured[1], position_measured[2],
      position_measured[3],
      position_measured[4], position_measured[5]);
    // printf(" %.2f \t%.2f \t%.2f \t- %.2f \t- %.2f \t%.2f",
    //   position_target[0], position_target[1], position_target[2],
    //   position_target[3],
    //   position_target[4], position_target[5]);
    printf("\n");
  #endif

  /* base_quat */
  lowlevel_states[0] = imu->quaternion[0];
  lowlevel_states[1] = imu->quaternion[1];
  lowlevel_states[2] = imu->quaternion[2];
  lowlevel_states[3] = imu->quaternion[3];

  /* base_ang_vel */
  lowlevel_states[4] = imu->angular_velocity[0];
  lowlevel_states[5] = imu->angular_velocity[1];
  lowlevel_states[6] = imu->angular_velocity[2];

  /* joint_positions */
  for (int i = 0; i < N_JOINTS; i += 1) {
    lowlevel_states[7 + i] = position_measured[i];
  }

  /* joint_velocities */
  for (int i = 0; i < N_JOINTS; i += 1) {
    lowlevel_states[19 + i] = velocity_measured[i];
  }

  /* mode */
  lowlevel_states[31] = state;

  /* command velocity */
  lowlevel_states[32] = stick_command_velocity_x_;
  lowlevel_states[33] = stick_command_velocity_y_;
  lowlevel_states[34] = stick_command_velocity_yaw_;

  // execute every 4 control loops
  if (control_loop_count >= (int)std::round(config_policy_dt_/ config_control_dt_)) {
    if (state == STATE_IDLE || state == STATE_RL_RUNNING) {
      size_t expected_bytes = sizeof(float) * N_LOWLEVEL_STATES;
      ssize_t actual_bytes = sendto(udp.sockfd, lowlevel_states, expected_bytes, 0, (const struct sockaddr *)&udp.send_addr, sizeof(udp.send_addr));

      if (actual_bytes < 0 || actual_bytes != expected_bytes) {
        printf("[Error] <UDP> Error sending: %s\n", strerror(errno));
      }
    }
    size_t expected_bytes = sizeof(float) * N_LOWLEVEL_STATES;
    struct sockaddr_in visualize_addr = {0};
    visualize_addr.sin_family = AF_INET;
    visualize_addr.sin_port = htons(VISUALIZE_PORT);
    visualize_addr.sin_addr.s_addr = inet_addr(HOST_IP_ADDR);
    ssize_t actual_bytes = sendto(udp.sockfd, lowlevel_states, expected_bytes, 0, (const struct sockaddr *)&visualize_addr, sizeof(visualize_addr));
    control_loop_count = 0;
  }

  control_loop_count += 1;
}

void RealHumanoid::imu_loop() {
  imu->update_reading();
}

void RealHumanoid::keyboard_loop() {
  termios term;
  tcgetattr(0, &term);

  termios term2 = term;

  term2.c_lflag &= ~ICANON;
  tcsetattr(0, TCSANOW, &term2);

  int byteswaiting;
  ioctl(0, FIONREAD, &byteswaiting);

  tcsetattr(0, TCSANOW, &term);

  if (byteswaiting > 0) {
    char c = fgetc(stdin);
    printf("key pressed: %c\n", c);

    switch (c) {
      case 'r':
        next_state = STATE_RL_INIT;
        break;
      case 't':
        next_state = STATE_RL_RUNNING;
        break;
      case 'q':
        next_state = STATE_IDLE;
        break;
    }


  }

  tcsetattr(0, TCSANOW, &term);
}

void RealHumanoid::joystick_loop() {
  size_t expected_bytes = 13;
  uint8_t udp_buffer[13];
  ssize_t actual_bytes = recvfrom(udp_joystick.sockfd, udp_buffer, expected_bytes, MSG_WAITALL, NULL, NULL);

  if (actual_bytes < 0 || actual_bytes != expected_bytes) {
    printf("[Error] <UDPStick> Error receiving: %s\n", strerror(errno));

    return;
  }

  uint8_t command_mode = udp_buffer[0];
  stick_command_velocity_x_ = *(float *)(udp_buffer + 1);
  stick_command_velocity_y_ = *(float *)(udp_buffer + 5);
  stick_command_velocity_yaw_ = *(float *)(udp_buffer + 9);

  if (command_mode != 0) {
    switch (command_mode) {
      case 1:
        next_state = STATE_IDLE;
        break;
      case 2:
        next_state = STATE_RL_INIT;
        break;
      case 3:
        next_state = STATE_RL_RUNNING;
        break;
      default:
        next_state = STATE_IDLE;
    }
  }
}

void RealHumanoid::process_actions() {

}

void RealHumanoid::process_observations() {

}

void RealHumanoid::policy_forward() {

}

void RealHumanoid::udp_recv() {
  size_t expected_bytes = sizeof(float) * N_LOWLEVEL_COMMANDS;
  ssize_t actual_bytes = recvfrom(udp.sockfd, lowlevel_commands, expected_bytes, MSG_WAITALL, NULL, NULL);

  if (actual_bytes < 0 || actual_bytes != expected_bytes) {
    printf("[Error] <UDP> Error receiving: %s\n", strerror(errno));
  }
  //TODO: detect if the action is delayed

}

void RealHumanoid::update_joints() {
  /* set target positions to joint controller */
  for (size_t i = 0; i < N_JOINTS; i += 1) {
    joint_ptrs[i]->set_target_position((position_target[i] + position_offsets[i]) * joint_axis_directions[i]);
  }

  joint_ptrs[LEG_LEFT_HIP_ROLL_JOINT]->write_pdo_2();
  joint_ptrs[LEG_RIGHT_HIP_ROLL_JOINT]->write_pdo_2();
  joint_ptrs[LEG_LEFT_HIP_ROLL_JOINT]->read_pdo_2();
  joint_ptrs[LEG_RIGHT_HIP_ROLL_JOINT]->read_pdo_2();

  joint_ptrs[LEG_LEFT_HIP_YAW_JOINT]->write_pdo_2();
  joint_ptrs[LEG_RIGHT_HIP_YAW_JOINT]->write_pdo_2();
  joint_ptrs[LEG_LEFT_HIP_YAW_JOINT]->read_pdo_2();
  joint_ptrs[LEG_RIGHT_HIP_YAW_JOINT]->read_pdo_2();

  joint_ptrs[LEG_LEFT_HIP_PITCH_JOINT]->write_pdo_2();
  joint_ptrs[LEG_RIGHT_HIP_PITCH_JOINT]->write_pdo_2();
  joint_ptrs[LEG_LEFT_HIP_PITCH_JOINT]->read_pdo_2();
  joint_ptrs[LEG_RIGHT_HIP_PITCH_JOINT]->read_pdo_2();

  joint_ptrs[LEG_LEFT_KNEE_PITCH_JOINT]->write_pdo_2();
  joint_ptrs[LEG_RIGHT_KNEE_PITCH_JOINT]->write_pdo_2();
  joint_ptrs[LEG_LEFT_KNEE_PITCH_JOINT]->read_pdo_2();
  joint_ptrs[LEG_RIGHT_KNEE_PITCH_JOINT]->read_pdo_2();

  joint_ptrs[LEG_LEFT_ANKLE_PITCH_JOINT]->write_pdo_2();
  joint_ptrs[LEG_RIGHT_ANKLE_PITCH_JOINT]->write_pdo_2();
  joint_ptrs[LEG_LEFT_ANKLE_PITCH_JOINT]->read_pdo_2();
  joint_ptrs[LEG_RIGHT_ANKLE_PITCH_JOINT]->read_pdo_2();

  joint_ptrs[LEG_LEFT_ANKLE_ROLL_JOINT]->write_pdo_2();
  joint_ptrs[LEG_RIGHT_ANKLE_ROLL_JOINT]->write_pdo_2();
  joint_ptrs[LEG_LEFT_ANKLE_ROLL_JOINT]->read_pdo_2();
  joint_ptrs[LEG_RIGHT_ANKLE_ROLL_JOINT]->read_pdo_2();

  /* update measured positions from joint controller */
  for (size_t i = 0; i < N_JOINTS; i += 1) {
    position_measured[i] = joint_ptrs[i]->get_measured_position() * joint_axis_directions[i] - position_offsets[i];
    velocity_measured[i] = joint_ptrs[i]->get_measured_velocity() * joint_axis_directions[i];
  }
}


void RealHumanoid::stop() {
  if (stopped == 0) {
    stopped = 1;

    loop_udp_recv->shutdown();
    loop_control->shutdown();
    loop_keyboard->shutdown();
    loop_joystick->shutdown();

    #if DEBUG_DISABLE_TRANSPORTS == 0
      for (int i = 0; i < N_JOINTS; i += 1) {
        joint_ptrs[i]->set_mode(MODE_DAMPING);
      }
    #endif

    printf("Entered damping mode. Press Ctrl+C again to exit.\n");
  }
  else if (stopped == 1) {
    printf("exiting...\n");

    #if DEBUG_DISABLE_TRANSPORTS == 0
      for (int i = 0; i < N_JOINTS; i += 1) {
        joint_ptrs[i]->set_mode(MODE_IDLE);
      }
    #endif

    stopped = 2;

    sleep(1);
  }


  printf("RealHumanoid stopped\n");
}

void RealHumanoid::initialize() {
  ssize_t status;

  #if DEBUG_DISABLE_TRANSPORTS == 0
    // // left arm
    // left_arm_bus.open("can0");
    // if (!left_arm_bus.isOpen()) {
    //   printf("[ERROR] <Main>: Error initializing left arm transport\n");
    //   exit(1);
    // }

    // // right arm
    // right_arm_bus.open("can1");
    // if (!right_arm_bus.isOpen()) {
    //   printf("[ERROR] <Main>: Error initializing right arm transport\n");
    //   exit(1);
    // }

    // left leg
    left_leg_bus.open("can0");
    if (!left_leg_bus.isOpen()) {
      printf("[ERROR] <Main>: Error initializing left leg transport\n");
      exit(1);
    }

    // right leg
    right_leg_bus.open("can1");
    if (!right_leg_bus.isOpen()) {
      printf("[ERROR] <Main>: Error initializing right arm transport\n");
      exit(1);
    }
  #endif

  // // left arm
  // joint_ptrs[ARM_LEFT_SHOULDER_PITCH_JOINT] = std::make_shared<MotorController>(&left_arm_bus, 1);
  // joint_ptrs[ARM_LEFT_SHOULDER_ROLL_JOINT] = std::make_shared<MotorController>(&left_arm_bus, 3);
  // joint_ptrs[ARM_LEFT_SHOULDER_YAW_JOINT] = std::make_shared<MotorController>(&left_arm_bus, 5);
  // joint_ptrs[ARM_LEFT_ELBOW_PITCH_JOINT] = std::make_shared<MotorController>(&left_arm_bus, 7);
  // joint_ptrs[ARM_LEFT_ELBOW_ROLL_JOINT] = std::make_shared<MotorController>(&left_arm_bus, 9);

  // // right arm
  // joint_ptrs[ARM_RIGHT_SHOULDER_PITCH_JOINT] = std::make_shared<MotorController>(&right_arm_bus, 2);
  // joint_ptrs[ARM_RIGHT_SHOULDER_ROLL_JOINT] = std::make_shared<MotorController>(&right_arm_bus, 4);
  // joint_ptrs[ARM_RIGHT_SHOULDER_YAW_JOINT] = std::make_shared<MotorController>(&right_arm_bus, 6);
  // joint_ptrs[ARM_RIGHT_ELBOW_PITCH_JOINT] = std::make_shared<MotorController>(&right_arm_bus, 8);
  // joint_ptrs[ARM_RIGHT_ELBOW_ROLL_JOINT] = std::make_shared<MotorController>(&right_arm_bus, 10);

  // left leg
  joint_ptrs[LEG_LEFT_HIP_ROLL_JOINT] = std::make_shared<MotorController>(&left_leg_bus, 1);
  joint_ptrs[LEG_LEFT_HIP_YAW_JOINT] = std::make_shared<MotorController>(&left_leg_bus, 3);
  joint_ptrs[LEG_LEFT_HIP_PITCH_JOINT] = std::make_shared<MotorController>(&left_leg_bus, 5);
  joint_ptrs[LEG_LEFT_KNEE_PITCH_JOINT] = std::make_shared<MotorController>(&left_leg_bus, 7);
  joint_ptrs[LEG_LEFT_ANKLE_PITCH_JOINT] = std::make_shared<MotorController>(&left_leg_bus, 11);
  joint_ptrs[LEG_LEFT_ANKLE_ROLL_JOINT] = std::make_shared<MotorController>(&left_leg_bus, 13);

  // right leg
  joint_ptrs[LEG_RIGHT_HIP_ROLL_JOINT] = std::make_shared<MotorController>(&right_leg_bus, 2);
  joint_ptrs[LEG_RIGHT_HIP_YAW_JOINT] = std::make_shared<MotorController>(&right_leg_bus, 4);
  joint_ptrs[LEG_RIGHT_HIP_PITCH_JOINT] = std::make_shared<MotorController>(&right_leg_bus, 6);
  joint_ptrs[LEG_RIGHT_KNEE_PITCH_JOINT] = std::make_shared<MotorController>(&right_leg_bus, 8);
  joint_ptrs[LEG_RIGHT_ANKLE_PITCH_JOINT] = std::make_shared<MotorController>(&right_leg_bus, 12);
  joint_ptrs[LEG_RIGHT_ANKLE_ROLL_JOINT] = std::make_shared<MotorController>(&right_leg_bus, 14);


  #if DEBUG_DISABLE_TRANSPORTS == 0
    for (int i = 0; i < N_JOINTS; i += 1) {
      joint_ptrs[i]->set_mode(MODE_IDLE);
    }
  #endif

  imu = new IMU(IMU_PATH, IMU_BAUDRATE);

  status = imu->init();
  if (status < 0) {
    printf("[ERROR] <Main>: Error initializing IMU\n");
    exit(1);
  }

  status = initialize_udp(&udp, "0.0.0.0", POLICY_ACS_PORT, ROBOT_IP_ADDR, POLICY_OBS_PORT);
  if (status < 0) {
    printf("[ERROR] <Main>: Error initializing UDP\n");
    exit(1);
  }

  // Initialize joystick UDP socket to listen on ROBOT_IP:10011 and send to ROBOT_IP:10011
  status = initialize_udp(&udp_joystick, "0.0.0.0", JOYSTICK_PORT, "127.0.0.1", JOYSTICK_PORT);
  if (status < 0) {
    printf("[ERROR] <Main>: Error initializing UDP joystick\n");
    exit(1);
  }

  // 500 Hz UDP receive
  loop_udp_recv = std::make_shared<LoopFunc>("loop_udp_recv", 0.002, [this] { udp_recv(); }, 1, true, 49);

  // 500 Hz IMU input
  loop_imu = std::make_shared<LoopFunc>("loop_imu", 0.002, [this] { imu_loop(); }, 1, true, 49);

  // 20 Hz keyboard input
  loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, [this] { keyboard_loop(); }, 0, true, 19);

  // 20 Hz joystick input
  loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.05, [this] { joystick_loop(); }, 1, true, 19);

  // we want to ensure all packages are transmitted before killing this thread
  // 100 Hz control loop
  loop_control = std::make_shared<LoopFunc>("loop_control", 0.004, [this] { control_loop(); }, 0, false, 50, true);
}


void RealHumanoid::run() {
  initialize();

  printf("Enabling motors...\n");

  printf("read config joint_kp:\n [");
  for (int i = 0; i < N_JOINTS; i += 1) {
    printf("%.3f ", joint_kp[i]);
  }
  printf("]\n");
  printf("read config joint_kd:\n [");
  for (int i = 0; i < N_JOINTS; i += 1) {
    printf("%.3f ", joint_kd[i]);
  }
  printf("]\n");
  printf("read config torque_limit:\n [");
  for (int i = 0; i < N_JOINTS; i += 1) {
    printf("%.3f ", torque_limit[i]);
  }
  printf("]\n");

  #if DEBUG_DISABLE_TRANSPORTS == 0
    for (int i = 0; i < N_JOINTS; i += 1) {
      usleep(10);
      joint_ptrs[i]->write_position_kp(joint_kp[i]);
      usleep(10);
      joint_ptrs[i]->write_position_kd(joint_kd[i]);
      usleep(10);
      joint_ptrs[i]->write_torque_limit(torque_limit[i]);
      usleep(100);
      joint_ptrs[i]->feed();
      joint_ptrs[i]->set_mode(MODE_DAMPING);
    }
  #endif

  printf("Motors enabled\n");

  loop_udp_recv->start();
  loop_keyboard->start();
  loop_joystick->start();
  loop_imu->start();
  loop_control->start();

  while (stopped != 2) {
    sleep(1);
  }
}
