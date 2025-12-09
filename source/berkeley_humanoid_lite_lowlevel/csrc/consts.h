// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#pragma once


#define HOST_IP_ADDR   "127.0.0.1"
#define ROBOT_IP_ADDR  "127.0.0.1"

#define POLICY_OBS_PORT  10000
#define POLICY_ACS_PORT  10001
#define VISUALIZE_PORT  10002
#define JOYSTICK_PORT  10011

#define N_JOINTS            12


#define POLICY_CONFIG_PATH          "configs/policy_biped.yaml"


#define DEBUG_DISABLE_TRANSPORTS     0
#define DEBUG_FREQUENCY_LOGGING      0
#define DEBUG_JOINT_DATA_LOGGING     0


#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#define NANOSECOND_PER_SECOND       1000000000


#define IMU_PATH      "/dev/serial/by-path/pci-0000:00:14.0-usb-0:4:1.0"
#define IMU_BAUDRATE  B1000000

#define N_DOF     12   //24


