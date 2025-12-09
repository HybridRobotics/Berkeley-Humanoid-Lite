// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "serializer.h"


#define SYNC_1 0x75
#define SYNC_2 0x65


class IMU {
  public:
    float quaternion[4];        // quaternion, (w, x, y, z)
    float angular_velocity[3];  // angular velocity, (x, y, z), rad/s
    float gravity_vector[3];    // gravity vector, (x, y, z), m/s^2

    IMU(const char *path, speed_t baudrate);

    ~IMU();

    ssize_t init();

    void update_reading();
    
  private:
    int fd;
    const char *path_;
    speed_t baudrate_;
};


