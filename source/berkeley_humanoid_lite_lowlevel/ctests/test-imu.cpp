// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>

#include "imu.h"


#define IMU_PATH      "/dev/serial/by-path/pci-0000:00:14.0-usb-0:4:1.0"
#define IMU_BAUDRATE  B1000000


int main() {
  IMU imu = IMU(IMU_PATH, IMU_BAUDRATE);
  ssize_t ret = imu.init();
  if (ret < 0) {
    printf("Error initializing IMU: %s\n", strerror(errno));
    return -1;
  }

  const sched_param sched{.sched_priority = 49};
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &sched);
  

  
  while (1) {
    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // imu.trigger_update();
    imu.update_reading();

    clock_gettime(CLOCK_MONOTONIC, &current_time);
    double elapsed_time = (current_time.tv_sec - start_time.tv_sec) + (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
    // printf("Elapsed time: %.2f ms, Hz: %.2f\n", elapsed_time * 1e3, 1.0 / elapsed_time);

    printf("qpos: %f %f %f %f\n", imu.quaternion[0], imu.quaternion[1], imu.quaternion[2], imu.quaternion[3]);
  }
  
  

  printf("Hello, World!\n");
}

