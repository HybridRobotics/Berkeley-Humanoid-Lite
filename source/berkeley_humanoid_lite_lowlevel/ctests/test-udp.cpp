// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <time.h>

#include "udp.h"
#include "consts.h"

#define N_OBSERVATIONS  50
#define N_ACTIONS       14

#define FREQUENCY 100


float obs[N_OBSERVATIONS];
float acs[N_ACTIONS];


int main() {
  UDP udp;
  initialize_udp(&udp, "0.0.0.0", JOYSTICK_PORT, "127.0.0.1", JOYSTICK_PORT);

  while (true) {
    uint8_t udp_buffer[13];
    ssize_t actual_bytes = recvfrom(udp.sockfd, udp_buffer, 13, MSG_WAITALL, NULL, NULL);
    printf("actual_bytes: %ld\n", actual_bytes);
    printf("udp_buffer: %d %.2f %.2f %.2f\n", udp_buffer[0], *(float *)(udp_buffer + 1), *(float *)(udp_buffer + 5), *(float *)(udp_buffer + 9));
  }

  return 0;
}