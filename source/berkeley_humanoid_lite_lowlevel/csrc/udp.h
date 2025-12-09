// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#ifndef UDP_H
#define UDP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>


#define NANOSECOND_PER_SECOND       1000000000


typedef struct {
  int sockfd;
  struct sockaddr_in recv_addr;
  struct sockaddr_in send_addr;
} UDP;


typedef struct {
  UDP udp;
  size_t n_obs;
  size_t n_acs;
} PolicyComm;

typedef struct {
  UDP udp;
  size_t n_obs;
  size_t n_acs;

  pthread_t thread_obs;
  pthread_t thread_acs;

  size_t frequency;

  float *obs;
  float *acs;
} EnvironmentComm;


ssize_t initialize_udp(
    UDP *udp,
    const char *recv_ip, const uint16_t recv_port,
    const char *send_ip, const uint16_t send_port
);

ssize_t initialize_policy(
    PolicyComm *comm,
    const char *compute_ip, int compute_port,
    const char *env_ip, int env_port,
    size_t n_obs, size_t n_acs
);

#ifdef __cplusplus
}
#endif

#endif