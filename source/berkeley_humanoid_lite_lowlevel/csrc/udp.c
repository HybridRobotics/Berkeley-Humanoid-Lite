// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include "udp.h"


ssize_t initialize_udp(
    UDP *udp,
    const char *recv_ip, const uint16_t recv_port,
    const char *send_ip, const uint16_t send_port
) {
  memset(udp, 0, sizeof(UDP));
  memset(&udp->recv_addr, 0, sizeof(udp->recv_addr));
  memset(&udp->send_addr, 0, sizeof(udp->send_addr));

  udp->recv_addr.sin_family = AF_INET;
  udp->recv_addr.sin_addr.s_addr = inet_addr(recv_ip);
  udp->recv_addr.sin_port = htons(recv_port);
  
  udp->send_addr.sin_family = AF_INET;
  udp->send_addr.sin_addr.s_addr = inet_addr(send_ip);
  udp->send_addr.sin_port = htons(send_port);

  if ((udp->sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    printf("[Error] <UDP> Error creating socket: %s\n", strerror(errno));
    return -1;
  }

  if (bind(udp->sockfd, (struct sockaddr *)&udp->recv_addr, sizeof(udp->recv_addr)) < 0) {
    printf("[Error] <UDP> Error binding socket: %s\n", strerror(errno));
    return -1;
  }

  printf("[INFO] <UDP> Server listening on %s:%d\n", recv_ip, recv_port);
  return 0;
}
