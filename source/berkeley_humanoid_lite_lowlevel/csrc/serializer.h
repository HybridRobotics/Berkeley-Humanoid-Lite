// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#ifndef SERIALIZER_H
#define SERIALIZER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>


#define ENCODING_END                0x0A
#define ENCODING_ESC                0x0B
#define ENCODING_ESC_END            0x1A
#define ENCODING_ESC_ESC            0x1B

int Serial_init(const char *path, speed_t baudrate);

size_t Serial_receive(int fd, char *buffer);

void Serial_transmit(int fd, char *buffer, size_t len);

void Serial_stop(int fd);

#ifdef __cplusplus
}
#endif

#endif
