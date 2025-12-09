// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include "serializer.h"


int Serial_init(const char *path, speed_t baudrate) {
  int fd = open(path, O_RDWR);

  if (fd < 0) {
    printf("[ERROR] <Serial>: Error %i from open: %s\n", errno, strerror(errno));
  }

  struct termios tty;
  memset(&tty, 0, sizeof tty);

  if (tcgetattr(fd, &tty) != 0) {
    printf("[ERROR] <Serial>: Error %i from tcgetattr: %s\n", errno, strerror(errno));
  }

  tty.c_cflag &= ~PARENB;  // Disable parity
  tty.c_cflag &= ~CSTOPB;  // 1 stop bit
  tty.c_cflag &= ~CSIZE;   // Clear the mask
  tty.c_cflag |= CS8;      // 8 data bits
  tty.c_cflag &= ~CRTSCTS; // No hardware flow control
  tty.c_cflag |= CREAD | CLOCAL; // Turn on READ & ignore ctrl lines

  tty.c_lflag &= ~ICANON; // Disable canonical mode
  tty.c_lflag &= ~ECHO;   // Disable echo
  tty.c_lflag &= ~ECHOE;  // Disable erasure
  tty.c_lflag &= ~ECHONL; // Disable new-line echo
  tty.c_lflag &= ~ISIG;   // Disable interpretation of INTR, QUIT and SUSP
  tty.c_iflag &= ~(IXON | IXOFF | IXANY); // Turn off s/w flow ctrl
  tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL); // Disable any special handling of received bytes

  tty.c_oflag &= ~OPOST; // Prevent special interpretation of output bytes (e.g. newline chars)
  tty.c_oflag &= ~ONLCR; // Prevent conversion of newline to carriage return/line feed

  tty.c_cc[VTIME] = 10; // Wait for up to 0.2s (1 deciseconds), returning as soon as any data is received.
  tty.c_cc[VMIN] = 0;

  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    printf("[ERROR] <Serial>: Error %i from cfsetattr: %s\n", errno, strerror(errno));
  }

  if (cfsetospeed(&tty, baudrate) != 0) {
    printf("[ERROR] <Serial>: Error %i from cfsetospeed: %s\n", errno, strerror(errno));
  }

  printf("[INFO] <Serial>: Serial port %s initialized\n", path);
  return fd;
}

size_t Serial_receive(int fd, char *buffer) {
  size_t index = 0;
  char c = 0;
  ssize_t len;

  len = read(fd, &c, 1);

  while (len > 0 && c != ENCODING_END) {
    if (c == ENCODING_ESC) {
      len = read(fd, &c, 1);
      if (c == ENCODING_ESC_END) {
        buffer[index] = ENCODING_END;
      } else if (c == ENCODING_ESC_ESC) {
        buffer[index] = ENCODING_ESC;
      } else {
        buffer[index] = c;
      }
    } else {
      buffer[index] = c;
    }
    index += 1;
    len = read(fd, &c, 1);
  }

  tcflush(fd, TCIFLUSH);

  return index;
}

void Serial_transmit(int fd, char *buffer, size_t len) {
  size_t index = 0;
  ssize_t len_written;

  while (index < len) {
    char c = buffer[index];
    if (c == 0xC0) {
      char escape;
      escape = ENCODING_END;
      len_written = write(fd, &escape, 1);
      if (len_written < 0) {
        printf("[ERROR] <Serial>: Error %i from write: %s\n", errno, strerror(errno));
      }
      escape = ENCODING_ESC_END;
      len_written = write(fd, &escape, 1);
      if (len_written < 0) {
        printf("[ERROR] <Serial>: Error %i from write: %s\n", errno, strerror(errno));
      }
    } else if (c == 0xDB) {
      char escape;
      escape = ENCODING_ESC;
      len_written = write(fd, &escape, 1);
      if (len_written < 0) {
        printf("[ERROR] <Serial>: Error %i from write: %s\n", errno, strerror(errno));
      }
      escape = ENCODING_ESC_ESC;
      len_written = write(fd, &escape, 1);
      if (len_written < 0) {
        printf("[ERROR] <Serial>: Error %i from write: %s\n", errno, strerror(errno));
      }
    } else {
      len_written = write(fd, &c, 1);
      if (len_written < 0) {
        printf("[ERROR] <Serial>: Error %i from write: %s\n", errno, strerror(errno));
      }
    }
    index += 1;
  }
  char end = ENCODING_END;
  len_written = write(fd, &end, 1);
  if (len_written < 0) {
    printf("[ERROR] <Serial>: Error %i from write: %s\n", errno, strerror(errno));
  }
  
  tcflush(fd, TCOFLUSH);
}

void Serial_stop(int fd) {
  close(fd);
}
