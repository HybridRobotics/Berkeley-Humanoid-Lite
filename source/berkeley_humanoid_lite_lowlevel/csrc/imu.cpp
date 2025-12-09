// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include "imu.h"

IMU::IMU(const char *path, speed_t baudrate) {
  path_ = path;
  baudrate_ = baudrate;
}

IMU::~IMU() {
  close(fd);
}

ssize_t IMU::init() {
  // imu->fd = Serial_init(path, baudrate);
  fd = open(path_, O_RDWR);

  if (fd < 0) {
    printf("[ERROR] <Serial>: Error %i from open: %s\n", errno, strerror(errno));
    return -1;
  }

  struct termios tty;
  memset(&tty, 0, sizeof tty);

  if (tcgetattr(fd, &tty) != 0) {
    printf("[ERROR] <Serial>: Error %i from tcgetattr: %s\n", errno, strerror(errno));
    return -1;
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

  tty.c_cc[VTIME] = 1; // Wait for up to 0.1s (1 deciseconds), returning as soon as any data is received.
  tty.c_cc[VMIN] = 0;

  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    printf("[ERROR] <Serial>: Error %i from cfsetattr: %s\n", errno, strerror(errno));
    return -1;
  }

  if (cfsetospeed(&tty, baudrate_) != 0) {
    printf("[ERROR] <Serial>: Error %i from cfsetospeed: %s\n", errno, strerror(errno));
    return -1;
  }

  printf("[INFO] <IMU>: Serial port %s initialized\n", path_);

  if (fd < 0) {
    printf("[ERROR] <IMU>: Error %i from open: %s\n", errno, strerror(errno));
    return -1;
  }

  quaternion[0] = 1.0;
  quaternion[1] = 0.0;
  quaternion[2] = 0.0;
  quaternion[3] = 0.0;

  angular_velocity[0] = 0.0;
  angular_velocity[1] = 0.0;
  angular_velocity[2] = 0.0;

  gravity_vector[0] = 0.0;
  gravity_vector[1] = 0.0;
  gravity_vector[2] = 0.0;

  // wait for the IMU to start for 0.1 seconds
  usleep(100000);

  return 0;
}

void IMU::update_reading() {
  uint8_t sync_byte;
  size_t bytes_read;
  
  bytes_read = read(fd, &sync_byte, 1);
  if (bytes_read < 1 || sync_byte != SYNC_1) {
    printf("Sync byte 1 error: %s\n", strerror(errno));
    return;
  }

  bytes_read = read(fd, &sync_byte, 1);
  if (bytes_read < 1 || sync_byte != SYNC_2) {
    printf("Sync byte 2 error: %s\n", strerror(errno));
    return;
  }

  uint16_t size;
  bytes_read = read(fd, &size, sizeof(size));
  if (bytes_read < 2) {
    printf("Size error: %s\n", strerror(errno));
    return;
  }

  float uart_buffer[7];
  bytes_read = read(fd, uart_buffer, size);
  if (bytes_read < size) {
    printf("Error reading: %s, expected %d bytes, got %ld bytes\n", strerror(errno), size, bytes_read);
    return;
  }

  quaternion[0] = uart_buffer[0];
  quaternion[1] = uart_buffer[1];
  quaternion[2] = uart_buffer[2];
  quaternion[3] = uart_buffer[3];

  angular_velocity[0] = uart_buffer[4];
  angular_velocity[1] = uart_buffer[5];
  angular_velocity[2] = uart_buffer[6];
}
