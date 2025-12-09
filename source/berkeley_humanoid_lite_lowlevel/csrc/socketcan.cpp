/*******************************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2021, Qiayuan Liao
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

//
// Created by qiayuan on 3/3/21.
//
#include "socketcan.h"

#include <iostream>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <utility>


  /* ref:
   * https://github.com/JCube001/socketcan-demo
   * http://blog.mbedded.ninja/programming/operating-systems/linux/how-to-use-socketcan-with-c-in-linux
   * https://github.com/linux-can/can-utils/blob/master/candump.c
   */

SocketCan::~SocketCan()
{
  if (isOpen())
    close();
}

bool SocketCan::open(const std::string &interface)
{
  // Request a socket
  sock_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (sock_fd_ == -1)
  {
    std::cerr << "Error: Unable to create a CAN socket" << std::endl;
    return false;
  }
  char name[16] = {}; // avoid stringop-truncation
  strncpy(name, interface.c_str(), interface.size());
  strncpy(interface_request_.ifr_name, name, IFNAMSIZ);
  // Get the index of the network interface
  if (ioctl(sock_fd_, SIOCGIFINDEX, &interface_request_) == -1)
  {
    std::cerr << "Unable to select CAN interface " << name << ": I/O control error" << std::endl;
    // Invalidate unusable socket
    close();
    return false;
  }
  // Bind the socket to the network interface
  address_.can_family = AF_CAN;
  address_.can_ifindex = interface_request_.ifr_ifindex;
  const int rc = bind(sock_fd_, reinterpret_cast<struct sockaddr *>(&address_),
                      sizeof(address_));
  if (rc == -1)
  {
    std::cerr << "Failed to bind socket to " << name << " network interface" << std::endl;
    close();
    return false;
  }
  return true;
}

void SocketCan::close()
{
  if (!isOpen())
    return;
  // Close the file descriptor for our socket
  ::close(sock_fd_);
  sock_fd_ = -1;
}

bool SocketCan::isOpen() const { return (sock_fd_ != -1); }

void SocketCan::write(can_frame *frame) const
{
  if (!isOpen())
  {
    std::cerr << "Unable to write: Socket " << interface_request_.ifr_name << " not open" << std::endl;
    return;
  }
  if (::write(sock_fd_, frame, sizeof(can_frame)) == -1)
    std::cerr << "Unable to write: The " << interface_request_.ifr_name << " tx buffer may be full" << std::endl;
}

can_frame SocketCan::read()
{

  // Holds the set of descriptors, that 'select' shall monitor
  fd_set descriptors;
  // Highest file descriptor in set
  int maxfd = sock_fd_;
  // How long 'select' shall wait before returning with timeout
  timeval timeout{};
  // Buffer to store incoming frame
  can_frame rx_frame{};
  // Run until termination signal received
  timeout.tv_sec = 1; // Should be set each loop
  // Clear descriptor set
  FD_ZERO(&descriptors);
  // Add socket descriptor
  FD_SET(sock_fd_, &descriptors);
  // Wait until timeout or activity on any descriptor
  if (select(maxfd + 1, &descriptors, nullptr, nullptr, &timeout))
  {
    ::read(sock_fd_, &rx_frame, CAN_MTU);
    return rx_frame;
  } else {  
    printf("[ERROR] <SocketCan>: Timeout reading from socket\n");
    return {};
  }
}
