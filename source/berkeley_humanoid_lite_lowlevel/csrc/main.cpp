// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include <signal.h>
#include "real_humanoid.h"


RealHumanoid humanoid;


void handle_keyboard_interrupt(int sig) {
  printf("\n<Main> Caught signal %d\n", sig);
  humanoid.stop();
}

int main() {
  printf("<Humanoid> Starting...\n");
  signal(SIGINT, handle_keyboard_interrupt);

  humanoid.run();

  return 0;
}
