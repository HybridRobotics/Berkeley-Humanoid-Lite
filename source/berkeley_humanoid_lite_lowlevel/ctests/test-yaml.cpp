// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#include <stdio.h>
#include <string.h>
#include <yaml-cpp/yaml.h>


int main() {
  const std::string config_path = "config.yaml";
  
  YAML::Node config = YAML::LoadFile(config_path);

  printf("Joint names: %s\n", config["joint_names"][0].as<std::string>().c_str());
  return 0;
}
