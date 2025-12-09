# ================================================================
# cmake -S ./ -B ./build/
# cmake --build ./build/ --target all
# ================================================================

cmake_minimum_required(VERSION 3.10)
project(humanoid)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

#set(TORCH_PATH "/home/tk/Documents/libtorch/")

#find_package(Torch REQUIRED PATHS ${TORCH_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(yaml-cpp)


# ================================ main application ================================

add_executable(main
    csrc/main.cpp
    csrc/udp.c
    csrc/serializer.c
    csrc/imu.cpp
    csrc/socketcan.cpp
    csrc/motor_controller.cpp
    csrc/real_humanoid.cpp
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)

target_include_directories(main PRIVATE csrc)

target_link_libraries(main yaml-cpp::yaml-cpp)


# ================================ test applications ================================

# Torch
add_executable(test-torch ctests/test-torch.cpp)
target_link_libraries(test-torch "${TORCH_LIBRARIES}")
set_property(TARGET test-torch PROPERTY CXX_STANDARD 17)


# IMU
add_executable(test-imu ctests/test-imu.cpp csrc/imu.cpp)
target_include_directories(test-imu PRIVATE csrc)
set_property(TARGET test-imu PROPERTY CXX_STANDARD 17)


# UDP
add_executable(test-udp ctests/test-udp.cpp csrc/udp.c)
target_include_directories(test-udp PRIVATE csrc)
set_property(TARGET test-udp PROPERTY CXX_STANDARD 17)


# YAML
add_executable(test-yaml ctests/test-yaml.cpp)
target_include_directories(test-yaml PRIVATE csrc)
target_link_libraries(test-yaml yaml-cpp::yaml-cpp)
set_property(TARGET test-yaml PROPERTY CXX_STANDARD 17)
