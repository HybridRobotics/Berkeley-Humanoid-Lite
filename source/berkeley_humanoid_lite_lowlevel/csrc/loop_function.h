// Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstring>
#include <sstream>
#include <iomanip>

class LoopFunc {
public:
  LoopFunc(const std::string &name, float period, std::function<void()> func, int bind_cpu = -1, bool detach = false, int priority = -1, bool log = false)
      : _name(name), _period(period), _func(func), _bind_cpu(bind_cpu), _running(false), _detach(detach), _priority(priority), _log(log) {}

  void start() {
    _running = true;
    if (_log) {
      log("[Loop Start] name: " + _name + ", period: " + format_period_ms() + " ms, freq: " + format_freq_hz() + " Hz" + (_bind_cpu != -1 ? ", run at cpu: " + std::to_string(_bind_cpu) : ", cpu unspecified") + (_priority != -1 ? ", priority: " + std::to_string(_priority) : ""));
    }
    _thread = std::thread(&LoopFunc::loop, this);
    if (_bind_cpu != -1) {
      set_thread_affinity(_thread.native_handle(), _bind_cpu);
    }
    if (_priority != -1) {
      set_thread_priority(_thread.native_handle(), _priority);
    }
    if (_detach) {
      _thread.detach();
    }
  }

  void shutdown() {
    _running = false;
    if (_thread.joinable()) {
      printf("Joining thread %s\n", _name.c_str());
      _thread.join();
    }
    log("[Loop End] name: " + _name);
  }

private:
  std::string _name;
  float _period;
  std::function<void()> _func;
  int _bind_cpu;
  std::atomic<bool> _running;
  std::thread _thread;
  bool _detach;
  int _priority;
  bool _log;
  void loop() {
    while (_running) {
      const auto target_time = std::chrono::high_resolution_clock::now() + std::chrono::duration<double>(_period);
      _func();
      const auto finished_time = std::chrono::high_resolution_clock::now();
      if (finished_time < target_time) {
        std::this_thread::sleep_for(target_time - finished_time);
      } else {
        if (_log) {
          auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finished_time - target_time).count();
          printf("[ERROR] <LoopFunc>: Task %s use %d us more to complete\n", _name.c_str(), static_cast<int>(elapsed));
        }
      }
    }
  }

  std::string format_period_ms() const {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(0) << _period * 1000;
    return stream.str();
  }

  std::string format_freq_hz() const {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(0) << 1 / _period;
    return stream.str();
  }

  void log(const std::string &message) {
    static std::mutex logMutex;
    std::lock_guard<std::mutex> lock(logMutex);
    std::cout << message << std::endl;
  }

  void set_thread_affinity(std::thread::native_handle_type thread_handle, int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (pthread_setaffinity_np(thread_handle, sizeof(cpu_set_t), &cpuset) != 0) {
      std::ostringstream oss;
      oss << "Error setting thread affinity: CPU " << cpu_id << " may not be valid or accessible.";
      throw std::runtime_error(oss.str());
    }
  }

  void set_thread_priority(std::thread::native_handle_type thread_handle, int priority) {
    sched_param sched{};
    sched.sched_priority = priority;

    if (pthread_setschedparam(thread_handle, SCHED_FIFO, &sched) != 0) {
      std::cerr << "Failed to set thread priority for worker [" << _name << "]: " << strerror(errno) << std::endl;
    }
  
  }
};

