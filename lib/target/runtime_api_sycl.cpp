#include <cstdio>
#include <sycl/sycl.hpp>
#include <runtime_api.h>

sycl::queue sycl_queue;

#define sycl_error_check(_call, _file, _line)                                                                          \
  do {                                                                                                                 \
    try {                                                                                                              \
      _call;                                                                                                           \
    } catch (sycl::exception const &e) {                                                                               \
      fprintf(stderr, "SYCL error in %s at line %d: %s\n", _file, _line, e.what());                                    \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  } while (0)

namespace target
{
  char *buffer = nullptr;

  void set_device(int device, const char *file, int line)
  {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
      fprintf(stderr, "SYCL error in %s at line %d: No GPU devices found\n", file, line);
      exit(-1);
    }
    if (device < 0 || device >= static_cast<int>(devices.size())) {
      fprintf(stderr, "SYCL error in %s at line %d: Device index %d out of range (0-%zu)\n", file, line, device,
              devices.size() - 1);
      exit(-1);
    }
    sycl::property_list properties {sycl::property::queue::in_order {}, sycl::property::queue::enable_profiling {}};
    sycl_error_check(sycl_queue = sycl::queue(devices[device], properties), file, line);
  }

  void *malloc(size_t size, const char *file, int line)
  {
    void *dev_ptr = nullptr;
    sycl_error_check(dev_ptr = sycl::malloc_device(size, sycl_queue), file, line);
    return dev_ptr;
  }

  void free(void *dev_ptr, const char *file, int line)
  { sycl_error_check(sycl::free(dev_ptr, sycl_queue), file, line); }

  void memcpy_to_buffer(const void *src, size_t count, const char *file, int line)
  {
    if (buffer == nullptr) { buffer = static_cast<char *>(malloc(constant_buffer_size(), file, line)); }
    sycl_error_check(sycl_queue.memcpy(reinterpret_cast<void *>(buffer), src, count).wait(), file, line);
  }

  void launch_kernel(const void *func, unsigned int grid_dim, unsigned int block_dim, const char *file, int line)
  {
    fprintf(stderr, "SYCL error in %s at line %d: launch_kernel not implemented\n", file, line);
    exit(-1);
  }

  event_t event_create(const char *file, int line)
  {
    event_t event;
    event.event = new sycl::event();
    return event;
  }

  void event_destory(event_t event, const char *file, int line) { delete static_cast<sycl::event *>(event.event); }

  void event_record(event_t event, const char *file, int line)
  { sycl_error_check(*static_cast<sycl::event *>(event.event) = sycl_queue.ext_oneapi_submit_barrier(), file, line); }

  void event_synchronize(event_t event, const char *file, int line)
  { sycl_error_check(static_cast<sycl::event *>(event.event)->wait(), file, line); }

  float event_elapsed_time(event_t start, event_t end, const char *file, int line)
  {
    float ms = 0.0f;
    sycl_error_check(
      ms = static_cast<float>(
             static_cast<sycl::event *>(end.event)->get_profiling_info<sycl::info::event_profiling::command_end>()
             - static_cast<sycl::event *>(start.event)->get_profiling_info<sycl::info::event_profiling::command_end>())
        / 1e6f,
      file, line);
    return ms;
  }
} // namespace target
