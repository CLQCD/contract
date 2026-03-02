#include <cstdio>
#include <chrono>
#include <sycl/sycl.hpp>
#include <runtime_api.h>

static sycl::queue *global_queue = nullptr;

sycl::queue &get_sycl_queue()
{
  if (global_queue == nullptr) {
    fprintf(stderr, "SYCL error: queue not initialized. Call init(device) first.\n");
    exit(-1);
  }
  return *global_queue;
}

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

  void set_device(int device, const char *file, int line)
  {
    try {
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
      if (global_queue != nullptr) { delete global_queue; }
      global_queue = new sycl::queue(
        devices[device],
        sycl::property_list {sycl::property::queue::in_order {}, sycl::property::queue::enable_profiling {}});
      auto dev = global_queue->get_device();
      fprintf(stdout, "SYCL: Using device %d: %s\n", device, dev.get_info<sycl::info::device::name>().c_str());
    } catch (sycl::exception const &e) {
      fprintf(stderr, "SYCL error in %s at line %d: %s\n", file, line, e.what());
      exit(-1);
    }
  }

  void *malloc(size_t size, const char *file, int line)
  {
    void *dev_ptr = nullptr;
    sycl_error_check(dev_ptr = sycl::malloc_device(size, get_sycl_queue()), file, line);
    return dev_ptr;
  }

  void free(void *dev_ptr, const char *file, int line)
  {
    sycl_error_check(sycl::free(dev_ptr, get_sycl_queue()), file, line);
  }

  void memcpy_to_symbol(const void *symbol, const void *src, size_t count, const char *file, int line)
  {
    sycl_error_check(get_sycl_queue().memcpy(const_cast<void *>(symbol), src, count).wait(), file, line);
  }

  void launch_kernel(const void *func, unsigned int grid_dim, unsigned int block_dim, const char *file, int line)
  {
    // SYCL kernels are launched via contract::launch_kernel<> template, not through function pointers.
    // This function should not be called in the SYCL backend.
    fprintf(stderr,
            "SYCL error in %s at line %d: launch_kernel via function pointer is not supported in SYCL backend. "
            "Use contract::launch_kernel<> template instead.\n",
            file, line);
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
  {
    sycl_error_check(*static_cast<sycl::event *>(event.event) = get_sycl_queue().ext_oneapi_submit_barrier(), file, line);
  }

  void event_synchronize(event_t event, const char *file, int line)
  {
    sycl_error_check(static_cast<sycl::event *>(event.event)->wait(), file, line);
  }

  float event_elapsed_time(event_t start, event_t end, const char *file, int line)
  {
    float ms = 0.0f;
    try {
      auto start_event = *static_cast<sycl::event *>(start.event);
      auto end_event = *static_cast<sycl::event *>(end.event);
      auto start_time = start_event.get_profiling_info<sycl::info::event_profiling::command_end>();
      auto end_time = end_event.get_profiling_info<sycl::info::event_profiling::command_end>();
      ms = static_cast<float>(end_time - start_time) / 1e6f;
    } catch (sycl::exception const &e) {
      fprintf(stderr, "SYCL error in %s at line %d: %s\n", file, line, e.what());
      exit(-1);
    }
    return ms;
  }
} // namespace target
