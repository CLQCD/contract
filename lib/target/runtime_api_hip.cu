#include <cstdio>
#include <hip/hip_runtime.h>
#include <runtime_api.h>

#define hip_error_check(_call, _file, _line)                                                                           \
  do {                                                                                                                 \
    hipError_t error = _call;                                                                                          \
    if (error != hipSuccess) {                                                                                         \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", _file, _line, hipGetErrorString(error));                    \
      exit(error);                                                                                                     \
    }                                                                                                                  \
  } while (0)

namespace target
{

  void set_device(int device, const char *file, int line) { hip_error_check(hipSetDevice(device), file, line); }

  void *malloc(size_t size, const char *file, int line)
  {
    void *dev_ptr;
    hip_error_check(hipMalloc(&dev_ptr, size), file, line);
    return dev_ptr;
  }

  void free(void *dev_ptr, const char *file, int line) { hip_error_check(hipFree(dev_ptr), file, line); }

  void memcpy_to_symbol(const void *symbol, const void *src, size_t count, const char *file, int line)
  {
    hip_error_check(hipMemcpyToSymbol(symbol, src, count), file, line);
  }

  void launch_kernel(void (*func)(), unsigned int grid_dim, unsigned int block_dim, const char *file, int line)
  {
    hip_error_check(hipLaunchKernel(func, dim3(grid_dim), dim3(block_dim), {}), file, line);
  }

  event_t event_create(const char *file, int line)
  {
    hipEvent_t hip_event;
    hip_error_check(hipEventCreate(&hip_event), file, line);
    event_t event;
    event.event = static_cast<void *>(hip_event);
    return event;
  }

  void event_destory(event_t event, const char *file, int line)
  {
    hip_error_check(hipEventDestroy(static_cast<hipEvent_t>(event.event)), file, line);
  }

  void event_record(event_t event, const char *file, int line)
  {
    hip_error_check(hipEventRecord(static_cast<hipEvent_t>(event.event)), file, line);
  }

  void event_synchronize(event_t event, const char *file, int line)
  {
    hip_error_check(hipEventSynchronize(static_cast<hipEvent_t>(event.event)), file, line);
  }

  float event_elapsed_time(event_t start, event_t end, const char *file, int line)
  {
    float ms;
    hip_error_check(hipEventElapsedTime(&ms, static_cast<hipEvent_t>(start.event), static_cast<hipEvent_t>(end.event)),
                    file, line);
    return ms;
  }
} // namespace target
