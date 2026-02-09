#include <cstdio>
#include <cuda_runtime.h>
#include <runtime_api.h>

#define cuda_error_check(_call, _file, _line)                                                                          \
  do {                                                                                                                 \
    cudaError_t error = _call;                                                                                         \
    if (error != cudaSuccess) {                                                                                        \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", _file, _line, cudaGetErrorString(error));                   \
      exit(error);                                                                                                     \
    }                                                                                                                  \
  } while (0)

namespace target
{

  void set_device(int device, const char *file, int line) { cuda_error_check(cudaSetDevice(device), file, line); }

  void *malloc(size_t size, const char *file, int line)
  {
    void *dev_ptr;
    cuda_error_check(cudaMalloc(&dev_ptr, size), file, line);
    return dev_ptr;
  }

  void free(void *dev_ptr, const char *file, int line) { cuda_error_check(cudaFree(dev_ptr), file, line); }

  void memcpy_to_symbol(const void *symbol, const void *src, size_t count, const char *file, int line)
  {
    cuda_error_check(cudaMemcpyToSymbol(symbol, src, count), file, line);
  }

  void launch_kernel(void (*func)(), unsigned int grid_dim, unsigned int block_dim, const char *file, int line)
  {
    cuda_error_check(cudaLaunchKernel(func, dim3(grid_dim), dim3(block_dim), {}), file, line);
  }

  event_t event_create(const char *file, int line)
  {
    cudaEvent_t cuda_event;
    cuda_error_check(cudaEventCreate(&cuda_event), file, line);
    event_t event;
    event.event = static_cast<void *>(cuda_event);
    return event;
  }

  void event_destory(event_t event, const char *file, int line)
  {
    cuda_error_check(cudaEventDestroy(static_cast<cudaEvent_t>(event.event)), file, line);
  }

  void event_record(event_t event, const char *file, int line)
  {
    cuda_error_check(cudaEventRecord(static_cast<cudaEvent_t>(event.event)), file, line);
  }

  void event_synchronize(event_t event, const char *file, int line)
  {
    cuda_error_check(cudaEventSynchronize(static_cast<cudaEvent_t>(event.event)), file, line);
  }

  float event_elapsed_time(event_t start, event_t end, const char *file, int line)
  {
    float ms;
    cuda_error_check(
      cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start.event), static_cast<cudaEvent_t>(end.event)), file, line);
    return ms;
  }
} // namespace target
