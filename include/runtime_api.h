#pragma once

#include <stdlib.h>

namespace target
{

  struct event_t {
    void *event;
  };

  void set_device(int device, const char *file, int line);
  void *malloc(size_t size, const char *file, int line);
  void free(void *dev_ptr, const char *file, int line);
  void memcpy_to_symbol(const void *symbol, const void *src, size_t count, const char *file, int line);
  void launch_kernel(void (*func)(), unsigned int grid_dim, unsigned int block_dim, const char *file, int line);
  event_t event_create(const char *file, int line);
  void event_destory(event_t event, const char *file, int line);
  void event_record(event_t event, const char *file, int line);
  void event_synchronize(event_t event, const char *file, int line);
  float event_elapsed_time(event_t start, event_t end, const char *file, int line);

} // namespace target

#define target_set_device(device) target::set_device(device, __FILE__, __LINE__)
#define target_malloc(size) target::malloc(size, __FILE__, __LINE__)
#define target_free(dev_ptr) target::free(dev_ptr, __FILE__, __LINE__)
#define target_memcpy_to_symbol(symbol, src, count) target::memcpy_to_symbol(symbol, src, count, __FILE__, __LINE__)
#define target_launch_kernel(func, grid_dim, block_dim)                                                                \
  target::launch_kernel(func, grid_dim, block_dim, __FILE__, __LINE__)
#define target_event_create() target::event_create(__FILE__, __LINE__)
#define target_event_destory(event) target::event_destory(event, __FILE__, __LINE__)
#define target_event_record(event) target::event_record(event, __FILE__, __LINE__)
#define target_event_synchronize(event) target::event_synchronize(event, __FILE__, __LINE__)
#define target_event_elapsed_time(start, end) target::event_elapsed_time(start, end, __FILE__, __LINE__)
