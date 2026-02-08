#include <runtime_api.h>

#include <hip/runtime.h>

void device_set(int device) { hipSetDevice(device); }

void device_memcpy_host_to_device(void *dst, const void *src, size_t count)
{
  hipMemcpy(dst, src, count, hipMemcpyHostToDevice);
}

void device_launch_kernel(void (*func)(), unsigned int grid_dim, unsigned int block_dim)
{
  hipLaunchKernel(func, dim3(grid_dim), dim3(block_dim), {});
}