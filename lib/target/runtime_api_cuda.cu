#include <runtime_api.h>

void device_set(int device) { cudaSetDevice(device); }

void device_memcpy_host_to_device(void *dst, const void *src, size_t count)
{
  cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

void device_launch_kernel(void (*func)(), unsigned int grid_dim, unsigned int block_dim)
{
  cudaLaunchKernel(func, dim3(grid_dim), dim3(block_dim), {});
}