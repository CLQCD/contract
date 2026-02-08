#pragma once

void device_set(int device);
void device_memcpy_host_to_device(void *dst, const void *src, size_t count);
void device_launch_kernel(void (*func)(), unsigned int grid_dim, unsigned int block_dim);