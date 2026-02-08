#pragma once

#include <runtime_api.h>
#include <constant.cuh>
#include <load_store.cuh>

#define for_abc_def                                                                                                    \
  for (int a = 0; a < 3; ++a)                                                                                          \
    for (int b = (a + 1) % 3, c = (a + 2) % 3, _once_abc = 1; _once_abc; _once_abc = 0)                                \
      for (int d = 0; d < 3; ++d)                                                                                      \
        for (int e = (d + 1) % 3, f = (d + 2) % 3, ad = a * 3 + d, _once_def = 1; _once_def; _once_def = 0)

#define for_a_d                                                                                                        \
  for (int a = 0; a < 3; ++a)                                                                                          \
    for (int d = 0; d < 3; ++d)                                                                                        \
      for (int ad = a * 3 + d, _once = 1; _once; _once = 0)

#define epsilon_abc_def(_out, _in_1, _in_2)                                                                            \
  do {                                                                                                                 \
    _out = 0;                                                                                                          \
    _out += _in_1[b * 3 + e] * _in_2[c * 3 + f];                                                                       \
    _out -= _in_1[c * 3 + e] * _in_2[b * 3 + f];                                                                       \
    _out -= _in_1[b * 3 + f] * _in_2[c * 3 + e];                                                                       \
    _out += _in_1[c * 3 + f] * _in_2[b * 3 + e];                                                                       \
  } while (0)

namespace contract
{

  template <typename Args, int BLOCK_SIZE_, int LANE_SIZE_> struct BaseKernel {
    const Args &args;
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int LANE_SIZE = LANE_SIZE_;
    static constexpr int TILE_SIZE = BLOCK_SIZE_ / LANE_SIZE_;

    constexpr BaseKernel(const Args &args) : args(args) { }

    virtual __device__ __forceinline__ void operator()(size_t x_offset) = 0;
  };

  template <typename Kernel, typename Args> __global__ void kernel()
  {
    Kernel functor(get_args<Args>());

    const size_t x_offset = blockIdx.x * Kernel::TILE_SIZE;
    functor(x_offset);
  }

  template <typename Kernel, typename Args> void launch_kernel(Args &args, size_t volume)
  {
    unsigned int grid = (volume * Kernel::LANE_SIZE + Kernel::BLOCK_SIZE - 1) / Kernel::BLOCK_SIZE;
    unsigned int block = Kernel::BLOCK_SIZE;

    static_assert(sizeof(Args) <= buffer_size, "Parameter struct is greater than max constant size");
    device_memcpy_host_to_device(get_buffer<Args>(), &args, sizeof(Args));
    device_launch_kernel(kernel<Kernel, Args>, grid, block);
  }

} // namespace contract