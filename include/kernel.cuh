#pragma once

#include <type_traits>

#include <runtime_api.h>

#if defined(GPU_TARGET_CUDA)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#elif defined(GPU_TARGET_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <rocprim/rocprim.hpp>
#endif

namespace cg = cooperative_groups;

#define for_abc_def                                                                                                    \
  for (int a = 0; a < 3; ++a)                                                                                          \
    for (int b = (a + 1) % 3, c = (a + 2) % 3, _once_abc = 1; _once_abc; _once_abc = 0)                                \
      for (int d = 0; d < 3; ++d)                                                                                      \
        for (int e = (d + 1) % 3, f = (d + 2) % 3, ad = a * 3 + d, _once_def = 1; _once_def; _once_def = 0)

#define for_a_d                                                                                                        \
  for (int a = 0; a < 3; ++a)                                                                                          \
    for (int d = 0; d < 3; ++d)                                                                                        \
      for (int ad = a * 3 + d, _once = 1; _once; _once = 0)

#define epsilon_abc_def(_in_1, _in_2)                                                                                  \
  ({                                                                                                                   \
    auto _v1 = _in_1[b * 3 + e] * _in_2[c * 3 + f] + _in_1[c * 3 + f] * _in_2[b * 3 + e];                              \
    auto _v2 = _in_1[c * 3 + e] * _in_2[b * 3 + f] + _in_1[b * 3 + f] * _in_2[c * 3 + e];                              \
    _v1 - _v2;                                                                                                         \
  })

namespace contract
{

  using ThreadBlock = cg::thread_block;
#if defined(GPU_TARGET_CUDA)
  template <unsigned int TILE_SIZE> using ThreadTile = cg::thread_block_tile<TILE_SIZE, ThreadBlock>;
#elif defined(GPU_TARGET_HIP)
  template <unsigned int TILE_SIZE> using ThreadTile = cg::thread_block_tile<TILE_SIZE>;
#endif

  template <typename T, unsigned int BLOCK_SIZE, unsigned int TILE_SIZE> struct WarpReduce {
#if defined(GPU_TARGET_CUDA)
    using TargetWarpReduce = cub::WarpReduce<T, TILE_SIZE>;
    using TargetWarpReduceStorage = typename cub::WarpReduce<T, TILE_SIZE>::TempStorage;
#elif defined(GPU_TARGET_HIP)
    using TargetWarpReduce = rocprim::warp_reduce<T, TILE_SIZE>;
    using TargetWarpReduceStorage = typename rocprim::warp_reduce<T, TILE_SIZE>::storage_type;
#endif

    WarpReduce() = default;

    static __device__ __forceinline__ T plus(const unsigned int gid, T input)
    {
      __shared__ TargetWarpReduceStorage storage[BLOCK_SIZE / TILE_SIZE];
#if defined(GPU_TARGET_CUDA)
      return TargetWarpReduce(storage[gid]).Reduce(input, cuda::std::plus<>());
#elif defined(GPU_TARGET_HIP)
      T output;
      TargetWarpReduce().reduce(input, output, storage[gid], rocprim::plus<T>());
      return output;
#endif
    }
  };

  constexpr size_t constant_buffer_size() { return 32764; };

  __constant__ char buffer[constant_buffer_size()];

  template <typename Args> constexpr Args &get_args() { return reinterpret_cast<Args &>(buffer); }

  template <typename Args, unsigned int BLOCK_SIZE_, unsigned int TILE_SIZE_> struct BaseKernel {
    const Args &args;
    static constexpr unsigned int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr unsigned int TILE_SIZE = TILE_SIZE_;
    static constexpr unsigned int TILES_PER_BLOCK = BLOCK_SIZE_ / TILE_SIZE_;

    constexpr BaseKernel(const Args &args) : args(args) { }
  };

  template <typename Args, unsigned int BLOCK_SIZE_, unsigned int TILE_SIZE_>
  struct TileKernel : public BaseKernel<Args, BLOCK_SIZE_, TILE_SIZE_> {
    constexpr TileKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE_, TILE_SIZE_>(args) { }

    virtual __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE_> tile) = 0;
  };

  template <typename Kernel, typename Args> __global__ void kernel()
  {
    const size_t x_offset = blockIdx.x * Kernel::TILES_PER_BLOCK;
    Kernel functor(get_args<Args>());

    if constexpr (std::is_base_of_v<TileKernel<Args, Kernel::BLOCK_SIZE, Kernel::TILE_SIZE>, Kernel>) {
      ThreadBlock block = cg::this_thread_block();
      ThreadTile<Kernel::TILE_SIZE> tile = cg::tiled_partition<Kernel::TILE_SIZE>(block);
      functor(x_offset, tile);
    } else {
      functor(x_offset);
    }
  }

  template <typename Kernel, typename Args> void launch_kernel(Args &args, size_t volume)
  {
    unsigned int grid = (volume * Kernel::TILE_SIZE + Kernel::BLOCK_SIZE - 1) / Kernel::BLOCK_SIZE;
    unsigned int block = Kernel::BLOCK_SIZE;

    static_assert(sizeof(Args) <= constant_buffer_size(), "Parameter struct is greater than max constant buffer size");
    target_memcpy_to_symbol(buffer, &args, sizeof(Args));
    const void *func = reinterpret_cast<const void *>(kernel<Kernel, Args>);
    target_launch_kernel(func, grid, block);
  }

} // namespace contract