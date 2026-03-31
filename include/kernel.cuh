#pragma once

#include <type_traits>

#include <runtime_api.h>

#if defined(GPU_TARGET_CUDA)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
namespace cg = cooperative_groups;
#elif defined(GPU_TARGET_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <rocprim/rocprim.hpp>
namespace cg = cooperative_groups;
#elif defined(GPU_TARGET_SYCL)
#include <sycl/sycl.hpp>
extern sycl::queue sycl_queue;
#endif

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
  [&]() {                                                                                                              \
    auto _v1 = _in_1[b * 3 + e] * _in_2[c * 3 + f] + _in_1[c * 3 + f] * _in_2[b * 3 + e];                              \
    auto _v2 = _in_1[c * 3 + e] * _in_2[b * 3 + f] + _in_1[b * 3 + f] * _in_2[c * 3 + e];                              \
    return _v1 - _v2;                                                                                                  \
  }()

namespace contract
{
#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
  using ThreadBlock = cg::thread_block;
#elif defined(GPU_TARGET_SYCL)
  using ThreadBlock = sycl::sub_group;
#endif

#if defined(GPU_TARGET_CUDA)
  template <unsigned int TILE_SIZE> using ThreadTile = cg::thread_block_tile<TILE_SIZE, ThreadBlock>;
#elif defined(GPU_TARGET_HIP)
  template <unsigned int TILE_SIZE> using ThreadTile = cg::thread_block_tile<TILE_SIZE>;
#elif defined(GPU_TARGET_SYCL)
  template <unsigned int TILE_SIZE> struct ThreadTile {
    sycl::sub_group sg;
    sycl::nd_item<1> item;

    ThreadTile(sycl::sub_group sg_, sycl::nd_item<1> item_) : sg(sg_), item(item_) { }

    unsigned int thread_rank() const { return sg.get_local_id()[0]; }
    unsigned int meta_group_rank() const { return sg.get_group_id()[0]; }

    template <typename T> T shfl(T val, unsigned int src) const { return sycl::select_from_group(sg, val, src); }
    template <typename T> T shfl_down(T val, unsigned int delta) const
    { return sycl::shift_group_left(sg, val, delta); }

    void sync() const { sycl::group_barrier(sg); }
  };
#endif

  template <typename T, unsigned int BLOCK_SIZE, unsigned int TILE_SIZE> struct WarpReduce {
    WarpReduce() = default;

#if defined(GPU_TARGET_CUDA)
    static __device__ __forceinline__ T plus(const unsigned int gid, T input)
    {
      __shared__ typename cub::WarpReduce<T, TILE_SIZE>::TempStorage storage[BLOCK_SIZE / TILE_SIZE];
      return cub::WarpReduce<T, TILE_SIZE>(storage[gid]).Reduce(input, cuda::std::plus<>());
    }
#elif defined(GPU_TARGET_HIP)
    static __device__ __forceinline__ T plus(const unsigned int gid, T input)
    {
      __shared__ typename rocprim::warp_reduce<T, TILE_SIZE>::storage_type storage[BLOCK_SIZE / TILE_SIZE];
      T output;
      rocprim::warp_reduce<T, TILE_SIZE>().reduce(input, output, storage[gid], rocprim::plus<T>());
      return output;
    }
#elif defined(GPU_TARGET_SYCL)
    static T plus(const unsigned int gid, T input, sycl::sub_group sg)
    {
      for (unsigned int stride = TILE_SIZE / 2; stride > 0; stride /= 2) {
        T other;
        if constexpr (std::is_arithmetic_v<T>) {
          other = sycl::shift_group_left(sg, input, stride);
        } else if constexpr (sizeof(T) % sizeof(double) == 0) {
          constexpr int N = sizeof(T) / sizeof(double);
          double *src = reinterpret_cast<double *>(&input);
          double *dst = reinterpret_cast<double *>(&other);
          for (int i = 0; i < N; ++i) dst[i] = sycl::shift_group_left(sg, src[i], stride);
        } else {
          constexpr int N = sizeof(T) / sizeof(float);
          static_assert(sizeof(T) % sizeof(float) == 0);
          float *src = reinterpret_cast<float *>(&input);
          float *dst = reinterpret_cast<float *>(&other);
          for (int i = 0; i < N; ++i) dst[i] = sycl::shift_group_left(sg, src[i], stride);
        }
        input += other;
      }
      return input;
    }
#endif
  };

  constexpr size_t constant_buffer_size() { return 32764; };

#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
  __constant__ char buffer[constant_buffer_size()];
#elif defined(GPU_TARGET_SYCL)
  static sycl::ext::oneapi::experimental::device_global<std::array<char, constant_buffer_size()>> buffer;
#endif

  static char *get_buffer()
  {
#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
    return buffer;
#elif defined(GPU_TARGET_SYCL)
    return buffer.get().data();
#endif
  }

  template <typename Args> static constexpr Args &get_args() { return *reinterpret_cast<Args *>(get_buffer()); }

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
#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
    const size_t x_offset = blockIdx.x * Kernel::TILES_PER_BLOCK;
    Kernel functor(get_args<Args>());

    if constexpr (std::is_base_of_v<TileKernel<Args, Kernel::BLOCK_SIZE, Kernel::TILE_SIZE>, Kernel>) {
      ThreadBlock block = cg::this_thread_block();
      ThreadTile<Kernel::TILE_SIZE> tile = cg::tiled_partition<Kernel::TILE_SIZE>(block);
      functor(x_offset, tile);
    } else {
      functor(x_offset);
    }
#endif
  }

  template <typename Kernel, typename Args> void launch_kernel(Args &args, size_t volume)
  {
    unsigned int grid_dim = (volume * Kernel::TILE_SIZE + Kernel::BLOCK_SIZE - 1) / Kernel::BLOCK_SIZE;
    unsigned int block_dim = Kernel::BLOCK_SIZE;

    static_assert(sizeof(Args) <= constant_buffer_size(), "Parameter struct is greater than max constant buffer size");
    target_memcpy_to_symbol(get_buffer(), &args, sizeof(Args));
#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
    auto func = kernel<Kernel, Args>;
    target_launch_kernel(Kernel::TILE_SIZE, func, grid_dim, block_dim);
#elif defined(GPU_TARGET_SYCL)
    sycl_queue
      .submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_dim * block_dim), sycl::range<1>(block_dim)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Kernel::TILE_SIZE)]] {
                         const size_t x_offset = item.get_group(0) * Kernel::TILES_PER_BLOCK;
                         Kernel functor(get_args<Args>());
                         if constexpr (std::is_base_of_v<TileKernel<Args, Kernel::BLOCK_SIZE, Kernel::TILE_SIZE>, Kernel>) {
                           ThreadBlock block = item.get_sub_group();
                           ThreadTile<Kernel::TILE_SIZE> tile(block, item);
                           functor(x_offset, tile);
                         } else {
                           functor(x_offset);
                         }
                       });
      })
      .wait();
#endif
  }

} // namespace contract