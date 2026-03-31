#pragma once

#include <type_traits>

#include <kernel_define.cuh>
#include <complex.cuh>
#include <runtime_api.h>

#if defined(GPU_TARGET_CUDA)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#define shared_memory(_type, _name, _shape) __shared__ _type _name##_shape
#elif defined(GPU_TARGET_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <rocprim/rocprim.hpp>
#define shared_memory(_type, _name, _shape) __shared__ _type _name _shape
#elif defined(GPU_TARGET_SYCL)
#include <sycl/sycl.hpp>
#define shared_memory(_type, _name, _shape)                                                                            \
  auto &_name = *sycl::ext::oneapi::group_local_memory_for_overwrite<_type _shape>(tile.block)

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

namespace target
{
#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
  __constant__ char buffer[constant_buffer_size()];
#elif defined(GPU_TARGET_SYCL)
  extern sycl::queue sycl_queue;
  extern char *buffer;
#endif
}; // namespace target

namespace contract
{
  template <unsigned int TILE_SIZE> struct ThreadTile {
#if defined(GPU_TARGET_CUDA)
    using cg = cooperative_groups;
    const cg::thread_block_tile<TILE_SIZE, cg::thread_block> tile;

    __device__ __forceinline__ ThreadTile() : tile(cg::tiled_partition<TILE_SIZE>(cg::this_thread_block())) { }

    __device__ __forceinline__ unsigned int thread_rank() const { return tile.thread_rank(); }
    __device__ __forceinline__ unsigned int meta_group_rank() const { return tile.meta_group_rank(); }

    template <typename T> __device__ __forceinline__ T shfl(T val, unsigned int src) const
    { return tile.shfl(val, src); }
    template <typename T> __device__ __forceinline__ T shfl_down(T val, unsigned int delta) const
    { return tile.shfl_down(val, delta); }

    __device__ __forceinline__ void sync() const { tile.sync(); }

    template <unsigned int BLOCK_SIZE, typename T> __device__ __forceinline__ T plus(T input)
    {
      __shared__ typename cub::WarpReduce<T, TILE_SIZE>::TempStorage storage[BLOCK_SIZE / TILE_SIZE];
      return cub::WarpReduce<T, TILE_SIZE>(storage[meta_group_rank()]).Reduce(input, cuda::std::plus<>());
    }
#elif defined(GPU_TARGET_HIP)
    using cg = cooperative_groups;
    const cg::thread_block_tile<TILE_SIZE> tile;

    __device__ __forceinline__ ThreadTile() : tile(cg::tiled_partition<TILE_SIZE>(cg::this_thread_block())) { }

    __device__ __forceinline__ unsigned int thread_rank() const { return tile.thread_rank(); }
    __device__ __forceinline__ unsigned int meta_group_rank() const { return tile.meta_group_rank(); }

    template <typename T> __device__ __forceinline__ T shfl(T val, unsigned int src) const
    { return tile.shfl(val, src); }
    template <typename F> __device__ __forceinline__ Complex<F> shfl(Complex<F> val, unsigned int src) const
    {
      F real = tile.shfl(val.real(), src);
      F imag = tile.shfl(val.imag(), src);
      return {real, imag};
    }
    template <typename T> __device__ __forceinline__ T shfl_down(T val, unsigned int delta) const
    { return tile.shfl_down(val, delta); }
    template <typename F> __device__ __forceinline__ Complex<F> shfl_down(Complex<F> val, unsigned int delta) const
    {
      F real = tile.shfl_down(val.real(), delta);
      F imag = tile.shfl_down(val.imag(), delta);
      return {real, imag};
    }

    __device__ __forceinline__ void sync() const { tile.sync(); }

    template <unsigned int BLOCK_SIZE, typename T> __device__ __forceinline__ T plus(T input)
    {
      __shared__ typename rocprim::warp_reduce<T, TILE_SIZE>::storage_type storage[BLOCK_SIZE / TILE_SIZE];
      T output;
      rocprim::warp_reduce<T, TILE_SIZE>().reduce(input, output, storage[meta_group_rank()], rocprim::plus<T>());
      return output;
    }
#elif defined(GPU_TARGET_SYCL)
    const sycl::group<1> block;
    const sycl::sub_group tile;

    __device__ __forceinline__ ThreadTile(sycl::nd_item<1> &block) :
      block(block.get_group()), tile(block.get_sub_group())
    {
    }

    __device__ __forceinline__ unsigned int thread_rank() const { return tile.get_local_id()[0]; }
    __device__ __forceinline__ unsigned int meta_group_rank() const { return tile.get_group_id()[0]; }

    template <typename T> __device__ __forceinline__ T shfl(T val, unsigned int src) const
    { return sycl::select_from_group(tile, val, src); }
    template <typename T> __device__ __forceinline__ T shfl_down(T val, unsigned int delta) const
    { return sycl::shift_group_left(tile, val, delta); }

    __device__ __forceinline__ void sync() const { sycl::group_barrier(tile); }

    template <unsigned int BLOCK_SIZE, typename T> __device__ __forceinline__ T plus(T input)
    { return sycl::reduce_over_group(tile, input, sycl::plus<>()); }
    template <unsigned int BLOCK_SIZE, typename F> __device__ __forceinline__ Complex<F> plus(Complex<F> input)
    {
      F real = sycl::reduce_over_group(tile, input.real(), sycl::plus<>());
      F imag = sycl::reduce_over_group(tile, input.imag(), sycl::plus<>());
      return {real, imag};
    }
#endif
  };

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

#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
  template <typename Kernel, typename Args> __global__ void kernel()
  {
    const size_t x_offset = blockIdx.x * Kernel::TILES_PER_BLOCK;
    Kernel functor(reinterpret_cast<Args &>(target::buffer));

    if constexpr (std::is_base_of_v<TileKernel<Args, Kernel::BLOCK_SIZE, Kernel::TILE_SIZE>, Kernel>) {
      ThreadTile<Kernel::TILE_SIZE> tile();
      functor(x_offset, tile);
    } else {
      functor(x_offset);
    }
  }
#endif

  template <typename Kernel, typename Args> void launch_kernel_(Args &args, size_t volume, const char *file, int line)
  {
    unsigned int grid_dim = (volume * Kernel::TILE_SIZE + Kernel::BLOCK_SIZE - 1) / Kernel::BLOCK_SIZE;
    unsigned int block_dim = Kernel::BLOCK_SIZE;

    static_assert(sizeof(Args) <= target::constant_buffer_size(),
                  "Parameter struct is greater than max constant buffer size");
    target::memcpy_to_symbol(reinterpret_cast<const void *>(target::buffer), &args, sizeof(Args), file, line);
#if defined(GPU_TARGET_CUDA) || defined(GPU_TARGET_HIP)
    const void *func = reinterpret_cast<const void *>(kernel<Kernel, Args>);
    target::launch_kernel(func, grid_dim, block_dim, file, line);
#elif defined(GPU_TARGET_SYCL)
    Args *args_buffer = reinterpret_cast<Args *>(target::buffer);
    target::sycl_queue
      .submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_dim * block_dim), sycl::range<1>(block_dim)),
                       [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Kernel::TILE_SIZE)]] {
                         const size_t x_offset = item.get_group(0) * Kernel::TILES_PER_BLOCK;
                         Kernel functor(*args_buffer);
                         if constexpr (std::is_base_of_v<TileKernel<Args, Kernel::BLOCK_SIZE, Kernel::TILE_SIZE>, Kernel>) {
                           ThreadTile<Kernel::TILE_SIZE> tile(item);
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

#define contract_launch_kernel(Kernel, args, volume) contract::launch_kernel_<Kernel>(args, volume, __FILE__, __LINE__)