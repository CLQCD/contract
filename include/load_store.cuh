#pragma once

#include <complex.cuh>
#include <kernel.cuh>

namespace contract
{

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ T tile_shfl_down(ThreadTile<TILE_SIZE> tile, T var, unsigned int delta)
  { return tile.shfl_down(var, delta); }

  template <typename F, unsigned int TILE_SIZE>
  __device__ __forceinline__ Complex<F> tile_shfl_down(ThreadTile<TILE_SIZE> tile, Complex<F> var, unsigned int delta)
  {
    F r = tile.shfl_down(var.real(), delta);
    F i = tile.shfl_down(var.imag(), delta);
    return Complex<F>(r, i);
  }

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ T tile_shfl(ThreadTile<TILE_SIZE> tile, T var, int src)
  { return tile.shfl(var, src); }

  template <typename F, unsigned int TILE_SIZE>
  __device__ __forceinline__ Complex<F> tile_shfl(ThreadTile<TILE_SIZE> tile, Complex<F> var, int src)
  {
    F r = tile.shfl(var.real(), src);
    F i = tile.shfl(var.imag(), src);
    return Complex<F>(r, i);
  }

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ void tile_load_bcast(ThreadTile<TILE_SIZE> tile, T *dst, void *src, size_t x_offset)
  {
    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();
    const size_t offset = x_offset + gid;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    if constexpr (TILE_SIZE == 1) {
      dst[0] = src_ptr[offset];
    } else {
      T var;
      if (tid == 0) { var = src_ptr[offset]; }
      dst[tid] = tile.shfl(var, 0);
    }
  }

  template <unsigned int BLOCK_SIZE, typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ void tile_reduce_store(ThreadTile<TILE_SIZE> tile, void *dst, T *src, size_t x_offset)
  {
    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();
    const size_t offset = x_offset + gid;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    T var = tile.template plus<BLOCK_SIZE>(src[tid]);
    if constexpr (TILE_SIZE == 1) {
      dst_ptr[offset] = var;
    } else {
      if (tid == 0) { dst_ptr[offset] = var; }
    }
  }

  template <unsigned int BLOCK_SIZE, typename T, unsigned int TILE_SIZE, unsigned int VECTOR_SIZE>
  __device__ __forceinline__ void tile_allreduce_vector(ThreadTile<TILE_SIZE> tile, T (*dst)[VECTOR_SIZE],
                                                        T (*src)[VECTOR_SIZE])
  {
    const auto tid = tile.thread_rank();
    if constexpr (TILE_SIZE == 1) {
    } else {
#pragma unroll
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        T var = tile.template plus<BLOCK_SIZE>(src[tid][v]);
        dst[tid][v] = tile.shfl(var, 0);
      }
    }
  }

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ void tile_load(ThreadTile<TILE_SIZE> tile, T *dst, void *src, size_t x_offset)
  {
    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();
    const size_t offset = x_offset * TILE_SIZE + gid * TILE_SIZE + tid;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    dst[tid] = src_ptr[offset];
  }

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ void tile_store(ThreadTile<TILE_SIZE> tile, void *dst, T *src, size_t x_offset)
  {
    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();
    const size_t offset = x_offset * TILE_SIZE + gid * TILE_SIZE + tid;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    dst_ptr[offset] = src[tid];
  }

  template <typename T, unsigned int TILE_SIZE, unsigned int VECTOR_SIZE>
  __device__ __forceinline__ void tile_load_vector(ThreadTile<TILE_SIZE> tile, T (*dst)[VECTOR_SIZE], void *src,
                                                   size_t x_offset)
  {
    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();
    const size_t offset = (x_offset + gid) * TILE_SIZE * VECTOR_SIZE + tid;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; ++v) { (&dst[0][0])[tid + v * TILE_SIZE] = src_ptr[offset + v * TILE_SIZE]; }
  }

  template <typename T, unsigned int TILE_SIZE, unsigned int VECTOR_SIZE>
  __device__ __forceinline__ void tile_store_vector(ThreadTile<TILE_SIZE> tile, void *dst, T (*src)[VECTOR_SIZE],
                                                    size_t x_offset)
  {
    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();
    const size_t offset = (x_offset + gid) * TILE_SIZE * VECTOR_SIZE + tid;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; ++v) { dst_ptr[offset + v * TILE_SIZE] = (&src[0][0])[tid + v * TILE_SIZE]; }
  }

#ifdef GPU_TARGET_SYCL

  // SYCL versions of non-tile load/store that use nd_item instead of threadIdx.x / blockDim.x

  template <int TILE_SIZE, int VECTOR_SIZE, typename T>
  inline __attribute__((always_inline)) void load_vector(T (*dst)[TILE_SIZE][VECTOR_SIZE], void *src, size_t x_offset,
                                                         sycl::nd_item<1> item)
  {
    const size_t local_id = item.get_local_id(0);
    const size_t local_range = item.get_local_range(0);
    const size_t offset = x_offset * TILE_SIZE * VECTOR_SIZE + local_id;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      (&dst[0][0][0])[local_id + v * local_range] = src_ptr[offset + v * local_range];
    }
  }

  template <int TILE_SIZE, int VECTOR_SIZE, typename T>
  inline __attribute__((always_inline)) void store_vector(void *dst, T (*src)[TILE_SIZE][VECTOR_SIZE], size_t x_offset,
                                                          sycl::nd_item<1> item)
  {
    const size_t local_id = item.get_local_id(0);
    const size_t local_range = item.get_local_range(0);
    const size_t offset = x_offset * TILE_SIZE * VECTOR_SIZE + local_id;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      dst_ptr[offset + v * local_range] = (&src[0][0][0])[local_id + v * local_range];
    }
  }

#endif // GPU_TARGET_SYCL

}; // namespace contract
