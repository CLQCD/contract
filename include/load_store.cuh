#pragma once

#include <complex.cuh>
#include <kernel.cuh>

namespace contract
{

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ T tile_shfl_down(ThreadTile<TILE_SIZE> tile, T var, unsigned int delta)
  {
    return tile.shfl_down(var, delta);
  }

  template <typename F, unsigned int TILE_SIZE>
  __device__ __forceinline__ Complex<F> tile_shfl_down(ThreadTile<TILE_SIZE> tile, Complex<F> var, unsigned int delta)
  {
    F r = tile.shfl_down(var.real(), delta);
    F i = tile.shfl_down(var.imag(), delta);
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

  template <typename Reduce, typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ void tile_reduce_store(ThreadTile<TILE_SIZE> tile, void *dst, T *src, size_t x_offset)
  {
    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();
    const size_t offset = x_offset + gid;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
#if defined(GPU_TARGET_SYCL)
    T var = Reduce::plus(gid, src[tid], tile.sg);
#else
    T var = Reduce::plus(gid, src[tid]);
#endif
    if constexpr (TILE_SIZE == 1) {
      dst_ptr[offset] = var;
    } else {
      if (tid == 0) { dst_ptr[offset] = var; }
    }
  }

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ void tile_load_scalar(ThreadTile<TILE_SIZE> tile, T *dst, void *src, size_t x_offset)
  {
    const auto tid = tile.thread_rank();
#if defined(GPU_TARGET_SYCL)
    const size_t offset = x_offset * TILE_SIZE + tile.item.get_local_id(0);
#else
    const size_t offset = x_offset * TILE_SIZE + threadIdx.x;
#endif
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    dst[tid] = src_ptr[offset];
  }

  template <typename T, unsigned int TILE_SIZE>
  __device__ __forceinline__ void tile_store_scalar(ThreadTile<TILE_SIZE> tile, void *dst, T *src, size_t x_offset)
  {
    const auto tid = tile.thread_rank();
#if defined(GPU_TARGET_SYCL)
    const size_t offset = x_offset * TILE_SIZE + tile.item.get_local_id(0);
#else
    const size_t offset = x_offset * TILE_SIZE + threadIdx.x;
#endif
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

#if !defined(GPU_TARGET_SYCL)

  template <typename F> __device__ __forceinline__ F warp_shfl_down(F var, unsigned int delta)
  {
#if defined(GPU_TARGET_CUDA)
    return __shfl_down_sync(0xffffffff, var, delta);
#elif defined(GPU_TARGET_HIP)
    return __shfl_down(var, delta);
#endif
  }

  template <typename F> __device__ __forceinline__ Complex<F> warp_shfl_down(Complex<F> var, unsigned int delta)
  {
    using T = Complex<F>;
#if defined(GPU_TARGET_CUDA)
    F r = __shfl_down_sync(0xffffffff, var.real(), delta);
    F i = __shfl_down_sync(0xffffffff, var.imag(), delta);
#elif defined(GPU_TARGET_HIP)
    F r = __shfl_down(var.real(), delta);
    F i = __shfl_down(var.imag(), delta);
#endif
    return T(r, i);
  }

  template <int TILE_SIZE, typename T> __device__ __forceinline__ void bcast_lane(T data[TILE_SIZE])
  {
    static_assert(TILE_SIZE <= 32, "TILE_SIZE must be smaller than the warp size");
    const int l_idx = threadIdx.x % TILE_SIZE;
    if constexpr (TILE_SIZE == 1) {
    } else {
      if (l_idx > 0) { data[l_idx] = data[0]; }
    }
  }

  template <int TILE_SIZE, typename T> __device__ __forceinline__ void reduce_lane(T data[TILE_SIZE])
  {
    static_assert(TILE_SIZE <= 32, "TILE_SIZE must be smaller than the warp size");
    const int l_idx = threadIdx.x % TILE_SIZE;
    if constexpr (TILE_SIZE == 1) {
    } else if constexpr ((TILE_SIZE & (TILE_SIZE - 1)) == 0) {
      T var = data[l_idx];
#pragma unroll
      for (int stride = TILE_SIZE / 2; stride > 0; stride /= 2) { var += warp_shfl_down(var, stride); }
      if (l_idx == 0) { data[0] = var; }
    } else {
      if (l_idx == 0) {
#pragma unroll
        for (int i = 1; i < TILE_SIZE; i++) { data[0] += data[i]; }
      }
    }
  }

  template <int TILE_SIZE, typename T> __device__ __forceinline__ void bcast_lane(T (*data)[TILE_SIZE])
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    if constexpr (TILE_SIZE == 1) {
    } else {
      if (l_idx > 0) { data[t_idx][l_idx] = data[t_idx][0]; }
    }
  }

  template <int TILE_SIZE, typename T> __device__ __forceinline__ void reduce_lane(T (*data)[TILE_SIZE])
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    if constexpr (TILE_SIZE == 1) {
    } else if constexpr ((TILE_SIZE & (TILE_SIZE - 1)) == 0) {
#pragma unroll
      for (int stride = TILE_SIZE / 2; stride > 1; stride /= 2) {
        if (l_idx < stride) { data[t_idx][l_idx] += data[t_idx][l_idx + stride]; }
        __syncthreads();
      }
      if (l_idx == 0) { data[t_idx][0] += data[t_idx][1]; }
    } else {
      if (l_idx == 0) {
#pragma unroll
        for (int i = 1; i < TILE_SIZE; i++) { data[t_idx][0] += data[t_idx][i]; }
      }
    }
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void load_tile(T dst[TILE_SIZE], void *src, size_t x_offset)
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = x_offset + t_idx;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    if constexpr (TILE_SIZE == 1) {
      dst[0] = src_ptr[offset];
    } else {
      if (l_idx == 0) { dst[0] = src_ptr[offset]; }
    }
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void store_tile(void *dst, T src[TILE_SIZE], size_t x_offset)
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = x_offset + t_idx;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    if constexpr (TILE_SIZE == 1) {
      dst_ptr[offset] = src[0];
    } else {
      if (l_idx == 0) { dst_ptr[offset] = src[0]; }
    }
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void load_tile(T (*dst)[TILE_SIZE], void *src, size_t x_offset)
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = x_offset + t_idx;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    if constexpr (TILE_SIZE == 1) {
      dst[t_idx][0] = src_ptr[offset];
    } else {
      if (l_idx == 0) { dst[t_idx][0] = src_ptr[offset]; }
    }
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void store_tile(void *dst, T (*src)[TILE_SIZE], size_t x_offset)
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = x_offset + t_idx;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    if constexpr (TILE_SIZE == 1) {
      dst_ptr[offset] = src[t_idx][0];
    } else {
      if (l_idx == 0) { dst_ptr[offset] = src[t_idx][0]; }
    }
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void load_lane(T dst[TILE_SIZE], void *src, size_t x_offset)
  {
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = x_offset * TILE_SIZE + threadIdx.x;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    dst[l_idx] = src_ptr[offset];
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void store_lane(void *dst, T src[TILE_SIZE], size_t x_offset)
  {
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = x_offset * TILE_SIZE + threadIdx.x;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    dst_ptr[offset] = src[l_idx];
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void load_lane(T (*dst)[TILE_SIZE], void *src, size_t x_offset)
  {
    size_t offset = x_offset * TILE_SIZE + threadIdx.x;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    (&dst[0][0])[threadIdx.x] = src_ptr[offset];
  }

  template <int TILE_SIZE, typename T>
  __device__ __forceinline__ void store_lane(void *dst, T (*src)[TILE_SIZE], size_t x_offset)
  {
    size_t offset = x_offset * TILE_SIZE + threadIdx.x;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    dst_ptr[offset] = (&src[0][0])[threadIdx.x];
  }

  template <int TILE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void load_vector(T dst[TILE_SIZE][VECTOR_SIZE], void *src, size_t x_offset)
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = (x_offset + t_idx) * TILE_SIZE * VECTOR_SIZE + l_idx;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; ++v) { (&dst[0][0])[l_idx + v * TILE_SIZE] = src_ptr[offset + v * TILE_SIZE]; }
  }

  template <int TILE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void store_vector(void *dst, T src[TILE_SIZE][VECTOR_SIZE], size_t x_offset)
  {
    const int t_idx = threadIdx.x / TILE_SIZE;
    const int l_idx = threadIdx.x % TILE_SIZE;
    const size_t offset = (x_offset + t_idx) * TILE_SIZE * VECTOR_SIZE + l_idx;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; ++v) { dst_ptr[offset + v * TILE_SIZE] = (&src[0][0])[l_idx + v * TILE_SIZE]; }
  }

  template <int TILE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void load_vector(T (*dst)[TILE_SIZE][VECTOR_SIZE], void *src, size_t x_offset)
  {
    const size_t offset = x_offset * TILE_SIZE * VECTOR_SIZE + threadIdx.x;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      (&dst[0][0][0])[threadIdx.x + v * blockDim.x] = src_ptr[offset + v * blockDim.x];
    }
  }

  template <int TILE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void store_vector(void *dst, T (*src)[TILE_SIZE][VECTOR_SIZE], size_t x_offset)
  {
    const size_t offset = x_offset * TILE_SIZE * VECTOR_SIZE + threadIdx.x;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      dst_ptr[offset + v * blockDim.x] = (&src[0][0][0])[threadIdx.x + v * blockDim.x];
    }
  }

#else // GPU_TARGET_SYCL

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

#endif // !GPU_TARGET_SYCL

}; // namespace contract
