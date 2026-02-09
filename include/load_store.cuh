#pragma once

#include <complex.cuh>

namespace contract
{

  template <typename F> __device__ __forceinline__ F warp_shfl_down(F var, unsigned int delta)
  {
    return __shfl_down_sync(0xffffffff, var, delta);
  }

  template <typename F> __device__ __forceinline__ Complex<F> warp_shfl_down(Complex<F> var, unsigned int delta)
  {
    using T = Complex<F>;
    F r = __shfl_down_sync(0xffffffff, var.real(), delta);
    F i = __shfl_down_sync(0xffffffff, var.imag(), delta);
    return T(r, i);
  }

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void bcast_lane(T data[LANE_SIZE])
  {
    static_assert(LANE_SIZE <= 32, "LANE_SIZE must be smaller than the warp size");
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
    } else {
      if (l_idx > 0) { data[l_idx] = data[0]; }
    }
  }

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void reduce_lane(T data[LANE_SIZE])
  {
    static_assert(LANE_SIZE <= 32, "LANE_SIZE must be smaller than the warp size");
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
    } else if constexpr (LANE_SIZE & (LANE_SIZE - 1) == 0) {
      T var = data[l_idx];
#pragma unroll
      for (int stride = LANE_SIZE / 2; stride > 0; stride /= 2) { var += warp_shfl_down(var, stride); }
      if (l_idx == 0) { data[0] = var; }
    } else {
      if (l_idx == 0) {
#pragma unroll
        for (int i = 1; i < LANE_SIZE; i++) { data[0] += data[i]; }
      }
    }
  }

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void bcast_lane(T (*data)[LANE_SIZE])
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
    } else {
      if (l_idx > 0) { data[t_idx][l_idx] = data[t_idx][0]; }
    }
  }

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void reduce_lane(T (*data)[LANE_SIZE])
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
    } else if constexpr (LANE_SIZE & (LANE_SIZE - 1) == 0) {
#pragma unroll
      for (int stride = LANE_SIZE / 2; stride > 1; stride /= 2) {
        if (l_idx < stride) { data[t_idx][l_idx] += data[t_idx][l_idx + stride]; }
        __syncthreads();
      }
      if (l_idx == 0) { data[t_idx][0] += data[t_idx][1]; }
    } else {
      if (l_idx == 0) {
#pragma unroll
        for (int i = 1; i < LANE_SIZE; i++) { data[t_idx][0] += data[t_idx][i]; }
      }
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void load_tile(T dst[LANE_SIZE], void *src, size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = x_offset + t_idx;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    if constexpr (LANE_SIZE == 1) {
      dst[0] = src_ptr[offset];
    } else {
      if (l_idx == 0) { dst[0] = src_ptr[offset]; }
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void store_tile(void *dst, T src[LANE_SIZE], size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = x_offset + t_idx;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    if constexpr (LANE_SIZE == 1) {
      dst_ptr[offset] = src[0];
    } else {
      if (l_idx == 0) { dst_ptr[offset] = src[0]; }
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void load_tile(T (*dst)[LANE_SIZE], void *src, size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = x_offset + t_idx;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    if constexpr (LANE_SIZE == 1) {
      dst[t_idx][0] = src_ptr[offset];
    } else {
      if (l_idx == 0) { dst[t_idx][0] = src_ptr[offset]; }
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void store_tile(void *dst, T (*src)[LANE_SIZE], size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = x_offset + t_idx;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    if constexpr (LANE_SIZE == 1) {
      dst_ptr[offset] = src[t_idx][0];
    } else {
      if (l_idx == 0) { dst_ptr[offset] = src[t_idx][0]; }
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void load_lane(T dst[LANE_SIZE], void *src, size_t x_offset)
  {
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = x_offset * LANE_SIZE + threadIdx.x;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    dst[l_idx] = src_ptr[offset];
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void store_lane(void *dst, T src[LANE_SIZE], size_t x_offset)
  {
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = x_offset * LANE_SIZE + threadIdx.x;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    dst_ptr[offset] = src[l_idx];
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void load_lane(T (*dst)[LANE_SIZE], void *src, size_t x_offset)
  {
    size_t offset = x_offset * LANE_SIZE + threadIdx.x;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
    (&dst[0][0])[threadIdx.x] = src_ptr[offset];
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void store_lane(void *dst, T (*src)[LANE_SIZE], size_t x_offset)
  {
    size_t offset = x_offset * LANE_SIZE + threadIdx.x;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
    dst_ptr[offset] = (&src[0][0])[threadIdx.x];
  }

  template <int LANE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void load_vector(T dst[LANE_SIZE][VECTOR_SIZE], void *src, size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = (x_offset + t_idx) * LANE_SIZE * VECTOR_SIZE + l_idx;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; ++v) { (&dst[0][0])[l_idx + v * LANE_SIZE] = src_ptr[offset + v * LANE_SIZE]; }
  }

  template <int LANE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void store_vector(void *dst, T src[LANE_SIZE][VECTOR_SIZE], size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    const size_t offset = (x_offset + t_idx) * LANE_SIZE * VECTOR_SIZE + l_idx;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; ++v) { dst_ptr[offset + v * LANE_SIZE] = (&src[0][0])[l_idx + v * LANE_SIZE]; }
  }

  template <int LANE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void load_vector(T (*dst)[LANE_SIZE][VECTOR_SIZE], void *src, size_t x_offset)
  {
    const size_t offset = x_offset * LANE_SIZE * VECTOR_SIZE + threadIdx.x;
    const T *__restrict__ src_ptr = static_cast<const T *>(src);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      (&dst[0][0][0])[threadIdx.x + v * blockDim.x] = src_ptr[offset + v * blockDim.x];
    }
  }

  template <int LANE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void store_vector(void *dst, T (*src)[LANE_SIZE][VECTOR_SIZE], size_t x_offset)
  {
    const size_t offset = x_offset * LANE_SIZE * VECTOR_SIZE + threadIdx.x;
    T *__restrict__ dst_ptr = static_cast<T *>(dst);
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      dst_ptr[offset + v * blockDim.x] = (&src[0][0][0])[threadIdx.x + v * blockDim.x];
    }
  }

}; // namespace contract
