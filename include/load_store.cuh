#pragma once

#include <array>

namespace contract
{

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void bcast_lane(T (*dst)[LANE_SIZE], T &src)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    dst[t_idx][l_idx] = src;
  }

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void reduce_lane(T &dst, T (*src)[LANE_SIZE])
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
      dst = src[t_idx][0];
    } else if constexpr (LANE_SIZE & (LANE_SIZE - 1) == 0) {
#pragma unroll
      for (int stride = LANE_SIZE / 2; stride > 1; stride /= 2) {
        if (l_idx < stride) { src[t_idx][l_idx] += src[t_idx][l_idx + stride]; }
        __syncthreads();
      }
      if (l_idx == 0) {
        src[t_idx][l_idx] += src[t_idx][l_idx + 1];
        dst = src[t_idx][0];
      }
      __syncthreads();
    } else {
      if (l_idx == 0) {
#pragma unroll
        for (int i = 1; i < LANE_SIZE; i++) { src[t_idx][0] += src[t_idx][i]; }
        dst = src[t_idx][0];
      }
      __syncthreads();
    }
  }

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void bcast_lane(T (*data)[LANE_SIZE])
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
    } else {
      if (l_idx > 0) { data[t_idx][l_idx] = data[t_idx][0]; }
      __syncthreads();
    }
  }

  template <int LANE_SIZE, typename T> __device__ __forceinline__ void reduce_lane(T (*data)[LANE_SIZE])
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
    } else if constexpr (LANE_SIZE & (LANE_SIZE - 1) == 0) {
#pragma unroll
      for (int stride = LANE_SIZE / 2; stride > 0; stride /= 2) {
        if (l_idx < stride) { data[t_idx][l_idx] += data[t_idx][l_idx + stride]; }
        __syncthreads();
      }
    } else {
      if (l_idx == 0) {
#pragma unroll
        for (int i = 1; i < LANE_SIZE; i++) { data[t_idx][0] += data[t_idx][i]; }
      }
      __syncthreads();
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void load_tile(T (*dst)[LANE_SIZE], void *src, size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
      dst[t_idx][l_idx] = static_cast<T *>(src)[x_offset + t_idx];
    } else {
      if (l_idx == 0) { dst[t_idx][0] = static_cast<T *>(src)[x_offset + t_idx]; }
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void store_tile(void *dst, T (*src)[LANE_SIZE], size_t x_offset)
  {
    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    if constexpr (LANE_SIZE == 1) {
      static_cast<T *>(dst)[x_offset + t_idx] = src[t_idx][l_idx];
    } else {
      if (l_idx == 0) { static_cast<T *>(dst)[x_offset + t_idx] = src[t_idx][0]; }
    }
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void load_lane(T (*dst)[LANE_SIZE], void *src, size_t x_offset)
  {
    size_t offset = x_offset * LANE_SIZE + threadIdx.x;
    (&dst[0][0])[threadIdx.x] = static_cast<T *>(src)[offset];
  }

  template <int LANE_SIZE, typename T>
  __device__ __forceinline__ void store_lane(void *dst, T (*src)[LANE_SIZE], size_t x_offset)
  {
    size_t offset = x_offset * LANE_SIZE + threadIdx.x;
    static_cast<T *>(dst)[offset] = (&src[0][0])[threadIdx.x];
  }

  template <int LANE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void load_vector(T (*dst)[LANE_SIZE][VECTOR_SIZE], void *src, size_t x_offset)
  {
    const size_t offset = x_offset * LANE_SIZE * VECTOR_SIZE + threadIdx.x;
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      (&dst[0][0][0])[threadIdx.x + v * blockDim.x] = static_cast<T *>(src)[offset + v * blockDim.x];
    }
  }

  template <int LANE_SIZE, int VECTOR_SIZE, typename T>
  __device__ __forceinline__ void store_vector(void *dst, T (*src)[LANE_SIZE][VECTOR_SIZE], size_t x_offset)
  {
    const size_t offset = x_offset * LANE_SIZE * VECTOR_SIZE + threadIdx.x;
#pragma unroll
    for (int v = 0; v < VECTOR_SIZE; v++) {
      static_cast<T *>(dst)[offset + v * blockDim.x] = (&src[0][0][0])[threadIdx.x + v * blockDim.x];
    }
  }

}; // namespace contract
