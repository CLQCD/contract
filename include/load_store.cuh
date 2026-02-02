#include <array>

template <int LANE_SIZE, typename T>
__inline__ __device__ void load_tile(T (*dst)[LANE_SIZE], void *src, size_t x_offset, int x_idx, int idx)
{
  if constexpr (LANE_SIZE == 1) {
    dst[x_idx][idx] = static_cast<T *>(src)[x_offset + x_idx];
  } else if constexpr (LANE_SIZE & (LANE_SIZE - 1) == 0) {
    if (idx < 1) {
      dst[x_idx][idx] = static_cast<T *>(src)[x_offset + x_idx];
      dst[x_idx][idx + 1] = dst[x_idx][idx];
    }
    __syncthreads();
#pragma unroll
    for (int stride = 2; stride < LANE_SIZE; stride *= 2) {
      if (idx < stride) { dst[x_idx][idx + stride] = dst[x_idx][idx]; }
      __syncthreads();
    }
  } else {
    if (idx < 1) {
#pragma unroll
      dst[x_idx][idx] = static_cast<T *>(src)[x_offset + x_idx];
      for (int i = 1; i < LANE_SIZE; i++) { dst[x_idx][i] = dst[x_idx][idx]; }
    }
    __syncthreads();
  }
}

template <int LANE_SIZE, typename T>
__inline__ __device__ void store_tile(void *dst, T (*src)[LANE_SIZE], size_t x_offset, int x_idx, int idx)
{
  if constexpr (LANE_SIZE == 1) {
    static_cast<T *>(dst)[x_offset + x_idx] = src[x_idx][idx];
  } else if constexpr (LANE_SIZE & (LANE_SIZE - 1) == 0) {
#pragma unroll
    for (int stride = LANE_SIZE / 2; stride > 1; stride /= 2) {
      if (idx < stride) { src[x_idx][idx] += src[x_idx][idx + stride]; }
      __syncthreads();
    }
    if (idx < 1) {
      src[x_idx][idx] += src[x_idx][idx + 1];
      static_cast<T *>(dst)[x_offset + x_idx] = src[x_idx][idx];
    }
    __syncthreads();
  } else {
    if (idx < 1) {
#pragma unroll
      for (int i = 1; i < LANE_SIZE; i++) { src[x_idx][idx] += src[x_idx][i]; }
      static_cast<T *>(dst)[x_offset + x_idx] = src[x_idx][idx];
    }
    __syncthreads();
  }
}

template <int LANE_SIZE, typename T>
__inline__ __device__ void load_lane(T (*dst)[LANE_SIZE], void *src, size_t x_offset, int x_idx, int idx)
{
  size_t offset = x_offset * LANE_SIZE;
  int pos = x_idx * LANE_SIZE + idx;
  dst[x_idx][idx] = static_cast<T *>(src)[offset + pos];
}

template <int LANE_SIZE, typename T>
__inline__ __device__ void store_lane(void *dst, T (*src)[LANE_SIZE], size_t x_offset, int x_idx, int idx)
{
  size_t offset = x_offset * LANE_SIZE;
  int pos = x_idx * LANE_SIZE + idx;
  static_cast<T *>(dst)[offset + pos] = src[x_idx][idx];
}

template <int LANE_SIZE, int VECTOR_SIZE, int BLOCK_SIZE, int N, typename T>
__inline__ __device__ void load_vector(std::array<T (*)[LANE_SIZE][VECTOR_SIZE], N> dst, std::array<void *, N> src,
                                       size_t x_offset, int x_idx, int idx)
{
  size_t offset = x_offset * LANE_SIZE * VECTOR_SIZE;
  for (int pos = x_idx * LANE_SIZE + idx; pos < BLOCK_SIZE * VECTOR_SIZE; pos += BLOCK_SIZE) {
    int t_idx = pos / VECTOR_SIZE / LANE_SIZE;
    int l_idx = pos / VECTOR_SIZE % LANE_SIZE;
    int v_idx = pos % VECTOR_SIZE;
#pragma unroll
    for (int i = 0; i < N; i++) { dst[i][t_idx][l_idx][v_idx] = static_cast<T *>(src[i])[offset + pos]; }
  }
  __syncthreads();
}

template <int LANE_SIZE, int VECTOR_SIZE, int BLOCK_SIZE, int N, typename T>
__inline__ __device__ void store_vector(std::array<void *, N> dst, std::array<T (*)[LANE_SIZE][VECTOR_SIZE], N> src,
                                        size_t x_offset, int x_idx, int idx)
{
  size_t offset = x_offset * LANE_SIZE * VECTOR_SIZE;
  for (int pos = x_idx * LANE_SIZE + idx; pos < BLOCK_SIZE * VECTOR_SIZE; pos += BLOCK_SIZE) {
    int t_idx = pos / VECTOR_SIZE / LANE_SIZE;
    int l_idx = pos / VECTOR_SIZE % LANE_SIZE;
    int v_idx = pos % VECTOR_SIZE;
#pragma unroll
    for (int i = 0; i < N; i++) { static_cast<T *>(dst[i])[offset + pos] = src[i][t_idx][l_idx][v_idx]; }
  }
  __syncthreads();
}

template <int LANE_SIZE, int VECTOR_SIZE, int BLOCK_SIZE, typename T>
__inline__ __device__ void load_vector(T (*dst)[LANE_SIZE][VECTOR_SIZE], void *src, size_t x_offset, int x_idx, int idx)
{
  load_vector<LANE_SIZE, VECTOR_SIZE, BLOCK_SIZE, 1, T>(std::array {dst}, std::array {src}, x_offset, x_idx, idx);
}

template <int LANE_SIZE, int VECTOR_SIZE, int BLOCK_SIZE, typename T>
__inline__ __device__ void store_vector(void *dst, T (*src)[LANE_SIZE][VECTOR_SIZE], size_t x_offset, int x_idx, int idx)
{
  store_vector<LANE_SIZE, VECTOR_SIZE, BLOCK_SIZE, 1, T>(std::array {dst}, std::array {src}, x_offset, x_idx, idx);
}
