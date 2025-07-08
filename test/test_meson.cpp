#include <cuda_runtime.h>
#include <contract.h>

void meson(void *correl, void *propag_a, void *propag_b, size_t volume, int gamma_ab, int gamma_dc)
{
  cudaEvent_t start, stop;
  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));
  meson_two_point(correl, propag_a, propag_b, volume, gamma_ab, gamma_dc);
  CUDA_ERROR_CHECK(cudaEventRecord(start));
  CUDA_ERROR_CHECK(cudaEventSynchronize(start));
  for (int i = 0; i < 100; ++i) { meson_two_point(correl, propag_a, propag_b, volume, gamma_ab, gamma_dc); }
  CUDA_ERROR_CHECK(cudaEventRecord(stop));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Time elapsed: %f ms, Bandwidth: %f GB/s\n", milliseconds / 100,
         (2.0 * volume * 4 * 4 * 3 * 3 * 16 + volume * 16) / (1024 * 1024 * 1024) / (milliseconds / 1000.0 / 100));
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  return;
}

int main(int argc, char *argv[])
{
  init(0);

  size_t volume = 24 * 24 * 24 * 18;
  void *correl, *propag_i, *propag_j, *propag_m;
  cudaMalloc(&correl, volume * 2 * sizeof(double));
  cudaMalloc(&propag_i, volume * 16 * 9 * 2 * sizeof(double));
  cudaMalloc(&propag_j, volume * 16 * 9 * 2 * sizeof(double));
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  meson(correl, propag_i, propag_j, volume, 5, 5);
  cudaFree(correl);
  cudaFree(propag_i);
  cudaFree(propag_j);
  cudaFree(propag_m);

  return 0;
}
