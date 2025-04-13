#include <cuda_runtime.h>
#include <contract.h>
#include <baryon.cuh>

void baryon_two_point(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                      size_t volume, int gamma_ij, int gamma_kl, int gamma_mn)
{
  if (volume % TILE_SIZE != 0) {
    fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
    exit(-1);
  }

  switch (contract_type) {
  case IK_JL_MN: baryon_ik_jl_mn(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IK_JN_ML: baryon_ik_jn_ml(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IL_JK_MN: baryon_il_jk_mn(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IL_JN_MK: baryon_il_jn_mk(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IN_JK_ML: baryon_in_jk_ml(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IN_JL_MK: baryon_in_jl_mk(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  default: break;
  }
}

void proton(void *correl, void *propag_i, void *propag_j, void *propag_m, int contract_type, size_t volume,
            int gamma_ij, int gamma_kl, int gamma_mn)
{
  if (volume % TILE_SIZE != 0) {
    fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
    exit(-1);
  }

  cudaEvent_t start, stop;
  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));
  CUDA_ERROR_CHECK(cudaEventRecord(start));
  CUDA_ERROR_CHECK(cudaEventSynchronize(start));
  for (int i = 0; i < 100; ++i) {
    switch ((BaryonContractType)contract_type) {
    case IK_JL_MN: baryon_ik_jl_mn(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IK_JN_ML: baryon_ik_jn_ml(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IL_JK_MN: baryon_il_jk_mn(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IL_JN_MK: baryon_il_jn_mk(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IN_JK_ML: baryon_in_jk_ml(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IN_JL_MK: baryon_in_jl_mk(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
    default: break;
    }
  }
  CUDA_ERROR_CHECK(cudaEventRecord(stop));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Time elapsed: %f ms, Bandwidth: %f GB/s\n", milliseconds / 100,
         (3.0 * volume * Ns * Ns * Nc * Nc * sizeof(Complex128) + volume * sizeof(Complex128)) / (1024 * 1024 * 1024)
           / (milliseconds / 1000.0 / 100));
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

void init(int device) { CUDA_ERROR_CHECK(cudaSetDevice(device)); }