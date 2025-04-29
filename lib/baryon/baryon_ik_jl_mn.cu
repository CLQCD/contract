#include <baryon.cuh>

void baryon_ik_jl_mn(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn)
{
  unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int block = BLOCK_SIZE;
  dim3 gridDim(grid, 1, 1);
  dim3 blockDim(block, 1, 1);

  Arguments args_h = {correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn};
  CUDA_ERROR_CHECK(cudaMemcpyToSymbol(args, &args_h, sizeof(Arguments)));
  CUDA_ERROR_CHECK(cudaLaunchKernel(instantiate<IK_JL_MN>(gamma_mn), gridDim, blockDim, {}));

  return;
}
