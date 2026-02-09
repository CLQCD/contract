#include <baryon_v2.cuh>
#include <baryon.h>

void baryon_il_jn_mk(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn)
{
  if (volume % TILE_SIZE != 0) {
    fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
    exit(-1);
  }

  unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int block = BLOCK_SIZE;
  dim3 gridDim(grid, 1, 1);
  dim3 blockDim(block, 1, 1);

  Arguments args_h = {correl, propag_m, propag_i, propag_j, gamma_ij, gamma_kl, gamma_mn};
  cudaMemcpyToSymbol(args, &args_h, sizeof(Arguments));
  cudaLaunchKernel(instantiate<IL_JN_MK>(gamma_kl), gridDim, blockDim, {});

  return;
}
