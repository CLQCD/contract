#include <meson.cuh>
#include <meson.h>

namespace meson
{
  void launch(void *correl, void *propag_i, void *propag_j, size_t volume, int gamma_ab, int gamma_dc)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int block = BLOCK_SIZE;
    dim3 gridDim(grid, 1U, 1U);
    dim3 blockDim(block, 1, 1);

    Arguments args_h = {correl, propag_i, propag_j, gamma_ab, gamma_dc};
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(args, &args_h, sizeof(Arguments)));
    CUDA_ERROR_CHECK(cudaLaunchKernel(reinterpret_cast<void *>(meson_kernel), gridDim, blockDim, {}));

    return;
  }
} // namespace meson
