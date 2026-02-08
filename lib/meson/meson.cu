#include <meson.cuh>
#include <meson.h>

namespace meson
{
  void launch(void *correl, void *propag_i, void *propag_j, size_t volume, int gamma_ij, int gamma_kl)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int block = BLOCK_SIZE;
    dim3 gridDim(grid, 1U, 1U);
    dim3 blockDim(block, 1, 1);

    using Args = contract::MesonArgs<double>;
    using Kernel = contract::MesonKernel<Args>;
    Args args(correl, propag_i, propag_j, gamma_ij, gamma_kl);
    contract::launch_kernel<Kernel>(args, volume);
    return;
  }
} // namespace meson
