#include <meson_all.cuh>
#include <meson.h>

namespace meson_all_source
{
  void launch(void *correl[Ns * Ns], void *propag_a, void *propag_b, size_t volume, int gamma_ab)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int block = BLOCK_SIZE;
    dim3 gridDim(grid, 1U, 1U);
    dim3 blockDim(block, 1, 1);

    Arguments args_h = {{}, propag_a, propag_b, volume, gamma_ab};
    for (int i = 0; i < Ns * Ns; ++i) { args_h.correl[i] = correl[i]; }
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(args, &args_h, sizeof(Arguments)));
    CUDA_ERROR_CHECK(cudaLaunchKernel(reinterpret_cast<void *>(meson_all_source_kernel), gridDim, blockDim, {}));

    return;
  }
} // namespace meson_all_source

namespace meson_all_sink
{
  void launch(void *correl[Ns * Ns], void *propag_a, void *propag_b, size_t volume, int gamma_dc)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int block = BLOCK_SIZE;
    dim3 gridDim(grid, 1U, 1U);
    dim3 blockDim(block, 1, 1);

    Arguments args_h = {{}, propag_a, propag_b, volume, gamma_dc};
    for (int i = 0; i < Ns * Ns; ++i) { args_h.correl[i] = correl[i]; }
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(args, &args_h, sizeof(Arguments)));
    CUDA_ERROR_CHECK(cudaLaunchKernel(reinterpret_cast<void *>(meson_all_sink_kernel), gridDim, blockDim, {}));

    return;
  }
} // namespace meson_all_sink
