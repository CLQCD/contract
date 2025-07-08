#include <baryon_sequential.cuh>
#include <baryon.h>

namespace baryon_sequential
{
  template <BaryonContractType CONTRACT, int GAMMA_FC> void *instantiate()
  {
    return reinterpret_cast<void *>(baryon_sequential_b_kernel<CONTRACT, GAMMA_FC>);
  }
  template <BaryonContractType CONTRACT> void *instantiate(int gamma_fc)
  {
    switch (gamma_fc) {
    case 0: return instantiate<CONTRACT, 0>(); break;
    case 1: return instantiate<CONTRACT, 1>(); break;
    case 2: return instantiate<CONTRACT, 2>(); break;
    case 3: return instantiate<CONTRACT, 3>(); break;
    case 4: return instantiate<CONTRACT, 4>(); break;
    case 5: return instantiate<CONTRACT, 5>(); break;
    case 6: return instantiate<CONTRACT, 6>(); break;
    case 7: return instantiate<CONTRACT, 7>(); break;
    case 8: return instantiate<CONTRACT, 8>(); break;
    case 9: return instantiate<CONTRACT, 9>(); break;
    case 10: return instantiate<CONTRACT, 10>(); break;
    case 11: return instantiate<CONTRACT, 11>(); break;
    case 12: return instantiate<CONTRACT, 12>(); break;
    case 13: return instantiate<CONTRACT, 13>(); break;
    case 14: return instantiate<CONTRACT, 14>(); break;
    case 15: return instantiate<CONTRACT, 15>(); break;
    default:
      fprintf(stderr, "Error: Invalid gamma_fc value %d\n", gamma_fc);
      exit(-1);
      break;
    }
    return nullptr;
  }
  template <BaryonContractType CONTRACT, BaryonSequentialType PRESERVE>
  void launch(void *correl, void *propag_a, void *propag_b, void *propag_c, size_t volume, int gamma_ab, int gamma_de,
              int gamma_fc)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int block = BLOCK_SIZE;
    dim3 gridDim(grid, 1U, 1U);
    dim3 blockDim(block, 1, 1);

    Arguments args_h = {correl, propag_a, propag_b, propag_c, gamma_ab, gamma_de};
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(args, &args_h, sizeof(Arguments)));
    CUDA_ERROR_CHECK(cudaLaunchKernel(instantiate<CONTRACT>(gamma_fc), gridDim, blockDim, {}));

    return;
  }
  template void launch<AD_BE_CF, PRESERVE_B>(void *correl, void *propag_a, void *propag_b, void *propag_c,
                                             size_t volume, int gamma_ab, int gamma_de, int gamma_fc);
  template void launch<AD_BF_CE, PRESERVE_B>(void *correl, void *propag_a, void *propag_b, void *propag_c,
                                             size_t volume, int gamma_ab, int gamma_de, int gamma_fc);
  template void launch<AE_BD_CF, PRESERVE_B>(void *correl, void *propag_a, void *propag_b, void *propag_c,
                                             size_t volume, int gamma_ab, int gamma_de, int gamma_fc);
  template void launch<AE_BF_CD, PRESERVE_B>(void *correl, void *propag_a, void *propag_b, void *propag_c,
                                             size_t volume, int gamma_ab, int gamma_de, int gamma_fc);
  template void launch<AF_BD_CE, PRESERVE_B>(void *correl, void *propag_a, void *propag_b, void *propag_c,
                                             size_t volume, int gamma_ab, int gamma_de, int gamma_fc);
  template void launch<AF_BE_CD, PRESERVE_B>(void *correl, void *propag_a, void *propag_b, void *propag_c,
                                             size_t volume, int gamma_ab, int gamma_de, int gamma_fc);
} // namespace baryon_sequential