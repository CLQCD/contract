#include <baryon.cuh>
#include <baryon.h>

namespace baryon
{

  template <BaryonContractType CONTRACT, int GAMMA_MN> void *instantiate()
  {
    return reinterpret_cast<void *>(baryon_kernel<CONTRACT, GAMMA_MN>);
  }

  template <BaryonContractType CONTRACT> void *instantiate(int gamma_mn)
  {
    switch (gamma_mn) {
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
      fprintf(stderr, "Error: Invalid gamma_mn value %d\n", gamma_mn);
      exit(-1);
      break;
    }
    return nullptr;
  }

  template <BaryonContractType CONTRACT>
  void launch(void *correl, void *propag_i, void *propag_j, void *propag_n, size_t volume, int gamma_ij, int gamma_kl,
              int gamma_mn)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int block = BLOCK_SIZE;
    dim3 gridDim(grid, 1U, 1U);
    dim3 blockDim(block, 1U, 1U);

    Arguments args_h = {correl, propag_i, propag_j, propag_n, gamma_ij, gamma_kl};
    CUDA_ERROR_CHECK(cudaMemcpyToSymbol(args, &args_h, sizeof(Arguments)));
    CUDA_ERROR_CHECK(cudaLaunchKernel(instantiate<CONTRACT>(gamma_mn), gridDim, blockDim, {}));
    return;
  }

  void launch(void *correl, void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
              size_t volume, int gamma_ij, int gamma_kl, int gamma_mn)
  {
    switch (contract_type) {
    case IK_JL_NM: launch<IK_JL_NM>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IK_JM_NL: launch<IK_JM_NL>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IL_JK_NM: launch<IL_JK_NM>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IL_JM_NK: launch<IL_JM_NK>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IM_JK_NL: launch<IM_JK_NL>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IM_JL_NK: launch<IM_JL_NK>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    default: break;
    }
    return;
  }

} // namespace baryon
