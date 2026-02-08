#include <baryon_sequential.cuh>
#include <baryon.h>

namespace baryon_sequential_j
{

  template <BaryonContractType CONTRACT, int GAMMA_MN>
  void launch(void *propag_i, void *propag_j, void *propag_n, size_t volume, int gamma_ij, int gamma_kl)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    unsigned int grid = (volume * (Ns * Ns) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int block = BLOCK_SIZE;
    dim3 gridDim(grid, 1U, 1U);
    dim3 blockDim(block, 1, 1);

    using Args = contract::BaryonSequentialArgs<double, CONTRACT, GAMMA_MN>;
    using Kernel = contract::BaryonSequentialJKernel<Args>;
    Args args(propag_i, propag_j, propag_n, gamma_ij, gamma_kl);
    contract::launch_kernel<Kernel>(args, volume);

    return;
  }

  template <BaryonContractType CONTRACT>
  void launch(void *propag_i, void *propag_j, void *propag_n, size_t volume, int gamma_ij, int gamma_kl, int gamma_mn)
  {
    switch (gamma_mn) {
    case 0: launch<CONTRACT, 0>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 1: launch<CONTRACT, 1>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 2: launch<CONTRACT, 2>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 3: launch<CONTRACT, 3>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 4: launch<CONTRACT, 4>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 5: launch<CONTRACT, 5>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 6: launch<CONTRACT, 6>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 7: launch<CONTRACT, 7>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 8: launch<CONTRACT, 8>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 9: launch<CONTRACT, 9>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 10: launch<CONTRACT, 10>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 11: launch<CONTRACT, 11>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 12: launch<CONTRACT, 12>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 13: launch<CONTRACT, 13>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 14: launch<CONTRACT, 14>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    case 15: launch<CONTRACT, 15>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl); break;
    default:
      fprintf(stderr, "Error: Invalid gamma_mn value %d\n", gamma_mn);
      exit(-1);
      break;
    }
  }

  void launch(void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type, size_t volume,
              int gamma_ij, int gamma_kl, int gamma_mn)
  {
    switch (contract_type) {
    case IK_JL_NM: launch<IK_JL_NM>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IK_JM_NL: launch<IK_JM_NL>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IL_JK_NM: launch<IL_JK_NM>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IL_JM_NK: launch<IL_JM_NK>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IM_JK_NL: launch<IM_JK_NL>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    case IM_JL_NK: launch<IM_JL_NK>(propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, gamma_mn); break;
    default:
      fprintf(stderr, "Error: Invalid contract_type value %d\n", contract_type);
      exit(-1);
      break;
    }
  }

} // namespace baryon_sequential