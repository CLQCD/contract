#include <baryon_general.cuh>
#include <baryon.h>

namespace baryon_general
{

  template <BaryonContractType CONTRACT>
  void launch(void *correl, void *propag_i, void *propag_j, void *propag_n, size_t volume, int gamma_ij, int gamma_kl,
              std::complex<double> *project_mn)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    using Args = contract::BaryonGeneralArgs<double, CONTRACT>;
    using Kernel = contract::BaryonGeneralKernel<Args>;
    Args args(correl, propag_i, propag_j, propag_n, gamma_ij, gamma_kl, project_mn);
    contract::launch_kernel<Kernel>(args, volume);
    return;
  }

  void launch(void *correl, void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
              size_t volume, int gamma_ij, int gamma_kl, std::complex<double> *project_mn)
  {
    switch (contract_type) {
    case IK_JL_NM:
      launch<IK_JL_NM>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, project_mn);
      break;
    case IK_JM_NL:
      launch<IK_JM_NL>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, project_mn);
      break;
    case IL_JK_NM:
      launch<IL_JK_NM>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, project_mn);
      break;
    case IL_JM_NK:
      launch<IL_JM_NK>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, project_mn);
      break;
    case IM_JK_NL:
      launch<IM_JK_NL>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, project_mn);
      break;
    case IM_JL_NK:
      launch<IM_JL_NK>(correl, propag_i, propag_j, propag_n, volume, gamma_ij, gamma_kl, project_mn);
      break;
    default:
      fprintf(stderr, "Error: Invalid contract_type value %d\n", contract_type);
      exit(-1);
      break;
    }
    return;
  }

} // namespace baryon_general
