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
