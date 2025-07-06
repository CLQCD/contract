#include <cuda_runtime.h>
#include <contract.h>
#include <baryon.h>
#include <meson.h>

void meson_two_point(void *correl, void *propag_a, void *propag_b, size_t volume, int gamma_ab, int gamma_dc)
{
  meson::launch(correl, propag_a, propag_b, volume, gamma_ab, gamma_dc);
  return;
}

void meson_all_source_two_point(void **correl, void *propag_a, void *propag_b, size_t volume, int gamma_ab)
{
  meson_all_source::launch(correl, propag_a, propag_b, volume, gamma_ab);
  return;
}

void meson_all_sink_two_point(void **correl, void *propag_a, void *propag_b, size_t volume, int gamma_dc)
{
  meson_all_sink::launch(correl, propag_a, propag_b, volume, gamma_dc);
  return;
}

void baryon_two_point(void *correl, void *propag_a, void *propag_b, void *propag_c, BaryonContractType contract_type,
                      size_t volume, int gamma_ab, int gamma_de, int gamma_fc)
{
  switch (contract_type) {
  case AD_BE_CF:
    baryon::launch<AD_BE_CF>(correl, propag_a, propag_b, propag_c, volume, gamma_ab, gamma_de, gamma_fc);
    break;
  case AD_BF_CE:
    baryon::launch<AD_BF_CE>(correl, propag_a, propag_b, propag_c, volume, gamma_ab, gamma_de, gamma_fc);
    break;
  case AE_BD_CF:
    baryon::launch<AE_BD_CF>(correl, propag_a, propag_b, propag_c, volume, gamma_ab, gamma_de, gamma_fc);
    break;
  case AE_BF_CD:
    baryon::launch<AE_BF_CD>(correl, propag_a, propag_b, propag_c, volume, gamma_ab, gamma_de, gamma_fc);
    break;
  case AF_BD_CE:
    baryon::launch<AF_BD_CE>(correl, propag_a, propag_b, propag_c, volume, gamma_ab, gamma_de, gamma_fc);
    break;
  case AF_BE_CD:
    baryon::launch<AF_BE_CD>(correl, propag_a, propag_b, propag_c, volume, gamma_ab, gamma_de, gamma_fc);
    break;
  default: break;
  }
  return;
}

void baryon_two_point_v2(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                         size_t volume, int gamma_ij, int gamma_kl, int gamma_mn)
{
  switch (contract_type) {
  case IK_JL_MN: baryon_ik_jl_mn(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IK_JN_ML: baryon_ik_jn_ml(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IL_JK_MN: baryon_il_jk_mn(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IL_JN_MK: baryon_il_jn_mk(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IN_JK_ML: baryon_in_jk_ml(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  case IN_JL_MK: baryon_in_jl_mk(correl, propag_i, propag_j, propag_m, volume, gamma_ij, gamma_kl, gamma_mn); break;
  default: break;
  }
  return;
}

void init(int device)
{
  CUDA_ERROR_CHECK(cudaSetDevice(device));
  return;
}