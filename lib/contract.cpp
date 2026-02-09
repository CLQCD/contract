#include <contract.h>
#include <baryon.h>
#include <meson.h>
#include <runtime_api.h>

void meson_two_point(void *correl, void *propag_i, void *propag_j, size_t volume, int gamma_ij, int gamma_kl)
{
  meson::launch(correl, propag_i, propag_j, volume, gamma_ij, gamma_kl);
  return;
}

void meson_all_source_two_point(void **correl, void *propag_i, void *propag_j, size_t volume, int gamma_ij)
{
  meson_all_source::launch(correl, propag_i, propag_j, volume, gamma_ij);
  return;
}

void meson_all_sink_two_point(void **correl, void *propag_i, void *propag_j, size_t volume, int gamma_kl)
{
  meson_all_sink::launch(correl, propag_i, propag_j, volume, gamma_kl);
  return;
}

void baryon_diquark(void *diquark, void *propag_i, void *propag_j, size_t volume, int gamma_ij, int gamma_kl)
{
  diquark::launch(diquark, propag_i, propag_j, volume, gamma_ij, gamma_kl);
  return;
}

void baryon_two_point(void *correl, void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
                      size_t volume, int gamma_ij, int gamma_kl, int gamma_mn)
{
  baryon::launch(correl, propag_i, propag_j, propag_n, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
  return;
}

void baryon_general_two_point(void *correl, void *propag_i, void *propag_j, void *propag_n,
                              BaryonContractType contract_type, size_t volume, int gamma_ij, int gamma_kl,
                              double _Complex *project_mn)
{
  std::complex<double> project_mn_[16];
  for (int ij = 0; ij < 16; ++ij) { project_mn_[ij] = project_mn[ij]; }
  baryon_general::launch(correl, propag_i, propag_j, propag_n, contract_type, volume, gamma_ij, gamma_kl, project_mn_);
  return;
}

void baryon_sequential_two_point(void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
                                 BaryonSequentialType sequential_type, size_t volume, int gamma_ij, int gamma_kl,
                                 int gamma_mn)
{
  switch (sequential_type) {
  case SEQUENTIAL_I:
    baryon_sequential_i::launch(propag_i, propag_j, propag_n, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
    break;
  case SEQUENTIAL_J:
    baryon_sequential_j::launch(propag_i, propag_j, propag_n, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
    break;
  case SEQUENTIAL_N:
    baryon_sequential_n::launch(propag_i, propag_j, propag_n, contract_type, volume, gamma_ij, gamma_kl, gamma_mn);
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
  target_set_device(device);
  return;
}