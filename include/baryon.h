#pragma once

#include <cstdlib>
#include <complex>

namespace baryon
{
  void launch(void *correl, void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
              size_t volume, int gamma_ij, int gamma_kl, int gamma_mn);
}

namespace baryon_general
{
  void launch(void *correl, void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
              size_t volume, int gamma_ij, int gamma_kl, std::complex<double> *project_mn);
}

namespace baryon_sequential_i
{
  void launch(void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type, size_t volume,
              int gamma_ij, int gamma_kl, int gamma_mn);
}

namespace baryon_sequential_j
{
  void launch(void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type, size_t volume,
              int gamma_ij, int gamma_kl, int gamma_mn);
}

namespace baryon_sequential_n
{
  void launch(void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type, size_t volume,
              int gamma_ij, int gamma_kl, int gamma_mn);
}

namespace diquark
{
  void launch(void *diquark, void *propag_i, void *propag_j, size_t volume, int gamma_ij, int gamma_kl);
}

void baryon_ik_jl_mn(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn);
void baryon_ik_jn_ml(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn);
void baryon_il_jk_mn(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn);
void baryon_il_jn_mk(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn);
void baryon_in_jk_ml(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn);
void baryon_in_jl_mk(void *correl, void *propag_i, void *propag_j, void *propag_m, size_t volume, int gamma_ij,
                     int gamma_kl, int gamma_mn);
