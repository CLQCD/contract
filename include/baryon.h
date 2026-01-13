#pragma once

#include <stdlib.h>

namespace baryon
{
  void launch(void *correl, void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
              size_t volume, int gamma_ij, int gamma_kl, int gamma_mn);
}

namespace baryon_sequential
{
  template <BaryonContractType CONTRACT, BaryonSequentialType PRESERVE>
  void launch(void *correl, void *propag_a, void *propag_b, void *propag_c, size_t volume, int gamma_ab, int gamma_de,
              int gamma_fc);
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
