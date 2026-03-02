#pragma once

#include <stdio.h>
#include <stdlib.h>
#ifndef __cplusplus
#include <complex.h>
#endif

#include <contract_enum.h>

#ifdef __cplusplus
extern "C" {
#endif

void init(int device);
void meson_two_point(void *correl, void *propag_i, void *propag_j, unsigned long volume, int gamma_ij, int gamma_kl);
void meson_all_source_two_point(void **correl, void *propag_i, void *propag_j, unsigned long volume, int gamma_ij);
void meson_all_sink_two_point(void **correl, void *propag_i, void *propag_j, unsigned long volume, int gamma_kl);
void baryon_diquark(void *diquark, void *propag_i, void *propag_j, size_t volume, int gamma_ij, int gamma_kl);
void baryon_two_point(void *correl, void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
                      unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn);
void baryon_general_two_point(void *correl, void *propag_i, void *propag_j, void *propag_n,
                              BaryonContractType contract_type, size_t volume, int gamma_ij, int gamma_kl,
                              double _Complex *project_mn);
void baryon_sequential_two_point(void *propag_i, void *propag_j, void *propag_n, BaryonContractType contract_type,
                                 BaryonSequentialType sequential_type, size_t volume, int gamma_ij, int gamma_kl,
                                 int gamma_mn);
#ifndef GPU_TARGET_SYCL
void baryon_two_point_v2(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                         unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn);
#endif

#ifdef __cplusplus
}
#endif
