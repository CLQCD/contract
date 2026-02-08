#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <contract_define.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  IK_JL_NM,
  IK_JM_NL,
  IL_JK_NM,
  IL_JM_NK,
  IM_JK_NL,
  IM_JL_NK,
  AD_BE_CF = IK_JL_NM,
  AD_BF_CE = IK_JM_NL,
  AE_BD_CF = IL_JK_NM,
  AE_BF_CD = IL_JM_NK,
  AF_BD_CE = IM_JK_NL,
  AF_BE_CD = IM_JL_NK,
  IK_JL_MN = IK_JL_NM,
  IK_JN_ML = IK_JM_NL,
  IL_JK_MN = IL_JK_NM,
  IL_JN_MK = IL_JM_NK,
  IN_JK_ML = IM_JK_NL,
  IN_JL_MK = IM_JL_NK,
} BaryonContractType;

typedef enum {
  SEQUENTIAL_I,
  SEQUENTIAL_J,
  SEQUENTIAL_N,
} BaryonSequentialType;

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
void baryon_two_point_v2(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                         unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn);

#ifdef __cplusplus
}
#endif
