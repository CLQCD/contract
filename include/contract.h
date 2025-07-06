#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <contract_define.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  AD_BE_CF,
  AD_BF_CE,
  AE_BD_CF,
  AE_BF_CD,
  AF_BD_CE,
  AF_BE_CD,
  IK_JL_MN = AD_BE_CF,
  IK_JN_ML = AD_BF_CE,
  IL_JK_MN = AE_BD_CF,
  IL_JN_MK = AE_BF_CD,
  IN_JK_ML = AF_BD_CE,
  IN_JL_MK = AF_BE_CD,
} BaryonContractType;

void init(int device);
void meson_two_point(void *correl, void *propag_a, void *propag_b, unsigned long volume, int gamma_ab, int gamma_dc);
void meson_all_source_two_point(void **correl, void *propag_a, void *propag_b, unsigned long volume, int gamma_ab);
void meson_all_sink_two_point(void **correl, void *propag_a, void *propag_b, unsigned long volume, int gamma_dc);
void baryon_two_point(void *correl, void *propag_a, void *propag_b, void *propag_c, BaryonContractType contract_type,
                      unsigned long volume, int gamma_ab, int gamma_de, int gamma_fc);
void baryon_two_point_v2(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                         unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn);

#ifdef __cplusplus
}
#endif
