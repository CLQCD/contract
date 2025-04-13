#pragma once

#include "contract_define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  IK_JL_MN,
  IK_JN_ML,
  IL_JK_MN,
  IL_JN_MK,
  IN_JK_ML,
  IN_JL_MK,
} BaryonContractType;

void init(int device);
void baryon_two_point(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                      unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn);
void proton(void *correl, void *propag_i, void *propag_j, void *propag_m, int contract_type, unsigned long volume,
            int gamma_ij, int gamma_kl, int gamma_mn);

#ifdef __cplusplus
}
#endif
