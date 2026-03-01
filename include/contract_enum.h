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
