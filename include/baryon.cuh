#pragma once

#include <contract.h>
#include <gamma.cuh>

const int Ns = 4;
const int Nc = 3;
const int BLOCK_SIZE = 64;
const int TILE_SIZE = BLOCK_SIZE / (Ns * Ns);

struct Arguments {
  void *correl;
  void *propag_a;
  void *propag_b;
  void *propag_c;
  size_t volume;
  int gamma_ab;
  int gamma_de;
};

__constant__ Arguments args {};

template <BaryonContractType CONTRACT, int GAMMA_FC> __global__ void baryon_kernel()
{
  // const size_t volume = args.volume;
  const size_t x_block = blockIdx.x * TILE_SIZE;
  const int thread_id = threadIdx.x;
  const int idx0 = threadIdx.x / (Ns * Ns);
  const int idx1 = threadIdx.x % (Ns * Ns);

  __shared__ Complex128 propag_a[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_b[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_c[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 correl[TILE_SIZE][Ns * Ns];
  correl[idx0][idx1] = 0;

  constexpr bool SWAP_AB = (CONTRACT == AF_BD_CE || CONTRACT == AF_BE_CD);
  constexpr bool SWAP_DE = (CONTRACT == AE_BD_CF || CONTRACT == AE_BF_CD || CONTRACT == AF_BE_CD);
  constexpr BaryonContractType MODE = (CONTRACT == AD_BE_CF || CONTRACT == AE_BD_CF) ? AD_BE_CF : AD_BF_CE;

  size_t offset = x_block * (Ns * Ns * Nc * Nc);
  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int AB = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    if constexpr (SWAP_AB) {
      propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_b)[offset + pos];
      propag_b[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
      propag_c[x][AB][ab] = static_cast<Complex128 *>(args.propag_c)[offset + pos];
    } else {
      propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
      propag_b[x][AB][ab] = static_cast<Complex128 *>(args.propag_b)[offset + pos];
      propag_c[x][AB][ab] = static_cast<Complex128 *>(args.propag_c)[offset + pos];
    }
  }
  __syncthreads();

  int AD = idx1;
  int A = AD / Ns;
  int D = AD % Ns;
  int B = gamma_index(args.gamma_ab, A);
  Complex128 gamma_ab_data = gamma_data<SWAP_AB>(args.gamma_ab, A);
  int E = gamma_index(args.gamma_de, D);
  Complex128 gamma_de_data = gamma_data<SWAP_DE>(args.gamma_de, D);
  if constexpr (MODE == AD_BE_CF) {
    int BE = B * Ns + E;
    for (int c = 0; c < Nc; ++c) {
      int a = (c + 1) % Nc, b = (c + 2) % Nc;
      for (int f = 0; f < Nc; ++f) {
        int d = (f + 1) % Nc, e = (f + 2) % Nc;
        Complex128 tmp = 0;
        for (int C = 0; C < Ns; ++C) {
          int F = gamma_index<GAMMA_FC>(C);
          Complex128 gamma_fc_data = gamma_data<GAMMA_FC, true>(C);
          int CF = C * Ns + F;
          tmp += gamma_fc_data * propag_c[idx0][CF][c * Nc + f];
        }
        correl[idx0][idx1] += gamma_ab_data * gamma_de_data * tmp
          * (propag_a[idx0][AD][a * Nc + d] * propag_b[idx0][BE][b * Nc + e]
             - propag_a[idx0][AD][b * Nc + d] * propag_b[idx0][BE][a * Nc + e]
             - propag_a[idx0][AD][a * Nc + e] * propag_b[idx0][BE][b * Nc + d]
             + propag_a[idx0][AD][b * Nc + e] * propag_b[idx0][BE][a * Nc + d]);
      }
    }
  } else if constexpr (MODE == AD_BF_CE) {
    for (int a = 0; a < Nc; ++a) {
      int b = (a + 1) % Nc, c = (a + 2) % Nc;
      for (int d = 0; d < Nc; ++d) {
        int e = (d + 1) % Nc, f = (d + 2) % Nc;
        Complex128 tmp = 0;
        for (int C = 0; C < Ns; ++C) {
          int F = gamma_index<GAMMA_FC>(C);
          Complex128 gamma_fc_data = gamma_data<GAMMA_FC, true>(C);
          int BF = B * Ns + F;
          int CE = C * Ns + E;
          tmp += gamma_fc_data
            * (propag_b[idx0][BF][b * Nc + e] * propag_c[idx0][CE][c * Nc + f]
               - propag_b[idx0][BF][c * Nc + e] * propag_c[idx0][CE][b * Nc + f]
               - propag_b[idx0][BF][b * Nc + f] * propag_c[idx0][CE][c * Nc + e]
               + propag_b[idx0][BF][c * Nc + f] * propag_c[idx0][CE][b * Nc + e]);
        }
        correl[idx0][idx1] += gamma_ab_data * gamma_de_data * tmp * propag_a[idx0][AD][a * Nc + d];
      }
    }
  }
  __syncthreads();

  if (idx1 < 8) { correl[idx0][idx1] += correl[idx0][idx1 + 8]; }
  __syncthreads();
  if (idx1 < 4) { correl[idx0][idx1] += correl[idx0][idx1 + 4]; }
  __syncthreads();
  if (idx1 < 2) { correl[idx0][idx1] += correl[idx0][idx1 + 2]; }
  __syncthreads();
  if (idx1 < 1) {
    correl[idx0][idx1] += correl[idx0][idx1 + 1];
    static_cast<Complex128 *>(args.correl)[x_block + idx0] = correl[idx0][idx1];
  }
  __syncthreads();

  return;
}
