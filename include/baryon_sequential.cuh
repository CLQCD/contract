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
  int gamma_ab;
  int gamma_de;
};

__constant__ Arguments args {};

template <BaryonContractType CONTRACT, int GAMMA_FC> __global__ void baryon_sequential_a_kernel()
{
  const size_t x_block = blockIdx.x * TILE_SIZE;
  const int thread_id = threadIdx.x;
  const int idx0 = threadIdx.x / (Ns * Ns);
  const int idx1 = threadIdx.x % (Ns * Ns);

  __shared__ Complex128 propag_a[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_b[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_c[TILE_SIZE][Ns * Ns][Nc * Nc];

  constexpr bool SWAP_AB = (CONTRACT == AF_BD_CE || CONTRACT == AF_BE_CD);
  constexpr bool SWAP_DE = (CONTRACT == AE_BD_CF || CONTRACT == AE_BF_CD || CONTRACT == AF_BE_CD);
  constexpr BaryonContractType MODE = (CONTRACT == AD_BE_CF || CONTRACT == AE_BD_CF) ? AD_BE_CF : AD_BF_CE;

  size_t offset = x_block * (Ns * Ns * Nc * Nc);
  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int AB = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    if constexpr (SWAP_AB) {
      // propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_b)[offset + pos];
      propag_b[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
      propag_c[x][AB][ab] = static_cast<Complex128 *>(args.propag_c)[offset + pos];
    } else {
      // propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
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
    for (int a = 0; a < Nc; ++a) {
      int b = (a + 1) % Nc, c = (a + 2) % Nc;
      for (int d = 0; d < Nc; ++d) {
        int e = (d + 1) % Nc, f = (d + 2) % Nc;
        Complex128 tmp = 0;
        for (int C = 0; C < Ns; ++C) {
          int F = gamma_index<GAMMA_FC>(C);
          Complex128 gamma_fc_data = gamma_data<GAMMA_FC, true>(C);
          int CF = C * Ns + F;
          tmp += gamma_fc_data
            * (propag_b[idx0][BE][b * Nc + e] * propag_c[idx0][CF][c * Nc + f]
               - propag_b[idx0][BE][c * Nc + e] * propag_c[idx0][CF][b * Nc + f]
               - propag_b[idx0][BE][b * Nc + f] * propag_c[idx0][CF][c * Nc + e]
               + propag_b[idx0][BE][c * Nc + f] * propag_c[idx0][CF][b * Nc + e]);
        }
        propag_a[idx0][AD][a * Nc + d] = gamma_ab_data * gamma_de_data * tmp;
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
        propag_a[idx0][AD][a * Nc + d] = gamma_ab_data * gamma_de_data * tmp;
      }
    }
  }
  __syncthreads();

  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int AB = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    if constexpr (SWAP_AB) {
      static_cast<Complex128 *>(args.propag_b)[offset + pos] = propag_a[x][AB][ab];
    } else {
      static_cast<Complex128 *>(args.propag_a)[offset + pos] = propag_a[x][AB][ab];
    }
  }
  __syncthreads();

  return;
}

template <BaryonContractType CONTRACT, int GAMMA_FC> __global__ void baryon_sequential_b_kernel()
{
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
      // propag_b[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
      propag_c[x][AB][ab] = static_cast<Complex128 *>(args.propag_c)[offset + pos];
    } else {
      propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
      // propag_b[x][AB][ab] = static_cast<Complex128 *>(args.propag_b)[offset + pos];
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
    for (int b = 0; b < Nc; ++b) {
      int c = (b + 1) % Nc, a = (b + 2) % Nc;
      for (int e = 0; e < Nc; ++e) {
        int f = (e + 1) % Nc, d = (e + 2) % Nc;
        Complex128 tmp = 0;
        for (int C = 0; C < Ns; ++C) {
          int F = gamma_index<GAMMA_FC>(C);
          Complex128 gamma_fc_data = gamma_data<GAMMA_FC, true>(C);
          int CF = C * Ns + F;
          tmp += gamma_fc_data
            * (propag_c[idx0][CF][c * Nc + f] * propag_a[idx0][AD][a * Nc + d]
               - propag_c[idx0][CF][a * Nc + f] * propag_a[idx0][AD][c * Nc + d]
               - propag_c[idx0][CF][c * Nc + d] * propag_a[idx0][AD][a * Nc + f]
               + propag_c[idx0][CF][a * Nc + d] * propag_a[idx0][AD][c * Nc + f]);
        }
        propag_b[idx0][BE][b * Nc + e] += gamma_ab_data * gamma_de_data * tmp;
      }
    }
  } else if constexpr (MODE == AD_BF_CE) {
    for (int b = 0; b < Nc; ++b) {
      int c = (b + 1) % Nc, a = (b + 2) % Nc;
      for (int e = 0; e < Nc; ++e) {
        int f = (e + 1) % Nc, d = (e + 2) % Nc;
        Complex128 tmp = 0;
        for (int C = 0; C < Ns; ++C) {
          int F = gamma_index<GAMMA_FC>(C);
          Complex128 gamma_fc_data = gamma_data<GAMMA_FC, true>(C);
          int BF = B * Ns + F;
          int CE = C * Ns + E;
          tmp = gamma_fc_data
            * (propag_c[idx0][CE][c * Nc + f] * propag_a[idx0][AD][a * Nc + d]
               - propag_c[idx0][CE][a * Nc + f] * propag_a[idx0][AD][c * Nc + d]
               - propag_c[idx0][CE][c * Nc + d] * propag_a[idx0][AD][a * Nc + f]
               + propag_c[idx0][CE][a * Nc + d] * propag_a[idx0][AD][c * Nc + f]);
          propag_b[idx0][BF][b * Nc + e] += gamma_ab_data * gamma_de_data * tmp;
        }
      }
    }
  }
  __syncthreads();

  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int AB = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    if constexpr (SWAP_AB) {
      static_cast<Complex128 *>(args.propag_a)[offset + pos] = propag_b[x][AB][ab];
    } else {
      static_cast<Complex128 *>(args.propag_b)[offset + pos] = propag_b[x][AB][ab];
    }
  }
  __syncthreads();

  return;
}

template <BaryonContractType CONTRACT, int GAMMA_FC> __global__ void baryon_sequential_c_kernel()
{
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
      // propag_c[x][AB][ab] = static_cast<Complex128 *>(args.propag_c)[offset + pos];
    } else {
      propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
      propag_b[x][AB][ab] = static_cast<Complex128 *>(args.propag_b)[offset + pos];
      // propag_c[x][AB][ab] = static_cast<Complex128 *>(args.propag_c)[offset + pos];
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
          tmp = gamma_fc_data
            * (propag_a[idx0][AD][a * Nc + d] * propag_b[idx0][BE][b * Nc + e]
               - propag_a[idx0][AD][b * Nc + d] * propag_b[idx0][BE][a * Nc + e]
               - propag_a[idx0][AD][a * Nc + e] * propag_b[idx0][BE][b * Nc + d]
               + propag_a[idx0][AD][b * Nc + e] * propag_b[idx0][BE][a * Nc + d]);
          propag_c[idx0][CF][c * Nc + f] += gamma_ab_data * gamma_de_data * tmp;
        }
      }
    }
  } else if constexpr (MODE == AD_BF_CE) {
    for (int c = 0; c < Nc; ++c) {
      int a = (c + 1) % Nc, b = (c + 2) % Nc;
      for (int f = 0; f < Nc; ++f) {
        int d = (f + 1) % Nc, e = (f + 2) % Nc;
        Complex128 tmp = 0;
        for (int C = 0; C < Ns; ++C) {
          int F = gamma_index<GAMMA_FC>(C);
          Complex128 gamma_fc_data = gamma_data<GAMMA_FC, true>(C);
          int BF = B * Ns + F;
          int CE = C * Ns + E;
          tmp = gamma_fc_data
            * (propag_a[idx0][AD][a * Nc + d] * propag_b[idx0][BF][b * Nc + e]
               - propag_a[idx0][AD][b * Nc + d] * propag_b[idx0][BF][a * Nc + e]
               - propag_a[idx0][AD][a * Nc + e] * propag_b[idx0][BF][b * Nc + d]
               + propag_a[idx0][AD][b * Nc + e] * propag_b[idx0][BF][a * Nc + d]);
          propag_c[idx0][CE][c * Nc + f] += gamma_ab_data * gamma_de_data * tmp;
        }
      }
    }
  }
  __syncthreads();

  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int AB = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    static_cast<Complex128 *>(args.propag_c)[offset + pos] = propag_c[x][AB][ab];
  }
  __syncthreads();

  return;
}
