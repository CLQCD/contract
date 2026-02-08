#pragma once

#include <contract.h>
#include <gamma.cuh>

const int Ns = 4;
const int Nc = 3;
const int BLOCK_SIZE = 64;
const int TILE_SIZE = BLOCK_SIZE / (Ns * Ns);

struct Arguments {
  void *correl;
  void *propag_i;
  void *propag_j;
  void *propag_n;
  int gamma_ij;
  int gamma_kl;
};

__constant__ Arguments args {};

template <BaryonContractType CONTRACT, int GAMMA_MN> __global__ void baryon_kernel()
{
  const size_t x_block = blockIdx.x * TILE_SIZE;
  const int thread_id = threadIdx.x;
  const int idx0 = threadIdx.x / (Ns * Ns);
  const int idx1 = threadIdx.x % (Ns * Ns);

  __shared__ Complex128 propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 correl[TILE_SIZE][Ns * Ns];
  correl[idx0][idx1] = 0;

  constexpr bool SWAP_IJ = (CONTRACT == IM_JK_NL || CONTRACT == IM_JL_NK);
  constexpr bool SWAP_KL = (CONTRACT == IL_JK_NM || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK);

  size_t offset = x_block * (Ns * Ns * Nc * Nc);
  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int ij = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    if constexpr (SWAP_IJ) {
      propag_i[x][ij][ab] = static_cast<Complex128 *>(args.propag_j)[offset + pos];
      propag_j[x][ij][ab] = static_cast<Complex128 *>(args.propag_i)[offset + pos];
    } else {
      propag_i[x][ij][ab] = static_cast<Complex128 *>(args.propag_i)[offset + pos];
      propag_j[x][ij][ab] = static_cast<Complex128 *>(args.propag_j)[offset + pos];
    }
    propag_n[x][ij][ab] = static_cast<Complex128 *>(args.propag_n)[offset + pos];
  }
  __syncthreads();

  int il = idx1;
  int i = il / Ns;
  int l = il % Ns;
  int j = gamma_index(args.gamma_ij, i);
  Complex128 gamma_ij_data = gamma_data<SWAP_IJ, double>(args.gamma_ij, i);
  int k = gamma_index(args.gamma_kl, l);
  Complex128 gamma_kl_data = gamma_data<!SWAP_KL, double>(args.gamma_kl, l);
  int ik = i * Ns + k;
  int jl = j * Ns + l;
  if constexpr (CONTRACT == IK_JL_NM || CONTRACT == IL_JK_NM) {
    // ik @ kl, ij @ jl, mn @ nm
    for (int c = 0; c < Nc; ++c) {
      int a = (c + 1) % Nc, b = (c + 2) % Nc;
      for (int f = 0; f < Nc; ++f) {
        int d = (f + 1) % Nc, e = (f + 2) % Nc;
        Complex128 tmp = 0;
        for (int n = 0; n < Ns; ++n) {
          int m = gamma_index<GAMMA_MN>(n);
          int nm = n * Ns + m;
          tmp += gamma_data<GAMMA_MN, true, double>(n) * propag_n[idx0][nm][c * Nc + f];
        }
        correl[idx0][idx1] += gamma_ij_data * gamma_kl_data * tmp
          * (propag_i[idx0][ik][a * Nc + d] * propag_j[idx0][jl][b * Nc + e]
             - propag_i[idx0][ik][b * Nc + d] * propag_j[idx0][jl][a * Nc + e]
             - propag_i[idx0][ik][a * Nc + e] * propag_j[idx0][jl][b * Nc + d]
             + propag_i[idx0][ik][b * Nc + e] * propag_j[idx0][jl][a * Nc + d]);
      }
    }
  } else if constexpr (CONTRACT == IK_JM_NL || CONTRACT == IM_JK_NL || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK) {
    // ik @ kl, ij @ jm, mn @ nl
    for (int a = 0; a < Nc; ++a) {
      int b = (a + 1) % Nc, c = (a + 2) % Nc;
      for (int d = 0; d < Nc; ++d) {
        int e = (d + 1) % Nc, f = (d + 2) % Nc;
        Complex128 tmp = 0;
        for (int n = 0; n < Ns; ++n) {
          int m = gamma_index<GAMMA_MN>(n);
          int nl = n * Ns + l;
          int jm = j * Ns + m;
          tmp += gamma_data<GAMMA_MN, true, double>(n)
            * (propag_n[idx0][nl][b * Nc + e] * propag_j[idx0][jm][c * Nc + f]
               - propag_n[idx0][nl][c * Nc + e] * propag_j[idx0][jm][b * Nc + f]
               - propag_n[idx0][nl][b * Nc + f] * propag_j[idx0][jm][c * Nc + e]
               + propag_n[idx0][nl][c * Nc + f] * propag_j[idx0][jm][b * Nc + e]);
        }
        correl[idx0][idx1] += gamma_ij_data * gamma_kl_data * tmp * propag_i[idx0][ik][a * Nc + d];
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
