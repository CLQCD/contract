#pragma once

#include <load_store.cuh>
#include <contract.h>
#include <gamma.cuh>

constexpr int Ns = 4;
constexpr int Nc = 3;
constexpr int BLOCK_SIZE = 64;
constexpr int LANE_SIZE = Ns * Ns;
constexpr int TILE_SIZE = BLOCK_SIZE / LANE_SIZE;

struct Arguments {
  void *correl;
  void *propag_i;
  void *propag_j;
  void *propag_n;
  int gamma_ij;
  int gamma_kl;
};

__constant__ Arguments args {};

template <BaryonContractType CONTRACT, int GAMMA_MN, typename T>
__inline__ __device__ void baryon_local(T correl[Ns * Ns], const T propag_i[Ns * Ns][Nc * Nc],
                                        const T propag_j[Ns * Ns][Nc * Nc], const T propag_n[Ns * Ns][Nc * Nc], int idx)
{
  constexpr bool SWAP_IJ = (CONTRACT == IM_JK_NL || CONTRACT == IM_JL_NK);
  constexpr bool SWAP_KL = (CONTRACT == IL_JK_NM || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK);

  int il = idx;
  int i = il / Ns;
  int l = il % Ns;
  int j = gamma_index(args.gamma_ij, i);
  Complex128 gamma_ij_data = gamma_data<SWAP_IJ>(args.gamma_ij, i);
  int k = gamma_index(args.gamma_kl, l);
  Complex128 gamma_kl_data = gamma_data<!SWAP_KL>(args.gamma_kl, l);
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
          tmp += gamma_data<GAMMA_MN, true>(n) * propag_n[nm][c * Nc + f];
        }
        correl[idx] += gamma_ij_data * gamma_kl_data * tmp
          * (propag_i[ik][a * Nc + d] * propag_j[jl][b * Nc + e] - propag_i[ik][b * Nc + d] * propag_j[jl][a * Nc + e]
             - propag_i[ik][a * Nc + e] * propag_j[jl][b * Nc + d] + propag_i[ik][b * Nc + e] * propag_j[jl][a * Nc + d]);
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
          tmp += gamma_data<GAMMA_MN, true>(n)
            * (propag_n[nl][b * Nc + e] * propag_j[jm][c * Nc + f] - propag_n[nl][c * Nc + e] * propag_j[jm][b * Nc + f]
               - propag_n[nl][b * Nc + f] * propag_j[jm][c * Nc + e]
               + propag_n[nl][c * Nc + f] * propag_j[jm][b * Nc + e]);
        }
        correl[idx] += gamma_ij_data * gamma_kl_data * tmp * propag_i[ik][a * Nc + d];
      }
    }
  }
  __syncthreads();
}

template <BaryonContractType CONTRACT, int GAMMA_MN>
__device__ void baryon_kernel_v3(size_t x_offset, int x_idx, int idx)
{
  __shared__ Complex128 propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 correl[TILE_SIZE][Ns * Ns];
  correl[x_idx][idx] = 0;

  constexpr bool SWAP_IJ = (CONTRACT == IM_JK_NL || CONTRACT == IM_JL_NK);

  if constexpr (SWAP_IJ) {
    load_vector<Ns * Ns, Nc * Nc, BLOCK_SIZE, 3>(std::array {propag_i, propag_j, propag_n},
                                                 std::array {args.propag_j, args.propag_i, args.propag_n}, x_offset,
                                                 x_idx, idx);
  } else {
    load_vector<Ns * Ns, Nc * Nc, BLOCK_SIZE, 3>(std::array {propag_i, propag_j, propag_n},
                                                 std::array {args.propag_i, args.propag_j, args.propag_n}, x_offset,
                                                 x_idx, idx);
  }

  baryon_local<CONTRACT, GAMMA_MN>(correl[x_idx], propag_i[x_idx], propag_j[x_idx], propag_n[x_idx], idx);

  store_tile<Ns * Ns>(args.correl, correl, x_offset, x_idx, idx);
}

template <BaryonContractType CONTRACT, int GAMMA_MN> __global__ void baryon_kernel()
{
  const size_t x_offset = blockIdx.x * TILE_SIZE;
  const int x_idx = threadIdx.x / (Ns * Ns);
  const int idx = threadIdx.x % (Ns * Ns);

  baryon_kernel_v3<CONTRACT, GAMMA_MN>(x_offset, x_idx, idx);
}
