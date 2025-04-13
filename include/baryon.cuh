#pragma once

#include <cuda_runtime.h>
#include <cuda/std/complex>
#include "contract.h"

const int Ns = 4;
const int Nc = 3;
const int BLOCK_SIZE = 64;
const int TILE_SIZE = BLOCK_SIZE / (Ns * Ns);

typedef cuda::std::complex<double> Complex128;

struct Arguments {
  void *correl;
  void *propag_k;
  void *propag_l;
  void *propag_n;
  size_t volume;
  int gamma_ij;
  int gamma_kl;
  int gamma_mn;
};

__device__ Arguments args {nullptr, nullptr, nullptr, nullptr, 0, -1, -1, -1};

// __device__ const int gamma_csr_index[Ns * Ns][Ns] {
//   {0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}, {0, 1, 2, 3}, {2, 3, 0, 1}, {1, 0, 3, 2}, {1, 0, 3, 2}, {2, 3, 0, 1},
//   {2, 3, 0, 1}, {1, 0, 3, 2}, {1, 0, 3, 2}, {2, 3, 0, 1}, {0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}, {0, 1, 2, 3},
// };

// __device__ const Complex128 I(0.0, 1.0);
// __device__ const Complex128 gamma_csr_data[Ns * Ns][Ns] {
//   {1, 1, 1, 1},     {I, I, -I, -I}, {-1, 1, 1, -1},   {-I, I, -I, I}, {I, -I, -I, I}, {-1, 1, -1, 1},
//   {-I, -I, -I, -I}, {1, 1, -1, -1}, {1, 1, 1, 1},     {I, I, -I, -I}, {-1, 1, 1, -1}, {-I, I, -I, I},
//   {I, -I, -I, I},   {-1, 1, -1, 1}, {-I, -I, -I, -I}, {1, 1, -1, -1},
// };

__device__ __forceinline__ int gamma_index(int gamma, int row)
{
  switch (gamma) {
  case 0:
  case 3:
  case 12:
  case 15: return row; break;
  case 1:
  case 2:
  case 13:
  case 14: return 3 - row; break;
  case 4:
  case 7:
  case 8:
  case 11: return (row + 2) % 4; break;
  case 5:
  case 6:
  case 9:
  case 10: return (3 - row + 2) % 4; break;
  default: break;
  }
  return -1;
}

// // T stands for transpose, CSR for false and CSC for true
// template <bool T> __device__ __forceinline__ Complex128 gamma_data(int gamma, int row)
// {
//   if constexpr (T) {
//     if (gamma == 1 || gamma == 4 || gamma == 5 || gamma == 7 || gamma == 10 || gamma == 13) {
//       return -gamma_csr_data[gamma][row];
//     } else {
//       return gamma_csr_data[gamma][row];
//     }
//   } else {
//     return gamma_csr_data[gamma][row];
//   }
// }

// T stands for transpose, CSR for false and CSC for true
template <bool T> __device__ __forceinline__ const Complex128 gamma_data(int gamma, int row)
{
  const Complex128 I(0.0, 1.0);
  if constexpr (!T) {
    switch (gamma) {
    case 0: return 1; break;
    case 1: return (row == 0 || row == 1) ? I : -I; break;
    case 2: return (row == 0 || row == 3) ? -1 : 1; break;
    case 3: return (row == 0 || row == 2) ? -I : I; break;
    case 4: return (row == 0 || row == 3) ? I : -I; break;
    case 5: return (row == 0 || row == 2) ? -1 : 1; break;
    case 6: return -I; break;
    case 7: return (row == 0 || row == 1) ? 1 : -1; break;
    case 8: return 1; break;
    case 9: return (row == 0 || row == 1) ? I : -I; break;
    case 10: return (row == 0 || row == 3) ? -1 : 1; break;
    case 11: return (row == 0 || row == 2) ? -I : I; break;
    case 12: return (row == 0 || row == 3) ? I : -I; break;
    case 13: return (row == 0 || row == 2) ? -1 : 1; break;
    case 14: return -I; break;
    case 15: return (row == 0 || row == 1) ? 1 : -1; break;
    default: break;
    }
  } else {
    switch (gamma) {
    case 0: return 1; break;
    case 1: return (row == 0 || row == 1) ? -I : I; break;
    case 2: return (row == 0 || row == 3) ? -1 : 1; break;
    case 3: return (row == 0 || row == 2) ? -I : I; break;
    case 4: return (row == 0 || row == 3) ? -I : I; break;
    case 5: return (row == 0 || row == 2) ? 1 : -1; break;
    case 6: return -I; break;
    case 7: return (row == 0 || row == 1) ? -1 : 1; break;
    case 8: return 1; break;
    case 9: return (row == 0 || row == 1) ? I : -I; break;
    case 10: return (row == 0 || row == 3) ? 1 : -1; break;
    case 11: return (row == 0 || row == 2) ? -I : I; break;
    case 12: return (row == 0 || row == 3) ? I : -I; break;
    case 13: return (row == 0 || row == 2) ? 1 : -1; break;
    case 14: return -I; break;
    case 15: return (row == 0 || row == 1) ? 1 : -1; break;
    default: break;
    }
  }
  return 0;
}

template <int GAMMA> __device__ __forceinline__ int gamma_index(int row)
{
  switch (GAMMA) {
  case 0:
  case 3:
  case 12:
  case 15: return row; break;
  case 1:
  case 2:
  case 13:
  case 14: return 3 - row; break;
  case 4:
  case 7:
  case 8:
  case 11: return (row + 2) % 4; break;
  case 5:
  case 6:
  case 9:
  case 10: return (3 - row + 2) % 4; break;
  default: break;
  }
  return -1;
}

// // T stands for transpose, CSR for false and CSC for true
// template <int GAMMA, bool T> __device__ __forceinline__ const Complex128 gamma_data(int row)
// {
//   if constexpr (T && (GAMMA == 1 || GAMMA == 4 || GAMMA == 5 || GAMMA == 7 || GAMMA == 10 || GAMMA == 13)) {
//     return -gamma_csr_data[GAMMA][row];
//   } else {
//     return gamma_csr_data[GAMMA][row];
//   }
// }

// T stands for transpose, CSR for false and CSC for true
template <int GAMMA, bool T> __device__ __forceinline__ const Complex128 gamma_data(int row)
{
  const Complex128 I(0.0, 1.0);
  if constexpr (!T) {
    switch (GAMMA) {
    case 0: return 1; break;
    case 1: return (row == 0 || row == 1) ? I : -I; break;
    case 2: return (row == 0 || row == 3) ? -1 : 1; break;
    case 3: return (row == 0 || row == 2) ? -I : I; break;
    case 4: return (row == 0 || row == 3) ? I : -I; break;
    case 5: return (row == 0 || row == 2) ? -1 : 1; break;
    case 6: return -I; break;
    case 7: return (row == 0 || row == 1) ? 1 : -1; break;
    case 8: return 1; break;
    case 9: return (row == 0 || row == 1) ? I : -I; break;
    case 10: return (row == 0 || row == 3) ? -1 : 1; break;
    case 11: return (row == 0 || row == 2) ? -I : I; break;
    case 12: return (row == 0 || row == 3) ? I : -I; break;
    case 13: return (row == 0 || row == 2) ? -1 : 1; break;
    case 14: return -I; break;
    case 15: return (row == 0 || row == 1) ? 1 : -1; break;
    default: break;
    }
  } else {
    switch (GAMMA) {
    case 0: return 1; break;
    case 1: return (row == 0 || row == 1) ? -I : I; break;
    case 2: return (row == 0 || row == 3) ? -1 : 1; break;
    case 3: return (row == 0 || row == 2) ? -I : I; break;
    case 4: return (row == 0 || row == 3) ? -I : I; break;
    case 5: return (row == 0 || row == 2) ? 1 : -1; break;
    case 6: return -I; break;
    case 7: return (row == 0 || row == 1) ? -1 : 1; break;
    case 8: return 1; break;
    case 9: return (row == 0 || row == 1) ? I : -I; break;
    case 10: return (row == 0 || row == 3) ? 1 : -1; break;
    case 11: return (row == 0 || row == 2) ? -I : I; break;
    case 12: return (row == 0 || row == 3) ? I : -I; break;
    case 13: return (row == 0 || row == 2) ? 1 : -1; break;
    case 14: return -I; break;
    case 15: return (row == 0 || row == 1) ? 1 : -1; break;
    default: break;
    }
  }
  return 0;
}

template <BaryonContractType CONTRACT, int GAMMA> __global__ void baryon()
{
  // const size_t volume = args.volume;
  const size_t x_block = blockIdx.x * TILE_SIZE;
  const int thread_id = threadIdx.x;
  const int idx0 = threadIdx.x / (Ns * Ns);
  const int idx1 = threadIdx.x % (Ns * Ns);

  __shared__ Complex128 propag_k[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_l[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 correl[TILE_SIZE][Ns * Ns];
  correl[idx0][idx1] = 0;

  size_t offset = x_block * (Ns * Ns * Nc * Nc);
  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int ij = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    propag_k[x][ij][ab] = static_cast<Complex128 *>(args.propag_k)[offset + pos];
    propag_l[x][ij][ab] = static_cast<Complex128 *>(args.propag_l)[offset + pos];
    propag_n[x][ij][ab] = static_cast<Complex128 *>(args.propag_n)[offset + pos];
  }
  __syncthreads();

  if constexpr (CONTRACT == IK_JN_ML || CONTRACT == IN_JK_ML || CONTRACT == IL_JN_MK || CONTRACT == IN_JL_MK) {
    // ik -> il, ml -> ml, jn -> in, k -> l
    // jk -> jl, ml -> ml, in -> jn, k -> l
    // mk -> mk, il -> il, jn -> in, k -> l
    // mk -> mk, jl -> jl, in -> jn, k -> l
    const int im = idx1;
    const int i = im / Ns;
    const int m = im % Ns;
    const int j = gamma_index(args.gamma_ij, i);
    const Complex128 gamma_ij_data = gamma_data<(CONTRACT == IN_JK_ML || CONTRACT == IN_JL_MK)>(args.gamma_ij, i);
    const int n = gamma_index(args.gamma_mn, m);
    const Complex128 gamma_mn_data = gamma_data<false>(args.gamma_mn, m);
    const int jn = j * Ns + n;
    for (int c = 0; c < Nc; ++c) {
      const int a = (c + 1) % Nc, b = (c + 2) % Nc;
      for (int f = 0; f < Nc; ++f) {
        const int d = (f + 1) % Nc, e = (f + 2) % Nc;
        Complex128 tmp = 0;
        for (int l = 0; l < Ns; ++l) {
          int k = gamma_index<GAMMA>(l);
          if constexpr (CONTRACT == IK_JN_ML || CONTRACT == IN_JK_ML) {
            int ik = i * Ns + k;
            int ml = m * Ns + l;
            tmp += gamma_data<GAMMA, true>(l)
              * (propag_k[idx0][ik][a * Nc + d] * propag_l[idx0][ml][b * Nc + e]
                 - propag_k[idx0][ik][b * Nc + d] * propag_l[idx0][ml][a * Nc + e]
                 - propag_k[idx0][ik][a * Nc + e] * propag_l[idx0][ml][b * Nc + d]
                 + propag_k[idx0][ik][b * Nc + e] * propag_l[idx0][ml][a * Nc + d]);
          } else if constexpr (CONTRACT == IL_JN_MK || CONTRACT == IN_JL_MK) {
            int mk = m * Ns + k;
            int il = i * Ns + l;
            tmp += gamma_data<GAMMA, true>(l)
              * (propag_k[idx0][mk][a * Nc + d] * propag_l[idx0][il][b * Nc + e]
                 - propag_k[idx0][mk][b * Nc + d] * propag_l[idx0][il][a * Nc + e]
                 - propag_k[idx0][mk][a * Nc + e] * propag_l[idx0][il][b * Nc + d]
                 + propag_k[idx0][mk][b * Nc + e] * propag_l[idx0][il][a * Nc + d]);
          }
        }
        correl[idx0][idx1] += gamma_ij_data * gamma_mn_data * tmp * propag_n[idx0][jn][c * Nc + f];
      }
    }
  } else if constexpr (CONTRACT == IK_JL_MN || CONTRACT == IL_JK_MN) {
    // ik -> ik, jl -> il, mn -> mn, j -> i
    // jk -> jk, il -> jl, mn -> mn, j -> i
    const int il = idx1;
    const int i = il / Ns;
    const int l = il % Ns;
    const int j = gamma_index(args.gamma_ij, i);
    const Complex128 gamma_ij_data = gamma_data<(CONTRACT == IL_JK_MN)>(args.gamma_ij, i);
    const int k = gamma_index(args.gamma_kl, l);
    const Complex128 gamma_kl_data = gamma_data<true>(args.gamma_kl, l);
    const int ik = i * Ns + k;
    const int jl = j * Ns + l;
    for (int c = 0; c < Nc; ++c) {
      const int a = (c + 1) % Nc, b = (c + 2) % Nc;
      for (int f = 0; f < Nc; ++f) {
        const int d = (f + 1) % Nc, e = (f + 2) % Nc;
        Complex128 tmp = 0;
        for (int m = 0; m < Ns; ++m) {
          int n = gamma_index<GAMMA>(m);
          int mn = m * Ns + n;
          tmp += gamma_data<GAMMA, false>(m) * propag_n[idx0][mn][c * Nc + f];
        }
        correl[idx0][idx1] += gamma_ij_data * gamma_kl_data * tmp
          * (propag_k[idx0][ik][a * Nc + d] * propag_l[idx0][jl][b * Nc + e]
             - propag_k[idx0][ik][b * Nc + d] * propag_l[idx0][jl][a * Nc + e]
             - propag_k[idx0][ik][a * Nc + e] * propag_l[idx0][jl][b * Nc + d]
             + propag_k[idx0][ik][b * Nc + e] * propag_l[idx0][jl][a * Nc + d]);
      }
    }
  }
  __syncthreads();

  offset = x_block + idx0;
  if (idx1 < 8) { correl[idx0][idx1] += correl[idx0][idx1 + 8]; }
  __syncthreads();
  if (idx1 < 4) { correl[idx0][idx1] += correl[idx0][idx1 + 4]; }
  __syncthreads();
  if (idx1 < 2) { correl[idx0][idx1] += correl[idx0][idx1 + 2]; }
  __syncthreads();
  if (idx1 < 1) {
    correl[idx0][idx1] += correl[idx0][idx1 + 1];
    static_cast<Complex128 *>(args.correl)[offset] = correl[idx0][idx1];
  }
  __syncthreads();
}

template <BaryonContractType CONTRACT, int GAMMA> void *instantiate()
{
  return reinterpret_cast<void *>(baryon<CONTRACT, GAMMA>);
}
template <BaryonContractType CONTRACT> void *instantiate(int gamma)
{
  switch (gamma) {
  case 0: return instantiate<CONTRACT, 0>(); break;
  case 1: return instantiate<CONTRACT, 1>(); break;
  case 2: return instantiate<CONTRACT, 2>(); break;
  case 3: return instantiate<CONTRACT, 3>(); break;
  case 4: return instantiate<CONTRACT, 4>(); break;
  case 5: return instantiate<CONTRACT, 5>(); break;
  case 6: return instantiate<CONTRACT, 6>(); break;
  case 7: return instantiate<CONTRACT, 7>(); break;
  case 8: return instantiate<CONTRACT, 8>(); break;
  case 9: return instantiate<CONTRACT, 9>(); break;
  case 10: return instantiate<CONTRACT, 10>(); break;
  case 11: return instantiate<CONTRACT, 11>(); break;
  case 12: return instantiate<CONTRACT, 12>(); break;
  case 13: return instantiate<CONTRACT, 13>(); break;
  case 14: return instantiate<CONTRACT, 14>(); break;
  case 15: return instantiate<CONTRACT, 15>(); break;
  default:
    fprintf(stderr, "Error: Invalid gamma value %d\n", gamma);
    exit(-1);
    break;
  }
  return nullptr;
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
