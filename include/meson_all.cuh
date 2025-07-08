#pragma once

#include <contract.h>
#include <gamma.cuh>

const unsigned int Ns = 4;
const unsigned int Nc = 3;
const unsigned int BLOCK_SIZE = 64;
const unsigned int TILE_SIZE = BLOCK_SIZE / (Ns * Ns);

struct Arguments {
  void *correl[Ns * Ns];
  void *propag_a;
  void *propag_b;
  int gamma;
};

__constant__ Arguments args {};

__global__ void meson_all_source_kernel()
{
  const size_t x_block = blockIdx.x * TILE_SIZE;
  const int thread_id = threadIdx.x;
  const int idx0 = threadIdx.x / (Ns * Ns);
  const int idx1 = threadIdx.x % (Ns * Ns);

  __shared__ Complex128 propag_a[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_b[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 correl[TILE_SIZE][Ns * Ns];
  correl[idx0][idx1] = 0;

  size_t offset = x_block * (Ns * Ns * Nc * Nc);
  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int AB = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
    propag_b[x][AB][ab] = conj(static_cast<Complex128 *>(args.propag_b)[offset + pos]);
  }
  __syncthreads();

  int AD = idx1;
  int A = AD / Ns;
  int D = AD % Ns;
  int B = gamma_index(args.gamma, A);
  Complex128 gamma_ab_data = gamma_gamma5_data<true>(args.gamma, A);
  for (int i = 0; i < Ns * Ns; ++i) {
    int C = gamma_index(i, D);
    Complex128 gamma_dc_data = gamma_gamma5_data<false>(i, D);
    int BC = B * Ns + C;
    Complex128 tmp = 0;
    for (int a = 0; a < Nc; ++a) {
      for (int b = 0; b < Nc; ++b) { tmp += propag_a[idx0][AD][a * Nc + b] * propag_b[idx0][BC][a * Nc + b]; }
    }
    correl[idx0][idx1] = gamma_ab_data * gamma_dc_data * tmp;
    __syncthreads();

    if (idx1 < 8) { correl[idx0][idx1] += correl[idx0][idx1 + 8]; }
    __syncthreads();
    if (idx1 < 4) { correl[idx0][idx1] += correl[idx0][idx1 + 4]; }
    __syncthreads();
    if (idx1 < 2) { correl[idx0][idx1] += correl[idx0][idx1 + 2]; }
    __syncthreads();
    if (idx1 < 1) {
      correl[idx0][idx1] += correl[idx0][idx1 + 1];
      static_cast<Complex128 *>(args.correl[i])[x_block + idx0] = correl[idx0][idx1];
    }
    __syncthreads();
  }

  return;
}

__global__ void meson_all_sink_kernel()
{
  // const size_t volume = args.volume;
  const size_t x_block = blockIdx.x * TILE_SIZE;
  const int thread_id = threadIdx.x;
  const int idx0 = threadIdx.x / (Ns * Ns);
  const int idx1 = threadIdx.x % (Ns * Ns);

  __shared__ Complex128 propag_a[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 propag_b[TILE_SIZE][Ns * Ns][Nc * Nc];
  __shared__ Complex128 correl[TILE_SIZE][Ns * Ns];
  correl[idx0][idx1] = 0;

  size_t offset = x_block * (Ns * Ns * Nc * Nc);
  for (int pos = thread_id; pos < TILE_SIZE * (Ns * Ns * Nc * Nc); pos += BLOCK_SIZE) {
    int x = pos / (Ns * Ns * Nc * Nc);
    int AB = pos / (Nc * Nc) % (Ns * Ns);
    int ab = pos % (Nc * Nc);
    propag_a[x][AB][ab] = static_cast<Complex128 *>(args.propag_a)[offset + pos];
    propag_b[x][AB][ab] = conj(static_cast<Complex128 *>(args.propag_b)[offset + pos]);
  }
  __syncthreads();

  int AD = idx1;
  int A = AD / Ns;
  int D = AD % Ns;
  int C = gamma_index(args.gamma, D);
  Complex128 gamma_dc_data = gamma_gamma5_data<false>(args.gamma, D);
  for (int i = 0; i < Ns * Ns; ++i) {
    int B = gamma_index(i, A);
    Complex128 gamma_ab_data = gamma_gamma5_data<true>(i, A);
    int BC = B * Ns + C;
    Complex128 tmp = 0;
    for (int a = 0; a < Nc; ++a) {
      for (int b = 0; b < Nc; ++b) { tmp += propag_a[idx0][AD][a * Nc + b] * propag_b[idx0][BC][a * Nc + b]; }
    }
    correl[idx0][idx1] = gamma_ab_data * gamma_dc_data * tmp;
    __syncthreads();

    if (idx1 < 8) { correl[idx0][idx1] += correl[idx0][idx1 + 8]; }
    __syncthreads();
    if (idx1 < 4) { correl[idx0][idx1] += correl[idx0][idx1 + 4]; }
    __syncthreads();
    if (idx1 < 2) { correl[idx0][idx1] += correl[idx0][idx1 + 2]; }
    __syncthreads();
    if (idx1 < 1) {
      correl[idx0][idx1] += correl[idx0][idx1 + 1];
      static_cast<Complex128 *>(args.correl[i])[x_block + idx0] = correl[idx0][idx1];
    }
    __syncthreads();
  }

  return;
}
