#pragma once

#include <kernel.cuh>
#include <contract.h>
#include <gamma.cuh>

const unsigned int Ns = 4;
const unsigned int Nc = 3;
const unsigned int BLOCK_SIZE = 64;
const unsigned int LANE_SIZE = Ns * Ns;
const unsigned int TILE_SIZE = BLOCK_SIZE / LANE_SIZE;

namespace contract
{

  template <typename F> struct MesonArgs {
    using T = Complex<F>;

    void *correl;
    void *propag_i;
    void *propag_j;
    int gamma_ij;
    int gamma_kl;

    MesonArgs(void *correl, void *propag_i, void *propag_j, int gamma_ij, int gamma_kl) :
      correl(correl), propag_i(propag_i), propag_j(propag_j), gamma_ij(gamma_ij), gamma_kl(gamma_kl)
    {
    }
  };

  template <typename F>
  __device__ __forceinline__ void meson_local(Complex<F> correl[Ns * Ns], const Complex<F> propag_i[Ns * Ns][Nc * Nc],
                                              const Complex<F> propag_j[Ns * Ns][Nc * Nc], int gamma_ij, int gamma_kl,
                                              int idx)
  {
    using T = Complex<F>;
    int il = idx;
    int i = il / Ns;
    int l = il % Ns;
    int j = gamma_index(gamma_ij, i);
    T gamma_ij_data = gamma_gamma5_data<true, F>(gamma_ij, i); // We actually need ji
    int k = gamma_index(gamma_kl, l);
    T gamma_kl_data = gamma_gamma5_data<true, F>(gamma_kl, l);
    int ik = i * Ns + k;
    int jl = j * Ns + l;
    T tmp = 0;
    for_a_d { tmp += propag_i[ik][ad] * conj(propag_j[jl][ad]); }
    correl[idx] = gamma_ij_data * gamma_kl_data * tmp;
  }

  template <typename Args> __device__ void meson_kernel(const Args &args, size_t x_offset)
  {
    __shared__ typename Args::T propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T correl[TILE_SIZE][Ns * Ns];

    int t_idx = threadIdx.x / LANE_SIZE;
    int l_idx = threadIdx.x % LANE_SIZE;

    load_vector<Ns * Ns, Nc * Nc>(propag_i[t_idx], args.propag_i, x_offset);
    load_vector<Ns * Ns, Nc * Nc>(propag_j[t_idx], args.propag_j, x_offset);
    __syncwarp(); // Seems better?

    meson_local(correl[t_idx], propag_i[t_idx], propag_j[t_idx], args.gamma_ij, args.gamma_kl, l_idx);
    __syncwarp();

    reduce_lane<Ns * Ns>(correl[t_idx]);
    store_tile<Ns * Ns>(args.correl, correl[t_idx], x_offset);
  }

  template <typename Args> struct MesonKernel : public BaseKernel<Args, BLOCK_SIZE, LANE_SIZE> {
    constexpr MesonKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, LANE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset) { meson_kernel(this->args, x_offset); }
  };

}; // namespace contract