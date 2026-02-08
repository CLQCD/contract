#pragma once

#include <kernel.cuh>
#include <contract.h>
#include <gamma.cuh>

constexpr int Ns = 4;
constexpr int Nc = 3;
constexpr int BLOCK_SIZE = 64;
constexpr int LANE_SIZE = Ns * Ns;
constexpr int TILE_SIZE = BLOCK_SIZE / LANE_SIZE;

namespace contract
{
  template <int GAMMA_IJ_, typename F> struct DiquarkArgs {
    using T = thrust::complex<F>;
    static constexpr int GAMMA_IJ = GAMMA_IJ_;

    void *diquark;
    void *propag_i;
    void *propag_j;
    int gamma_ij;
    int gamma_kl;

    DiquarkArgs(void *diquark, void *propag_i, void *propag_j, int gamma_kl) :
      diquark(diquark), propag_i(propag_i), propag_j(propag_j), gamma_kl(gamma_kl)
    {
    }
  };

  template <int GAMMA_IJ, typename F>
  __device__ __forceinline__ void diquark_local(Complex<F> diquark[Ns * Ns][Nc * Nc],
                                                const Complex<F> propag_i[Ns * Ns][Nc * Nc],
                                                const Complex<F> propag_j[Ns * Ns][Nc * Nc], int gamma_kl, int idx)
  {
    using T = Complex<F>;
    int ml = idx;
    int m = ml / Ns;
    int l = ml % Ns;
    T gamma5_m_l = 1 - 2 * ((m >> 1) ^ (l >> 1)); // Special case for Ns = 4
    int k = gamma_index(gamma_kl, l);
    T gamma_kl_data = gamma_data<true, F>(gamma_kl, l);

    // ik @ kl, ij @ jm
    for_abc_def
    {
      T tmp = 0, tmp_color;
      for (int i = 0; i < Ns; ++i) {
        int j = gamma_index(GAMMA_IJ, i);
        int ik = i * Ns + k;
        int jm = j * Ns + m;
        epsilon_abc_def(tmp_color, propag_i[ik], propag_j[jm]);
        tmp += gamma_data<false, F>(GAMMA_IJ, i) * tmp_color;
      }
      diquark[ml][ad] = gamma5_m_l * conj(gamma_kl_data * tmp);
    }
    __syncthreads();
  }

  template <typename Args> __device__ void diquark_kernel(const Args &args, size_t x_offset)
  {
    __shared__ typename Args::T propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T diquark[TILE_SIZE][Ns * Ns][Nc * Nc];

    load_vector<Ns * Ns, Nc * Nc>(propag_i, args.propag_i, x_offset);
    load_vector<Ns * Ns, Nc * Nc>(propag_j, args.propag_j, x_offset);
    __syncthreads();

    int t_idx = threadIdx.x / LANE_SIZE;
    int l_idx = threadIdx.x % LANE_SIZE;
    diquark_local<Args::GAMMA_IJ>(diquark[t_idx], propag_i[t_idx], propag_j[t_idx], args.gamma_kl, l_idx);

    store_vector<Ns * Ns, Nc * Nc>(args.diquark, diquark, x_offset);
  }

  template <typename Args> struct DiquarkKernel : public BaseKernel<Args, BLOCK_SIZE, LANE_SIZE> {
    constexpr DiquarkKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, LANE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset) { diquark_kernel(this->args, x_offset); }
  };

}; // namespace contract
