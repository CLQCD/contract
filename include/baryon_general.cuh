#pragma once

#include <complex>
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
  template <typename F, BaryonContractType CONTRACT_> struct BaryonGeneralArgs {
    using T = Complex<F>;
    static constexpr BaryonContractType CONTRACT = CONTRACT_;

    void *correl;
    void *propag_i;
    void *propag_j;
    void *propag_n;
    int gamma_ij;
    int gamma_kl;
    T project_mn[Ns * Ns];

    BaryonGeneralArgs(void *correl, void *propag_i, void *propag_j, void *propag_n, int gamma_ij, int gamma_kl,
                      std::complex<double> project_mn[Ns * Ns]) :
      correl(correl), propag_i(propag_i), propag_j(propag_j), propag_n(propag_n), gamma_ij(gamma_ij), gamma_kl(gamma_kl)
    {
      for (int ij = 0; ij < Ns * Ns; ++ij) { this->project_mn[ij] = project_mn[ij]; }
    }
  };

  template <BaryonContractType CONTRACT, typename F>
  __device__ __forceinline__ void
  baryon_general_local(Complex<F> correl[Ns * Ns], const Complex<F> propag_i[Ns * Ns][Nc * Nc],
                       const Complex<F> propag_j[Ns * Ns][Nc * Nc], const Complex<F> propag_n[Ns * Ns][Nc * Nc],
                       int gamma_ij, int gamma_kl, const Complex<F> project_mn[Ns * Ns], int idx)
  {
    using T = Complex<F>;
    constexpr bool SWAP_IJ = (CONTRACT == IM_JK_NL || CONTRACT == IM_JL_NK);
    constexpr bool SWAP_KL = (CONTRACT == IL_JK_NM || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK);

    correl[idx] = 0;

    int il = idx;
    int i = il / Ns;
    int l = il % Ns;
    int j = gamma_index(gamma_ij, i);
    T gamma_ij_data = gamma_data<SWAP_IJ, F>(gamma_ij, i);
    int k = gamma_index(gamma_kl, l);
    T gamma_kl_data = gamma_data<!SWAP_KL, F>(gamma_kl, l);
    int ik = i * Ns + k;
    int jl = j * Ns + l;
    if constexpr (CONTRACT == IK_JL_NM || CONTRACT == IL_JK_NM) {
      // ik @ kl, ij @ jl, mn @ nm
      for_abc_def
      {
        T tmp = 0, tmp_color;
        for (int n = 0; n < Ns; ++n) {
          for (int m = 0; m < Ns; ++m) {
            int nm = n * Ns + m;
            int mn = m * Ns + n;
            tmp += project_mn[mn] * propag_n[nm][ad];
          }
        }
        epsilon_abc_def(tmp_color, propag_i[ik], propag_j[jl]);
        correl[idx] += gamma_ij_data * gamma_kl_data * tmp * tmp_color;
      }
    } else if constexpr (CONTRACT == IK_JM_NL || CONTRACT == IM_JK_NL || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK) {
      // ik @ kl, ij @ jm, mn @ nl
      for_abc_def
      {
        T tmp = 0, tmp_color;
        for (int n = 0; n < Ns; ++n) {
          for (int m = 0; m < Ns; ++m) {
            int jm = j * Ns + m;
            int nl = n * Ns + l;
            int mn = m * Ns + n;
            epsilon_abc_def(tmp_color, propag_j[jm], propag_n[nl]);
            tmp += project_mn[mn] * tmp_color;
          }
        }
        correl[idx] += gamma_ij_data * gamma_kl_data * tmp * propag_i[ik][ad];
      }
    }
    __syncthreads();
  }

  template <typename Args> __device__ void baryon_general_kernel(const Args &args, size_t x_offset)
  {
    __shared__ typename Args::T propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T correl[TILE_SIZE][Ns * Ns];

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      load_vector<Ns * Ns, Nc * Nc>(propag_i, args.propag_j, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_j, args.propag_i, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_n, args.propag_n, x_offset);
    } else {
      load_vector<Ns * Ns, Nc * Nc>(propag_i, args.propag_i, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_j, args.propag_j, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_n, args.propag_n, x_offset);
    }
    __syncthreads();

    int t_idx = threadIdx.x / LANE_SIZE;
    int l_idx = threadIdx.x % LANE_SIZE;
    baryon_general_local<Args::CONTRACT>(correl[t_idx], propag_i[t_idx], propag_j[t_idx], propag_n[t_idx],
                                         args.gamma_ij, args.gamma_kl, args.project_mn, l_idx);
    reduce_lane<Ns * Ns>(correl);

    store_tile<Ns * Ns>(args.correl, correl, x_offset);
  }

  template <typename Args> struct BaryonGeneralKernel : public BaseKernel<Args, BLOCK_SIZE, LANE_SIZE> {
    constexpr BaryonGeneralKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, LANE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset) { baryon_general_kernel(this->args, x_offset); }
  };

}; // namespace contract
