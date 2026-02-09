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
  template <typename F, BaryonContractType CONTRACT_, int GAMMA_MN_> struct BaryonArgs {
    using T = Complex<F>;
    static constexpr BaryonContractType CONTRACT = CONTRACT_;
    static constexpr int GAMMA_MN = GAMMA_MN_;

    void *correl;
    void *propag_i;
    void *propag_j;
    void *propag_n;
    int gamma_ij;
    int gamma_kl;

    BaryonArgs(void *correl, void *propag_i, void *propag_j, void *propag_n, int gamma_ij, int gamma_kl) :
      correl(correl), propag_i(propag_i), propag_j(propag_j), propag_n(propag_n), gamma_ij(gamma_ij), gamma_kl(gamma_kl)
    {
    }
  };

  template <BaryonContractType CONTRACT, int GAMMA_MN, typename F>
  __device__ __forceinline__ void baryon_local(Complex<F> correl[Ns * Ns], const Complex<F> propag_i[Ns * Ns][Nc * Nc],
                                               const Complex<F> propag_j[Ns * Ns][Nc * Nc],
                                               const Complex<F> propag_n[Ns * Ns][Nc * Nc], int gamma_ij, int gamma_kl,
                                               int idx)
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
          int m = gamma_index(GAMMA_MN, n);
          int nm = n * Ns + m;
          tmp += gamma_data<true, F>(GAMMA_MN, n) * propag_n[nm][ad];
        }
        tmp_color = epsilon_abc_def(propag_i[ik], propag_j[jl]);
        correl[idx] += gamma_ij_data * gamma_kl_data * tmp * tmp_color;
      }
    } else if constexpr (CONTRACT == IK_JM_NL || CONTRACT == IM_JK_NL || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK) {
      // ik @ kl, ij @ jm, mn @ nl
      for_abc_def
      {
        T tmp = 0, tmp_color;
        for (int n = 0; n < Ns; ++n) {
          int m = gamma_index(GAMMA_MN, n);
          int jm = j * Ns + m;
          int nl = n * Ns + l;
          tmp_color = epsilon_abc_def(propag_j[jm], propag_n[nl]);
          tmp += gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
        }
        correl[idx] += gamma_ij_data * gamma_kl_data * tmp * propag_i[ik][ad];
      }
    }
  }

  template <typename Args> __device__ void baryon_kernel(const Args &args, size_t x_offset)
  {
    __shared__ typename Args::T propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T correl[TILE_SIZE][Ns * Ns];

    const int t_idx = threadIdx.x / LANE_SIZE;
    const int l_idx = threadIdx.x % LANE_SIZE;
    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      load_vector<Ns * Ns, Nc * Nc>(propag_i[t_idx], args.propag_j, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_j[t_idx], args.propag_i, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_n[t_idx], args.propag_n, x_offset);
    } else {
      load_vector<Ns * Ns, Nc * Nc>(propag_i[t_idx], args.propag_i, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_j[t_idx], args.propag_j, x_offset);
      load_vector<Ns * Ns, Nc * Nc>(propag_n[t_idx], args.propag_n, x_offset);
    }
    __syncwarp();

    baryon_local<Args::CONTRACT, Args::GAMMA_MN>(correl[t_idx], propag_i[t_idx], propag_j[t_idx], propag_n[t_idx],
                                                 args.gamma_ij, args.gamma_kl, l_idx);
    __syncwarp();

    reduce_lane<Ns * Ns>(correl[t_idx]);
    store_tile<Ns * Ns>(args.correl, correl[t_idx], x_offset);
  }

  template <typename Args> struct BaryonKernel : public BaseKernel<Args, BLOCK_SIZE, LANE_SIZE> {
    constexpr BaryonKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, LANE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset) { baryon_kernel(this->args, x_offset); }
  };

}; // namespace contract
