#pragma once

#include <kernel.cuh>
#include <load_store.cuh>
#include <contract_enum.h>
#include <gamma.cuh>

const unsigned int Ns = 4;
const unsigned int Nc = 3;
const unsigned int BLOCK_SIZE = 64;
const unsigned int TILE_SIZE = Ns * Ns;
const unsigned int TILES_PER_BLOCK = BLOCK_SIZE / TILE_SIZE;

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

  template <typename Args> __device__ void baryon_kernel(const Args &args, size_t x_offset, ThreadTile<TILE_SIZE> tile)
  {
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T correl[TILES_PER_BLOCK][Ns * Ns];

    using Reduce = WarpReduce<typename Args::T, BLOCK_SIZE, TILE_SIZE>;

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      tile_load_vector(tile, propag_i[gid], args.propag_j, x_offset);
      tile_load_vector(tile, propag_j[gid], args.propag_i, x_offset);
      tile_load_vector(tile, propag_n[gid], args.propag_n, x_offset);
    } else {
      tile_load_vector(tile, propag_i[gid], args.propag_i, x_offset);
      tile_load_vector(tile, propag_j[gid], args.propag_j, x_offset);
      tile_load_vector(tile, propag_n[gid], args.propag_n, x_offset);
    }
    tile.sync();

    baryon_local<Args::CONTRACT, Args::GAMMA_MN>(correl[gid], propag_i[gid], propag_j[gid], propag_n[gid],
                                                 args.gamma_ij, args.gamma_kl, tid);
    tile.sync();

    tile_reduce_store<Reduce>(tile, args.correl, correl[gid], x_offset);
  }

  template <typename Args> struct BaryonKernel : public TileKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr BaryonKernel(const Args &args) : TileKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile) override
    {
      baryon_kernel(this->args, x_offset, tile);
    }
  };

}; // namespace contract
