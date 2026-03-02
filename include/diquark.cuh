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
  template <int GAMMA_IJ_, typename F> struct DiquarkArgs {
    using T = Complex<F>;
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
        tmp_color = epsilon_abc_def(propag_i[ik], propag_j[jm]);
        tmp += gamma_data<false, F>(GAMMA_IJ, i) * tmp_color;
      }
      diquark[ml][ad] = gamma5_m_l * conj(gamma_kl_data * tmp);
    }
  }

  template <typename Args> __device__ void diquark_kernel(const Args &args, size_t x_offset, ThreadTile<TILE_SIZE> tile)
  {
#if defined(GPU_TARGET_SYCL)
    auto &propag_i
      = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns][Nc * Nc]>(
        tile.item.get_group());
    auto &propag_j
      = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns][Nc * Nc]>(
        tile.item.get_group());
    auto &diquark
      = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns][Nc * Nc]>(
        tile.item.get_group());
#else
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T diquark[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
#endif

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    tile_load_vector(tile, propag_i[gid], args.propag_i, x_offset);
    tile_load_vector(tile, propag_j[gid], args.propag_j, x_offset);
    tile.sync();

    diquark_local<Args::GAMMA_IJ>(diquark[gid], propag_i[gid], propag_j[gid], args.gamma_kl, tid);
    tile.sync();

    tile_store_vector(tile, args.diquark, diquark[gid], x_offset);
  }

  template <typename Args> struct DiquarkKernel : public TileKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr DiquarkKernel(const Args &args) : TileKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

#if defined(GPU_TARGET_SYCL)
    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile)
#else
    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile) override
#endif
    {
      diquark_kernel(this->args, x_offset, tile);
    }
  };

}; // namespace contract
