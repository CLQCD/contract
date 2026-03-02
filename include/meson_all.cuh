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

  template <typename F> struct MesonAllArgs {
    using T = Complex<F>;

    void *correl[Ns * Ns];
    void *propag_i;
    void *propag_j;
    int gamma;

    MesonAllArgs(void *correl[Ns * Ns], void *propag_i, void *propag_j, int gamma) :
      propag_i(propag_i), propag_j(propag_j), gamma(gamma)
    {
      for (int i = 0; i < Ns * Ns; i++) { this->correl[i] = correl[i]; }
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

  template <typename Args>
  __device__ void meson_all_source_kernel(const Args &args, size_t x_offset, ThreadTile<TILE_SIZE> tile)
  {
#if defined(GPU_TARGET_SYCL)
    auto &propag_i
      = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns][Nc * Nc]>(
        tile.item.get_group());
    auto &propag_j
      = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns][Nc * Nc]>(
        tile.item.get_group());
    auto &correl = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns]>(
      tile.item.get_group());
#else
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T correl[TILES_PER_BLOCK][Ns * Ns];
#endif

    using Reduce = WarpReduce<typename Args::T, BLOCK_SIZE, TILE_SIZE>;

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    tile_load_vector(tile, propag_i[gid], args.propag_i, x_offset);
    tile_load_vector(tile, propag_j[gid], args.propag_j, x_offset);
    tile.sync();

    for (int gamma_kl = 0; gamma_kl < Ns * Ns; ++gamma_kl) {
      meson_local(correl[gid], propag_i[gid], propag_j[gid], args.gamma, gamma_kl, tid);
      tile.sync();

      tile_reduce_store<Reduce>(tile, args.correl[gamma_kl], correl[gid], x_offset);
      tile.sync();
    }
  }

  template <typename Args> struct MesonAllSourceKernel : public TileKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr MesonAllSourceKernel(const Args &args) : TileKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

#if defined(GPU_TARGET_SYCL)
    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile)
#else
    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile) override
#endif
    {
      meson_all_source_kernel(this->args, x_offset, tile);
    }
  };

  template <typename Args>
  __device__ void meson_all_sink_kernel(const Args &args, size_t x_offset, ThreadTile<TILE_SIZE> tile)
  {
#if defined(GPU_TARGET_SYCL)
    auto &propag_i
      = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns][Nc * Nc]>(
        tile.item.get_group());
    auto &propag_j
      = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns][Nc * Nc]>(
        tile.item.get_group());
    auto &correl = *sycl::ext::oneapi::group_local_memory_for_overwrite<typename Args::T[TILES_PER_BLOCK][Ns * Ns]>(
      tile.item.get_group());
#else
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T correl[TILES_PER_BLOCK][Ns * Ns];
#endif

    using Reduce = WarpReduce<typename Args::T, BLOCK_SIZE, TILE_SIZE>;

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    tile_load_vector(tile, propag_i[gid], args.propag_i, x_offset);
    tile_load_vector(tile, propag_j[gid], args.propag_j, x_offset);
    tile.sync();

    for (int gamma_ij = 0; gamma_ij < Ns * Ns; ++gamma_ij) {
      meson_local(correl[gid], propag_i[gid], propag_j[gid], gamma_ij, args.gamma, tid);
      tile.sync();

      tile_reduce_store<Reduce>(tile, args.correl[gamma_ij], correl[gid], x_offset);
      tile.sync();
    }
  }

  template <typename Args> struct MesonAllSinkKernel : public TileKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr MesonAllSinkKernel(const Args &args) : TileKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

#if defined(GPU_TARGET_SYCL)
    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile)
#else
    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile) override
#endif
    {
      meson_all_sink_kernel(this->args, x_offset, tile);
    }
  };

} // namespace contract
