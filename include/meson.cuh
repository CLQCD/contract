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

  template <typename Args> __device__ void meson_kernel(const Args &args, size_t x_offset, ThreadTile<TILE_SIZE> tile)
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

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    load_vector<Ns * Ns, Nc * Nc>(propag_i, args.propag_i, x_offset, tile.item);
    load_vector<Ns * Ns, Nc * Nc>(propag_j, args.propag_j, x_offset, tile.item);
    sycl::group_barrier(tile.item.get_group());

    meson_local(correl[gid], propag_i[gid], propag_j[gid], args.gamma_ij, args.gamma_kl, tid);
    tile.sync();

    tile_reduce_store<BLOCK_SIZE>(tile, args.correl, correl[gid], x_offset);
#else
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T correl[TILES_PER_BLOCK][Ns * Ns];

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    load_vector<Ns * Ns, Nc * Nc>(propag_i, args.propag_i, x_offset);
    load_vector<Ns * Ns, Nc * Nc>(propag_j, args.propag_j, x_offset);
    __syncthreads(); // Seems to be faster on P100

    meson_local(correl[gid], propag_i[gid], propag_j[gid], args.gamma_ij, args.gamma_kl, tid);
    tile.sync();

    tile_reduce_store<BLOCK_SIZE>(tile, args.correl, correl[gid], x_offset);
#endif
  }

  template <typename Args> struct MesonKernel : public TileKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr MesonKernel(const Args &args) : TileKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset, ThreadTile<TILE_SIZE> tile) override
    {
      meson_kernel(this->args, x_offset, tile);
    }
  };

}; // namespace contract