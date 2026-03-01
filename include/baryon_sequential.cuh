#pragma once

#include <atomic>

#include <kernel.cuh>
#include <load_store.cuh>
#include <contract_enum.h>
#include <gamma.cuh>

const int Ns = 4;
const int Nc = 3;
const int BLOCK_SIZE = 64;
const int TILE_SIZE = Ns * Ns;
const int TILES_PER_BLOCK = BLOCK_SIZE / TILE_SIZE;

namespace contract
{

  template <typename F, BaryonContractType CONTRACT_, int GAMMA_MN_> struct BaryonSequentialArgs {
    using T = Complex<F>;
    static constexpr BaryonContractType CONTRACT = CONTRACT_;
    static constexpr int GAMMA_MN = GAMMA_MN_;

    void *propag_i;
    void *propag_j;
    void *propag_n;
    int gamma_ij;
    int gamma_kl;

    BaryonSequentialArgs(void *propag_i, void *propag_j, void *propag_n, int gamma_ij, int gamma_kl) :
      propag_i(propag_i), propag_j(propag_j), propag_n(propag_n), gamma_ij(gamma_ij), gamma_kl(gamma_kl)
    {
    }
  };

  template <BaryonContractType CONTRACT, BaryonSequentialType SEQUENTIAL, int GAMMA_MN, typename F>
  __device__ __forceinline__ void
  baryon_sequential_local(Complex<F> propag_i[Ns * Ns][Nc * Nc], Complex<F> propag_j[Ns * Ns][Nc * Nc],
                          Complex<F> propag_n[Ns * Ns][Nc * Nc], int gamma_ij, int gamma_kl, int idx)
  {
    cg_block block = cg::this_thread_block();
    cg_tile<TILE_SIZE> tile = cg::tiled_partition<TILE_SIZE>(block);

    using T = Complex<F>;
    constexpr bool SWAP_IJ = (CONTRACT == IM_JK_NL || CONTRACT == IM_JL_NK);
    constexpr bool SWAP_KL = (CONTRACT == IL_JK_NM || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK);

    int il = idx;
    int i = il / Ns;
    int l = il % Ns;
    int j = gamma_index(gamma_ij, i);
    T gamma_ij_data = gamma_data<SWAP_IJ, F>(gamma_ij, i);
    int k = gamma_index(gamma_kl, l);
    T gamma_kl_data = gamma_data<!SWAP_KL, F>(gamma_kl, l);
    int ik = i * Ns + k;
    if constexpr (CONTRACT == IK_JL_NM || CONTRACT == IL_JK_NM) {
      int jl = j * Ns + l;
      if constexpr (SEQUENTIAL == SEQUENTIAL_I) {
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int nm = n * Ns + m;
            tmp_color = epsilon_abc_def(propag_j[jl], propag_n[nm]);
            tmp += gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
          }
          propag_i[ik][ad] = gamma_ij_data * gamma_kl_data * tmp;
        }
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_J) {
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int nm = n * Ns + m;
            tmp_color = epsilon_abc_def(propag_i[ik], propag_n[nm]);
            tmp += gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
          }
          propag_j[jl][ad] = gamma_ij_data * gamma_kl_data * tmp;
        }
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_N) {
        for_abc_def
        {
          T tmp_color;
          tmp_color = epsilon_abc_def(propag_i[ik], propag_j[jl]);
          propag_n[idx][ad] = gamma_ij_data * gamma_kl_data * tmp_color;
        }
        tile.sync();
#pragma unroll
        for (int stride = TILE_SIZE / 2; stride > 0; stride /= 2) {
          if (idx < stride) {
            for_a_d { propag_n[idx][ad] += propag_n[idx + stride][ad]; }
          }
          tile.sync();
        }
        if (idx > 0) {
          for_a_d { propag_n[idx][ad] = propag_n[0][ad]; }
        }
        tile.sync();
        int nm = idx;
        int n = nm / Ns;
        int m = nm % Ns;
        T gamma_mn_data = gamma_data<true, F>(GAMMA_MN, n, m);
        for_a_d { propag_n[nm][ad] *= gamma_mn_data; }
      }
    } else if constexpr (CONTRACT == IK_JM_NL || CONTRACT == IM_JK_NL || CONTRACT == IL_JM_NK || CONTRACT == IM_JL_NK) {
      if constexpr (SEQUENTIAL == SEQUENTIAL_I) {
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
          propag_i[ik][ad] = gamma_ij_data * gamma_kl_data * tmp;
        }
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_J) {
        for_a_d { propag_j[idx][ad] = 0; }
        tile.sync();
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int jm = j * Ns + m;
            int nl = n * Ns + l;
            tmp_color = epsilon_abc_def(propag_i[ik], propag_n[nl]);
            tmp = gamma_ij_data * gamma_kl_data * gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
            F *propag_j_ptr = reinterpret_cast<F *>(&propag_j[jm][ad]);
            atomicAdd(&propag_j_ptr[0], tmp.real());
            atomicAdd(&propag_j_ptr[1], tmp.imag());
          }
        }
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_N) {
        for_a_d { propag_n[idx][ad] = 0; }
        tile.sync();
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int jm = j * Ns + m;
            int nl = n * Ns + l;
            tmp_color = epsilon_abc_def(propag_i[ik], propag_j[jm]);
            tmp = gamma_ij_data * gamma_kl_data * gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
            F *propag_n_ptr = reinterpret_cast<double *>(&propag_n[nl][ad]);
            atomicAdd(&propag_n_ptr[0], tmp.real());
            atomicAdd(&propag_n_ptr[1], tmp.imag());
          }
        }
      }
    }
  }

  template <typename Args>
  __device__ void baryon_sequential_i_kernel(const Args &args, size_t x_offset, cg_tile<TILE_SIZE> tile)
  {
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      load_vector<Ns * Ns, Nc * Nc>(propag_i[gid], args.propag_j, x_offset, tile);
      // load_vector<Ns * Ns, Nc * Nc>(propag_j[t_idx], args.propag_i, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_n[gid], args.propag_n, x_offset, tile);
      tile.sync();

      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_J, Args::GAMMA_MN>(propag_i[gid], propag_j[gid], propag_n[gid],
                                                                            args.gamma_ij, args.gamma_kl, tid);
      tile.sync();

      store_vector<Ns * Ns, Nc * Nc>(args.propag_i, propag_j[gid], x_offset, tile);
    } else {
      // load_vector<Ns * Ns, Nc * Nc>(propag_i[t_idx], args.propag_i, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_j[gid], args.propag_j, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_n[gid], args.propag_n, x_offset, tile);
      tile.sync();

      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_I, Args::GAMMA_MN>(propag_i[gid], propag_j[gid], propag_n[gid],
                                                                            args.gamma_ij, args.gamma_kl, tid);
      tile.sync();

      store_vector<Ns * Ns, Nc * Nc>(args.propag_i, propag_i[gid], x_offset, tile);
    }
  }

  template <typename Args>
  __device__ void baryon_sequential_j_kernel(const Args &args, size_t x_offset, cg_tile<TILE_SIZE> tile)
  {
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      // load_vector<Ns * Ns, Nc * Nc>(propag_i[t_idx], args.propag_j, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_j[gid], args.propag_i, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_n[gid], args.propag_n, x_offset, tile);
      tile.sync();

      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_I, Args::GAMMA_MN>(propag_i[gid], propag_j[gid], propag_n[gid],
                                                                            args.gamma_ij, args.gamma_kl, tid);
      tile.sync();

      store_vector<Ns * Ns, Nc * Nc>(args.propag_j, propag_i[gid], x_offset, tile);
    } else {
      load_vector<Ns * Ns, Nc * Nc>(propag_i[gid], args.propag_i, x_offset, tile);
      // load_vector<Ns * Ns, Nc * Nc>(propag_j[t_idx], args.propag_j, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_n[gid], args.propag_n, x_offset, tile);
      tile.sync();

      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_J, Args::GAMMA_MN>(propag_i[gid], propag_j[gid], propag_n[gid],
                                                                            args.gamma_ij, args.gamma_kl, tid);
      tile.sync();

      store_vector<Ns * Ns, Nc * Nc>(args.propag_j, propag_j[gid], x_offset, tile);
    }
  }

  template <typename Args>
  __device__ void baryon_sequential_n_kernel(const Args &args, size_t x_offset, cg_tile<TILE_SIZE> tile)
  {
    __shared__ typename Args::T propag_i[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILES_PER_BLOCK][Ns * Ns][Nc * Nc];

    const auto gid = tile.meta_group_rank();
    const auto tid = tile.thread_rank();

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      load_vector<Ns * Ns, Nc * Nc>(propag_i[gid], args.propag_j, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_j[gid], args.propag_i, x_offset, tile);
      // load_vector<Ns * Ns, Nc * Nc>(propag_n[t_idx], args.propag_n, x_offset, tile);
    } else {
      load_vector<Ns * Ns, Nc * Nc>(propag_i[gid], args.propag_i, x_offset, tile);
      load_vector<Ns * Ns, Nc * Nc>(propag_j[gid], args.propag_j, x_offset, tile);
      // load_vector<Ns * Ns, Nc * Nc>(propag_n[t_idx], args.propag_n, x_offset, tile);
    }
    tile.sync();

    baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_N, Args::GAMMA_MN>(propag_i[gid], propag_j[gid], propag_n[gid],
                                                                          args.gamma_ij, args.gamma_kl, tid);
    tile.sync();

    store_vector<Ns * Ns, Nc * Nc>(args.propag_n, propag_n[gid], x_offset, tile);
  }

  template <typename Args> struct BaryonSequentialIKernel : public BaseKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr BaryonSequentialIKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset, cg_tile<TILE_SIZE> tile) override
    {
      baryon_sequential_i_kernel(this->args, x_offset, tile);
    }
  };

  template <typename Args> struct BaryonSequentialJKernel : public BaseKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr BaryonSequentialJKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset, cg_tile<TILE_SIZE> tile) override
    {
      baryon_sequential_j_kernel(this->args, x_offset, tile);
    }
  };

  template <typename Args> struct BaryonSequentialNKernel : public BaseKernel<Args, BLOCK_SIZE, TILE_SIZE> {
    constexpr BaryonSequentialNKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, TILE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset, cg_tile<TILE_SIZE> tile) override
    {
      baryon_sequential_n_kernel(this->args, x_offset, tile);
    }
  };

}; // namespace contract
