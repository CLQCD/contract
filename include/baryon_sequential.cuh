#pragma once

#include <atomic>

#include <kernel.cuh>
#include <contract.h>
#include <gamma.cuh>

const int Ns = 4;
const int Nc = 3;
const int BLOCK_SIZE = 64;
const int LANE_SIZE = Ns * Ns;
const int TILE_SIZE = BLOCK_SIZE / LANE_SIZE;

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
    int jl = j * Ns + l;
    if constexpr (CONTRACT == IK_JL_NM || CONTRACT == IL_JK_NM) {
      if constexpr (SEQUENTIAL == SEQUENTIAL_I) {
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int nm = n * Ns + m;
            epsilon_abc_def(tmp_color, propag_j[jl], propag_n[nm]);
            tmp += gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
          }
          propag_i[ik][ad] = gamma_ij_data * gamma_kl_data * tmp;
        }
        __syncthreads();
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_J) {
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int nm = n * Ns + m;
            epsilon_abc_def(tmp_color, propag_i[ik], propag_n[nm]);
            tmp += gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
          }
          propag_j[jl][ad] = gamma_ij_data * gamma_kl_data * tmp;
        }
        __syncthreads();
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_N) {
        for_abc_def
        {
          T tmp_color;
          epsilon_abc_def(tmp_color, propag_i[ik], propag_j[jl]);
          propag_n[il][ad] = gamma_ij_data * gamma_kl_data * tmp_color;
        }
        __syncthreads();
#pragma unroll
        for (int stride = LANE_SIZE / 2; stride > 1; stride /= 2) {
          if (idx < stride) {
            for_a_d { propag_n[idx][ad] += propag_n[idx + stride][ad]; }
          }
          __syncthreads();
        }
        if (idx > 0) {
          for_a_d { propag_n[idx][ad] = propag_n[0][ad]; }
        }
        __syncthreads();
        int nm = idx;
        int n = nm / Ns;
        int m = nm % Ns;
        T gamma_mn_data = gamma_data<true, F>(GAMMA_MN, n, m);
        for_a_d { propag_n[nm][ad] *= gamma_mn_data * propag_n[nm][ad]; }
        __syncthreads();
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
            epsilon_abc_def(tmp_color, propag_j[jm], propag_n[nl]);
            tmp += gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
          }
          propag_i[ik][ad] = gamma_ij_data * gamma_kl_data * tmp;
        }
        __syncthreads();
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_J) {
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int jm = j * Ns + m;
            int nl = n * Ns + l;
            epsilon_abc_def(tmp_color, propag_i[ik], propag_n[nl]);
            tmp = gamma_ij_data * gamma_kl_data * gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
            F *propag_j_ptr = reinterpret_cast<double *>(&propag_j[jm][ad]);
            atomicAdd(&propag_j_ptr[0], tmp.real());
            atomicAdd(&propag_j_ptr[1], tmp.imag());
          }
        }
        __syncthreads();
      } else if constexpr (SEQUENTIAL == SEQUENTIAL_N) {
        for_abc_def
        {
          T tmp = 0, tmp_color;
          for (int n = 0; n < Ns; ++n) {
            int m = gamma_index(GAMMA_MN, n);
            int jm = j * Ns + m;
            int nl = n * Ns + l;
            epsilon_abc_def(tmp_color, propag_i[ik], propag_j[jm]);
            tmp = gamma_ij_data * gamma_kl_data * gamma_data<true, F>(GAMMA_MN, n) * tmp_color;
            F *propag_n_ptr = reinterpret_cast<double *>(&propag_n[nl][ad]);
            atomicAdd(&propag_n_ptr[0], tmp.real());
            atomicAdd(&propag_n_ptr[1], tmp.imag());
          }
        }
        __syncthreads();
      }
    }
  }

  template <typename Args> __device__ void baryon_sequential_i_kernel(const Args &args, size_t x_offset)
  {
    __shared__ typename Args::T propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      load_vector(propag_i, args.propag_j, x_offset);
      // load_vector(propag_j, args.propag_i, x_offset);
      load_vector(propag_n, args.propag_n, x_offset);
      __syncthreads();

      int t_idx = threadIdx.x / LANE_SIZE;
      int l_idx = threadIdx.x % LANE_SIZE;
      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_J, Args::GAMMA_MN>(
        propag_i[t_idx], propag_j[t_idx], propag_n[t_idx], args.gamma_ij, args.gamma_kl, l_idx);

      store_vector(args.propag_i, propag_j, x_offset);
    } else {
      // load_vector(propag_i, args.propag_i, x_offset);
      load_vector(propag_j, args.propag_j, x_offset);
      load_vector(propag_n, args.propag_n, x_offset);
      __syncthreads();

      int t_idx = threadIdx.x / LANE_SIZE;
      int l_idx = threadIdx.x % LANE_SIZE;
      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_I, Args::GAMMA_MN>(
        propag_i[t_idx], propag_j[t_idx], propag_n[t_idx], args.gamma_ij, args.gamma_kl, l_idx);

      store_vector(args.propag_i, propag_i, x_offset);
    }
  }

  template <typename Args> __device__ void baryon_sequential_j_kernel(const Args &args, size_t x_offset)
  {
    __shared__ typename Args::T propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      // load_vector(propag_i, args.propag_j, x_offset);
      load_vector(propag_j, args.propag_i, x_offset);
      load_vector(propag_n, args.propag_n, x_offset);
      __syncthreads();

      int t_idx = threadIdx.x / LANE_SIZE;
      int l_idx = threadIdx.x % LANE_SIZE;
      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_I, Args::GAMMA_MN>(
        propag_i[t_idx], propag_j[t_idx], propag_n[t_idx], args.gamma_ij, args.gamma_kl, l_idx);

      store_vector(args.propag_j, propag_i, x_offset);
    } else {
      load_vector(propag_i, args.propag_i, x_offset);
      // load_vector(propag_j, args.propag_j, x_offset);
      load_vector(propag_n, args.propag_n, x_offset);
      __syncthreads();

      int t_idx = threadIdx.x / LANE_SIZE;
      int l_idx = threadIdx.x % LANE_SIZE;
      baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_J, Args::GAMMA_MN>(
        propag_i[t_idx], propag_j[t_idx], propag_n[t_idx], args.gamma_ij, args.gamma_kl, l_idx);

      store_vector(args.propag_j, propag_j, x_offset);
    }
  }

  template <typename Args> __device__ void baryon_sequential_n_kernel(const Args &args, size_t x_offset)
  {
    __shared__ typename Args::T propag_i[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_j[TILE_SIZE][Ns * Ns][Nc * Nc];
    __shared__ typename Args::T propag_n[TILE_SIZE][Ns * Ns][Nc * Nc];

    constexpr bool SWAP_IJ = (Args::CONTRACT == IM_JK_NL || Args::CONTRACT == IM_JL_NK);

    if constexpr (SWAP_IJ) {
      load_vector(propag_i, args.propag_j, x_offset);
      load_vector(propag_j, args.propag_i, x_offset);
      // load_vector(propag_n, args.propag_n, x_offset);
    } else {
      load_vector(propag_i, args.propag_i, x_offset);
      load_vector(propag_j, args.propag_j, x_offset);
      // load_vector(propag_n, args.propag_n, x_offset);
    }
    __syncthreads();

    int t_idx = threadIdx.x / LANE_SIZE;
    int l_idx = threadIdx.x % LANE_SIZE;
    baryon_sequential_local<Args::CONTRACT, SEQUENTIAL_N, Args::GAMMA_MN>(
      propag_i[t_idx], propag_j[t_idx], propag_n[t_idx], args.gamma_ij, args.gamma_kl, l_idx);

    store_vector(args.propag_n, propag_n, x_offset);
  }

  template <typename Args> struct BaryonSequentialIKernel : public BaseKernel<Args, BLOCK_SIZE, LANE_SIZE> {
    constexpr BaryonSequentialIKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, LANE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset) { baryon_sequential_i_kernel(this->args, x_offset); }
  };

  template <typename Args> struct BaryonSequentialJKernel : public BaseKernel<Args, BLOCK_SIZE, LANE_SIZE> {
    constexpr BaryonSequentialJKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, LANE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset) { baryon_sequential_j_kernel(this->args, x_offset); }
  };

  template <typename Args> struct BaryonSequentialNKernel : public BaseKernel<Args, BLOCK_SIZE, LANE_SIZE> {
    constexpr BaryonSequentialNKernel(const Args &args) : BaseKernel<Args, BLOCK_SIZE, LANE_SIZE>(args) { }

    __device__ __forceinline__ void operator()(size_t x_offset) { baryon_sequential_n_kernel(this->args, x_offset); }
  };

}; // namespace contract
