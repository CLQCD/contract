#pragma once

#include <complex.cuh>

// __device__ const int gamma_csr_index[Ns * Ns][Ns] {
//   {0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}, {0, 1, 2, 3}, {2, 3, 0, 1}, {1, 0, 3, 2}, {1, 0, 3, 2}, {2, 3, 0, 1},
//   {2, 3, 0, 1}, {1, 0, 3, 2}, {1, 0, 3, 2}, {2, 3, 0, 1}, {0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}, {0, 1, 2, 3},
// };

// __device__ const Complex<double> I(0.0, 1.0);
// __device__ const Complex<double> gamma_csr_data[Ns * Ns][Ns] {
//   {1, 1, 1, 1},     {I, I, -I, -I}, {-1, 1, 1, -1},   {-I, I, -I, I}, {I, -I, -I, I}, {-1, 1, -1, 1},
//   {-I, -I, -I, -I}, {1, 1, -1, -1}, {1, 1, 1, 1},     {I, I, -I, -I}, {-1, 1, 1, -1}, {-I, I, -I, I},
//   {I, -I, -I, I},   {-1, 1, -1, 1}, {-I, -I, -I, -I}, {1, 1, -1, -1},
// };

// template <bool TRANSPOSE> __device__ __forceinline__ Complex<double> gamma_data(int gamma, int row)
// {
//   if constexpr (TRANSPOSE) {
//     if (gamma == 1 || gamma == 4 || gamma == 5 || gamma == 7 || gamma == 10 || gamma == 13) {
//       return -gamma_csr_data[gamma][row];
//     } else {
//       return gamma_csr_data[gamma][row];
//     }
//   } else {
//     return gamma_csr_data[gamma][row];
//   }
// }

// template <int GAMMA, bool TRANSPOSE> __device__ __forceinline__ const Complex<double> gamma_data(int row)
// {
//   if constexpr (TRANSPOSE && (GAMMA == 1 || GAMMA == 4 || GAMMA == 5 || GAMMA == 7 || GAMMA == 10 || GAMMA == 13)) {
//     return -gamma_csr_data[GAMMA][row];
//   } else {
//     return gamma_csr_data[GAMMA][row];
//   }
// }

namespace contract
{
  __device__ __forceinline__ constexpr int row_01(int row)
  {
    // return (row == 0 || row == 1) ? 1 : -1
    return 1 - 2 * (row >> 1);
  }

  __device__ __forceinline__ constexpr int row_02(int row)
  {
    // return (row == 0 || row == 2) ? 1 : -1
    return 1 - 2 * (row & 1);
  }

  __device__ __forceinline__ constexpr int row_03(int row)
  { // return (row == 0 || row == 3) ? 1 : -1
    return 1 - 2 * ((row >> 1) ^ (row & 1));
  }

  __device__ __forceinline__ int gamma_index(int gamma, int row)
  {
    switch (gamma) {
    case 0:
    case 3:
    case 12:
    case 15: return row; break;
    case 1:
    case 2:
    case 13:
    case 14: return 3 - row; break;
    case 4:
    case 7:
    case 8:
    case 11: return (row + 2) % 4; break;
    case 5:
    case 6:
    case 9:
    case 10: return (3 - row + 2) % 4; break;
    default: break;
    }
    return -1;
  }

  template <bool TRANSPOSE, typename F> __device__ __forceinline__ const Complex<F> gamma_data(int gamma, int row)
  {
    const Complex<F> I(0.0, 1.0);
    if constexpr (!TRANSPOSE) {
      switch (gamma) {
      case 0: return 1; break;
      case 1: return row_01(row) * I; break;
      case 2: return row_03(row) * -1; break;
      case 3: return row_02(row) * -I; break;
      case 4: return row_03(row) * I; break;
      case 5: return row_02(row) * -1; break;
      case 6: return -I; break;
      case 7: return row_01(row) * 1; break;
      case 8: return 1; break;
      case 9: return row_01(row) * I; break;
      case 10: return row_03(row) * -1; break;
      case 11: return row_02(row) * -I; break;
      case 12: return row_03(row) * I; break;
      case 13: return row_02(row) * -1; break;
      case 14: return -I; break;
      case 15: return row_01(row) * 1; break;
      default: break;
      }
    } else {
      switch (gamma) {
      case 0: return 1; break;
      case 1: return row_01(row) * -I; break;
      case 2: return row_03(row) * -1; break;
      case 3: return row_02(row) * -I; break;
      case 4: return row_03(row) * -I; break;
      case 5: return row_02(row) * 1; break;
      case 6: return -I; break;
      case 7: return row_01(row) * -1; break;
      case 8: return 1; break;
      case 9: return row_01(row) * I; break;
      case 10: return row_03(row) * 1; break;
      case 11: return row_02(row) * -I; break;
      case 12: return row_03(row) * I; break;
      case 13: return row_02(row) * 1; break;
      case 14: return -I; break;
      case 15: return row_01(row) * 1; break;
      default: break;
      }
    }
    return 0;
  }

  template <bool TRANSPOSE, typename F> __device__ __forceinline__ const Complex<F> gamma_data_old(int gamma, int row)
  {
    const Complex<F> I(0.0, 1.0);
    if constexpr (!TRANSPOSE) {
      switch (gamma) {
      case 0: return 1; break;
      case 1: return (row == 0 || row == 1) ? I : -I; break;
      case 2: return (row == 0 || row == 3) ? -1 : 1; break;
      case 3: return (row == 0 || row == 2) ? -I : I; break;
      case 4: return (row == 0 || row == 3) ? I : -I; break;
      case 5: return (row == 0 || row == 2) ? -1 : 1; break;
      case 6: return -I; break;
      case 7: return (row == 0 || row == 1) ? 1 : -1; break;
      case 8: return 1; break;
      case 9: return (row == 0 || row == 1) ? I : -I; break;
      case 10: return (row == 0 || row == 3) ? -1 : 1; break;
      case 11: return (row == 0 || row == 2) ? -I : I; break;
      case 12: return (row == 0 || row == 3) ? I : -I; break;
      case 13: return (row == 0 || row == 2) ? -1 : 1; break;
      case 14: return -I; break;
      case 15: return (row == 0 || row == 1) ? 1 : -1; break;
      default: break;
      }
    } else {
      switch (gamma) {
      case 0: return 1; break;
      case 1: return (row == 0 || row == 1) ? -I : I; break;
      case 2: return (row == 0 || row == 3) ? -1 : 1; break;
      case 3: return (row == 0 || row == 2) ? -I : I; break;
      case 4: return (row == 0 || row == 3) ? -I : I; break;
      case 5: return (row == 0 || row == 2) ? 1 : -1; break;
      case 6: return -I; break;
      case 7: return (row == 0 || row == 1) ? -1 : 1; break;
      case 8: return 1; break;
      case 9: return (row == 0 || row == 1) ? I : -I; break;
      case 10: return (row == 0 || row == 3) ? 1 : -1; break;
      case 11: return (row == 0 || row == 2) ? -I : I; break;
      case 12: return (row == 0 || row == 3) ? I : -I; break;
      case 13: return (row == 0 || row == 2) ? 1 : -1; break;
      case 14: return -I; break;
      case 15: return (row == 0 || row == 1) ? 1 : -1; break;
      default: break;
      }
    }
    return 0;
  }

  template <bool TRANSPOSE, typename F>
  __device__ __forceinline__ const Complex<F> gamma_data(int gamma, int row, int col)
  {
    const Complex<F> I(0.0, 1.0);
    if (gamma_index(gamma, row) != col) { return 0; }
    if constexpr (!TRANSPOSE) {
      switch (gamma) {
      case 0: return 1; break;
      case 1: return row_01(row) * I; break;
      case 2: return row_03(row) * -1; break;
      case 3: return row_02(row) * -I; break;
      case 4: return row_03(row) * I; break;
      case 5: return row_02(row) * -1; break;
      case 6: return -I; break;
      case 7: return row_01(row) * 1; break;
      case 8: return 1; break;
      case 9: return row_01(row) * I; break;
      case 10: return row_03(row) * -1; break;
      case 11: return row_02(row) * -I; break;
      case 12: return row_03(row) * I; break;
      case 13: return row_02(row) * -1; break;
      case 14: return -I; break;
      case 15: return row_01(row) * 1; break;
      default: break;
      }
    } else {
      switch (gamma) {
      case 0: return 1; break;
      case 1: return row_01(row) * -I; break;
      case 2: return row_03(row) * -1; break;
      case 3: return row_02(row) * -I; break;
      case 4: return row_03(row) * -I; break;
      case 5: return row_02(row) * 1; break;
      case 6: return -I; break;
      case 7: return row_01(row) * -1; break;
      case 8: return 1; break;
      case 9: return row_01(row) * I; break;
      case 10: return row_03(row) * 1; break;
      case 11: return row_02(row) * -I; break;
      case 12: return row_03(row) * I; break;
      case 13: return row_02(row) * 1; break;
      case 14: return -I; break;
      case 15: return row_01(row) * 1; break;
      default: break;
      }
    }
    return 0;
  }

  template <bool TRANSPOSE, typename F>
  __device__ __forceinline__ const Complex<F> gamma_gamma5_data(int gamma, int row)
  {
    const Complex<F> I(0.0, 1.0);
    if constexpr (!TRANSPOSE) {
      switch (gamma) {
      case 0: return row_01(row) * 1; break;
      case 1: return -I; break;
      case 2: return row_02(row) * 1; break;
      case 3: return row_03(row) * -I; break;
      case 4: return row_02(row) * -I; break;
      case 5: return row_03(row) * -1; break;
      case 6: return row_01(row) * -I; break;
      case 7: return -1; break;
      case 8: return row_01(row) * -1; break;
      case 9: return I; break;
      case 10: return row_02(row) * -1; break;
      case 11: return row_03(row) * I; break;
      case 12: return row_02(row) * I; break;
      case 13: return row_03(row) * 1; break;
      case 14: return row_01(row) * I; break;
      case 15: return 1; break;
      default: break;
      }
    } else {
      switch (gamma) {
      case 0: return row_01(row) * 1; break;
      case 1: return I; break;
      case 2: return row_02(row) * 1; break;
      case 3: return row_03(row) * -I; break;
      case 4: return row_02(row) * I; break;
      case 5: return row_03(row) * 1; break;
      case 6: return row_01(row) * -I; break;
      case 7: return 1; break;
      case 8: return row_01(row) * -1; break;
      case 9: return I; break;
      case 10: return row_02(row) * 1; break;
      case 11: return row_03(row) * I; break;
      case 12: return row_02(row) * I; break;
      case 13: return row_03(row) * -1; break;
      case 14: return row_01(row) * I; break;
      case 15: return 1; break;
      default: break;
      }
    }
    return 0;
  }

  template <bool TRANSPOSE, typename F>
  __device__ __forceinline__ const Complex<F> gamma_gamma5_data_old(int gamma, int row)
  {
    const Complex<F> I(0.0, 1.0);
    if constexpr (!TRANSPOSE) {
      switch (gamma) {
      case 0: return (row == 0 || row == 1) ? 1 : -1; break;
      case 1: return -I; break;
      case 2: return (row == 0 || row == 2) ? 1 : -1; break;
      case 3: return (row == 0 || row == 3) ? -I : I; break;
      case 4: return (row == 0 || row == 2) ? -I : I; break;
      case 5: return (row == 0 || row == 3) ? -1 : 1; break;
      case 6: return (row == 0 || row == 1) ? -I : I; break;
      case 7: return -1; break;
      case 8: return (row == 0 || row == 1) ? -1 : 1; break;
      case 9: return I; break;
      case 10: return (row == 0 || row == 2) ? -1 : 1; break;
      case 11: return (row == 0 || row == 3) ? I : -I; break;
      case 12: return (row == 0 || row == 2) ? I : -I; break;
      case 13: return (row == 0 || row == 3) ? 1 : -1; break;
      case 14: return (row == 0 || row == 1) ? I : -I; break;
      case 15: return 1; break;
      default: break;
      }
    } else {
      switch (gamma) {
      case 0: return (row == 0 || row == 1) ? 1 : -1; break;
      case 1: return I; break;
      case 2: return (row == 0 || row == 2) ? 1 : -1; break;
      case 3: return (row == 0 || row == 3) ? -I : I; break;
      case 4: return (row == 0 || row == 2) ? I : -I; break;
      case 5: return (row == 0 || row == 3) ? 1 : -1; break;
      case 6: return (row == 0 || row == 1) ? -I : I; break;
      case 7: return 1; break;
      case 8: return (row == 0 || row == 1) ? -1 : 1; break;
      case 9: return I; break;
      case 10: return (row == 0 || row == 2) ? 1 : -1; break;
      case 11: return (row == 0 || row == 3) ? I : -I; break;
      case 12: return (row == 0 || row == 2) ? I : -I; break;
      case 13: return (row == 0 || row == 3) ? -1 : 1; break;
      case 14: return (row == 0 || row == 1) ? I : -I; break;
      case 15: return 1; break;
      default: break;
      }
    }
    return 0;
  }

  template <int GAMMA> __device__ __forceinline__ int gamma_index(int row)
  {
    switch (GAMMA) {
    case 0:
    case 3:
    case 12:
    case 15: return row; break;
    case 1:
    case 2:
    case 13:
    case 14: return 3 - row; break;
    case 4:
    case 7:
    case 8:
    case 11: return (row + 2) % 4; break;
    case 5:
    case 6:
    case 9:
    case 10: return (3 - row + 2) % 4; break;
    default: break;
    }
    return -1;
  }

  template <int GAMMA, bool TRANSPOSE, typename F> __device__ __forceinline__ const Complex<F> gamma_data(int row)
  {
    const Complex<F> I(0.0, 1.0);
    if constexpr (!TRANSPOSE) {
      switch (GAMMA) {
      case 0: return 1; break;
      case 1: return (row == 0 || row == 1) ? I : -I; break;
      case 2: return (row == 0 || row == 3) ? -1 : 1; break;
      case 3: return (row == 0 || row == 2) ? -I : I; break;
      case 4: return (row == 0 || row == 3) ? I : -I; break;
      case 5: return (row == 0 || row == 2) ? -1 : 1; break;
      case 6: return -I; break;
      case 7: return (row == 0 || row == 1) ? 1 : -1; break;
      case 8: return 1; break;
      case 9: return (row == 0 || row == 1) ? I : -I; break;
      case 10: return (row == 0 || row == 3) ? -1 : 1; break;
      case 11: return (row == 0 || row == 2) ? -I : I; break;
      case 12: return (row == 0 || row == 3) ? I : -I; break;
      case 13: return (row == 0 || row == 2) ? -1 : 1; break;
      case 14: return -I; break;
      case 15: return (row == 0 || row == 1) ? 1 : -1; break;
      default: break;
      }
    } else {
      switch (GAMMA) {
      case 0: return 1; break;
      case 1: return (row == 0 || row == 1) ? -I : I; break;
      case 2: return (row == 0 || row == 3) ? -1 : 1; break;
      case 3: return (row == 0 || row == 2) ? -I : I; break;
      case 4: return (row == 0 || row == 3) ? -I : I; break;
      case 5: return (row == 0 || row == 2) ? 1 : -1; break;
      case 6: return -I; break;
      case 7: return (row == 0 || row == 1) ? -1 : 1; break;
      case 8: return 1; break;
      case 9: return (row == 0 || row == 1) ? I : -I; break;
      case 10: return (row == 0 || row == 3) ? 1 : -1; break;
      case 11: return (row == 0 || row == 2) ? -I : I; break;
      case 12: return (row == 0 || row == 3) ? I : -I; break;
      case 13: return (row == 0 || row == 2) ? 1 : -1; break;
      case 14: return -I; break;
      case 15: return (row == 0 || row == 1) ? 1 : -1; break;
      default: break;
      }
    }
    return 0;
  }

} // namespace contract