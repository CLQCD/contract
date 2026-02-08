#include <thrust/complex.h>

namespace contract
{

  template <typename F> using Complex = thrust::complex<F>;

  typedef Complex<double> Complex128;
  typedef Complex<float> Complex64;

} // namespace contract
