#include <meson_all.cuh>
#include <meson.h>

namespace meson_all_source
{
  void launch(void *correl[Ns * Ns], void *propag_i, void *propag_j, size_t volume, int gamma_ij)
  {
    if (volume % TILES_PER_BLOCK != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILES_PER_BLOCK\n");
      exit(-1);
    }

    using Args = contract::MesonAllArgs<double>;
    using Kernel = contract::MesonAllSourceKernel<Args>;
    Args args(correl, propag_i, propag_j, gamma_ij);
    contract::launch_kernel<Kernel>(args, volume);
  }
} // namespace meson_all_source

namespace meson_all_sink
{
  void launch(void *correl[Ns * Ns], void *propag_i, void *propag_j, size_t volume, int gamma_kl)
  {
    if (volume % TILES_PER_BLOCK != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILES_PER_BLOCK\n");
      exit(-1);
    }

    using Args = contract::MesonAllArgs<double>;
    using Kernel = contract::MesonAllSinkKernel<Args>;
    Args args(correl, propag_i, propag_j, gamma_kl);
    contract::launch_kernel<Kernel>(args, volume);
  }
} // namespace meson_all_sink
