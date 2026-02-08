#include <diquark.cuh>
#include <baryon.h>

namespace diquark
{

  template <int GAMMA_IJ> void launch(void *diquark, void *propag_i, void *propag_j, size_t volume, int gamma_kl)
  {
    if (volume % TILE_SIZE != 0) {
      fprintf(stderr, "Error: Volume must be a multiple of TILE_SIZE\n");
      exit(-1);
    }

    using Args = contract::DiquarkArgs<GAMMA_IJ, double>;
    using Kernel = contract::DiquarkKernel<Args>;
    Args args(diquark, propag_i, propag_j, gamma_kl);
    contract::launch_kernel<Kernel>(args, volume);
    return;
  }

  void launch(void *diquark, void *propag_i, void *propag_j, size_t volume, int gamma_ij, int gamma_kl)
  {
    switch (gamma_ij) {
    case 0: launch<0>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 1: launch<1>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 2: launch<2>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 3: launch<3>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 4: launch<4>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 5: launch<5>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 6: launch<6>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 7: launch<7>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 8: launch<8>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 9: launch<9>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 10: launch<10>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 11: launch<11>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 12: launch<12>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 13: launch<13>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 14: launch<14>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    case 15: launch<15>(diquark, propag_i, propag_j, volume, gamma_kl); break;
    default: break;
    }
  }

} // namespace diquark
