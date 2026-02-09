#pragma once

#include <cstdlib>

namespace meson
{
  void launch(void *correl, void *propag_i, void *propag_j, size_t volume, int gamma_ij, int gamma_kl);
}

namespace meson_all_source
{
  void launch(void **correl, void *propag_i, void *propag_j, size_t volume, int gamma_ij);
}

namespace meson_all_sink
{
  void launch(void **correl, void *propag_i, void *propag_j, size_t volume, int gamma_kl);
}