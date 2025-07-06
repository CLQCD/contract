#pragma once

#include <stdlib.h>

namespace meson
{
  void launch(void *correl, void *propag_a, void *propag_b, size_t volume, int gamma_ab, int gamma_dc);
}

namespace meson_all_source
{
  void launch(void **correl, void *propag_a, void *propag_b, size_t volume, int gamma_ab);
}

namespace meson_all_sink
{
  void launch(void **correl, void *propag_a, void *propag_b, size_t volume, int gamma_dc);
}