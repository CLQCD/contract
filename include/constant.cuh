#pragma once

namespace contract
{

  constexpr size_t buffer_size = 32764;

  __constant__ char buffer[buffer_size];

  template <typename Arg> constexpr Arg &get_args() { return reinterpret_cast<Arg &>(buffer); }

  template <typename Arg> void *get_buffer()
  {
    void *ptr;
    cudaGetSymbolAddress(&ptr, buffer);
    return ptr;
  }

} // namespace contract