add_library(meson STATIC
    meson.cu
    meson_all.cu
)
set_property(TARGET meson PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(meson PRIVATE "-Wno-deprecated-gpu-targets" "--expt-relaxed-constexpr")
target_link_libraries(meson PUBLIC CUDA::cudart)
