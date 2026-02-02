add_library(baryon STATIC
    baryon.cu
    baryon_sequential_a.cu
    baryon_sequential_b.cu
    baryon_sequential_c.cu
    baryon_ik_jl_mn.cu
    baryon_ik_jn_ml.cu
    baryon_il_jk_mn.cu
    baryon_il_jn_mk.cu
    baryon_in_jk_ml.cu
    baryon_in_jl_mk.cu
)
set_property(TARGET baryon PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(baryon PRIVATE "-Wno-deprecated-gpu-targets" "--expt-relaxed-constexpr")
target_link_libraries(baryon PUBLIC CUDA::cudart)
