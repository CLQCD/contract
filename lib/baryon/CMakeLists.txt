set(baryon_sources
    diquark.cu
    baryon.cu
    baryon_general.cu
    baryon_sequential_i.cu
    baryon_sequential_j.cu
    baryon_sequential_n.cu
    baryon_ik_jl_mn.cu
    baryon_ik_jn_ml.cu
    baryon_il_jk_mn.cu
    baryon_il_jn_mk.cu
    baryon_in_jk_ml.cu
    baryon_in_jl_mk.cu
)

set_source_files_language(${baryon_sources})
add_library(baryon STATIC ${baryon_sources})
set_property(TARGET baryon PROPERTY POSITION_INDEPENDENT_CODE ON)
