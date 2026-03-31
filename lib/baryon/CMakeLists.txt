set(baryon_sources
    diquark.cu
    baryon.cu
    baryon_general.cu
    baryon_sequential_i.cu
    baryon_sequential_j.cu
    baryon_sequential_n.cu
)

set_source_files_language(${baryon_sources})
add_library(baryon STATIC ${baryon_sources})
set_property(TARGET baryon PROPERTY POSITION_INDEPENDENT_CODE ON)
