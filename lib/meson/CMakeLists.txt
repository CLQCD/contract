set(meson_sources
    meson.cu
    meson_all.cu
)

set_source_files_language(${meson_sources})
add_library(meson STATIC ${meson_sources})
set_property(TARGET meson PROPERTY POSITION_INDEPENDENT_CODE ON)
