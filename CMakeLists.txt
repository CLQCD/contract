project(contract LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.18)


set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
set(CMAKE_CXX_STANDARD 17)


set(GPU_TARGET cuda CACHE STRING "GPU target (cuda hip)")
set_property(CACHE GPU_TARGET PROPERTY STRINGS "cuda;hip")
if(${GPU_TARGET} STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GPU_ARCH 60 CACHE STRING "GPU architecture (60 61 70 75 80 86 89 90)")
    set_property(CACHE GPU_ARCH PROPERTY STRINGS "60;61;70;75;80;86;89;90")
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
elseif(${GPU_TARGET} STREQUAL "hip")
    enable_language(HIP)
    find_package(HIP REQUIRED)
    set(GPU_ARCH gfx906 CACHE STRING "GPU architecture (gfx906 gfx908 gfx90a gfx942 gfx950)")
    set_property(CACHE GPU_ARCH PROPERTY STRINGS "gfx906;gfx908;gfx90a;gfx942;gfx950")
    set(CMAKE_HIP_ARCHITECTURES ${GPU_ARCH})
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(lib)

install(
    TARGETS contract
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

add_subdirectory(test)
