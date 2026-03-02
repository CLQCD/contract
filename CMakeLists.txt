cmake_minimum_required(VERSION 3.18)
project(contract LANGUAGES C CXX)


set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
set(CMAKE_CXX_STANDARD 17)


set(GPU_TARGET cuda CACHE STRING "GPU target (cuda hip sycl)")
set_property(CACHE GPU_TARGET PROPERTY STRINGS "cuda;hip;sycl")
if(${GPU_TARGET} STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(GPU_ARCH 60 CACHE STRING "GPU architecture (60 61 70 75 80 86 89 90)")
    set_property(CACHE GPU_ARCH PROPERTY STRINGS "60;61;70;75;80;86;89;90")
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

    add_compile_definitions(
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:GPU_TARGET_CUDA>
    )
    add_compile_options(
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Wno-deprecated-gpu-targets>
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>
    )
elseif(${GPU_TARGET} STREQUAL "hip")
    enable_language(HIP)
    find_package(HIP REQUIRED)
    set(GPU_ARCH gfx906 CACHE STRING "GPU architecture (gfx906 gfx908 gfx90a gfx942 gfx950)")
    set_property(CACHE GPU_ARCH PROPERTY STRINGS "gfx906;gfx908;gfx90a;gfx942;gfx950")
    set(CMAKE_HIP_ARCHITECTURES ${GPU_ARCH})

    add_compile_definitions(
        $<$<COMPILE_LANG_AND_ID:HIP,Clang>:GPU_TARGET_HIP>
    )
elseif(${GPU_TARGET} STREQUAL "sycl")
    add_compile_definitions(GPU_TARGET_SYCL)
    add_compile_options(-fsycl)
    add_link_options(-fsycl)
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
