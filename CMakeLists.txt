project(contract LANGUAGES C CXX CUDA)
cmake_minimum_required(VERSION 3.18)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "..." FORCE)
set(CMAKE_CXX_STANDARD 17)

set(GPU_ARCH 60 CACHE STRING "CUDA architectures (60 61 70 75 80 86 89 90)")
set_property(CACHE GPU_ARCH PROPERTY STRINGS "60;61;70;75;80;86;89;90")
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})

include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(lib)

install(
    TARGETS contract
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES
    DESTINATION include
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
